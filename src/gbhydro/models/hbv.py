"""Regional differentiable HBV: static attributes -> small MLP -> 14 HBV parameters, constant in time.

The water-balance core mirrors HBV_Lab (github.com/abdallaox/HBV_python_implementation): same
14 parameters, same order of operations in the snow, soil and response routines, same triangular
MAXBAS routing and the same fixed initial stores. Differences, all for differentiability:

  * rain/snow split is a sigmoid over about 2 degC around TT (HBV-light's TTI option), not a step
  * MAXBAS is continuous; the kernel is a triangle of base MAXBAS integrated over whole days
  * upper-zone outflow Q0 + Q1 is capped at the available store instead of clamping the store at 0

Every window starts from the HBV_Lab initial stores; the 364 days before the target are the warm-up.
Parameter ranges are HBV_Lab's except FC, PERC and K2, widened to the Seibert values (FC 50-500 mm,
PERC 0.5-10 mm/d, K2 0.0005-0.1 /d) after the first run (14 Nov 2026) pinned 41% of catchments at FC = 300,
20% at PERC = 3 and a third at K2's bounds (runs hbv_narrow_s*). Ranges are hard bounds (sigmoid rescaling),
identical for every catchment.
"""
import torch
import torch.nn as nn

PARAMS = ["TT", "CFMAX", "SFCF", "CFR", "CWH", "FC", "LP", "BETA", "K0", "K1", "K2", "UZL", "PERC", "MAXBAS"]
RANGES = {"TT": (-2.0, 2.0), "CFMAX": (1.0, 6.0), "SFCF": (0.5, 1.5), "CFR": (0.0, 0.2), "CWH": (0.0, 0.2),
          "FC": (50.0, 500.0), "LP": (0.3, 1.0), "BETA": (1.0, 5.0),
          "K0": (0.1, 0.9), "K1": (0.05, 0.5), "K2": (0.0005, 0.1), "UZL": (5.0, 50.0), "PERC": (0.5, 10.0),
          "MAXBAS": (1.0, 10.0)}
INITIAL = {"snowpack": 0.0, "liquid_water": 0.0, "soil_moisture": 30.0, "upper_storage": 10.0, "lower_storage": 20.0}
STORES = list(INITIAL)
SNOW_SPLIT_WIDTH = 0.5      # degC; snow fraction = sigmoid((TT - T) / 0.5): 88% snow 1 degC below TT
MAX_ROUTING_DAYS = 10       # = upper bound of MAXBAS, so the kernel always sums to one


def hbv_step(p, t, pet, s: dict, k: dict) -> tuple[dict, dict]:
    """One day of HBV for a batch. All tensors [B]. Returns (new stores, fluxes)."""
    sp, lw, sm, suz, slz = (s[n] for n in STORES)
    # --- snow routine (HBV_Lab snow.py) ---
    f_snow = torch.sigmoid((k["TT"] - t) / SNOW_SPLIT_WIDTH)
    snowfall = p * f_snow * k["SFCF"]
    rain = p * (1 - f_snow)
    sp = sp + snowfall
    refreeze = torch.minimum(torch.relu(k["CFR"] * k["CFMAX"] * (k["TT"] - t)), lw)
    melt = torch.minimum(torch.relu(k["CFMAX"] * (t - k["TT"])), sp)
    sp = sp + refreeze - melt
    lw = lw - refreeze + melt + rain
    release = torch.relu(lw - k["CWH"] * sp)             # liquid water above the holding capacity
    lw = lw - release
    # --- soil routine (HBV_Lab soil.py): ET first, recharge from the pre-update soil moisture ---
    et = torch.minimum(pet * torch.clamp(sm / (k["LP"] * k["FC"]), max=1.0), sm)
    recharge = release * torch.clamp(sm / k["FC"], 0.0, 1.0) ** k["BETA"]
    sm = sm + release - et - recharge
    overflow = torch.relu(sm - k["FC"])
    sm = sm - overflow
    # --- response routine (HBV_Lab response.py) ---
    suz = suz + recharge + overflow
    perc = torch.minimum(k["PERC"], suz)
    suz, slz = suz - perc, slz + perc
    q0 = k["K0"] * torch.relu(suz - k["UZL"])
    q1 = k["K1"] * suz
    cap = torch.clamp(suz / (q0 + q1 + 1e-9), max=1.0)   # never release more than the store holds
    q0, q1 = q0 * cap, q1 * cap
    suz = suz - q0 - q1
    q2 = k["K2"] * slz
    slz = slz - q2
    new = {"snowpack": sp, "liquid_water": lw, "soil_moisture": sm, "upper_storage": suz, "lower_storage": slz}
    flux = {"q": q0 + q1 + q2, "q0": q0, "q1": q1, "q2": q2, "et": et, "recharge": recharge + overflow,
            "melt": melt, "p_eff": rain + snowfall, "perc": perc}
    return new, flux


def routing_weights(maxbas: torch.Tensor, n: int = MAX_ROUTING_DAYS) -> torch.Tensor:
    """Triangular unit hydrograph of base `maxbas` (continuous, [B]) integrated over whole days -> [B, n].
    Weight k applies to the runoff generated k days ago. Differentiable in maxbas; rows sum to one."""
    m = maxbas[:, None]
    edges = torch.arange(n + 1, device=maxbas.device, dtype=maxbas.dtype)[None, :].clamp(max=m)   # [B, n+1]
    cdf = torch.where(edges <= m / 2, 2 * edges ** 2 / m ** 2, 1 - 2 * (m - edges) ** 2 / m ** 2)
    return cdf[:, 1:] - cdf[:, :-1]


def route(q: torch.Tensor, maxbas: torch.Tensor) -> torch.Tensor:
    """Route generated runoff q [B, L] with per-basin triangular kernels -> [B, L] (zero inflow before day 0)."""
    w = routing_weights(maxbas)                                       # [B, n]
    n = w.shape[1]
    qp = torch.nn.functional.pad(q, (n - 1, 0))                       # [B, L + n - 1]
    win = qp.unfold(1, n, 1)                                          # [B, L, n]: win[:, t, j] = q[t - (n-1) + j]
    return torch.einsum("blj,bj->bl", win, w.flip(1))


class ParameterNet(nn.Module):
    """Standardised static attributes -> HBV parameters inside RANGES. Zero final bias = mid-range start."""

    def __init__(self, n_static: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        layers, d = [], n_static
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, len(PARAMS))
        nn.init.zeros_(self.head.bias)
        self.register_buffer("lo", torch.tensor([RANGES[p][0] for p in PARAMS]))
        self.register_buffer("hi", torch.tensor([RANGES[p][1] for p in PARAMS]))

    def forward(self, x_s: torch.Tensor) -> dict:
        z = self.lo + (self.hi - self.lo) * torch.sigmoid(self.head(self.body(x_s)))     # [B, 14]
        return {p: z[:, i] for i, p in enumerate(PARAMS)}


class RegionalHBV(nn.Module):
    name = "hbv"
    INITIAL = INITIAL                     # fixed cold-start stores; the trainer's drift penalty reads the total

    def __init__(self, n_static: int, hidden: int = 64, n_layers: int = 2, compile_step: bool = False):
        super().__init__()
        self.pnet = ParameterNet(n_static, hidden, n_layers)
        self.step = torch.compile(hbv_step) if compile_step else hbv_step

    def parameters_for(self, batch: dict) -> dict:
        """HBV parameters for the basins of a batch, dict of [B] tensors (regional: from the attributes)."""
        return self.pnet(batch["x_s"])

    def forward(self, batch: dict, state: dict | None = None) -> dict:
        """state: stores at the start of the first day (dict of [B]); default = INITIAL. The result carries
        'state' = stores after the last day, so continuous runs can be chained chunk by chunk."""
        p, t, pet = batch["p"], batch["t"], batch["pet"]              # [B, L] raw mm/d, degC, mm/d
        B, L = p.shape
        k = self.parameters_for(batch)
        s = state if state is not None else {n: torch.full((B,), v, device=p.device, dtype=p.dtype) for n, v in INITIAL.items()}
        stores, fluxes = {n: [] for n in STORES}, None
        for i in range(L):
            s, f = self.step(p[:, i], t[:, i], pet[:, i], s, k)
            for n in STORES:
                stores[n].append(s[n])
            if fluxes is None:
                fluxes = {n: [] for n in f}
            for n, v in f.items():
                fluxes[n].append(v)
        stores = {n: torch.stack(v, 1) for n, v in stores.items()}    # [B, L], value at the end of each day
        fluxes = {n: torch.stack(v, 1) for n, v in fluxes.items()}
        q_routed = route(fluxes["q"], k["MAXBAS"])
        return {"y_hat": q_routed[:, -1], "q_routed": q_routed, "fluxes": fluxes, "stores": stores, "params": k, "state": s}

    ranges = {k: list(v) for k, v in RANGES.items()}      # saved with every run (ranges may change between runs)

    def basin_parameters(self, x_s: torch.Tensor) -> dict:
        with torch.no_grad():
            return {p: v.cpu().numpy() for p, v in self.pnet(x_s).items()}


class LocalHBV(RegionalHBV):
    """Per-catchment HBV: the same core, ranges and initial stores as RegionalHBV, but 14 free parameters per basin
    instead of the attribute network. Parameters are held in unit space u in [0, 1] (rescaled to RANGES) and set by
    the DDS calibration in scripts/training/calibrate_hbv.py, not by gradient descent. Batches must carry the basin row index
    ('rows', or 'basin' from Study.window). This is the conventional, gauged-only baseline of the benchmark: it
    cannot predict a catchment it was not calibrated on."""
    name = "hbv_local"

    def __init__(self, n_basins: int, compile_step: bool = False):
        nn.Module.__init__(self)
        self.u = nn.Parameter(torch.full((n_basins, len(PARAMS)), 0.5), requires_grad=False)
        self.register_buffer("lo", torch.tensor([RANGES[p][0] for p in PARAMS]))
        self.register_buffer("hi", torch.tensor([RANGES[p][1] for p in PARAMS]))
        self.step = torch.compile(hbv_step) if compile_step else hbv_step

    def params_from_unit(self, u: torch.Tensor) -> dict:
        z = self.lo + (self.hi - self.lo) * u.clamp(0.0, 1.0)
        return {p: z[:, i] for i, p in enumerate(PARAMS)}

    def parameters_for(self, batch: dict) -> dict:
        rows = batch["rows"] if "rows" in batch else batch["basin"]
        return self.params_from_unit(self.u[rows])

    def basin_parameters(self, rows) -> dict:
        if torch.is_tensor(rows) and rows.is_floating_point():
            raise TypeError("LocalHBV.basin_parameters takes basin row indices, not attributes")
        with torch.no_grad():
            r = torch.as_tensor(rows, device=self.u.device)
            return {p: v.cpu().numpy() for p, v in self.params_from_unit(self.u[r]).items()}
