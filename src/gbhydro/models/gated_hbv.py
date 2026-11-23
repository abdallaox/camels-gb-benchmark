"""RFL-HBV: HBV's four stores and flux paths, with every flux rule learned by a small gate.

Specification:

  * state: SP, SM, SUZ, SLZ, each split into n parallel units (mm); a store's level is the sum over units
  * seven gates, one per flux; each reads its own store total (scaled), the day's dynamic inputs and the static
    attributes, through  h = tanh(W1 x + b1)  (n hidden),  g = sigmoid(W2 h + b2)  (n fractions, one per unit).
    The snow gate reads inputs and attributes only and gives one fraction for the catchment.
  * fluxes in HBV order, each later tap acting on what the earlier one left; inflows shared equally over units:
        Snow = g_snow f(T) P -> SP;  Rain = P - Snow -> SM;  Melt_i = g_melt_i SP_i -> SM
        ET_i = min(g_et_i PET / n, SM_i) -> air
        Rech_i = g_rech_i (SM_i - ET_i) -> SUZ;  Perc_i = g_perc_i SUZ_i -> SLZ
        Fast_i = g_fast_i (SUZ_i - Perc_i) -> river;  Base_i = g_base_i SLZ_i -> river
    where f(T) = sigmoid((TT_max - T) / 0.5) is the RPL-HBV's rain/snow switch set at the warm end of the HBV
    calibration range (TT_max = 2 degC, decision of 20 Nov 2026): it is a ceiling, every rain/snow split an HBV
    with TT in [-2, 2] can produce lies under it, and the snow gate, which reads T, learns where in that range the
    catchment sits. Runs gatedc2_* used TT_max = 0 (15 Nov 2026), which cut off the warm half of the range and
    left the marginal snow catchments without a snowpack. ET tied to PET since 15 Nov 2026. Before that
    Snow = g_snow P and ET_i = g_et_i SM_i, kept under constrained=False, run gatedc_s0.
  * Q = sum Fast + sum Base; ET = sum ET; discharge is never an input; no routing, no hidden memory vector,
    no flux feedback, no learned spread, no HBV equations or initialisation, no weight constraints.
  * one capacity hyperparameter n (units per store = hidden neurons per gate)
  * unit output biases start ordered, unit 1 fastest to unit n slowest

Gates read the stores as they stand at the start of the day; taps are taken from those values and the day's
inflows are added afterwards (the store balances in the paper supplement). Every window starts from the same small
stores as the RPL-HBV (soil 30, upper 10, lower 20 mm, split equally over units; snow 0).
"""
import torch
import torch.nn as nn

STORES = ["SP", "SM", "SUZ", "SLZ"]
SCALE = {"SP": 100.0, "SM": 200.0, "SUZ": 50.0, "SLZ": 200.0}        # typical sizes used to scale gate inputs, mm
INITIAL = {"SP": 0.0, "SM": 30.0, "SUZ": 10.0, "SLZ": 20.0}            # whole-store totals at the start of a window
GATES = {"snow": None, "melt": "SP", "et": "SM", "rech": "SM", "perc": "SUZ", "fast": "SUZ", "base": "SLZ"}
BIAS_FAST, BIAS_SLOW = 0.0, -4.0     # output biases ordered from unit 1 (sigmoid 0.5) to unit n (sigmoid 0.018)
BIAS_SINGLE = -2.0                   # n = 1: one moderate start (sigmoid 0.12/d), the middle of the n-unit range (22 Nov)
SNOW_TT_MAX, SNOW_SPLIT_WIDTH = 2.0, 0.5  # degC; snow possible = sigmoid((TT_max - T) / width): the RPL-HBV switch at the
                                          # top of its TT range (-2 to 2), so the gate can reach any calibrated HBV split


class Gate(nn.Module):
    def __init__(self, n_in: int, n: int, n_out: int, ordered_bias: bool, depth: int = 1):
        super().__init__()
        self.l1, self.l2 = nn.Linear(n_in, n), nn.Linear(n, n_out)
        self.extra = nn.ModuleList([nn.Linear(n, n) for _ in range(depth - 1)])   # depth > 1: 22 Nov test only
        with torch.no_grad():
            if ordered_bias:
                if n_out == 1:
                    self.l2.bias.fill_(BIAS_SINGLE)
                else:
                    self.l2.bias.copy_(torch.linspace(BIAS_FAST, BIAS_SLOW, n_out))
            else:
                self.l2.bias.zero_()

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        for layer in self.extra:
            h = torch.tanh(layer(h))
        return torch.sigmoid(self.l2(h))


def gated_step(p, t, pet, s: dict, ctx: torch.Tensor, gates: dict, constrained: bool, snow_tt: float) -> tuple[dict, dict]:
    """One day. p, t, pet [B] raw precipitation (mm), temperature (degC) and PET (mm); s: stores [B, n];
    ctx = cat(x_d, x_s) [B, n_dyn + n_static]. Returns (new stores, fluxes).
    constrained: snow only below about snow_tt degC and ET a fraction of PET (capped at the unit's soil water)."""
    n = s["SM"].shape[1]
    totals = {k: s[k].sum(1, keepdim=True) / SCALE[k] for k in STORES}
    g = {name: gates[name](ctx if store is None else torch.cat([totals[store], ctx], 1)) for name, store in GATES.items()}
    if constrained:
        snow = g["snow"][:, 0] * torch.sigmoid((snow_tt - t) / SNOW_SPLIT_WIDTH) * p      # [B]
        et = torch.minimum(g["et"] * pet[:, None] / n, s["SM"])                           # [B, n], sum <= PET
    else:
        snow = g["snow"][:, 0] * p
        et = g["et"] * s["SM"]
    rain = p - snow
    melt = g["melt"] * s["SP"]                      # [B, n]
    rech = g["rech"] * (s["SM"] - et)
    perc = g["perc"] * s["SUZ"]
    fast = g["fast"] * (s["SUZ"] - perc)
    base = g["base"] * s["SLZ"]
    new = {"SP": s["SP"] - melt + snow[:, None] / n,
           "SM": s["SM"] - et - rech + (rain + melt.sum(1))[:, None] / n,
           "SUZ": s["SUZ"] - perc - fast + rech.sum(1, keepdim=True) / n,
           "SLZ": s["SLZ"] - base + perc.sum(1, keepdim=True) / n}
    flux = {"q": fast.sum(1) + base.sum(1), "fast": fast.sum(1), "base": base.sum(1), "et": et.sum(1),
            "melt": melt.sum(1), "rech": rech.sum(1), "perc": perc.sum(1), "snow": snow, "rain": rain}
    return new, flux


class GatedHBV(nn.Module):
    name = "gated"
    INITIAL = INITIAL                     # fixed cold-start stores; the trainer's drift penalty reads the total

    def __init__(self, n_dyn: int, n_static: int, n: int = 8, constrained: bool = True, snow_tt: float = SNOW_TT_MAX,
                 hidden: int | None = None, depth: int = 1, compile_step: bool = False):
        super().__init__()
        self.n, self.constrained, self.snow_tt = n, constrained, float(snow_tt)
        self.hidden = n if hidden is None else int(hidden)    # hidden neurons per gate; default n (the spec), 22 Nov test
        ctx = n_dyn + n_static
        self.gates = nn.ModuleDict({name: Gate(ctx + (0 if store is None else 1), self.hidden, 1 if store is None else n,
                                               ordered_bias=store is not None, depth=depth) for name, store in GATES.items()})
        self.step = torch.compile(gated_step) if compile_step else gated_step

    def forward(self, batch: dict, state: dict | None = None) -> dict:
        """state: unit stores at the start of the first day (dict of [B, n]); default = INITIAL split equally.
        The result carries 'state' = unit stores after the last day for chaining chunks."""
        p, t, pet, x_d, x_s = batch["p"], batch["t"], batch["pet"], batch["x_d"], batch["x_s"]
        B, L = p.shape
        s = state if state is not None else {k: torch.full((B, self.n), v / self.n, device=p.device, dtype=p.dtype)
                                             for k, v in INITIAL.items()}
        gates = dict(self.gates.items())
        stores, units, fluxes = {k: [] for k in STORES}, {k: [] for k in STORES}, None
        for i in range(L):
            ctx = torch.cat([x_d[:, i], x_s], 1)
            s, f = self.step(p[:, i], t[:, i], pet[:, i], s, ctx, gates, self.constrained, self.snow_tt)
            for k in STORES:
                stores[k].append(s[k].sum(1)); units[k].append(s[k])
            if fluxes is None:
                fluxes = {k: [] for k in f}
            for k, v in f.items():
                fluxes[k].append(v)
        stores = {k: torch.stack(v, 1) for k, v in stores.items()}         # [B, L] store totals, end of day
        units = {k: torch.stack(v, 1) for k, v in units.items()}           # [B, L, n]
        fluxes = {k: torch.stack(v, 1) for k, v in fluxes.items()}         # [B, L]
        return {"y_hat": fluxes["q"][:, -1], "fluxes": fluxes, "stores": stores, "units": units, "state": s}
