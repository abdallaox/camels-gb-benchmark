"""Maps of store realism for the three HBV-structured models, same style as figures/spatial_maps.png.
Top row: anomaly correlation between the model soil store and ERA5-Land 0-100 cm soil water, all 669 catchments.
Bottom row: correlation between the model snowpack and ERA5-Land snow water equivalent in the catchments whose median
annual peak SWE exceeds 20 mm (the others are drawn as small grey dots). Seed median per catchment (restart median
for the LC-HBV). Usage: uv run python scripts/figures/plot_store_maps.py -> figures/store_maps.png"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from gbhydro.data import ROOT, load_attributes, load_outline_rings

INK, MUTED = "#222222", "#555555"
MODELS = {"RPL-HBV": [f"hbvc_b128_s{s}" for s in range(5)], "RFL-HBV (n = 16)": [f"gatedc3_n16_s{s}" for s in range(5)],
          "LC-HBV": [f"hbv_local_r{s}" for s in range(5)]}
ROWS = {"soil_corr_anom": ("soil realism (compared with ERA5-Land)", 0.5, 0.9, False, "YlOrBr"),   # soil: tan to dark brown
        "snow_corr": ("snow realism (compared with ERA5-Land)", 0.2, 0.9, True, "Blues")}           # snow: pale to dark blue


def ramp(name):
    """Sequential ramp without its palest fifth, so the weakest catchments stay visible on white; darker is better."""
    base = plt.get_cmap(name)
    return colors.LinearSegmentedColormap.from_list(name + "_trim", base(np.linspace(0.2, 1.0, 256)))
SNOWY_PEAK_MM = 20
TAB = pd.read_csv(ROOT / "results/overview/table7_store_realism.csv", index_col=0)   # medians as in Table 3
TABCOL = {"soil_corr_anom": "soil_anom_med", "snow_corr": "snow_corr_med"}

att = load_attributes()
rings = load_outline_rings()
real = {name: pd.concat([pd.read_csv(ROOT / f"results/{r}/store_realism.csv", index_col="basin") for r in runs]).groupby("basin").median(numeric_only=True)
        for name, runs in MODELS.items()}

fig, axes = plt.subplots(2, 3, figsize=(11, 10.4))
fig.subplots_adjust(left=0.01, right=0.88, top=0.95, bottom=0.01, wspace=0.02, hspace=0.08)
for (col, (label, vmin, vmax, snowy_only, cname)), row_axes in zip(ROWS.items(), axes):
    norm = colors.Normalize(vmin, vmax); cmap = ramp(cname)
    for ax, (name, d) in zip(row_axes, real.items()):
        v = d[col]
        if snowy_only:
            v = v[d.era_peak_swe_median_mm > SNOWY_PEAK_MM]
        a = att.loc[v.index]
        for rx, ry in rings:
            ax.plot(rx, ry, color="#cccccc", lw=0.6, zorder=1)
        if snowy_only:
            rest = att.loc[d.index.difference(v.index)]
            ax.scatter(rest.gauge_easting / 1e3, rest.gauge_northing / 1e3, color="#e4e4e4", s=3, lw=0, zorder=1.5)
        sc = ax.scatter(a.gauge_easting / 1e3, a.gauge_northing / 1e3, c=v, cmap=cmap, norm=norm, s=14 if snowy_only else 11, lw=0, zorder=2)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{name}\nmedian {TAB.loc[name, TABCOL[col]]:.2f}" + (f" ({len(v)} catchments)" if snowy_only else ""), loc="left", fontsize=10, color=INK)
    cb = fig.colorbar(sc, ax=list(row_axes), shrink=0.75, pad=0.01, extend="min")
    cb.set_label(label, fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK)
fig.savefig(ROOT / "figures/store_maps.png", dpi=150, bbox_inches="tight"); print("wrote figures/store_maps.png")
