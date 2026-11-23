"""Table of the 14 HBV parameters and the ranges used by the regional and the LC-HBV, read from
gbhydro.models.hbv.RANGES so the paper never disagrees with the code. Writes paper/tables/table4_hbv_params.tex.
Usage: uv run python scripts/tables/make_param_table.py"""
from gbhydro.data import ROOT
from gbhydro.models.hbv import PARAMS, RANGES

DESC = {"TT": ("threshold temperature for snow", "$^{\\circ}$C"), "CFMAX": ("degree-day melt factor", "mm $^{\\circ}$C$^{-1}$ d$^{-1}$"),
        "SFCF": ("snowfall correction factor", "--"), "CFR": ("refreezing coefficient", "--"),
        "CWH": ("water holding capacity of the snowpack", "--"), "FC": ("maximum soil moisture storage", "mm"),
        "LP": ("soil moisture threshold for full evapotranspiration, fraction of FC", "--"),
        "BETA": ("shape of the recharge function", "--"), "K0": ("recession coefficient, upper store above UZL", "d$^{-1}$"),
        "K1": ("recession coefficient, upper store", "d$^{-1}$"), "K2": ("recession coefficient, lower store", "d$^{-1}$"),
        "UZL": ("threshold for the fast upper-store outflow", "mm"), "PERC": ("maximum percolation to the lower store", "mm d$^{-1}$"),
        "MAXBAS": ("length of the triangular routing function", "d")}


def fmt(v):
    return f"{v:g}"


body = "parameter & description & unit & range \\\\\n\\midrule\n"
for p in PARAMS:
    lo, hi = RANGES[p]
    body += f"{p} & {DESC[p][0]} & {DESC[p][1]} & {fmt(lo)} to {fmt(hi)} \\\\\n"
tex = ("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
       "\\caption{The 14 HBV parameters and their ranges, identical for every catchment and shared by the RPL-HBV "
       "(sigmoid outputs rescaled to the range) and the per-catchment LC-HBV (search bounds). The ranges follow "
       "the first author's earlier Python implementation of HBV (HBV\\_Lab), with FC, PERC and K2 widened towards those of \\citet{seibert2012} for the "
       "groundwater catchments.}\n\\label{tab:hbv_params}\n"
       f"\\begin{{tabular}}{{llll}}\n\\toprule\n{body}\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
out = ROOT / "paper/tables/table4_hbv_params.tex"
out.write_text(tex, encoding="utf-8"); print("wrote", out)
