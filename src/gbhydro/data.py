"""Loaders for CAMELS-GB (Coxon et al., 2020, ESSD 12:2459)."""
from functools import reduce
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "camels_gb" / "data"
NH_DATA_DIR = ROOT / "data" / "nh"  # NeuralHydrology layout: attributes/ + timeseries/
ATTR_GROUPS = [
    "topographic", "climatic", "hydrologic", "hydrometry",
    "landcover", "soil", "hydrogeology", "humaninfluence",
]


def load_attributes(groups=ATTR_GROUPS) -> pd.DataFrame:
    """All static catchment attributes, one row per gauge_id."""
    frames = [
        pd.read_csv(DATA_DIR / f"CAMELS_GB_{g}_attributes.csv", index_col="gauge_id")
        for g in groups
    ]
    return reduce(lambda a, b: a.join(b), frames)


def load_timeseries(gauge_id: int) -> pd.DataFrame:
    """Daily hydromet series (1970-10-01 .. 2015-09-30) for one gauge."""
    path = next((DATA_DIR / "timeseries").glob(f"CAMELS_GB_hydromet_timeseries_{gauge_id}_*.csv"))
    return pd.read_csv(path, index_col="date", parse_dates=True)


def load_outline_rings(min_area_m2: float = 5e8) -> list:
    """Light land outline for the maps: exterior rings (km, British National Grid) of the union of all catchment
    polygons, gaps closed with a 5 km buffer in and out. The union takes minutes, so it is cached in
    data/processed/outline_rings.npz after the first call."""
    import numpy as np

    cache = ROOT / "data" / "processed" / "outline_rings.npz"
    if cache.exists():
        z = np.load(cache)
        return [z[k] for k in sorted(z.files, key=lambda k: int(k[1:]))]
    g = load_boundaries().to_crs(27700).geometry.simplify(200).buffer(0)     # simplify first: the union then takes seconds
    land = g.union_all().buffer(5000).buffer(-5000).simplify(500)
    rings = [np.asarray(g.exterior.xy) / 1e3 for g in getattr(land, "geoms", [land]) if g.area > min_area_m2]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"r{i}": r for i, r in enumerate(rings)})
    return rings


def load_boundaries():
    import geopandas as gpd

    gdf = gpd.read_file(DATA_DIR / "boundaries" / "CAMELS_GB_catchment_boundaries.shp")
    gdf["ID_STRING"] = gdf["ID_STRING"].astype(int)
    return gdf.rename(columns={"ID_STRING": "gauge_id"}).set_index("gauge_id")
