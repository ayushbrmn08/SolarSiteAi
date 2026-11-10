"""Data layer.

Real data:      load_real_layers, load_plants, plants_to_grid, farm_mask   (need rasterio)
Synthetic demo: make_synthetic_layers, synthetic_farm_locations           (demonstrates the CODE, not real findings)
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt

LAYER_NAMES = ["ghi", "temp", "elevation", "slope", "lulc", "ndvi",
               "dist_road", "dist_grid", "flood", "protected", "income"]   # model features
REAL_LAYERS = LAYER_NAMES + ["inside"]                                      # 'inside' = state-boundary mask
PLANT_COLUMNS = ["name", "district", "lat", "lon", "capacity_mw", "annual_generation_gwh"]

def load_layers_from_geotiffs(paths: dict):
    """paths: {layer_name: 'file.tif'}; all rasters must share one grid. Returns dict of 2-D arrays."""
    import rasterio  # optional dependency
    out = {}
    for k, p in paths.items():
        with rasterio.open(p) as src:
            out[k] = src.read(1).astype(float)
    return out

def _field(rng, shape, sigma):
    f = gaussian_filter(rng.standard_normal(shape), sigma)
    return (f - f.min()) / (f.max() - f.min())

def make_synthetic_layers(cfg):
    rng = np.random.default_rng(cfg.seed)
    r, c = cfg.shape
    yy = np.linspace(0, 1, r)[:, None] * np.ones((1, c))
    elevation = 100 + 900 * _field(rng, cfg.shape, 18)
    gy, gx = np.gradient(elevation, cfg.cell_size_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    ghi = 1500 + 600 * (1 - yy) * 0.6 + 600 * _field(rng, cfg.shape, 25) * 0.4  # kWh/m2/yr
    temp = 32 - 8 * (elevation - 100) / 900 + 2 * _field(rng, cfg.shape, 15)
    lulc = np.digitize(_field(rng, cfg.shape, 10), [0.08, 0.30, 0.62, 0.90])  # 0 water,1 forest,2 barren,3 crop,4 built
    ndvi = np.clip(0.15 + 0.6 * (lulc == 1) + 0.3 * (lulc == 3) + 0.1 * rng.standard_normal(cfg.shape), -0.1, 0.9)
    roads = _field(rng, cfg.shape, 3) > 0.93
    grid = _field(rng, cfg.shape, 2) > 0.965
    dist_road = distance_transform_edt(~roads) * cfg.cell_size_m / 1000   # km
    dist_grid = distance_transform_edt(~grid) * cfg.cell_size_m / 1000    # km
    flood = np.clip(_field(rng, cfg.shape, 12) * 1.3 - 0.15 - 0.3 * (elevation - 100) / 900, 0, 1)
    protected = (_field(rng, cfg.shape, 20) > 0.88).astype(float)
    income = _field(rng, cfg.shape, 22)
    return dict(ghi=ghi, temp=temp, elevation=elevation, slope=slope, lulc=lulc.astype(float),
                ndvi=ndvi, dist_road=dist_road, dist_grid=dist_grid, flood=flood,
                protected=protected, income=income)

def exclusion_mask(layers, cfg):
    """True = cell is allowed. Thresholds come from Config and must be justified in the report."""
    ok = (layers["slope"] <= cfg.max_slope_deg) & (layers["flood"] <= cfg.max_flood_risk)
    ok &= layers["protected"] == 0
    ok &= ~np.isin(layers["lulc"], cfg.excluded_lulc)
    return ok

def synthetic_farm_locations(layers, allowed, cfg, n=250):
    """Stand-in for an existing-plant inventory: farms drawn from a hidden 'true' suitability."""
    rng = np.random.default_rng(cfg.seed + 1)
    z = (0.9 * (layers["ghi"] - 1500) / 600 - 0.12 * layers["slope"]
         - 0.25 * layers["dist_grid"] - 0.15 * layers["dist_road"] + 0.15 * layers["income"])
    p = np.exp(2.0 * z) * allowed
    p = (p / p.sum()).ravel()
    idx = rng.choice(p.size, size=n, replace=False, p=p)
    y = np.zeros(p.size, dtype=int); y[idx] = 1
    return y.reshape(cfg.shape)

def stack_features(layers):
    return np.stack([layers[k] for k in LAYER_NAMES], axis=-1)   # (r, c, n_features)


# ---------------------------------------------------------------- real data helpers
def load_real_layers(cfg):
    """Load every real raster from cfg.data_dir and set cfg.shape / cfg.cell_size_m from the grid."""
    import rasterio
    layers = load_layers_from_geotiffs({n: f"{cfg.data_dir}/{n}.tif" for n in REAL_LAYERS})
    cfg.shape = layers["ghi"].shape
    with rasterio.open(f"{cfg.data_dir}/ghi.tif") as src:
        cfg.cell_size_m = float(src.res[0])
    return layers

def load_plants(path):
    """Read the plant CSV and keep only rows that have every required value."""
    plants = pd.read_csv(path)
    missing = [c for c in PLANT_COLUMNS if c not in plants.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return plants.dropna(subset=["lat", "lon", "capacity_mw", "annual_generation_gwh"]).reset_index(drop=True)

def plants_to_grid(plants, ref_raster):
    """Add raster_row / raster_col for each plant; plants outside the raster are dropped."""
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(ref_raster) as src:
        xs, ys = transform("EPSG:4326", src.crs, plants["lon"].tolist(), plants["lat"].tolist())
        cells = [src.index(x, y) for x, y in zip(xs, ys)]
        height, width = src.height, src.width
    out = plants.copy()
    out["raster_row"] = [r for r, _ in cells]
    out["raster_col"] = [c for _, c in cells]
    inside = out["raster_row"].between(0, height - 1) & out["raster_col"].between(0, width - 1)
    return out[inside].reset_index(drop=True)

def farm_mask(plants_on_grid, shape):
    """1 in every cell that holds at least one plant, else 0."""
    farms = np.zeros(shape, dtype=np.uint8)
    farms[plants_on_grid["raster_row"].to_numpy(), plants_on_grid["raster_col"].to_numpy()] = 1
    return farms
