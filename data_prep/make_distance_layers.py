"""Creates dist_road.tif and dist_grid.tif (distance in km) on the SAME grid as real_data/ghi.tif.

Setup:   pip install osmnx geopandas rasterio scipy shapely
Run:     python data_prep/make_distance_layers.py
Paths:   resolved from PROJECT_ROOT (env var, default = current directory), so the script works
         from any working directory. Expects <PROJECT_ROOT>/real_data/ghi.tif.

Design notes
- The query area is the bounding box of ghi.tif plus BUFFER_KM, NOT the "Jharkhand, India" place polygon.
  A line or road just outside the border still serves a cell just inside it, so clipping to the state
  overstates distance near the edges.
- The box is split into TILE_DEG tiles, so one Overpass timeout does not lose the whole download.
  Each tile is retried; osmnx caches responses, so a re-run only fetches what is missing.
- Distances are computed on the raster's projected grid (needs a metric CRS such as EPSG:32645).

"""
import os
import time
from pathlib import Path

import numpy as np
import osmnx as ox
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
REF = ROOT / "real_data" / "ghi.tif"       # grid reference: outputs copy its grid exactly
OUT_DIR = ROOT / "real_data"

BUFFER_KM = 50        # extra margin around the raster so border cells see features just outside
TILE_DEG = 1.0        # query tile size in degrees (about 100 km)
RETRIES = 3

QUERIES = {
    "dist_road": {"highway": ["motorway", "trunk", "primary", "secondary"]},
    "dist_grid": {"power": ["line", "substation"]},
}

ox.settings.requests_timeout = 300          # seconds per Overpass request
ox.settings.max_query_area_size = 2e10      # m2; keeps one 1-degree tile as one request
ox.settings.use_cache = True                # re-runs reuse downloaded tiles
ox.settings.log_console = False


def query_bounds_wgs84(src, buffer_km):
    """Raster bounds + buffer, as (west, south, east, north) in EPSG:4326."""
    if src.crs is None or not src.crs.is_projected:
        raise ValueError(f"{REF} needs a projected (metric) CRS, got {src.crs}")
    b = src.bounds
    pad = buffer_km * 1000.0
    west, south, east, north = transform_bounds(
        src.crs, "EPSG:4326", b.left - pad, b.bottom - pad, b.right + pad, b.top + pad, densify_pts=21)
    return west, south, east, north


def tiles(west, south, east, north, step):
    xs = np.arange(west, east, step)
    ys = np.arange(south, north, step)
    for x in xs:
        for y in ys:
            yield box(x, y, min(x + step, east), min(y + step, north))


def fetch_tile(poly, tags):
    """Return a GeoDataFrame for one tile, or None if the tile has no matching features."""
    for attempt in range(1, RETRIES + 1):
        try:
            return ox.features_from_polygon(poly, tags)
        except Exception as e:
            if "InsufficientResponse" in type(e).__name__:   # empty tile is normal, not an error
                return None
            print(f"    attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}")
            if attempt == RETRIES:
                raise
            time.sleep(10 * attempt)


def burn_features(name, tags, bounds, shape, transform, crs):
    """Rasterize every OSM feature of this query onto the reference grid (1 = feature, 0 = none)."""
    burned = np.zeros(shape, dtype="uint8")
    n_features = 0
    tile_list = list(tiles(*bounds, TILE_DEG))
    for i, poly in enumerate(tile_list, 1):
        gdf = fetch_tile(poly, tags)
        if gdf is None or gdf.empty:
            continue
        geoms = [g for g in gdf.to_crs(crs).geometry if g is not None and not g.is_empty]
        if not geoms:
            continue
        n_features += len(geoms)
        burned |= rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform,
                            all_touched=True, fill=0, dtype="uint8")
        print(f"  [{name}] tile {i}/{len(tile_list)}: {len(geoms)} features")
    return burned, n_features


def main():
    if not REF.exists():
        raise FileNotFoundError(f"Reference raster not found: {REF}  (set PROJECT_ROOT or run from the project root)")

    with rasterio.open(REF) as src:
        shape, transform, crs, profile = (src.height, src.width), src.transform, src.crs, src.profile
        bounds = query_bounds_wgs84(src, BUFFER_KM)
    cell_y, cell_x = abs(transform.e), abs(transform.a)          # metres (CRS is metric)
    print(f"Reference grid: {shape[1]} x {shape[0]} cells, CRS {crs}, pixel {cell_x:g} x {cell_y:g} m")
    print(f"Query bbox (W,S,E,N): {tuple(round(v, 3) for v in bounds)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tags in QUERIES.items():
        print(f"Downloading {name} features from OpenStreetMap ...")
        burned, n = burn_features(name, tags, bounds, shape, transform, crs)
        if burned.sum() == 0:
            raise RuntimeError(f"{name}: no OSM features landed on the grid; distances would be meaningless. "
                               "Check connectivity, the tags, and that ghi.tif covers the intended area.")
        print(f"  {n} features, {int(burned.sum())} burned cells")

        # Distance in metres to the nearest burned cell, using the true cell size on each axis.
        dist_km = distance_transform_edt(burned == 0, sampling=(cell_y, cell_x)) / 1000.0

        out = dict(profile, dtype="float32", count=1, nodata=None)
        out.pop("compress", None)
        path = OUT_DIR / f"{name}.tif"
        with rasterio.open(path, "w", **out) as dst:
            dst.write(dist_km.astype("float32"), 1)

        # Verify the written file really sits on the reference grid.
        with rasterio.open(path) as chk, rasterio.open(REF) as ref:
            assert (chk.crs, chk.transform, chk.width, chk.height) == (ref.crs, ref.transform, ref.width, ref.height), \
                f"{path} is not on the reference grid"
        p = np.percentile(dist_km, [50, 90, 99])
        print(f"  saved {path}  max {dist_km.max():.1f} km | p50 {p[0]:.1f} p90 {p[1]:.1f} p99 {p[2]:.1f} | grid check OK")


if __name__ == "__main__":
    main()
