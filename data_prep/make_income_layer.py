"""Turn the district income CSV into income.tif on the SAME grid as real_data/ghi.tif.

Usage (from the project root):
    python data_prep/make_income_layer.py path/to/districts.geojson   # recommended: real district polygons
    python data_prep/make_income_layer.py                             # fallback: nearest district centroid (crude)

Needs: pip install rasterio geopandas scipy pandas numpy
Input : real_data/jharkhand_district_income_24.csv  (columns: district, per_capita_income_inr, centroid_lat, centroid_lon)
Output: real_data/income.tif      income scaled 0-1 across the 24 districts (what run_stage1.py expects)
        real_data/income_inr.tif  raw per-capita income in INR, for reference
An existing income.tif is OVERWRITTEN (a note is printed). Copy your VIIRS night-lights income.tif elsewhere first if you want to keep it.

Every cell in a district gets that district's value, so the layer is blocky by construction.
"""
import difflib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
REF = ROOT / "real_data" / "ghi.tif"
CSV = ROOT / "real_data" / "jharkhand_district_income_24.csv"
OUT = ROOT / "real_data"

# Boundary-file columns. The first district column found is used (geoBoundaries: shapeName, GADM: NAME_2).
# The state filter applies only if the file has STATE_COL (GADM does, geoBoundaries does not).
DISTRICT_COLS = ("shapeName", "NAME_2", "DISTRICT", "dt_name")
STATE_COL, STATE_VALUE = "NAME_1", "Jharkhand"
MAX_FILL_CELLS = 3          # fill border slivers (boundary files disagree slightly) up to this many cells


def norm(s):
    """Lowercase letters only, so 'Saraikela-Kharsawan' and 'Saraikela Kharsawan' compare equal."""
    return re.sub(r"[^a-z]", "", str(s).lower().replace("district", ""))


def aliases(name):
    """'East Singhbhum (Purbi Singhbhum)' -> {'eastsinghbhum', 'purbisinghbhum'}"""
    parts = [p.strip() for p in re.split(r"[()]", name) if p.strip()]
    return {norm(p) for p in parts}


def match_districts(csv_names, poly_names, cutoff=0.85, margin=0.03):
    """Map each CSV district to one boundary-file district. Exact (normalised) match first, fuzzy for the rest.
    Raises if a district is unmatched, ambiguous, or two CSV rows claim the same polygon."""
    poly_norm = {norm(p): p for p in poly_names}
    mapping = {}
    for name in csv_names:
        hit = [poly_norm[a] for a in aliases(name) if a in poly_norm]
        if hit:
            mapping[name] = hit[0]
    for name in (n for n in csv_names if n not in mapping):
        scores = {}
        for pn, p in poly_norm.items():
            scores[p] = max(difflib.SequenceMatcher(None, a, pn).ratio() for a in aliases(name))
        ranked = sorted(scores.items(), key=lambda t: -t[1])
        if not ranked or ranked[0][1] < cutoff:
            raise ValueError(f"No boundary match for '{name}' (best: {ranked[:2]}). Add a spelling fix or check the file.")
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
            raise ValueError(f"Ambiguous match for '{name}': {ranked[:2]}. Resolve by hand.")
        mapping[name] = ranked[0][0]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Two CSV districts map to the same polygon: {mapping}")
    return mapping


def fill_gaps(idx, max_cells):
    """Give NaN cells within max_cells of a district the nearest district id; leave farther cells NaN."""
    nan = np.isnan(idx)
    if nan.all():
        raise RuntimeError("No cell received a district id: the polygons do not overlap the raster grid.")
    if not nan.any():
        return idx
    dist, (ri, ci) = distance_transform_edt(nan, return_indices=True)
    fill = nan & (dist <= max_cells)
    out = idx.copy()
    out[fill] = idx[ri[fill], ci[fill]]
    return out


def district_ids_from_polygons(boundary_path, df, shape, transform, crs):
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import array_bounds
    gdf = gpd.read_file(boundary_path)
    print(f"Boundary file columns: {list(gdf.columns)}")
    DISTRICT_COL = next((c for c in DISTRICT_COLS if c in gdf.columns), None)
    if DISTRICT_COL is None:
        raise KeyError(f"None of {DISTRICT_COLS} found in the boundary file; add its district-name column to DISTRICT_COLS.")
    if STATE_COL in gdf.columns:
        gdf = gdf[gdf[STATE_COL].astype(str).str.lower() == STATE_VALUE.lower()]
    gdf = gdf.to_crs(crs)
    # Keep only districts that touch the raster. Needed for all-India files with no state column: it avoids
    # matching Jharkhand names against ~700 districts and picking a same-name district elsewhere.
    west, south, east, north = array_bounds(shape[0], shape[1], transform)
    gdf = gdf.cx[west:east, south:north]
    print(f"Using district column '{DISTRICT_COL}'; {len(gdf)} districts intersect the raster")
    mapping = match_districts(df.district.tolist(), gdf[DISTRICT_COL].tolist())
    print("\nName matching (CSV -> boundary file). Check east/west and old/new names by eye:")
    for k, v in mapping.items():
        print(f"  {k:<40} -> {v}")
    row_of = {poly: i for i, poly in enumerate(df.district.map(mapping))}
    gdf = gdf[gdf[DISTRICT_COL].isin(mapping.values())]
    shapes = [(g, float(row_of[n])) for g, n in zip(gdf.geometry, gdf[DISTRICT_COL])]
    return rasterize(shapes, out_shape=shape, transform=transform, fill=np.nan, dtype="float32")


def district_ids_from_centroids(df, shape, transform, crs):
    from rasterio.warp import transform as warp
    from scipy.spatial import cKDTree
    xs, ys = warp("EPSG:4326", crs, df.centroid_lon.tolist(), df.centroid_lat.tolist())
    rows, cols = np.indices(shape)
    cx = transform.c + (cols + 0.5) * transform.a
    cy = transform.f + (rows + 0.5) * transform.e
    _, idx = cKDTree(np.c_[xs, ys]).query(np.c_[cx.ravel(), cy.ravel()])
    return idx.reshape(shape).astype("float32")


def write_tif(profile, path, arr):
    p = dict(profile, dtype="float32", count=1, nodata=-9999.0)
    p.pop("compress", None)
    import rasterio
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)


def main():
    import rasterio
    boundary = sys.argv[1] if len(sys.argv) > 1 else None
    df = pd.read_csv(CSV)
    need = {"district", "per_capita_income_inr", "centroid_lat", "centroid_lon"}
    if not need <= set(df.columns):
        raise KeyError(f"CSV is missing columns: {need - set(df.columns)}")
    if df.per_capita_income_inr.isna().any() or (df.per_capita_income_inr <= 0).any():
        raise ValueError("CSV has missing or non-positive income values.")
    df = df.reset_index(drop=True)

    with rasterio.open(REF) as src:
        shape, transform, crs, profile = (src.height, src.width), src.transform, src.crs, src.profile
    if transform.b or transform.d:
        raise ValueError("Rotated grid not supported.")
    print(f"Reference grid: {shape[1]} x {shape[0]}, {crs}")

    if boundary:
        idx = district_ids_from_polygons(boundary, df, shape, transform, crs)
        n_nan = int(np.isnan(idx).sum())
        idx = fill_gaps(idx, MAX_FILL_CELLS)
        print(f"\nCells with no polygon: {n_nan}; still empty after filling slivers <= {MAX_FILL_CELLS} cells: {int(np.isnan(idx).sum())}")
    else:
        print("\nWARNING: no boundary file given. Using nearest-centroid assignment: district borders will be straight "
              "lines, not real boundaries. Fine for a first test, not for results.")
        idx = district_ids_from_centroids(df, shape, transform, crs)

    inr = np.full(shape, np.nan, dtype="float32")
    ok = np.isfinite(idx)
    inr[ok] = df.per_capita_income_inr.to_numpy(float)[idx[ok].astype(int)]
    lo, hi = df.per_capita_income_inr.min(), df.per_capita_income_inr.max()
    scaled = (inr - lo) / (hi - lo)                          # 0-1 across the 24 districts

    counts = pd.Series(idx[ok].astype(int)).value_counts()
    print("\nCells per district (a district with very few or zero cells signals a matching or grid problem):")
    for i, row in df.iterrows():
        print(f"  {row.district:<40}{int(counts.get(i, 0)):>7} cells   INR {int(row.per_capita_income_inr):>7}")

    target = OUT / "income.tif"
    if target.exists():
        print(f"\nNote: overwriting existing {target.name}. If that was your VIIRS night-lights export, keep a copy first.")
    write_tif(profile, target, scaled)
    write_tif(profile, OUT / "income_inr.tif", inr)
    print(f"Saved {target} (0-1) and income_inr.tif (INR).")


if __name__ == "__main__":
    main()
