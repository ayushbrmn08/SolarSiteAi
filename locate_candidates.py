"""Add latitude / longitude to outputs/candidate_areas.csv  ->  outputs/candidate_locations.csv

Run after run_pipeline.py:   python locate_candidates.py
"""
import pandas as pd
import rasterio
from rasterio.warp import transform

REF_RASTER = "real_data/ghi.tif"
IN_CSV = "outputs/candidate_areas.csv"
OUT_CSV = "outputs/candidate_locations.csv"


def main():
    areas = pd.read_csv(IN_CSV)
    with rasterio.open(REF_RASTER) as src:
        xy = [src.xy(r, c) for r, c in zip(areas["centroid_row"], areas["centroid_col"])]
        lon, lat = transform(src.crs, "EPSG:4326", [x for x, _ in xy], [y for _, y in xy])
    result = pd.DataFrame({"id": areas["id"], "latitude": lat, "longitude": lon,
                           "area_ha": areas["area_ha"], "annual_energy_GWh": areas["annual_energy_GWh"]})
    result.to_csv(OUT_CSV, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
