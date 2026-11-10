"""Central configuration. Values here are assumptions to be justified in the report."""
from dataclasses import dataclass


@dataclass
class Config:
    # Paths
    data_dir: str = "real_data"
    output_dir: str = "outputs"

    # Grid (overwritten from the real rasters when real data is loaded)
    shape: tuple = (200, 200)          # rows, cols of the common grid
    cell_size_m: float = 100.0         # metres per cell
    seed: int = 42

    # Hard exclusions (documented thresholds)
    max_slope_deg: float = 10.0
    max_flood_risk: float = 0.5        # 0-1 index
    excluded_lulc: tuple = (0, 1, 4)   # 0 water, 1 forest, 4 built-up

    # Presence-background sampling
    background_ratio: int = 5          # background samples per positive

    # Spatial validation
    block_size_cells: int = 40         # blocks of 40x40 cells are the spatial CV groups
    n_splits: int = 5

    # Candidate areas
    suitability_threshold: float = 0.6
    min_candidate_cells: int = 20      # patches smaller than this are dropped

    # Energy model  E = A * H * eta * PR
    module_eff_stc: float = 0.20       # module efficiency at 25 C
    temp_coeff: float = -0.004         # per deg C above 25 C
    performance_ratio: float = 0.80
    packing_factor: float = 0.40       # fraction of each cell covered by panels
