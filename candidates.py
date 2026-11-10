"""Group suitable cells into contiguous candidate areas and attach annual energy."""
import numpy as np
import pandas as pd
from scipy.ndimage import label

def extract_candidates(suit, allowed, energy_cell, cfg):
    mask = (suit >= cfg.suitability_threshold) & allowed
    lab, n = label(mask)                      # 4-connected patches
    rows, keep = [], np.zeros_like(lab)
    for i in range(1, n + 1):
        cells = lab == i
        k = int(cells.sum())
        if k < cfg.min_candidate_cells:
            continue
        keep[cells] = i
        rr, cc = np.nonzero(cells)
        rows.append(dict(id=i, area_ha=k * cfg.cell_size_m ** 2 / 1e4, mean_suitability=float(suit[cells].mean()),
                         annual_energy_GWh=float(energy_cell[cells].sum() / 1e6),
                         centroid_row=float(rr.mean()), centroid_col=float(cc.mean())))
    df = pd.DataFrame(rows).sort_values("annual_energy_GWh", ascending=False).reset_index(drop=True)
    return keep, df
