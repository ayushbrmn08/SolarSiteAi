"""End-to-end run.

    python run_pipeline.py              # real data from ./real_data
    python run_pipeline.py --synthetic  # demo on synthetic data (checks the code only)

Results are written to ./outputs
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.metrics import roc_auc_score

import candidates as C
import data
import energy as E
import plotting
import suitability as S
from config import Config


def load_inputs(cfg, synthetic):
    """Return layers, the allowed-cell mask, the farm (label) mask and the plant table (None if synthetic)."""
    if synthetic:
        layers = data.make_synthetic_layers(cfg)
        allowed = data.exclusion_mask(layers, cfg)
        return layers, allowed, data.synthetic_farm_locations(layers, allowed, cfg), None

    layers = data.load_real_layers(cfg)
    allowed = data.exclusion_mask(layers, cfg) & (layers["inside"] >= 0.5)
    plants = data.plants_to_grid(data.load_plants(f"{cfg.data_dir}/plants.csv"), f"{cfg.data_dir}/ghi.tif")
    farms = data.farm_mask(plants, cfg.shape)
    on_eligible = int(allowed[plants["raster_row"], plants["raster_col"]].sum())
    print(f"Plants mapped to the grid: {len(plants)} ({int(farms.sum())} distinct cells)")
    if on_eligible < len(plants):
        print(f"WARNING: only {on_eligible} of {len(plants)} plants lie on eligible cells; the rest fall on cells "
              f"removed by the exclusion rules (for example built-up or forest at this resolution).")
    return layers, allowed, farms, plants


def run_suitability(cfg, layers, allowed, farms):
    """Task 1: compare models with spatial CV, then map suitability with the best one."""
    X_all = data.stack_features(layers)
    X_flat = X_all.reshape(-1, X_all.shape[-1])
    groups = S.block_ids(cfg.shape, cfg.block_size_cells).ravel()

    idx, y = S.sample_presence_background(farms, allowed, cfg)
    cv = S.spatial_cv(S.get_models(cfg.seed), X_flat[idx], y, groups[idx], cfg)
    print("\nTask 1 - suitability, spatial-block CV\n", cv.round(3).to_string())

    best = cv["PR_AUC"].idxmax()
    model, suit = S.fit_and_map(S.get_models(cfg.seed)[best], X_all, idx, y, cfg.shape, allowed)
    print(f"\nBest suitability model (PR-AUC): {best}")
    print("Feature importance:", {k: round(float(v), 3) for k, v in S.importance(model, X_flat[idx], y).items()})

    # Map comparison on eligible sample cells only (excluded cells are forced to 0 in both maps).
    ahp = S.ahp_weighted_overlay(layers, allowed)
    ok = allowed.ravel()[idx]
    print(f"AHP baseline ROC-AUC: {roc_auc_score(y[ok], ahp.ravel()[idx][ok]):.3f}  |  "
          f"ML (in-sample, optimistic): {roc_auc_score(y[ok], suit.ravel()[idx][ok]):.3f}  (eligible cells, {int(y[ok].sum())} positives)")
    return suit, cv


def run_energy(cfg, layers, allowed):
    """Task 2: physics-based annual energy per eligible cell (kWh/year)."""
    cell_area_m2 = cfg.cell_size_m ** 2 * cfg.packing_factor
    phys = E.annual_energy_kwh(cell_area_m2, layers["ghi"], layers["temp"], cfg)
    energy_cell = np.where(allowed, phys, 0.0)
    stats = E.energy_statistics(energy_cell, allowed, cfg)
    print("\nTask 2 - physics-based solar energy potential\n", stats.round(3).to_string(index=False))
    return energy_cell, stats


def main(synthetic=False):
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    out = lambda name: os.path.join(cfg.output_dir, name)

    layers, allowed, farms, plants = load_inputs(cfg, synthetic)
    suit, task1_metrics = run_suitability(cfg, layers, allowed, farms)
    energy_cell, task2_metrics = run_energy(cfg, layers, allowed)

    cand, table = C.extract_candidates(suit, allowed, energy_cell, cfg)
    print(f"\n{len(table)} candidate areas, total {table.annual_energy_GWh.sum():.1f} GWh/yr, "
          f"{table.area_ha.sum():.0f} ha")
    print(table.head(5).round(2).to_string(index=False))

    task1_metrics.to_csv(out("task1_metrics.csv"))
    task2_metrics.to_csv(out("task2_metrics.csv"), index=False)
    table.to_csv(out("candidate_areas.csv"), index=False)
    if plants is not None:
        E.plant_reference_table(plants, layers, allowed).to_csv(out("plant_energy_reference.csv"), index=False)
        print("Note: plant generation values are estimates, not measured readings; energy map is physics-only.")
    plotting.save_overview_maps(out("decision_map.png"), allowed, suit, energy_cell, cand)
    plotting.save_ranked_map(out("ranked_candidate_map.png"), allowed, suit, energy_cell, cand, table)
    print(f"\nSaved results to {cfg.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true", help="run on synthetic demo data")
    main(parser.parse_args().synthetic)
