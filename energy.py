"""Task 2: energy. Physics formula E = A * H * eta * PR plus an ML regressor that predicts per-cell yield."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def module_efficiency(temp_c, cfg):
    """eta with temperature adjustment; cell temp ~ ambient + 20 C under load (simple NOCT-style assumption)."""
    cell_t = temp_c + 20.0
    return cfg.module_eff_stc * (1 + cfg.temp_coeff * (cell_t - 25.0))

def annual_energy_kwh(area_m2, ghi_kwh_m2_yr, temp_c, cfg):
    """E = A x H x eta x PR, with A = usable (panel) area. Output kWh/year."""
    return area_m2 * ghi_kwh_m2_yr * module_efficiency(temp_c, cfg) * cfg.performance_ratio

def reference_yield_synthetic(layers, cfg, noise=0.03):
    """Stand-in for measured PV output (kWh/year per cell). Replace with real plant data."""
    rng = np.random.default_rng(cfg.seed + 3)
    cell_area = cfg.cell_size_m ** 2 * cfg.packing_factor
    e = annual_energy_kwh(cell_area, layers["ghi"], layers["temp"], cfg)
    soiling = 1 - 0.02 * layers["ndvi"].clip(0)          # vegetated surroundings -> slightly more soiling
    return e * soiling * (1 + noise * rng.standard_normal(e.shape))

def get_regressors(seed):
    return {
        "LR": make_pipeline(StandardScaler(), LinearRegression()),
        "RF": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "XGBoost-like (HistGB)": HistGradientBoostingRegressor(max_depth=4, random_state=seed),
        "SVR": TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR(C=10.0)), transformer=StandardScaler()),
    }

def regression_metrics(y, yhat):
    mask = np.abs(y) > 1e-6 * np.abs(y).max()     # MAPE ignores near-zero references
    return dict(MAE=mean_absolute_error(y, yhat), RMSE=mean_squared_error(y, yhat) ** 0.5, R2=r2_score(y, yhat),
                MAPE=100 * np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))

def spatial_cv_regression(models, X, y, groups, n_splits):
    import pandas as pd
    from sklearn.base import clone
    rows = []
    for name, m in models.items():
        res = [regression_metrics(y[te], clone(m).fit(X[tr], y[tr]).predict(X[te]))
               for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups)]
        rows.append({"Model": name, **pd.DataFrame(res).mean().to_dict()})
    return pd.DataFrame(rows).set_index("Model")


def spatial_cv_hybrid(models, X, phys, ref, groups, n_splits):
    """Hybrid physics + ML. ML learns the correction ratio ref/phys; final energy = phys * ratio_hat.
    A 'Physics only' row (ratio = 1) is the baseline the ML correction must beat."""
    import pandas as pd
    from sklearn.base import clone
    ratio = ref / phys
    rows = [{"Model": "Physics only", **pd.DataFrame(
        [regression_metrics(ref[te], phys[te]) for _, te in GroupKFold(n_splits=n_splits).split(X, ref, groups)]).mean().to_dict()}]
    for name, m in models.items():
        res = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, ref, groups):
            r_hat = clone(m).fit(X[tr], ratio[tr]).predict(X[te])
            res.append(regression_metrics(ref[te], phys[te] * r_hat))
        rows.append({"Model": f"Physics + {name}", **pd.DataFrame(res).mean().to_dict()})
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------- reporting helpers
def energy_statistics(energy_cell, allowed, cfg):
    """Summary table of the physics-based energy over eligible cells (GWh and MWh per year)."""
    vals = energy_cell[allowed]
    if vals.size == 0:
        raise ValueError("No eligible pixels available for Task 2.")
    mwh = vals / 1000.0
    rows = [
        ("Eligible pixels", int(allowed.sum()), "pixels", "Number of pixels passing the exclusion criteria"),
        ("Eligible land area", allowed.sum() * cfg.cell_size_m ** 2 / 1e4, "hectares", "Total area of eligible pixels"),
        ("Total annual energy potential", vals.sum() / 1e6, "GWh/year", "Sum of physics-based annual energy estimates"),
        ("Mean energy per pixel", mwh.mean(), "MWh/year", "Average annual energy potential per eligible pixel"),
        ("Median energy per pixel", np.median(mwh), "MWh/year", "Median annual energy potential per eligible pixel"),
        ("Minimum energy per pixel", mwh.min(), "MWh/year", "Minimum annual energy potential"),
        ("Maximum energy per pixel", mwh.max(), "MWh/year", "Maximum annual energy potential"),
        ("10th percentile energy", np.percentile(mwh, 10), "MWh/year", "Value below which 10% of eligible pixels fall"),
        ("90th percentile energy", np.percentile(mwh, 90), "MWh/year", "Value below which 90% of eligible pixels fall"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit", "Description"])

def plant_reference_table(plants, layers, allowed):
    """Plants (already placed on the grid) with the raster values at their location."""
    r, c = plants["raster_row"].to_numpy(), plants["raster_col"].to_numpy()
    gen = plants["annual_generation_gwh"]
    return pd.DataFrame({
        "name": plants["name"], "district": plants["district"], "capacity_mw": plants["capacity_mw"],
        "annual_generation_gwh_estimated": gen,
        "estimated_capacity_factor_pct": gen * 1000 / (plants["capacity_mw"] * 8760) * 100,
        "raster_row": r, "raster_col": c,
        "ghi": layers["ghi"][r, c], "temperature": layers["temp"][r, c],
        "inside_jharkhand": layers["inside"][r, c] >= 0.5, "eligible_pixel": allowed[r, c],
    })
