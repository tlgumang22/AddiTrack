import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Optional: only used if you want to calibrate noise from video
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# ---------------------------
# Config
# ---------------------------
EXCEL_PATH = r"C:\Users\dosiu\OneDrive\Desktop\python_vscode\BTP\data\all_folders_results.xlsx"
MODEL_PATH = r"models\curve_param_model.pkl"
NOISE_PATH = r"models\noise_stats.json"
RANDOM_STATE = 42

# ---------------------------
# 1) Mean curve (physics-inspired)
# ---------------------------
def growth_func(t, a, b, c):
    """
    Exponential-like growth; flexible but simple:
        Height(t) = a * (1 - exp(-b * t)) + c
    a: asymptotic height gain
    b: rate
    c: initial offset
    """
    return a * (1 - np.exp(-b * t)) + c

def fit_curve_to_group(time_arr: np.ndarray, height_arr: np.ndarray) -> Tuple[float, float, float]:
    """Fit growth_func to one experiment time/height array; return (a,b,c)."""
    a0 = max(height_arr) - min(height_arr) if len(height_arr) else 1.0
    b0 = 0.1
    c0 = min(height_arr) if len(height_arr) else 0.0
    popt, _ = curve_fit(growth_func, time_arr, height_arr, p0=[a0, b0, c0], maxfev=10000)
    return tuple(map(float, popt))

# ---------------------------
# 2) Learn noise from residuals (σ, ρ)
# ---------------------------
def estimate_noise_stats(residual_series_list) -> Dict[str, float]:
    """
    Learn global noise stats across experiments:
      - sigma: std of residuals
      - rho: lag-1 autocorrelation (AR(1))
    """
    all_res = []
    for r in residual_series_list:
        r = np.asarray(r, dtype=float)
        if r.size > 0:
            all_res.append(r)
            all_res.append(np.array([np.nan]))
    if not all_res:
        return {"sigma": 0.05, "rho": 0.3}

    cat = np.concatenate(all_res)
    sigma = np.nanstd(cat)

    rhos = []
    for r in residual_series_list:
        r = np.asarray(r, dtype=float)
        if r.size >= 2:
            r0 = r[:-1] - np.nanmean(r[:-1])
            r1 = r[1:]  - np.nanmean(r[1:])
            denom = (np.sqrt(np.nansum(r0**2)) * np.sqrt(np.nansum(r1**2)))
            if denom > 0:
                rho = float(np.nansum(r0 * r1) / denom)
                if np.isfinite(rho):
                    rhos.append(rho)
    rho = float(np.nanmean(rhos)) if rhos else 0.3

    rho = max(min(rho, 0.95), -0.95)
    sigma = float(sigma if sigma > 1e-6 else 0.05)
    return {"sigma": sigma, "rho": rho}

# ---------------------------
# 3) Train or load model + noise
# ---------------------------
def train_or_load_model_and_noise(excel_path=EXCEL_PATH):
    df = pd.read_excel(excel_path)
    df = df.rename(columns={
        "Voltage (V)": "Voltage",
        "Current (A)": "Current",
        "FeedRate (mm/min)": "FeedRate",
        "Time (s)": "Time",
        "Height (mm)": "Height",
        "Length (mm)": "Length"
    })
    required = {"Voltage", "Current", "FeedRate", "Time", "Height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    param_rows = []
    residual_series_list = []

    for (V, I, F), g in df.groupby(["Voltage", "Current", "FeedRate"]):
        g = g.sort_values("Time")
        t = g["Time"].values.astype(float)
        h = g["Height"].values.astype(float)
        if len(t) < 3:
            continue
        try:
            a, b, c = fit_curve_to_group(t, h)
            param_rows.append([V, I, F, a, b, c])
            h_hat = growth_func(t, a, b, c)
            residuals = h - h_hat
            residual_series_list.append(residuals)
        except Exception:
            continue

    if not param_rows:
        raise RuntimeError("Could not fit any experiment groups. Check data quality or column names.")

    param_df = pd.DataFrame(param_rows, columns=["Voltage", "Current", "FeedRate", "a", "b", "c"])

    X = param_df[["Voltage", "Current", "FeedRate"]]
    y = param_df[["a", "b", "c"]]
    if len(param_df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = RandomForestRegressor(
            n_estimators=1000, random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

    if X_test is not None:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
        # print(f"Parameter model R² (a,b,c): {r2:.3f}")
    else:
        print("Parameter model trained on full data (not enough groups to hold out a test set).")

    if os.path.exists(NOISE_PATH):
        with open(NOISE_PATH, "r") as f:
            noise_stats = json.load(f)
    else:
        noise_stats = estimate_noise_stats(residual_series_list)
        with open(NOISE_PATH, "w") as f:
            json.dump(noise_stats, f)

    print(f"Using noise stats: sigma={noise_stats['sigma']:.4f}, rho={noise_stats['rho']:.3f}")
    return model, noise_stats

# ---------------------------
# 4) Simulation: discrete, noisy series (Time vs Height)
# ---------------------------
def simulate_height_series(
    model: RandomForestRegressor,
    voltage: float,
    current: float,
    feedrate: float,
    t_max: int,
    noise_stats: Dict[str, float],
    seed: int = None,
    enforce_nonnegative: bool = True,
    monotonic_soft: bool = True,
) -> pd.DataFrame:
    """
    Returns a discrete table for t=1..t_max with realistic random variation.
    Height(t) = mean_curve(t) + AR(1) noise; Length = FeedRate * t
    monotonic_soft: softly prevent large negative dips by clipping small drops.
    """
    rng = np.random.default_rng(seed)
    x = pd.DataFrame([[voltage, current, feedrate]], columns=["Voltage", "Current", "FeedRate"])
    a, b, c = model.predict(x)[0]
    t = np.arange(1, int(t_max) + 1, dtype=float)

    mean_h = growth_func(t, a, b, c)

    sigma = float(noise_stats.get("sigma", 0.05))
    rho = float(noise_stats.get("rho", 0.3))
    eps = rng.normal(0.0, 1.0, size=len(t))
    noise = np.zeros_like(t)
    for i in range(len(t)):
        if i == 0:
            noise[i] = sigma * eps[i] / np.sqrt(1 - rho**2)
        else:
            noise[i] = rho * noise[i-1] + sigma * eps[i]

    h = mean_h + noise

    if monotonic_soft and len(h) > 1:
        for i in range(1, len(h)):
            if h[i] < h[i-1] - 0.15 * max(1.0, np.std(h)):
                h[i] = (h[i] + h[i-1]) / 2.0

    if enforce_nonnegative:
        h = np.maximum(h, 0.0)

    length = (feedrate * t)/60
    out = pd.DataFrame({
        "Time (s)": t.astype(int),
        "Height (mm)": h,
        "Length (mm)": length
    })
    return out

# ---------------------------
# 6) Main (example usage)
# ---------------------------
if __name__ == "__main__":
    model, noise_stats = train_or_load_model_and_noise(EXCEL_PATH)

    V, I, F, TMAX = 7, 14.5, 48, 100
    table = simulate_height_series(
        model, V, I, F, TMAX, noise_stats, seed=RANDOM_STATE,
        enforce_nonnegative=True, monotonic_soft=True
    )
    print("\nSimulated discrete Time vs Height table:")
    print(table.to_string(index=False))

    out_excel = f"C:\\Users\\dosiu\OneDrive\\Desktop\\python_vscode\\BTP\\data\\simulated_series_V{V}_I{I}_F{F}_T{TMAX}.xlsx"
    table.to_excel(out_excel, index=False, engine="openpyxl")
    print(f"\nSaved: {out_excel}")
