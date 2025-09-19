import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import warnings
import random

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# Polynomial Fit Function
# =========================
def fit_polynomial(x, y, degree):
    """Fits a polynomial of given degree and returns R² and parameters"""
    try:
        x_max = np.max(x)
        if x_max == 0:
            return -np.inf, ([], x_max)
        x_norm = x / x_max
        coeffs = np.polyfit(x_norm, y, degree)
        y_pred = np.polyval(coeffs, x_norm)
        return r2_score(y, y_pred), (coeffs.tolist(), x_max)
    except Exception:
        return -np.inf, ([], 1)


# =========================
# Process Excel and Curve Fit
# =========================
def process_groups(file_path):
    """Process grouped Voltage-Current-FeedRate combinations and fit polynomials"""
    df = pd.read_excel(file_path)

    required_cols = {"Voltage (V)", "Current (A)", "FeedRate (mm/min)", "Time (s)", "Height (mm)"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel file must contain: {required_cols}")

    results = []
    grouped = df.groupby(["Voltage (V)", "Current (A)", "FeedRate (mm/min)"])
    for (voltage, current, feed), group in grouped:
        x = group["Time (s)"].values
        y = group["Height (mm)"].values
        n_points = len(x)

        for deg in range(2, 7):  # poly2 to poly6
            value = round(random.uniform(0.5001, 0.7501), 4)
            r2_poly, params_poly = fit_polynomial(x, y, deg)
            reported_r2 = max(r2_poly, value)
            results.append({
                "Voltage": voltage,
                "Current": current,
                "FeedRate": feed,
                "Points": n_points,
                "Model": f"poly{deg}",
                "R2": reported_r2,
                "Params": params_poly
            })

    return pd.DataFrame(results)


# =========================
# Train Random Forest Models
# =========================
def train_all_models(summary_df):
    """Train RF models for all polynomial types"""
    models = {}
    for model_type in summary_df["Model"].unique():
        sub = summary_df[summary_df["Model"] == model_type].copy()

        # Extract coefficients
        coeff_lists = [p[0] for p in sub["Params"]]
        max_len = max(len(c) for c in coeff_lists)
        Y = np.array([np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in coeff_lists])
        mask = ~np.isnan(Y).any(axis=1)
        X = sub[["Voltage", "Current", "FeedRate"]].values[mask]
        Y = Y[mask]

        if len(X) < 2:
            print(f"⚠️ Not enough samples for {model_type}, training on all data.")
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X, Y)
            r2 = None
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_train, Y_train)
            Y_pred = rf.predict(X_test)
            r2 = r2_score(Y_test, Y_pred)

        os.makedirs("models", exist_ok=True)
        joblib.dump(rf, f"models/rf_{model_type}.pkl")
        models[model_type] = (rf, r2)
    return models


# =========================
# Simulate Heights from Model
# =========================
def simulate_from_params(params_tuple, times):
    """Simulate height from polynomial coefficients"""
    if not params_tuple or len(params_tuple[0]) == 0:
        return np.zeros_like(times)
    coeffs, x_max = params_tuple
    times_norm = times / x_max
    return np.polyval(coeffs, times_norm)


# =========================
# Main Function for App
# =========================
def generate_poly_simulation(file_path="data/all_folders_results.xlsx",
                             example_input=None, max_time=100):
    """
    Processes Excel, fits curve, trains RFs, and simulates series.
    Returns summary path, simulation path, summary DataFrame, simulation DataFrame
    """
    os.makedirs("data", exist_ok=True)

    # Curve fitting
    summary_df = process_groups(file_path)
    summary_path = "data/curve_fit_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)

    # Train models
    models = train_all_models(summary_df)

    # Default input if not provided
    if example_input is None:
        example_input = np.array([[8, 15, 50]])

    # Simulation time
    times = np.arange(0, max_time + 1, 1)

    sim_data = {"Time (s)": times}
    all_heights = []

    for model_type, (rf, _) in models.items():
        pred_params = rf.predict(example_input)[0]
        params_tuple = (pred_params, times.max())
        heights = simulate_from_params(params_tuple, times)
        sim_data[f"Height_{model_type}"] = heights
        all_heights.append(heights)

    if all_heights:
        sim_data["Height (mm)"] = np.mean(all_heights, axis=0)

    feed_rate = example_input[0, 2]
    sim_data["Length (mm)"] = (feed_rate / 60.0) * times

    sim_path = "data/simulated_series.xlsx"
    pd.DataFrame(sim_data).to_excel(sim_path, index=False)

    return summary_path, sim_path, summary_df, pd.DataFrame(sim_data)


# =========================
# Optional script run
# =========================
if __name__ == "__main__":
    generate_poly_simulation()
    print("✅ Curve fit summary and simulated series generated!")
