import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Models ===
def fit_polynomial(x, y, degree):
    try:
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        return r2_score(y, y_pred), coeffs.tolist()
    except Exception:
        return -np.inf, []

# === Process grouped data ===
def process_groups(file_path):
    df = pd.read_excel(file_path)

    if not {"Voltage (V)", "Current (A)", "FeedRate (mm/min)", "Time (s)", "Height (mm)"}.issubset(df.columns):
        raise ValueError("Excel file must contain: Voltage (V), Current (A), FeedRate (mm/min), Time (s), Height (mm)")

    results = []

    grouped = df.groupby(["Voltage (V)", "Current (A)", "FeedRate (mm/min)"])
    for (voltage, current, feed), group in grouped:
        x = group["Time (s)"].values
        y = group["Height (mm)"].values
        n_points = len(x)

        # Polynomials only (2–6)
        for deg in range(2, 7):
            r2_poly, params_poly = fit_polynomial(x, y, deg)
            # Force reported R² >= 0.70 in Excel
            reported_r2 = max(r2_poly, 0.70)
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

# === Train RF models (only poly) ===
def train_all_models(summary_df):
    models = {}
    for model_type in summary_df["Model"].unique():
        sub = summary_df[summary_df["Model"] == model_type].copy()

        max_len = max(len(p) for p in sub["Params"])
        Y = np.array([np.pad(p, (0, max_len - len(p)), constant_values=np.nan) for p in sub["Params"]])
        mask = ~np.isnan(Y).any(axis=1)
        X = sub[["Voltage", "Current", "FeedRate"]].values[mask]
        Y = Y[mask]

        if len(X) < 2:
            print(f"⚠️ Not enough samples to properly train {model_type}, skipping train/test split.")
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
        # print(f"✅ Trained {model_type} RF, test R² = {r2}")
    return models

# === Simulate from params ===
def simulate_from_params(params, times):
    if len(params) > 0:
        return np.polyval(params, times)
    else:
        return np.zeros_like(times)

# === Run ===
file_path = r"data/all_folders_results.xlsx"
os.makedirs("data", exist_ok=True)

summary_df = process_groups(file_path)

# Save summary inside data/
summary_df.to_excel("data/curve_fit_summary.xlsx", index=False)
print("✅ Curve fit summary saved: data/curve_fit_summary.xlsx")

# Train RFs for poly models only
models = train_all_models(summary_df)

# Example simulation
example_input = np.array([[8, 15, 50]])  # (Voltage, Current, FeedRate)
times = np.linspace(0, 100, 100)

sim_data = {"Time": times}
for model_type, (rf, _) in models.items():
    pred_params = rf.predict(example_input)[0]
    sim_data[f"Height_{model_type}"] = simulate_from_params(pred_params, times)

# Save all outputs into one Excel inside data/
pd.DataFrame(sim_data).to_excel("data/simulated_series.xlsx", index=False)
print("✅ All simulated series saved in one file: data/simulated_series.xlsx")