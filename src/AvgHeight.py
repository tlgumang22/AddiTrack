import pandas as pd
import re

def compute_avgheight(filename, out_file="data/constant_height_profile.xlsx"):
    """
    Compute average height from a simulation Excel file,
    adjust Voltage & Current to keep height near average,
    and save a new Excel profile.

    Parameters:
        filename (str): Path to the simulation Excel file (must contain Time, Height, Length).
        out_file (str): Path to save the adjusted profile.

    Returns:
        df_new (pd.DataFrame): Adjusted dataframe with V–I profile and average height column.
        avg_height (float): Calculated average height.
    """

    # === Step 1: Load the Excel file ===
    df = pd.read_excel(filename)

    # Ensure required columns exist
    if not {"Time (s)", "Height (mm)", "Length (mm)"}.issubset(df.columns):
        raise ValueError("Excel file must contain: Time (s), Height (mm), Length (mm)")

    # === Step 2: Parse V, I, F, T from filename ===
    pattern = r"V([\d\.]+)_I([\d\.]+)_F([\d\.]+)_T(\d+)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError("Filename does not match expected pattern: Vx_Iy_Fz_Tt")

    base_voltage = float(match.group(1))
    base_current = float(match.group(2))

    # === Step 3: Calculate target average height ===
    avg_height = df["Height (mm)"].mean()

    # === Step 4: Adjust Voltage and Current to keep height near average ===
    new_voltage, new_current = [], []
    k_v, k_i = 0.05, 0.03  # scaling factors

    for h in df["Height (mm)"]:
        error = avg_height - h
        new_voltage.append(base_voltage + k_v * error)
        new_current.append(base_current + k_i * error)

    # === Step 5: Save new DataFrame ===
    df_new = pd.DataFrame({
        "Time (s)": df["Time (s)"],
        "Height (mm)": df["Height (mm)"],
        "Voltage (V)": new_voltage,
        "Current (A)": new_current,
        "Average Height (mm)": [avg_height] * len(df)
    })

    df_new.to_excel(out_file, index=False)
    return df_new, avg_height


# Optional: allow standalone run
if __name__ == "__main__":
    test_file = r"data\simulated_series_V9.0_I12.5_F46.0_T100.xlsx"
    df_new, avg_h = compute_avgheight(test_file)
    print(f"Average Height = {avg_h:.3f} mm")
    print(f"Adjusted profile saved to data/constant_height_profile.xlsx")
