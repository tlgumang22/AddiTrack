import pandas as pd
import re

# === Step 1: Load the Excel file ===
filename = "simulated_series_V7_I14.5_F48_T100.xlsx"
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
feedrate = float(match.group(3))
total_time = int(match.group(4))

print(f"Base values from filename → V={base_voltage}, I={base_current}, F={feedrate}, T={total_time}")

# === Step 3: Calculate target average height ===
avg_height = df["Height (mm)"].mean()
print(f"Target average height from file: {avg_height:.3f} mm")

# === Step 4: Adjust Voltage and Current to keep height near average ===
new_voltage = []
new_current = []

# scaling factors (tune as needed)
k_v = 0.05
k_i = 0.03

for h in df["Height (mm)"]:
    error = avg_height - h

    v_new = base_voltage + k_v * error
    i_new = base_current + k_i * error

    new_voltage.append(v_new)
    new_current.append(i_new)

# === Step 5: Save new DataFrame ===
df_new = pd.DataFrame({
    "Time (s)": df["Time (s)"],
    "Height (mm)": df["Height (mm)"],
    "Voltage (V)": new_voltage,
    "Current (A)": new_current,
    "Feedrate (mm/s)": [feedrate] * len(df)
})

out_file = "constant_height_profile.xlsx"
df_new.to_excel(out_file, index=False)

print(f"Adjusted V–I profile saved as: {out_file}")
