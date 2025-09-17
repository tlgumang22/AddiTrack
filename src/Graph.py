import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import make_interp_spline

# ==============================
# Step 1: Load Excel Data
# ==============================
file_path = r"BTP\data\simulated_series_V7_I18.3_F45_T100.xlsx"
df = pd.read_excel(file_path)

if "Height%" in df.columns:
    df["Height"] = (df["Height%"] / 100.0) * 5.0
elif "Height" in df.columns:
    df["Height"] = df["Height"]
else:
    raise ValueError("Excel file must contain 'Height' or 'Height%' column.")

if "Time" not in df.columns:
    raise ValueError("Excel file must contain 'Time' column.")

x = df["Time"].to_numpy()
z = df["Height"].to_numpy()

# ==============================
# Step 2: Smooth Profile
# ==============================
x_smooth = np.linspace(x.min(), x.max(), 200)
spline = make_interp_spline(x, z, k=3)
z_smooth = spline(x_smooth)

# ==============================
# Step 3: Build Solid Extrusion
# ==============================
ymin, ymax = -1, 1
verts = []

for i in range(len(x_smooth) - 1):
    v1 = [x_smooth[i], ymin, z_smooth[i]]
    v2 = [x_smooth[i], ymax, z_smooth[i]]
    v3 = [x_smooth[i+1], ymax, z_smooth[i+1]]
    v4 = [x_smooth[i+1], ymin, z_smooth[i+1]]

    v1b, v2b, v3b, v4b = [v1[0], v1[1], 0], [v2[0], v2[1], 0], [v3[0], v3[1], 0], [v4[0], v4[1], 0]

    verts.append([v1, v2, v3, v4])
    verts.append([v1b, v2b, v3b, v4b])
    verts.append([v1, v4, v4b, v1b])
    verts.append([v2, v3, v3b, v2b])
    verts.append([v1, v2, v2b, v1b])
    verts.append([v4, v3, v3b, v4b])

# ==============================
# Step 4: Plot 3D Solid
# ==============================
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection="3d")

poly = Poly3DCollection(verts, facecolors="steelblue", edgecolors="k", alpha=0.85)
ax.add_collection3d(poly)

# Labels
ax.set_xlabel("Time (s)")
ax.set_ylabel("Width (mm)")
ax.set_zlabel("Height (mm)")
ax.set_title("3D Solid Extruded Height Profile vs Time")

# Set limits
ax.set_xlim(x.min(), x.max())
ax.set_ylim(ymin, ymax)
ax.set_zlim(0, max(z_smooth) + 1)

plt.tight_layout()
plt.show()
