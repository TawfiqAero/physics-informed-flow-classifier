import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

n_samples = 10000

# -------------------------------
# Generate Inputs
# -------------------------------
Re = np.random.uniform(30000, 600000, n_samples)
AoA = np.random.uniform(-5, 15, n_samples)

# -------------------------------
# Lift Coefficient (Thin Airfoil Theory)
# Cl = 2π * α (in radians)
# -------------------------------
AoA_rad = np.radians(AoA)
Cl = 2 * np.pi * AoA_rad

# -------------------------------
# Drag Coefficient (Parabolic Drag Polar)
# Cd = Cd0 + Cl^2 / (π * e * AR)
# -------------------------------
Cd0 = 0.01
e = 0.9
AR = 4

Cd = Cd0 + (Cl**2) / (np.pi * e * AR)

# -------------------------------
# Add Noise (Realistic Variation)
# -------------------------------
Cl += np.random.normal(0, 0.05, n_samples)
Cd += np.random.normal(0, 0.002, n_samples)

# -------------------------------
# Flow Regime Classification
# -------------------------------

# CURRENT MODEL (same as your project)
critical_Re = 150000 - (AoA * 5000)
# OPTIONAL (more physical - DO NOT USE unless retraining)
# critical_Re = 150000 - (AoA * 5000)

# Vectorized classification (FAST)
Flow_Regime = (Re >= critical_Re).astype(int)

# -------------------------------
# Create Dataset
# -------------------------------
df = pd.DataFrame({
    "Reynolds": Re,
    "AoA": AoA,
    "Cl": Cl,
    "Cd": Cd,
    "Flow_Regime": Flow_Regime
})

# -------------------------------
# Save File
# -------------------------------
df.to_csv("airfoil_dataset_10k.csv", index=False)

print("✅ Dataset created: airfoil_dataset_10k.csv")
print(df.head())
