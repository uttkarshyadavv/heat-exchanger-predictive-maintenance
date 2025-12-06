import numpy as np
import pandas as pd
import datetime as dt
import random

# -------------------------
# CONFIGURATION
# -------------------------
num_days = 7
num_points = num_days * 24 * 60   # 1 minute interval

start_time = dt.datetime(2024, 1, 1)


# -------------------------
# FUNCTION TO CREATE TIME INDEX
# -------------------------
def generate_timestamp_series(start, n_points):
    return [start + dt.timedelta(minutes=i) for i in range(n_points)]


timestamps = generate_timestamp_series(start_time, num_points)


# ================================================================
# 1) HEALTHY HEAT EXCHANGER DATA
# ================================================================

np.random.seed(42)

# Normal operating temperature ranges (°C)
T_in_hot = np.random.normal(120, 1, num_points)
T_out_hot = T_in_hot - np.random.normal(10, 0.5, num_points)

T_in_cold = np.random.normal(30, 1, num_points)
T_out_cold = T_in_cold + np.random.normal(12, 0.5, num_points)

# Flow rate (m³/hr)
flow_rate = np.random.normal(50, 1.5, num_points)

# Pressure drop (clean condition)
pressure_drop = np.random.normal(1.2, 0.05, num_points)

# Vibration (mm/s RMS)
vibration = np.random.normal(1.0, 0.1, num_points)


# Create DataFrame
df_healthy = pd.DataFrame({
    "timestamp": timestamps,
    "T_in_hot": T_in_hot,
    "T_out_hot": T_out_hot,
    "T_in_cold": T_in_cold,
    "T_out_cold": T_out_cold,
    "flow_rate": flow_rate,
    "pressure_drop": pressure_drop,
    "vibration": vibration
})


# Save Healthy file
df_healthy.to_csv("heat_exchanger_healthy.csv", index=False)

print("Healthy heat exchanger dataset saved as heat_exchanger_healthy.csv")



# ================================================================
# 2) FAULTY HEAT EXCHANGER DATA (fouling + blockage + vibration)
# ================================================================

# Add fouling: outlet temp drops
T_out_hot_fault = T_out_hot - np.random.normal(3, 0.5, num_points)
T_out_cold_fault = T_out_cold - np.random.normal(2, 0.5, num_points)

# Flow rate decreases (partial blockage)
flow_rate_fault = flow_rate - np.random.normal(5, 1, num_points)

# Pressure drop increases (fouling)
pressure_drop_fault = pressure_drop + np.random.normal(0.5, 0.1, num_points)

# Vibration rises (tube bundle issues)
vibration_fault = vibration + np.random.normal(0.8, 0.2, num_points)


# Create DataFrame
df_faulty = pd.DataFrame({
    "timestamp": timestamps,
    "T_in_hot": T_in_hot,
    "T_out_hot": T_out_hot_fault,
    "T_in_cold": T_in_cold,
    "T_out_cold": T_out_cold_fault,
    "flow_rate": flow_rate_fault,
    "pressure_drop": pressure_drop_fault,
    "vibration": vibration_fault
})


# Save Faulty file
df_faulty.to_csv("heat_exchanger_faulty.csv", index=False)

print("Faulty heat exchanger dataset saved as heat_exchanger_faulty.csv")
