# heat_exchanger_pdm_rf_with_detection_v2.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import r2_score # type: ignore
import joblib # type: ignore
import os

warnings.filterwarnings("ignore")

# USER INPUT

HEALTHY_CSV_PATH = "heat_exchanger_healthy.csv"   # baseline healthy data
TEST_CSV_PATH    = "heat_exchanger_faulty.csv"    # data to evaluate (unknown)
OUTPUT_PREFIX    = "heat_exchanger_output"

MIN_CONSECUTIVE_FAULTS = 30   # consecutive anomaly samples to declare fault
WARMUP_IGNORE = 60            # ignore first 60 samples for fault detection

# FUNCTION: load & compute UA + features

def prepare_heat_exchanger_df(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    required_cols = ['T_in_hot','T_out_hot','T_in_cold','T_out_cold',
                     'flow_rate','pressure_drop','vibration']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {csv_path}. Required: {required_cols}")

    df[required_cols] = df[required_cols].interpolate(limit=3).fillna(method='bfill').fillna(method='ffill')

    # Physical constants
    rho = 1000.0   # kg/m3
    Cp  = 4186.0   # J/(kg·K)

    flow_m3s = (df['flow_rate'].astype(float) / 3600.0).replace(0, 1e-6)

    Th_in  = df['T_in_hot'].astype(float)
    Th_out = df['T_out_hot'].astype(float)
    Tc_in  = df['T_in_cold'].astype(float)
    Tc_out = df['T_out_cold'].astype(float)

    # LMTD with safety against zero/neg
    delta1 = Th_in - Tc_out
    delta2 = Th_out - Tc_in
    eps = 1e-6
    lmtd = (delta1 - delta2) / (np.log((delta1 + eps) / (delta2 + eps)) + eps)

    Q_cold = rho * flow_m3s * Cp * (Tc_out - Tc_in)  # W
    UA = Q_cold / (lmtd + eps)
    df['UA_est'] = UA

    # Features
    df['deltaT_hot']   = Th_in - Th_out
    df['deltaT_cold']  = Tc_out - Tc_in
    df['pressure_drop'] = df['pressure_drop'].astype(float)
    df['vibration']     = df['vibration'].astype(float)

    win = 60
    df['vib_mean_60']  = df['vibration'].rolling(win, min_periods=1).mean()
    df['vib_std_60']   = df['vibration'].rolling(win, min_periods=1).std().fillna(0)
    df['dp_mean_60']   = df['pressure_drop'].rolling(win, min_periods=1).mean()
    df['flow_mean_60'] = df['flow_rate'].rolling(win, min_periods=1).mean()

    feature_cols = ['T_in_hot','T_out_hot','T_in_cold','T_out_cold',
                    'flow_rate','pressure_drop','vibration',
                    'deltaT_hot','deltaT_cold','vib_mean_60','vib_std_60',
                    'dp_mean_60','flow_mean_60']
    X = df[feature_cols].fillna(method='ffill').fillna(method='bfill').values
    y = df['UA_est'].values

    return df, X, y, feature_cols

# PREPARE HEALTHY BASELINE

df_healthy, X_healthy, y_healthy, feature_cols = prepare_heat_exchanger_df(HEALTHY_CSV_PATH)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_healthy, y_healthy)

y_healthy_pred = rf.predict(X_healthy)
resid_healthy  = y_healthy - y_healthy_pred
abs_resid_healthy = np.abs(resid_healthy)

# Robust threshold from healthy residuals: 99th percentile of |residual|
q99 = np.quantile(abs_resid_healthy, 0.99)

# Add small safety margin
threshold = q99 * 1.2 if q99 > 0 else 1e-6

print(f"[HE] Healthy |residual| 99th percentile={q99:.4e}, threshold={threshold:.4e}")

# APPLY TO TEST DATA
df_test, X_test, y_test, _ = prepare_heat_exchanger_df(TEST_CSV_PATH)
y_test_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
print(f"[HE] R2 on TEST data: {r2:.4f}")

resid_test = y_test - y_test_pred
abs_resid_test = np.abs(resid_test)
anomaly_flags = abs_resid_test > threshold

times_test = pd.to_datetime(df_test['timestamp'])

# FIND FIRST CONSISTENT FAULT (after warmup)
first_fault_idx = None
count = 0
for i in range(WARMUP_IGNORE, len(anomaly_flags)):
    if anomaly_flags[i]:
        count += 1
        if count >= MIN_CONSECUTIVE_FAULTS:
            first_fault_idx = i - MIN_CONSECUTIVE_FAULTS + 1
            break
    else:
        count = 0

if first_fault_idx is not None:
    first_fault_time = times_test.iloc[first_fault_idx]
    print(f"[HE] First consistent fault detected at index {first_fault_idx}, time {first_fault_time}")
else:
    print("[HE] No consistent fault detected (after warmup) based on residual threshold.")

# Only consider anomalies after warmup for global status
valid_flags = anomaly_flags.copy()
valid_flags[:WARMUP_IGNORE] = False
anom_fraction = valid_flags.mean()

if first_fault_idx is None and anom_fraction < 0.05:
    summary_msg = "HEALTHY: Heat exchanger working correctly; no immediate maintenance needed."
else:
    summary_msg = ("FAULTY: Heat exchanger shows consistent deviation from healthy behavior; "
                   "maintenance should be scheduled.")
print("[HE] FINAL STATUS:", summary_msg)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(rf, os.path.join('models', OUTPUT_PREFIX + '_rf.pkl'))

# PLOTS

os.makedirs('plots', exist_ok=True)

# 1) Sensor time-series
plt.figure(figsize=(14,10))
ax1 = plt.subplot(3,1,1)
ax1.plot(times_test, df_test['T_in_hot'], label='T_in_hot')
ax1.plot(times_test, df_test['T_out_hot'], label='T_out_hot')
ax1.plot(times_test, df_test['T_in_cold'], label='T_in_cold')
ax1.plot(times_test, df_test['T_out_cold'], label='T_out_cold')
ax1.set_ylabel('Temperature (°C)')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))

ax2 = plt.subplot(3,1,2, sharex=ax1)
ax2.plot(times_test, df_test['flow_rate'], label='Flow Rate (m3/hr)')
ax2.plot(times_test, df_test['pressure_drop'], label='Pressure Drop')
ax2.set_ylabel('Flow / Pressure')
ax2.legend(loc='upper right')

ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.plot(times_test, df_test['vibration'], label='Vibration (mm/s)', color='tab:orange')
ax3.set_ylabel('Vibration')
ax3.legend(loc='upper right')
ax3.set_xlabel('Timestamp')

if first_fault_idx is not None:
    for ax in [ax1, ax2, ax3]:
        ax.axvline(first_fault_time, color='red', linestyle='--', label='First Fault')
    handles, labels = ax3.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax3.legend(unique.values(), unique.keys(), loc='upper right')

plt.suptitle('Heat Exchanger: Test Sensor Time-Series')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join('plots', OUTPUT_PREFIX + "_test_sensors_timeseries.png"), dpi=200)

# 2) UA true vs predicted + anomalies
plt.figure(figsize=(12,6))
plt.plot(times_test, y_test, label='UA True (proxy)', linewidth=1)
plt.plot(times_test, y_test_pred, label='UA Predicted (RF)', linestyle='--', linewidth=1)
plt.scatter(times_test[valid_flags], y_test[valid_flags], color='red', s=8, label='Anomaly (after warmup)')
if first_fault_idx is not None:
    plt.axvline(first_fault_time, color='black', linestyle='--', label='First Fault')
plt.ylabel('UA (W/K)')
plt.xlabel('Time')
plt.title('Heat Exchanger: UA True vs Predicted & Anomalies')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('plots', OUTPUT_PREFIX + "_UA_true_pred_anom.png"), dpi=200)

# 3) Residuals with threshold
plt.figure(figsize=(12,4))
plt.plot(times_test, resid_test, label='Residual (true - pred)')
plt.axhline(threshold,  color='red', linestyle='--', label='+Threshold')
plt.axhline(-threshold, color='red', linestyle='--')
plt.axhline(0, color='k', linewidth=0.5)
if first_fault_idx is not None:
    plt.axvline(first_fault_time, color='black', linestyle='--', label='First Fault')
plt.ylabel('Residual')
plt.xlabel('Time')
plt.title('Heat Exchanger: Residuals & Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('plots', OUTPUT_PREFIX + "_residuals_threshold.png"), dpi=200)

# 4) Feature importance
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
plt.figure(figsize=(9,4))
plt.bar([feature_cols[i] for i in idx], importances[idx])
plt.xticks(rotation=45, ha='right')
plt.title('Heat Exchanger RF Feature Importances (healthy training)')
plt.tight_layout()
plt.savefig(os.path.join('plots', OUTPUT_PREFIX + "_feature_importances.png"), dpi=200)

# Save predictions CSV
df_out = df_test.copy()
df_out['UA_true'] = y_test
df_out['UA_pred'] = y_test_pred
df_out['residual'] = resid_test
df_out['abs_residual'] = abs_resid_test
df_out['anomaly_flag'] = valid_flags.astype(int)
if first_fault_idx is not None:
    df_out['first_fault_time'] = first_fault_time
else:
    df_out['first_fault_time'] = pd.NaT

df_out.to_csv(OUTPUT_PREFIX + "_test_predictions.csv", index=False)

print("Result:", summary_msg)
