import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
DATA_PATH = "model_outputs/predictions.csv"  

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['ds'])
    return df

df = load_data()

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.title("🔍 Filters")

# Target variables available in dataset
targets = ['rnti_count', 'rb_down', 'rb_up', 'down', 'up']
selected_target = st.sidebar.selectbox("Select Target Variable", targets)

# Model names assumed to follow pattern: <target>_<modelname>
# Example: rnti_count_TimesNet, rnti_count_Informer
available_models = sorted({col.split('_', 1)[1] for col in df.columns if '_' in col and col.split('_', 1)[0] in targets})
selected_model = st.sidebar.selectbox("Select Model", available_models)

stations = df['unique_id'].unique()
selected_station = st.sidebar.selectbox("Select Base Station", stations)

slices = df['slice_label'].unique()
selected_slice = st.sidebar.selectbox("Select Pseudo Slice", slices)

real_time_mode = st.sidebar.checkbox("Enable Real-Time Simulation", value=True)

# ===============================
# FILTER DATA
# ===============================
model_col = f"{selected_target}_{selected_model}"
actual_col = f"{selected_target}_actual"

if actual_col not in df.columns or model_col not in df.columns:
    st.error(f"Required columns '{actual_col}' and '{model_col}' not found in dataset.")
    st.stop()

filtered_df = df[
    (df['unique_id'] == selected_station) &
    (df['slice_label'] == selected_slice)
].sort_values(by="ds")

# ===============================
# DASHBOARD
# ===============================
st.title("📈 Slice-Aware Federated Learning Traffic Forecasting Dashboard")
st.markdown(f"**Target:** {selected_target} | **Model:** {selected_model} | **Station:** {selected_station} | **Pseudo-Slice:** {selected_slice}")

fig, ax = plt.subplots(figsize=(10, 5))
line_actual, = ax.plot([], [], label="Actual", color="blue")
line_pred, = ax.plot([], [], label="Predicted", color="orange")
ax.set_xlabel("Time")
ax.set_ylabel(f"{selected_target} value")
ax.legend()

placeholder = st.empty()

# ===============================
# REAL-TIME SIMULATION
# ===============================
if real_time_mode:
    actual_vals = []
    pred_vals = []
    time_vals = []

    for _, row in filtered_df.iterrows():
        actual_vals.append(row[actual_col])
        pred_vals.append(row[model_col])
        time_vals.append(row['ds'])

        line_actual.set_data(time_vals, actual_vals)
        line_pred.set_data(time_vals, pred_vals)

        ax.set_xlim(min(time_vals), max(time_vals))
        ax.set_ylim(min(min(actual_vals), min(pred_vals)) * 0.9,
                    max(max(actual_vals), max(pred_vals)) * 1.1)

        placeholder.pyplot(fig)
        time.sleep(0.1)  # Adjust speed

else:
    ax.plot(filtered_df['ds'], filtered_df[actual_col], label="Actual", color="blue")
    ax.plot(filtered_df['ds'], filtered_df[model_col], label="Predicted", color="orange")
    ax.legend()
    st.pyplot(fig)
