import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import h5py
from scipy.interpolate import splprep, splev
import plotly.io as pio
import imageio
from io import BytesIO
from plotly.subplots import make_subplots
import time
try:
    from stqdm import stqdm
except ImportError:
    st.warning("stqdm not found, progress bar disabled. Run `pip install stqdm` for better experience.")
    stqdm = lambda x, **kwargs: x

# --- CONFIG ---
data_file = 'fiber_all_processed.h5'  # Assumes it's in the repo root

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")
st.title("Ball Bearing FFT Visualization")

# --- Load Data Function ---
@st.cache_data(show_spinner=True)
def load_fiber_data(file_path):
    with h5py.File(file_path, 'r') as file:
        datasets = list(file.keys())
        fiber_data = {dataset: file[dataset][:] for dataset in datasets}
    return fiber_data

# --- Load ---
data = load_fiber_data(data_file)

# --- Extract ---
time_vector = data['time_vector']
freqs = data['freqs']
fibers = [data[f'fft_fiber_1_{i}'] for i in range(1, 5)] + [data[f'fft_fiber_2_{i}'] for i in range(1, 5)]
fiber_names = [f'1_{i}' for i in range(1, 5)] + [f'2_{i}' for i in range(1, 5)]

# --- Process Time ---
if time_vector[0] > 1e17:
    time_vector = time_vector / 1e9

time_formatted = pd.to_datetime(time_vector, unit='s')
num_times = len(time_vector)

# --- UI: Dropdown + Button ---
freqs_to_plot = [18.7, 37.4, 56, 57.6, 76.5, 115.6]
freq_selected = st.selectbox("Select Frequency (Hz)", freqs_to_plot, index=1)
start_button = st.button("Start Visualization")
export_button = st.button("Export Video")

# --- Helper: Peak Magnitude ---
def find_peak_magnitude(fft_data, time_idx, freq_target, window=0.2):
    indices = np.where((freqs >= freq_target - window) & (freqs <= freq_target + window))[0]
    if indices.size > 0:
        peak_vals = fft_data[indices, time_idx]
        max_idx = np.argmax(peak_vals)
        return peak_vals[max_idx]
    return 0

# --- Polar Fiber Angles (from original script) ---
fiber_angles_deg = [270, 306, 342, 18, 90, 126, 162, 198]
fiber_angles_rad = np.radians(fiber_angles_deg)

# --- Plot Functions ---
def plot_bearing(time_idx, freq_selected):
    magnitudes = [find_peak_magnitude(f, time_idx, freq_selected) for f in fibers]
    max_mag = max(magnitudes) or 1.0
    radii = [1 + (m / max_mag) * 1.5 for m in magnitudes]

    fig = go.Figure()

    # Bearing circle
    fig.add_trace(go.Scatterpolar(
        r=[1]*361,
        theta=list(range(361)),
        mode='lines',
        line=dict(color='gray', dash='dot'),
        showlegend=False))

    # Sensor locations
    fig.add_trace(go.Scatterpolar(
        r=[1]*len(fiber_angles_deg),
        theta=fiber_angles_deg,
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=[f"{name}" for name in fiber_names],
        textposition='top center',
        showlegend=False))

    # FFT magnitude lines
    for i in range(len(fiber_angles_rad)):
        fig.add_trace(go.Scatterpolar(
            r=[1, radii[i]],
            theta=[np.degrees(fiber_angles_rad[i])]*2,
            mode='lines',
            line=dict(color='green', width=2),
            showlegend=False))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 3])),
        margin=dict(t=40, l=0, r=0, b=0),
        title=f"Ball Bearing @ {freq_selected} Hz\n{time_formatted[time_idx].strftime('%H:%M:%S')}"
    )
    return fig

def plot_fft(time_idx):
    fig = go.Figure()
    for i, fiber in enumerate(fibers):
        fig.add_trace(go.Scatter(x=freqs, y=fiber[:, time_idx], name=fiber_names[i]))
    fig.update_layout(title="FFT Spectrum", xaxis_title='Frequency (Hz)', yaxis_title='Magnitude')
    return fig

def plot_magnitude_history(freq_selected):
    data = []
    for i, fiber in enumerate(fibers):
        mags = [find_peak_magnitude(fiber, t, freq_selected) for t in range(num_times)]
        data.append(go.Scatter(x=time_formatted, y=mags, name=fiber_names[i]))
    fig = go.Figure(data)
    fig.update_layout(title="Peak Magnitude History", xaxis_title='Time', yaxis_title='Magnitude')
    return fig

# --- Animate or Export ---
if start_button or export_button:
    img_dir = "frames"
    os.makedirs(img_dir, exist_ok=True)
    filenames = []

    for t in range(num_times):
        fig_bearing = plot_bearing(t, freq_selected)
        fig_fft = plot_fft(t)
        fig_hist = plot_magnitude_history(freq_selected)

        # Combine as subplot
        fig = make_subplots(rows=2, cols=2, specs=[[{"type": "polar"}, {}], [{"colspan": 2}, None]],
                            subplot_titles=("Ball Bearing", "FFT", "History"))
        for trace in fig_bearing.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig_fft.data:
            fig.add_trace(trace, row=1, col=2)
        for trace in fig_hist.data:
            fig.add_trace(trace, row=2, col=1)

        fig.update_layout(height=700, width=1200, showlegend=False)

        if export_button:
            file = f"{img_dir}/frame_{t:04d}.png"
            pio.write_image(fig, file, format='png', width=1200, height=700)
            filenames.append(file)
        else:
            # Create placeholders
            placeholder_bearing = st.empty()
            placeholder_fft = st.empty()
            placeholder_history = st.empty()

            # Animation loop
            for t in range(num_times):
                fig_bearing = plot_bearing(t, freq_selected)
                fig_fft = plot_fft(t)
                fig_hist = plot_magnitude_history(freq_selected)

                placeholder_bearing.plotly_chart(fig_bearing, use_container_width=True, key=f"bearing_{t}")
                placeholder_fft.plotly_chart(fig_fft, use_container_width=True, key=f"fft_{t}")
                placeholder_history.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{t}")

                time.sleep(0.2)

    if export_button:
        video_out = f"bearing_video_{freq_selected}Hz.mp4"
        with imageio.get_writer(video_out, fps=5) as writer:
            for fname in filenames:
                writer.append_data(imageio.imread(fname))
        st.success(f"Video saved as {video_out}")
