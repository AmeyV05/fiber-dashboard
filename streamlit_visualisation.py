import os
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import h5py
import plotly.io as pio
import imageio
from io import BytesIO
import tempfile
import base64
from scipy.interpolate import splprep, splev
import psutil
import gc

try:
    from stqdm import stqdm
except ImportError:
    st.warning("stqdm not found, progress bar disabled. Run `pip install stqdm` for better experience.")
    stqdm = lambda x, **kwargs: x

# --- CONFIG ---
FILE_URL = "https://etprojects.blob.core.windows.net/fiber-processed-test/fiber_all_processed.h5"

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")
st.title("Ball Bearing FFT Visualization")

# Memory usage monitoring
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    return memory_usage_mb

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'download_complete' not in st.session_state:
    st.session_state.download_complete = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'memory_usage' not in st.session_state:
    st.session_state.memory_usage = []

# --- File download ---
@st.cache_data(ttl=3600, max_entries=1)
def download_data():
    temp_file = os.path.join(st.session_state.temp_dir, "fiber_all_processed.h5")
    
    with st.spinner("Downloading data file... (~110MB)"):
        try:
            with requests.get(FILE_URL, stream=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
                r.raise_for_status()
                
                # Create a progress bar
                progress_bar = st.progress(0)
                file_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                # Save chunks
                with open(temp_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Update progress bar
                            if file_size:
                                progress_bar.progress(min(downloaded / file_size, 1.0))
            
            # Check if it's really an HDF5 file
            with open(temp_file, "rb") as f:
                preview = f.read(8)
            if not preview.startswith(b'\x89HDF'):
                st.error("File does not appear to be a valid HDF5 file (bad signature).")
                return None
            
            st.success(f"Download complete! File size: {os.path.getsize(temp_file) / 1e6:.2f} MB")
            return temp_file
            
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None

# --- Load Data Function ---
@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def load_fiber_data(file_path):
    with h5py.File(file_path, 'r') as file:
        datasets = list(file.keys())
        # Load data in a memory-efficient way
        fiber_data = {}
        for dataset in datasets:
            fiber_data[dataset] = file[dataset][:]
            
    return fiber_data

# --- Ensure data is available ---
if st.session_state.data is None:
    h5_file = download_data()
    if h5_file and os.path.exists(h5_file):
        st.session_state.data = load_fiber_data(h5_file)
        st.session_state.download_complete = True
    else:
        st.error("Could not load data file.")
        st.stop()

# --- Extract data ---
data = st.session_state.data
time_vector = data['time_vector']
freqs = data['freqs']
fibers = [data[f'fft_fiber_1_{i}'] for i in range(1, 5)] + [data[f'fft_fiber_2_{i}'] for i in range(1, 5)]
fiber_names = [f'1_{i}' for i in range(1, 5)] + [f'2_{i}' for i in range(1, 5)]

# --- Process Time ---
if time_vector[0] > 1e17:
    time_vector = time_vector / 1e9

time_formatted = pd.to_datetime(time_vector, unit='s')
num_times = len(time_vector)

# --- UI: Dropdown + Sliders ---
st.sidebar.header("Visualization Controls")
freqs_to_plot = [18.7, 37.4, 56, 57.6, 76.5, 115.6]
freq_selected = st.sidebar.selectbox("Select Frequency (Hz)", freqs_to_plot, index=1)

# Add animation speed control
animation_speed = st.sidebar.slider("Animation Speed", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

# Time index slider for manual control
time_idx = st.sidebar.slider("Time Index", min_value=0, max_value=num_times-1, value=0)

# Animation options
col1, col2 = st.sidebar.columns(2)
start_button = col1.button("Start Animation")
stop_button = col2.button("Stop Animation")

# Export options
export_button = st.sidebar.button("Export Video")
export_quality = st.sidebar.select_slider("Video Quality", options=["Low", "Medium", "High"], value="Medium")
export_fps = st.sidebar.slider("Frames per Second", min_value=1, max_value=30, value=5)

# Memory usage info
if st.sidebar.checkbox("Show Memory Usage", value=False):
    current_memory = get_memory_usage()
    st.session_state.memory_usage.append(current_memory)
    st.sidebar.write(f"Current Memory Usage: {current_memory:.2f} MB")
    if len(st.session_state.memory_usage) > 1:
        st.sidebar.line_chart(st.session_state.memory_usage)

if 'animate' not in st.session_state:
    st.session_state.animate = False

if start_button:
    st.session_state.animate = True
if stop_button:
    st.session_state.animate = False
    # Force garbage collection when animation stops
    gc.collect()

# --- Helper: Peak Magnitude ---
@st.cache_data(ttl=3600, max_entries=100)
def find_peak_magnitude_cached(freq_target, time_idx, window=0.2):
    results = []
    for fiber in fibers:
        indices = np.where((freqs >= freq_target - window) & (freqs <= freq_target + window))[0]
        if indices.size > 0:
            peak_vals = fiber[indices, time_idx]
            max_idx = np.argmax(peak_vals)
            results.append(peak_vals[max_idx])
        else:
            results.append(0)
    return results

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
@st.cache_data(ttl=60, max_entries=100)
def plot_bearing_cached(time_idx, freq_selected):
    magnitudes = find_peak_magnitude_cached(freq_selected, time_idx)
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

    # Add a colored heatmap on the circle based on magnitude
    theta_interp = np.linspace(0, 2*np.pi, 100)
    
    # Convert points to cartesian for interpolation
    points = np.array([np.cos(fiber_angles_rad) * radii, np.sin(fiber_angles_rad) * radii]).T
    
    # Only interpolate if we have enough unique points
    if len(np.unique(points, axis=0)) >= 3:  
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
        xi, yi = splev(np.linspace(0, 1, 100), tck)
        r_interp = np.sqrt(xi**2 + yi**2)
        theta_interp_deg = np.degrees(np.arctan2(yi, xi)) % 360
        
        # Sort by theta for proper line drawing
        sort_idx = np.argsort(theta_interp_deg)
        theta_interp_deg = theta_interp_deg[sort_idx]
        r_interp = r_interp[sort_idx]
        
        # Add interpolated line
        fig.add_trace(go.Scatterpolar(
            r=r_interp,
            theta=theta_interp_deg,
            mode='lines',
            line=dict(color='rgba(255,0,0,0.5)', width=3),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=False))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 3]),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        margin=dict(t=50, l=0, r=0, b=0),
        title=dict(
            text=f"Ball Bearing @ {freq_selected} Hz<br>{time_formatted[time_idx].strftime('%H:%M:%S')}",
            font=dict(size=16)
        ),
        height=400
    )
    return fig

def plot_bearing(time_idx, freq_selected):
    return plot_bearing_cached(time_idx, freq_selected)

@st.cache_data(ttl=60, max_entries=100)
def plot_fft_cached(time_idx, freq_selected):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    fig = go.Figure()
    for i, fiber in enumerate(fibers):
        fig.add_trace(go.Scatter(
            x=freqs, 
            y=fiber[:, time_idx], 
            name=fiber_names[i],
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Add vertical line at selected frequency
    fig.add_vline(x=freq_selected, line=dict(color="red", width=2, dash="dash"))
    
    fig.update_layout(
        title="FFT Spectrum",
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_fft(time_idx):
    return plot_fft_cached(time_idx, freq_selected)

@st.cache_data(ttl=60, max_entries=20)
def get_magnitude_history(freq_selected):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    data = []
    for i, fiber in enumerate(fibers):
        mags = [find_peak_magnitude(fiber, t, freq_selected) for t in range(num_times)]
        data.append({
            'x': time_formatted,
            'y': mags,
            'name': fiber_names[i],
            'line': dict(color=colors[i % len(colors)], width=2)
        })
    
    return data

def plot_magnitude_history(time_idx, freq_selected):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Get cached magnitude history data
    data_list = get_magnitude_history(freq_selected)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces from cached data
    for data_item in data_list:
        fig.add_trace(go.Scatter(**data_item))
    
    # Add vertical line at current time
    current_time = time_formatted[time_idx]
    fig.add_vline(x=current_time, line=dict(color="red", width=2, dash="dash"))
    
    fig.update_layout(
        title="Peak Magnitude History",
        xaxis_title='Time',
        yaxis_title='Magnitude',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Create layout ---
st.subheader(f"Visualization for {freq_selected} Hz")
col_bearing, col_fft = st.columns(2)
col_history = st.container()

placeholder_bearing = col_bearing.empty()
placeholder_fft = col_fft.empty()
placeholder_history = col_history.empty()

# --- Animation Function ---
def create_animation_frames(freq_selected, max_frames=None):
    frames = []
    frame_count = num_times if max_frames is None else min(max_frames, num_times)
    
    # Create a consistent color sequence
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Get magnitude history data for all fibers (reused across frames)
    magnitude_history_data = get_magnitude_history(freq_selected)
    
    for t in stqdm(range(frame_count), desc="Generating frames"):
        # Create the figure with correct subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "polar"}, {"type": "xy"}], [{"colspan": 2, "type": "xy"}, None]],
            subplot_titles=("Ball Bearing", "FFT Spectrum", "Peak Magnitude History")
        )
        
        # Add bearing plot traces
        bearing_fig = plot_bearing(t, freq_selected)
        for trace in bearing_fig.data:
            fig.add_trace(trace, row=1, col=1)
            
        # Add FFT plot traces with explicit colors
        for i, fiber in enumerate(fibers):
            fig.add_trace(
                go.Scatter(
                    x=freqs, 
                    y=fiber[:, t], 
                    name=fiber_names[i],
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=1, col=2
            )
        
        # Add vertical line at selected frequency on FFT plot
        fig.add_shape(
            type="line", 
            x0=freq_selected, 
            y0=0, 
            x1=freq_selected, 
            y1=1, 
            yref="paper",
            xref="x2",
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=2
        )
            
        # Add traces for history plot with explicit colors
        for i, data_item in enumerate(magnitude_history_data):
            fig.add_trace(
                go.Scatter(
                    x=data_item['x'],
                    y=data_item['y'],
                    name=data_item['name'],
                    line=data_item['line']
                ),
                row=2, col=1
            )
        
        # Add vertical line at current time on history plot
        current_time = time_formatted[t]
        fig.add_shape(
            type="line",
            x0=current_time,
            y0=0,
            x1=current_time,
            y1=1,
            yref="paper",
            xref="x3",
            line=dict(color="red", width=2, dash="dash"),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900, 
            width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f"Ball Bearing Visualization @ {freq_selected} Hz - {time_formatted[t].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        img_bytes = fig.to_image(format="png")
        frames.append(img_bytes)
        
        # Force garbage collection periodically to prevent memory buildup
        if t % 10 == 0:
            gc.collect()
    
    return frames

# Function to create a download link for the generated video
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# --- Live Animation ---
if st.session_state.animate:
    t = time_idx
    animation_cycle = 0  # Initialize a cycle counter
    while st.session_state.animate:
        fig_bearing = plot_bearing(t, freq_selected)
        fig_fft = plot_fft(t)
        fig_hist = plot_magnitude_history(t, freq_selected)

        # Use unique keys for each frame, including the animation cycle
        placeholder_bearing.plotly_chart(fig_bearing, use_container_width=True, key=f"bearing_{t}_{freq_selected}_{animation_cycle}")
        placeholder_fft.plotly_chart(fig_fft, use_container_width=True, key=f"fft_{t}_{freq_selected}_{animation_cycle}")
        placeholder_history.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{t}_{freq_selected}_{animation_cycle}")

        t += 1
        if t >= num_times:
            t = 0  # Reset to the beginning for looping
            animation_cycle += 1  # Increment the cycle counter
            # Force garbage collection at the end of each cycle
            gc.collect()
        time.sleep(animation_speed)

# Show static plots based on slider when not animating
else:
    fig_bearing = plot_bearing(time_idx, freq_selected)
    fig_fft = plot_fft(time_idx)
    fig_hist = plot_magnitude_history(time_idx, freq_selected)

    placeholder_bearing.plotly_chart(fig_bearing, use_container_width=True, key="static_bearing")
    placeholder_fft.plotly_chart(fig_fft, use_container_width=True, key="static_fft")
    placeholder_history.plotly_chart(fig_hist, use_container_width=True, key="static_history")

# --- Export Video ---
if export_button:
    with st.spinner("Exporting video... this may take some time"):
        # Set quality based on selection
        if export_quality == "Low":
            frame_count = min(50, num_times)
            video_dims = (800, 600)
        elif export_quality == "Medium":
            frame_count = min(100, num_times)
            video_dims = (1200, 900)
        else:  # High
            frame_count = num_times
            video_dims = (1920, 1080)
        
        try:
            # Generate frames
            frames = create_animation_frames(freq_selected, frame_count)
            
            # Create video file
            video_filename = os.path.join(st.session_state.temp_dir, f"bearing_video_{freq_selected}Hz.mp4")
            
            with imageio.get_writer(video_filename, fps=export_fps) as writer:
                for frame in frames:
                    img = imageio.imread(frame)
                    writer.append_data(img)
            
            # Provide download link
            st.markdown(get_binary_file_downloader_html(video_filename, f'Bearing Video ({freq_selected} Hz)'), unsafe_allow_html=True)
            st.success(f"Video created successfully! Click the link above to download.")
            
            # Clear memory after video export
            frames = None
            gc.collect()
            
        except Exception as e:
            st.error(f"Error creating video: {str(e)}")
            st.info("Try using a lower quality setting or fewer frames to reduce memory usage.")

# Cleanup temporary files when app is closed
# Note: This might not always run in Streamlit cloud environment
def cleanup():
    if os.path.exists(st.session_state.temp_dir):
        import shutil
        shutil.rmtree(st.session_state.temp_dir)

import atexit
atexit.register(cleanup)
