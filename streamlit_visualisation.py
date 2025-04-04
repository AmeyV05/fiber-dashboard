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
import threading
import datetime

# Configure Streamlit to minimize memory usage
try:
    # Reduce memory usage by setting Streamlit's cache sizes
    st.set_option('global.dataFrameSerialization', 'arrow')
    st.set_option('server.maxUploadSize', 10)  # Limit upload size
    
    # Set image mode to "PIL" for more memory efficient image handling
    st.set_option('deprecation.showPyplotGlobalUse', False)
except:
    pass

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

# Print initial memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    return memory_usage_mb

initial_memory = get_memory_usage()
print(f"Initial memory usage: {initial_memory:.2f} MB")

# Run garbage collection at startup
gc.collect()
post_gc_memory = get_memory_usage()
print(f"Memory after initial GC: {post_gc_memory:.2f} MB, Saved: {initial_memory - post_gc_memory:.2f} MB")

# --- Automatic Memory Cleanup --- 
# Moved this function up to avoid NameError
def perform_memory_cleanup(clear_streamlit_cache=False):
    """Perform a full memory cleanup including garbage collection and temp file removal"""
    # Run the Python garbage collector
    before = get_memory_usage()
    gc.collect()
    
    # Clear cache for heavyweight cached functions
    try:
        plot_bearing_cached.clear()
        plot_fft_cached.clear()
        get_magnitude_history.clear()
        find_peak_magnitude_cached.clear()
    except:
        pass
    
    # Optionally clear all Streamlit cache - use cautiously as this reloads all data
    if clear_streamlit_cache and before > 3000:  # Only clear if memory usage is very high
        try:
            st.cache_data.clear()
            print("Cleared all Streamlit cache due to high memory usage")
        except:
            pass
    
    # Clean temporary files that are no longer needed
    temp_dir = st.session_state.temp_dir
    if os.path.exists(temp_dir):
        frames_dir = os.path.join(temp_dir, "frames")
        if os.path.exists(frames_dir):
            for root, dirs, files in os.walk(frames_dir, topdown=False):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except:
                        pass
    
    # Free memory by reducing fiber data size if memory is high
    if before > 2000 and 'fiber_ids_to_load' in globals():
        # If memory usage is high, try to unload unused fibers
        unloaded_count = 0
        try:
            # Only keep currently visible fibers in memory
            visible_fibers = [f'fft_fiber_1_{i}' for i in range(1, 5)] + [f'fft_fiber_2_{i}' for i in range(1, 5)]
            for fiber_id in st.session_state.loaded_fiber_ids[:]:
                if fiber_id not in visible_fibers:
                    # Remove from session state
                    if fiber_id in st.session_state:
                        del st.session_state[fiber_id]
                        st.session_state.loaded_fiber_ids.remove(fiber_id)
                        unloaded_count += 1
            if unloaded_count > 0:
                print(f"Unloaded {unloaded_count} fiber datasets to free memory")
        except Exception as e:
            print(f"Error while unloading fiber data: {str(e)}")
    
    # Run garbage collection again after clearing files
    gc.collect()
    after = get_memory_usage()
    
    # Log cleanup in session state
    st.session_state.last_cleanup_time = datetime.datetime.now()
    cleanup_savings = before - after
    
    # Log to console for debugging
    print(f"Auto cleanup completed at {st.session_state.last_cleanup_time}. Memory before: {before:.2f}MB, after: {after:.2f}MB, saved: {cleanup_savings:.2f}MB")
    return before, after

def auto_cleanup_thread():
    """Background thread that periodically runs memory cleanup"""
    cleanup_count = 0
    while True:
        # Sleep for 10 minutes (600 seconds)
        time.sleep(600)
        
        # Check if auto cleanup is enabled
        if not st.session_state.auto_cleanup:
            continue
        
        cleanup_count += 1
        # Every 3rd cleanup (30 minutes), do a more aggressive cleanup that includes Streamlit cache
        clear_st_cache = (cleanup_count % 3 == 0)
        
        # Perform the cleanup
        print(f"Running scheduled cleanup #{cleanup_count}, clear_streamlit_cache={clear_st_cache}")
        perform_memory_cleanup(clear_streamlit_cache=clear_st_cache)

# --- Memory management ---
def check_memory_threshold(threshold_mb=2000):
    """Check if memory usage exceeds threshold and clear caches if needed"""
    current_memory = get_memory_usage()
    if current_memory > threshold_mb:
        print(f"Memory usage {current_memory:.2f}MB exceeds threshold {threshold_mb}MB, performing cleanup")
        perform_memory_cleanup()
        return True
    return False

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'download_complete' not in st.session_state:
    st.session_state.download_complete = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'memory_usage' not in st.session_state:
    st.session_state.memory_usage = []
if 'last_cleanup_time' not in st.session_state:
    st.session_state.last_cleanup_time = datetime.datetime.now()
if 'auto_cleanup' not in st.session_state:
    st.session_state.auto_cleanup = True
if 'loaded_fiber_ids' not in st.session_state:
    st.session_state.loaded_fiber_ids = []

# Start the auto-cleanup thread
cleanup_thread = threading.Thread(target=auto_cleanup_thread, daemon=True)
cleanup_thread.start()

# --- File download ---
@st.cache_data(ttl=24*3600, max_entries=1)
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

# --- Load Data Function --- (Optimized to reduce memory usage)
@st.cache_data(ttl=24*3600, max_entries=1, show_spinner=True)
def load_minimal_data(file_path):
    """Load only essential metadata and not the full fiber data"""
    try:
        with h5py.File(file_path, 'r') as file:
            # Only load time and frequency data
            print("Loading minimal metadata")
            fiber_data = {}
            
            # Extract time vector with downsampling if needed
            time_dset = file['time_vector']
            if len(time_dset) > 5000:
                # Downsample for better performance
                skip = max(1, len(time_dset) // 2000)
                fiber_data['time_vector'] = time_dset[::skip]
                fiber_data['downsampled'] = True
                fiber_data['downsample_factor'] = skip
                print(f"Downsampled time vector from {len(time_dset)} to {len(fiber_data['time_vector'])} points")
            else:
                fiber_data['time_vector'] = time_dset[:]
                fiber_data['downsampled'] = False
            
            # Load frequency data
            fiber_data['freqs'] = file['freqs'][:]
            
            # List available fibers but don't load them yet
            available_fibers = []
            for key in file.keys():
                if key.startswith('fft_fiber'):
                    available_fibers.append(key)
            
            fiber_data['available_fibers'] = available_fibers
            print(f"Found {len(available_fibers)} available fiber datasets")
                    
        return fiber_data
    except Exception as e:
        print(f"Error in load_minimal_data: {str(e)}")
        # Return minimal default data
        return {
            'time_vector': np.linspace(0, 100, 1000),
            'freqs': np.linspace(0, 200, 1000),
            'available_fibers': [],
            'downsampled': False
        }

# Load a specific fiber data on demand
@st.cache_data(ttl=24*3600, max_entries=16)
def load_fiber_data(file_path, fiber_id):
    """Load a specific fiber dataset on demand"""
    try:
        with h5py.File(file_path, 'r') as file:
            if fiber_id in file:
                # For large datasets, load in chunks
                dset = file[fiber_id]
                shape = dset.shape
                
                # Check if dataset is too large - if so, downsample to reduce memory
                if shape[0] * shape[1] > 5e6:  # Very large dataset
                    print(f"Downsampling large dataset {fiber_id} with shape {shape}")
                    # Downsample - take every Nth point
                    skip_factor = max(1, shape[0] // 1000)
                    return dset[::skip_factor, :]
                    
                elif shape[0] * shape[1] > 1e6:  # Large but manageable
                    print(f"Loading large dataset {fiber_id} in chunks with shape {shape}")
                    chunk_size = min(1000, shape[0])
                    chunks = []
                    for i in range(0, shape[0], chunk_size):
                        end = min(i + chunk_size, shape[0])
                        chunks.append(dset[i:end, :])
                    data = np.vstack(chunks)
                    del chunks
                    gc.collect()
                    return data
                else:
                    return dset[:]
            else:
                return None
    except Exception as e:
        print(f"Error loading fiber data {fiber_id}: {str(e)}")
        return np.zeros((100, 100))  # Return empty data on error

# --- Ensure data is available ---
if st.session_state.data is None:
    h5_file = download_data()
    if h5_file and os.path.exists(h5_file):
        # Only load minimal data at startup
        st.session_state.data = load_minimal_data(h5_file)
        st.session_state.h5_file_path = h5_file
        st.session_state.download_complete = True
    else:
        st.error("Could not load data file.")
        st.stop()

# --- Extract data ---
data = st.session_state.data
time_vector = data['time_vector']
freqs = data['freqs']

# Get fiber IDs to be loaded
fiber_ids_to_load = [f'fft_fiber_1_{i}' for i in range(1, 5)] + [f'fft_fiber_2_{i}' for i in range(1, 5)]
fiber_names = [f'1_{i}' for i in range(1, 5)] + [f'2_{i}' for i in range(1, 5)]

# Load fibers on demand
with st.spinner("Loading fiber data..."):
    # Check which fibers need to be loaded
    fibers_to_load = [f_id for f_id in fiber_ids_to_load if f_id not in st.session_state.loaded_fiber_ids]
    
    if fibers_to_load:
        fiber_loading_progress = st.progress(0)
        for i, fiber_id in enumerate(fibers_to_load):
            if fiber_id not in st.session_state:
                st.session_state[fiber_id] = load_fiber_data(st.session_state.h5_file_path, fiber_id)
                st.session_state.loaded_fiber_ids.append(fiber_id)
            fiber_loading_progress.progress((i+1)/len(fibers_to_load))
        
        # Perform cleanup after loading
        gc.collect()

# Create fiber list using references to session state to avoid duplication
fibers = [st.session_state[f_id] for f_id in fiber_ids_to_load]

# --- Process Time ---
if time_vector[0] > 1e17:
    time_vector = time_vector / 1e9

time_formatted = pd.to_datetime(time_vector, unit='s')
num_times = len(time_vector)

# --- Add a maintenance mode to reduce memory ---
if 'maintenance_mode' not in st.session_state:
    st.session_state.maintenance_mode = False

# Sidebar header
st.sidebar.header("Visualization Controls")

# Only show this option in the sidebar at the top
maintenance_toggle = st.sidebar.checkbox("Enable Low Memory Mode", value=st.session_state.maintenance_mode)
if maintenance_toggle != st.session_state.maintenance_mode:
    # State changed
    st.session_state.maintenance_mode = maintenance_toggle
    if maintenance_toggle:
        # Entering maintenance mode - clear caches and unload data
        perform_memory_cleanup(clear_streamlit_cache=True)
        # Unload all fiber data
        for fiber_id in st.session_state.loaded_fiber_ids[:]:
            if fiber_id in st.session_state:
                del st.session_state[fiber_id]
        st.session_state.loaded_fiber_ids = []
        gc.collect()
        st.experimental_rerun()
    else:
        # Exiting maintenance mode - will reload data on next interaction
        st.experimental_rerun()

# In maintenance mode, show minimal UI and data
if st.session_state.maintenance_mode:
    st.info("🔧 Low Memory Mode Active - Limited features available")
    st.write("Current memory usage: ", f"{get_memory_usage():.2f} MB")
    
    # Show minimal controls
    if st.button("Exit Low Memory Mode"):
        st.session_state.maintenance_mode = False
        st.experimental_rerun()
    
    # Show memory cleanup button
    if st.button("Clear Memory"):
        perform_memory_cleanup(clear_streamlit_cache=True)
        st.success("Memory cleared")
    
    # Early exit - don't load or display data
    st.stop()

# --- UI: Dropdown + Sliders --- (after maintenance mode check)
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

# Add a memory usage meter in the sidebar that updates every 10 seconds
if 'last_memory_update_time' not in st.session_state:
    st.session_state.last_memory_update_time = time.time()

current_time = time.time()
if current_time - st.session_state.last_memory_update_time > 10:  # Update every 10 seconds
    st.session_state.last_memory_update_time = current_time
    current_memory = get_memory_usage()
    st.session_state.memory_usage.append(current_memory)
    # Keep only the last 100 measurements to avoid growing too much
    if len(st.session_state.memory_usage) > 100:
        st.session_state.memory_usage = st.session_state.memory_usage[-100:]

# Add a small memory indicator in the sidebar
memory_indicator = st.sidebar.empty()
current_memory = get_memory_usage()
memory_color = 'green'
if current_memory > 1500:
    memory_color = 'orange'
if current_memory > 2500:
    memory_color = 'red'
memory_indicator.markdown(f"<span style='color:{memory_color};'>Memory: {current_memory:.1f} MB</span>", unsafe_allow_html=True)

# Add a button to enter low memory mode if memory is high
if current_memory > 2000 and not st.session_state.maintenance_mode:
    if st.sidebar.button("⚠️ Memory High - Click to Reduce"):
        st.session_state.maintenance_mode = True
        perform_memory_cleanup(clear_streamlit_cache=True)
        st.experimental_rerun()

# Memory usage info
if st.sidebar.checkbox("Show Memory Usage", value=False):
    st.sidebar.write(f"Current Memory Usage: {current_memory:.2f} MB")
    if len(st.session_state.memory_usage) > 1:
        st.sidebar.line_chart(st.session_state.memory_usage)
    
    # Add auto-cleanup toggle
    st.sidebar.checkbox("Enable Auto-Cleanup (every 10 min)", value=st.session_state.auto_cleanup, key="auto_cleanup")
    
    # Show when last cleanup occurred
    time_diff = datetime.datetime.now() - st.session_state.last_cleanup_time
    minutes_ago = int(time_diff.total_seconds() / 60)
    st.sidebar.write(f"Last cleanup: {minutes_ago} minutes ago")
    
    # Manual cleanup button
    if st.sidebar.button("Run Memory Cleanup Now"):
        before, after = perform_memory_cleanup()
        st.sidebar.success(f"Memory cleanup complete. Before: {before:.2f} MB, After: {after:.2f} MB, Saved: {before - after:.2f} MB")

if 'animate' not in st.session_state:
    st.session_state.animate = False

if start_button:
    st.session_state.animate = True
if stop_button:
    st.session_state.animate = False
    # Force garbage collection when animation stops
    gc.collect()

# --- Helper functions ---
def safe_max(arr):
    """Safely get maximum value from array, handling empty or all-nan arrays"""
    if len(arr) == 0:
        return 0
    
    # Handle NaN values
    arr_no_nan = arr[~np.isnan(arr)] if isinstance(arr, np.ndarray) else arr
    
    if len(arr_no_nan) == 0:
        return 0
    
    return max(arr_no_nan)

# --- Helper: Peak Magnitude ---
@st.cache_data(ttl=300, max_entries=20)
def find_peak_magnitude_cached(freq_target, time_idx, window=0.2):
    results = []
    for fiber in fibers:
        if fiber is None:
            results.append(0)
            continue
            
        try:
            indices = np.where((freqs >= freq_target - window) & (freqs <= freq_target + window))[0]
            if indices.size > 0:
                peak_vals = fiber[indices, time_idx]
                max_idx = np.argmax(peak_vals)
                results.append(peak_vals[max_idx])
            else:
                results.append(0)
        except Exception as e:
            print(f"Error in find_peak_magnitude_cached: {str(e)}")
            results.append(0)
    return results

def find_peak_magnitude(fft_data, time_idx, freq_target, window=0.2):
    if fft_data is None:
        return 0
        
    try:
        indices = np.where((freqs >= freq_target - window) & (freqs <= freq_target + window))[0]
        if indices.size > 0:
            peak_vals = fft_data[indices, time_idx]
            max_idx = np.argmax(peak_vals)
            return peak_vals[max_idx]
    except Exception as e:
        print(f"Error in find_peak_magnitude: {str(e)}")
    
    return 0

# --- Polar Fiber Angles (from original script) ---
fiber_angles_deg = [270, 306, 342, 18, 90, 126, 162, 198]
fiber_angles_rad = np.radians(fiber_angles_deg)

# --- Plot Functions ---
@st.cache_data(ttl=60, max_entries=10)
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
    
    # Debug print to verify point order
    point_order = []
    for i, (angle, r) in enumerate(zip(fiber_angles_deg, radii)):
        point_order.append(f"{fiber_names[i]}: {angle}° ({r:.2f})")
    
    # Add direct lines connecting the points in order (no interpolation)
    # This shows the raw sensor data connection
    ordered_thetas = fiber_angles_deg.copy()
    ordered_rs = radii.copy()
    # Add first point at the end to close the loop
    ordered_thetas.append(fiber_angles_deg[0])
    ordered_rs.append(radii[0])
    
    fig.add_trace(go.Scatterpolar(
        r=ordered_rs,
        theta=ordered_thetas,
        mode='lines',
        line=dict(color='blue', width=2, dash='dot'),
        name="Raw Data",
        showlegend=False
    ))
    
    # Only interpolate if we have enough unique points and valid data
    if len(np.unique(points, axis=0)) >= 3 and not np.isnan(points).any():
        try:
            # Add the first point to the end of the points array for a closed curve
            points_closed = np.vstack((points, points[0]))
            
            # Use periodic boundary condition for a closed curve
            tck, u = splprep([points_closed[:, 0], points_closed[:, 1]], s=0, per=True)
            
            # Use more points for smoother interpolation
            interp_points = 200
            t = np.linspace(0, 1, interp_points)
            xi, yi = splev(t, tck)
            
            # Convert back to polar coordinates
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
            
            # Show the actual interpolation points to visualize how the spline is constructed
            # Take 8 evenly spaced points from the interpolation (matching number of original points)
            sample_indices = np.linspace(0, interp_points-1, len(fiber_angles_deg), dtype=int)
            sample_r = [r_interp[i] for i in sample_indices]
            sample_theta = [theta_interp_deg[i] for i in sample_indices]
            
            fig.add_trace(go.Scatterpolar(
                r=sample_r,
                theta=sample_theta,
                mode='markers',
                marker=dict(size=8, color='red'),
                showlegend=False
            ))
        except Exception as e:
            # If interpolation fails, just skip it without crashing
            print(f"Interpolation skipped: {str(e)}")
            pass

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

@st.cache_data(ttl=60, max_entries=10)
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

@st.cache_data(ttl=60, max_entries=5)
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
def create_animation_frames(freq_selected, max_frames=None, quality="Medium"):
    # Create a temp directory for frames
    frames_dir = os.path.join(st.session_state.temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []
    
    # Set image resolution based on quality
    if quality == "Low":
        img_width, img_height = 800, 600
    elif quality == "Medium":
        img_width, img_height = 1200, 900
    else:  # High
        img_width, img_height = 1920, 1080
    
    frame_count = num_times if max_frames is None else min(max_frames, num_times)
    
    # Create a consistent color sequence
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Get magnitude history data for all fibers (reused across frames)
    magnitude_history_data = get_magnitude_history(freq_selected)
    
    # Pre-compute y-axis ranges for consistent scaling
    max_fft_value = 0
    max_history_value = 0
    
    # Find max values for consistent scaling
    for i, fiber in enumerate(fibers):
        max_fft_value = max(max_fft_value, np.max(fiber[:, :frame_count]))
        history_values = [find_peak_magnitude(fiber, t, freq_selected) for t in range(frame_count)]
        max_history_value = max(max_history_value, max(history_values) if history_values else 0)
    
    # Add small padding to max values and ensure minimum scale
    max_fft_value = max(max_fft_value * 1.1, 0.05)  # Ensure minimum y-scale for visibility
    max_history_value *= 1.1
    
    # Determine consistent x-axis range for FFT spectrum
    max_freq_to_show = min(450, max(freqs))
    
    # Create a progress bar and status for frame generation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Only load necessary data into memory
    for t in stqdm(range(frame_count), desc="Generating frames"):
        # Update progress
        progress = int((t + 1) / frame_count * 100)
        progress_bar.progress(progress / 100)
        status_text.text(f"Generating frame {t+1}/{frame_count} ({progress}%)")
        
        # Create a new figure for each frame to avoid memory build-up
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "polar", "rowspan": 1}, {"rowspan": 1}],
                   [{"colspan": 2}, None]],
            subplot_titles=("Bearing Visualization", "FFT Spectrum", "Magnitude History"),
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            row_heights=[0.5, 0.5]
        )
        
        # Similar to static plot functions but optimized for animation
        # Bearing plot (row 1, col 1) - polar plot
        magnitudes = []
        for fiber_idx, fiber in enumerate(fibers):
            # Calculate magnitude for current time step
            mag = find_peak_magnitude(fiber, t, freq_selected)
            magnitudes.append(mag)
        
        # Scale magnitudes for visualization
        max_mag = max(magnitudes) or 1.0
        radii = [1 + (m / max_mag) * 1.5 for m in magnitudes]
        
        # Add bearing circle
        fig.add_trace(
            go.Scatterpolar(
                r=[1]*361,
                theta=np.linspace(0, 360, 361),
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add fiber magnitudes
        for i, (r, ang, name) in enumerate(zip(radii, fiber_angles_deg, fiber_names)):
            # Plot fiber magnitude as line from center
            fig.add_trace(
                go.Scatterpolar(
                    r=[0, r],
                    theta=[ang, ang],
                    mode='lines+markers',
                    name=f"Fiber {name}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=[0, 8]),
                ),
                row=1, col=1
            )
        
        # FFT Plot (row 1, col 2)
        # Plot each fiber's FFT at the current time step
        for i, fiber in enumerate(fibers):
            # Plot subset of frequency range to improve performance
            plot_freqs = freqs[freqs <= max_freq_to_show]
            plot_values = fiber[freqs <= max_freq_to_show, t]
            
            fig.add_trace(
                go.Scatter(
                    x=plot_freqs,
                    y=plot_values,
                    name=f"Fiber {fiber_names[i]}",
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add vertical line at selected frequency
        fig.add_vline(
            x=freq_selected,
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=2
        )
        
        # Add a more visible point at the peak frequency for each fiber
        for i, fiber in enumerate(fibers):
            # Get the peak value around the selected frequency
            peak_value = find_peak_magnitude(fiber, t, freq_selected)
            # Highlight the peak with a marker
            if peak_value > 0:
                freq_indices = np.where((freqs >= freq_selected - 0.2) & (freqs <= freq_selected + 0.2))[0]
                if len(freq_indices) > 0:
                    max_idx = np.argmax(fiber[freq_indices, t])
                    peak_freq = freqs[freq_indices[max_idx]]
                    fig.add_trace(
                        go.Scatter(
                            x=[peak_freq],
                            y=[peak_value],
                            mode='markers',
                            marker=dict(size=10, color=colors[i % len(colors)]),
                            showlegend=False
                        ),
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
        
        # Update layout for consistent appearance
        fig.update_layout(
            title={
                'text': f"Ball Bearing FFT at {time_formatted[t]}",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            polar={
                'radialaxis': {'range': [0, 3], 'showticklabels': False, 'ticks': ''},
                'angularaxis': {'direction': 'clockwise', 'rotation': 90}
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            autosize=True,
            margin=dict(l=30, r=30, t=50, b=30),
            height=img_height,
            width=img_width
        )
        
        # Update y-axis range for FFT plot for consistent scaling
        fig.update_yaxes(range=[0, max_fft_value], row=1, col=2)
        
        # Update y-axis range for history plot for consistent scaling
        fig.update_yaxes(range=[0, max_history_value], row=2, col=1)
        
        # Update x-axis range for FFT plot
        fig.update_xaxes(range=[0, max_freq_to_show], title="Frequency (Hz)", row=1, col=2)
        
        # Update x-axis for history plot
        fig.update_xaxes(title="Time", row=2, col=1)
        fig.update_yaxes(title="Magnitude", row=2, col=1)
        
        # Update polar axis for bearing plot
        fig.update_layout(
            polar={
                'radialaxis': {'range': [0, 3], 'showticklabels': False, 'ticks': ''},
                'angularaxis': {'direction': 'clockwise', 'rotation': 90}
            },
        )
        
        # Update axes labels and styles
        fig.update_yaxes(title="Magnitude", row=1, col=2)
        fig.update_xaxes(title="Frequency (Hz)", row=1, col=2)
        fig.update_xaxes(title="Time", row=2, col=1)
        fig.update_yaxes(title="Magnitude", row=2, col=1)
        
        fig.update_xaxes(showticklabels=True, showgrid=True, zeroline=True, row=1, col=2)
        fig.update_yaxes(showticklabels=True, showgrid=True, zeroline=True, row=1, col=2)
        fig.update_xaxes(showticklabels=True, showgrid=True, zeroline=True, row=2, col=1)
        fig.update_yaxes(showticklabels=True, showgrid=True, zeroline=True, row=2, col=1)
        
        # Save frame to disk with higher quality
        frame_path = os.path.join(frames_dir, f"frame_{t:04d}.png")
        fig.write_image(frame_path, width=img_width, height=img_height, scale=2)  # scale=2 for higher resolution
        frame_paths.append(frame_path)
        
        # Close figure to free memory
        fig.data = []
        fig = None
        
        # Force garbage collection periodically
        if t % 5 == 0:
            gc.collect()
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return frame_paths, frames_dir

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
    memory_check_counter = 0  # Counter for memory checks
    while st.session_state.animate:
        fig_bearing = plot_bearing(t, freq_selected)
        fig_fft = plot_fft(t)
        fig_hist = plot_magnitude_history(t, freq_selected)

        # Use unique keys for each frame, including the animation cycle
        placeholder_bearing.plotly_chart(fig_bearing, use_container_width=True, key=f"bearing_{t}_{freq_selected}_{animation_cycle}")
        placeholder_fft.plotly_chart(fig_fft, use_container_width=True, key=f"fft_{t}_{freq_selected}_{animation_cycle}")
        placeholder_history.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{t}_{freq_selected}_{animation_cycle}")

        t += 1
        memory_check_counter += 1
        
        # Check memory usage every 5 frames
        if memory_check_counter >= 5:
            memory_check_counter = 0
            check_memory_threshold(1500)  # Lower threshold to be more proactive
        
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

# Add a recovery mechanism in case app runs out of memory
if 'out_of_memory' not in st.session_state:
    st.session_state.out_of_memory = False

# Check if we're in recovery mode
if st.session_state.out_of_memory:
    st.error("The application ran out of memory. We've cleared caches and reset data to recover.")
    
    # Clear all data
    for key in list(st.session_state.keys()):
        if key not in ['out_of_memory', 'temp_dir', 'auto_cleanup']:
            try:
                del st.session_state[key]
            except:
                pass
    
    st.session_state.data = None
    st.session_state.loaded_fiber_ids = []
    
    # Clear all caches
    try:
        st.cache_data.clear()
    except:
        pass
    
    # Force garbage collection
    gc.collect()
    
    # Reset the flag
    st.session_state.out_of_memory = False
    
    st.info("Memory has been cleared. Please refresh the page to restart the application.")
    st.stop()

# --- Add memory safeguards throughout the app ---
def check_memory_critical():
    """Check if memory usage is critical and we need to recover"""
    try:
        memory = get_memory_usage()
        if memory > 3500:  # Critical memory threshold - 3.5GB
            st.session_state.out_of_memory = True
            # Force reload
            st.experimental_rerun()
            return True
        return False
    except:
        return False

# Call this check in key parts of the application
check_memory_critical()

# --- Export Video ---
if export_button:
    # First verify memory is sufficient
    if get_memory_usage() > 2500:  # 2.5GB - risky for export
        st.warning("Memory usage is already high, which may cause the export to fail. Please try restarting the app first.")
        if st.button("Continue Anyway"):
            pass  # Continue with export
        else:
            st.stop()  # Stop execution
            
    with st.spinner("Preparing for video export..."):
        # Set quality based on selection
        if export_quality == "Low":
            frame_count = min(30, num_times)  # Reduced from 50 to 30
            batch_size = 5  # Reduced batch size
            img_width, img_height = 640, 480  # Smaller images
            scale = 1
        elif export_quality == "Medium":
            frame_count = min(60, num_times)  # Reduced from 100 to 60
            batch_size = 10  # Reduced batch size
            img_width, img_height = 1024, 768  # Medium sized images
            scale = 1.5
        else:  # High
            frame_count = min(100, num_times)  # Reduced from unlimited to max 100
            batch_size = 20
            img_width, img_height = 1280, 960  # Moderate high quality
            scale = 2
        
        memory_before = get_memory_usage()
        st.info(f"Memory usage before export: {memory_before:.2f} MB")
        
        try:
            video_filename = os.path.join(st.session_state.temp_dir, f"bearing_video_{freq_selected}Hz.mp4")
            
            # Create video writer - use minimal quality to save memory
            with imageio.get_writer(video_filename, fps=export_fps, quality=7) as writer:
                # Process frames in smaller batches to control memory usage
                num_batches = (frame_count + batch_size - 1) // batch_size
                
                # Create progress tracking
                video_progress = st.progress(0)
                status_text = st.empty()
                
                for batch_idx in range(num_batches):
                    # Check if memory is becoming critical during export
                    if check_memory_critical():
                        break
                        
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, frame_count)
                    batch_frames = range(start_idx, end_idx)
                    
                    status_text.text(f"Processing batch {batch_idx+1}/{num_batches} (frames {start_idx}-{end_idx-1})")
                    
                    # Generate frames for this batch only
                    with st.spinner(f"Generating frames {start_idx}-{end_idx-1}..."):
                        # Create a temp directory for this batch's frames
                        batch_frames_dir = os.path.join(st.session_state.temp_dir, f"frames_batch_{batch_idx}")
                        os.makedirs(batch_frames_dir, exist_ok=True)
                        
                        # Process each frame in the batch
                        for i, t in enumerate(batch_frames):
                            # Skip if memory is critical
                            if check_memory_critical():
                                break
                            
                            # Use simplified plotting for video export to save memory
                            # Just capture the essential info in a simple clean plot
                            fig = go.Figure()
                            
                            # Get magnitudes for this time step
                            magnitudes = []
                            for fiber_idx, fiber in enumerate(fibers):
                                mag = find_peak_magnitude(fiber, t, freq_selected)
                                magnitudes.append(mag)
                            
                            # Create a simple bar plot of magnitudes
                            fig.add_trace(go.Bar(
                                x=fiber_names,
                                y=magnitudes,
                                marker_color='blue'
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                title=f"Fiber Magnitudes at {freq_selected} Hz - {time_formatted[t]}",
                                xaxis_title="Fiber",
                                yaxis_title="Magnitude",
                                height=img_height,
                                width=img_width
                            )
                            
                            # Save frame
                            frame_path = os.path.join(batch_frames_dir, f"frame_{t:04d}.png")
                            fig.write_image(frame_path, width=img_width, height=img_height, scale=scale)
                            
                            # Close figure to free memory
                            fig.data = []
                            fig = None
                            
                            # Update batch progress
                            sub_progress = (i + 1) / len(batch_frames)
                            total_progress = (batch_idx * batch_size + i + 1) / frame_count
                            video_progress.progress(total_progress)
                            
                            # Force garbage collection for every frame in export mode
                            gc.collect()
                    
                    # Add frames to video
                    for t in batch_frames:
                        frame_path = os.path.join(batch_frames_dir, f"frame_{t:04d}.png")
                        # Read image and add to video
                        img = imageio.imread(frame_path)
                        writer.append_data(img)
                        # Remove frame after adding to video
                        os.remove(frame_path)
                    
                    # Clean up batch directory
                    try:
                        os.rmdir(batch_frames_dir)
                    except:
                        pass
                    
                    # Run memory cleanup after each batch
                    perform_memory_cleanup()
            
            # Clear progress indicators
            video_progress.empty()
            status_text.empty()
            
            # Report final memory usage
            memory_final = get_memory_usage()
            st.info(f"Final memory usage: {memory_final:.2f} MB")
            
            # Provide download link
            st.markdown(get_binary_file_downloader_html(video_filename, f'Bearing Video ({freq_selected} Hz)'), unsafe_allow_html=True)
            st.success(f"Video created successfully! Click the link above to download.")
            
            # Clear memory after video export
            perform_memory_cleanup()
            
        except Exception as e:
            st.error(f"Error creating video: {str(e)}")
            st.info("Try using a lower quality setting or fewer frames to reduce memory usage.")
            # Clear up any partial files
            cleanup_temp_files()

# Add memory info at the bottom of the page
if st.checkbox("Show Memory Information", value=False):
    current_memory = get_memory_usage()
    st.write(f"Current memory usage: {current_memory:.2f} MB")
    
    # Add button for garbage collection
    col1, col2 = st.columns(2)
    if col1.button("Standard Memory Cleanup"):
        before, after = perform_memory_cleanup(clear_streamlit_cache=False)
        st.success(f"Memory cleanup complete. Before: {before:.2f} MB, After: {after:.2f} MB, Saved: {before - after:.2f} MB")
    
    if col2.button("Full Cache Cleanup (Reload Data)"):
        before, after = perform_memory_cleanup(clear_streamlit_cache=True)
        st.warning(f"Full cache cleanup complete. Before: {before:.2f} MB, After: {after:.2f} MB, Saved: {before - after:.2f} MB. Data will reload.")
        st.session_state.data = None  # Force data reload on next interaction
        # Clear loaded fiber IDs to force reload
        st.session_state.loaded_fiber_ids = []
        for fiber_id in fiber_ids_to_load:
            if fiber_id in st.session_state:
                del st.session_state[fiber_id]
        gc.collect()

# Add function to clear temp files
def cleanup_temp_files():
    with st.spinner("Cleaning up temporary files..."):
        temp_dir = st.session_state.temp_dir
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except:
                        pass
                for dir in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir))
                    except:
                        pass
        st.success("Temporary files cleaned up.")
        
        # Update last cleanup time
        st.session_state.last_cleanup_time = datetime.datetime.now()

# Add button to manually clear temp files
if st.sidebar.button("Clean Temporary Files"):
    cleanup_temp_files()

# Cleanup temporary files when app is closed
# Note: This might not always run in Streamlit cloud environment
def cleanup():
    if os.path.exists(st.session_state.temp_dir):
        import shutil
        shutil.rmtree(st.session_state.temp_dir)

import atexit
atexit.register(cleanup)

# Add button to unload data and free memory
if st.sidebar.button("Unload Unused Data"):
    # Count how many fiber datasets are loaded
    loaded_count = len(st.session_state.loaded_fiber_ids)
    if loaded_count > 0:
        before = get_memory_usage()
        # Keep only the fibers needed for current view
        current_freq_fibers = []
        for f_id in st.session_state.loaded_fiber_ids:
            if f_id in fiber_ids_to_load:
                current_freq_fibers.append(f_id)
            else:
                # Remove fibers that aren't currently needed
                if f_id in st.session_state:
                    del st.session_state[f_id]
        
        # Update loaded fibers list
        st.session_state.loaded_fiber_ids = current_freq_fibers
        gc.collect()
        after = get_memory_usage()
        st.sidebar.success(f"Unloaded unused data. Before: {before:.2f} MB, After: {after:.2f} MB, Saved: {before - after:.2f} MB")
    else:
        st.sidebar.info("No unused fiber data to unload.")
