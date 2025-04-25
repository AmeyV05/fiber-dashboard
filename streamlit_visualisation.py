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
import imageio.v2 as imageio
from io import BytesIO
import tempfile
import base64
from scipy.interpolate import splprep, splev
import psutil
import gc
import threading
import datetime
import sys
import matplotlib.pyplot as plt
import scipy.interpolate

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
FILE_URL = "https://etprojects.blob.core.windows.net/fiber-processed-test/fiber_fft_all_processed.h5"

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
def auto_cleanup_thread():
    """Background thread that periodically checks memory usage and performs cleanup"""
    try:
        while True:
            time.sleep(300)  # Sleep for 5 minutes between checks
            
            # Skip if cleanup is disabled
            if not st.session_state.get('auto_cleanup', False):
                continue
                
            # Skip if we don't have the cleanup lock
            if not hasattr(st.session_state, 'cleanup_lock'):
                break
            
            try:
                current_memory = get_memory_usage()
                
                # Only perform cleanup if memory usage is high
                if current_memory > 2000:  # 2GB threshold
                    print(f"Auto cleanup triggered - Memory usage: {current_memory:.2f}MB")
                    
                    # Use threading lock to prevent concurrent cleanup
                    if st.session_state.cleanup_lock.acquire(blocking=False):
                        try:
                            perform_memory_cleanup()
                        finally:
                            st.session_state.cleanup_lock.release()
                
            except Exception as e:
                print(f"Error in cleanup cycle: {str(e)}")
                
    except Exception as e:
        print(f"Auto cleanup thread terminated: {str(e)}")

# --- Session State Initialization ---
if 'initialized' not in st.session_state:
    # Initialize all required session state variables
    st.session_state.initialized = True
    st.session_state.data = None
    st.session_state.download_complete = False
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.memory_usage = []
    st.session_state.last_cleanup_time = datetime.datetime.now()
    st.session_state.auto_cleanup = True
    st.session_state.loaded_fiber_ids = []
    st.session_state.maintenance_mode = False
    st.session_state.cleanup_lock = threading.Lock()
    st.session_state.out_of_memory = False
    st.session_state.animate = False

# Start the auto-cleanup thread only if it hasn't been started
if ('cleanup_thread' not in st.session_state or 
    not st.session_state.get('cleanup_thread') or 
    not st.session_state.cleanup_thread.is_alive()):
    
    cleanup_thread = threading.Thread(
        target=auto_cleanup_thread,
        daemon=True,
        name="AutoCleanupThread"
    )
    cleanup_thread.start()
    st.session_state.cleanup_thread = cleanup_thread
    print("Started new auto-cleanup thread")

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
                if key.startswith('fft_fiber') and ('_mags' in key or '_phases' in key):
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

# Get fiber IDs to be loaded (now with both mags and phases)
fiber_mags_ids_to_load = [f'fft_fiber_1_{i}_mags' for i in range(1, 5)] + [f'fft_fiber_2_{i}_mags' for i in range(1, 5)]
fiber_phases_ids_to_load = [f'fft_fiber_1_{i}_phases' for i in range(1, 5)] + [f'fft_fiber_2_{i}_phases' for i in range(1, 5)]
fiber_ids_to_load = fiber_mags_ids_to_load + fiber_phases_ids_to_load
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

# Create fiber lists using references to session state to avoid duplication
fibers_mags = [st.session_state[f_id] for f_id in fiber_mags_ids_to_load]
fibers_phases = [st.session_state[f_id] for f_id in fiber_phases_ids_to_load]

# --- Process Time ---
if time_vector[0] > 1e17:
    time_vector = time_vector / 1e9

time_formatted = pd.to_datetime(time_vector, unit='s')
num_times = len(time_vector)

# --- Add a maintenance mode to reduce memory ---
if 'maintenance_mode' not in st.session_state:
    st.session_state.maintenance_mode = False

# --- Sidebar header ---
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
        st.rerun()
    else:
        # Exiting maintenance mode - will reload data on next interaction
        st.rerun()

# In maintenance mode, show minimal UI and data
if st.session_state.maintenance_mode:
    st.info("🔧 Low Memory Mode Active - Limited features available")
    st.write("Current memory usage: ", f"{get_memory_usage():.2f} MB")
    
    # Show minimal controls with more detailed information
    st.warning("Low Memory Mode reduces functionality to minimize memory usage. Only essential features are available.")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    if col1.button("Exit Low Memory Mode"):
        st.session_state.maintenance_mode = False
        st.rerun()
    
    # Show memory cleanup button with more options
    if col2.button("Force Memory Cleanup"):
        before, after = perform_memory_cleanup(clear_streamlit_cache=True)
        st.success(f"Memory cleared: {before:.2f}MB → {after:.2f}MB (saved {before-after:.2f}MB)")
    
    # Add some diagnostics in low memory mode
    st.subheader("Memory Diagnostics")
    
    # Show loaded objects and their sizes
    if st.checkbox("Show Session State Objects", value=False):
        state_info = []
        for key in st.session_state:
            try:
                obj_size = sys.getsizeof(st.session_state[key]) / (1024 * 1024)  # Size in MB
                state_info.append({"Object": key, "Size (MB)": f"{obj_size:.2f}"})
            except:
                state_info.append({"Object": key, "Size": "Unknown"})
        
        # Sort by size (largest first)
        state_info.sort(key=lambda x: float(x["Size (MB)"].replace("Unknown", "0")), reverse=True)
        
        # Display as a table
        st.table(state_info[:20])  # Show top 20 largest objects
    
    # Show loaded fibers
    if st.checkbox("Show Loaded Fiber Data", value=False):
        st.write(f"Loaded fibers: {len(st.session_state.loaded_fiber_ids)}")
        st.write(st.session_state.loaded_fiber_ids)
    
    # Option to completely reset the app
    if st.button("⚠️ Reset Application (Clear All Data)"):
        # Clear all session state except maintenance mode flag
        maintenance_mode = st.session_state.maintenance_mode
        temp_dir = st.session_state.temp_dir if 'temp_dir' in st.session_state else None
        
        for key in list(st.session_state.keys()):
            if key not in ['out_of_memory', 'temp_dir', 'auto_cleanup']:
                try:
                    del st.session_state[key]
                except Exception as e:
                    print(f"Error deleting session state key {key}: {str(e)}")
                    pass
        
        # Restore maintenance mode and temp dir
        st.session_state.maintenance_mode = maintenance_mode
        if temp_dir:
            st.session_state.temp_dir = temp_dir
        
        # Clear all caches
        try:
            st.cache_data.clear()
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            pass
        
        # Force garbage collection
        gc.collect()
        
        st.success("Application reset complete. All data cleared from memory.")
        st.rerun()
    
    # Early exit - don't load or display data
    st.stop()

# --- UI: Dropdown + Sliders ---
freqs_to_plot = [18.7, 37.4, 56, 57.6, 76.5, 115.6]
freq_selected = st.sidebar.selectbox("Select Frequency (Hz)", freqs_to_plot, index=1)

# --- Animation speed control ---
animation_fps = st.sidebar.slider("Animation FPS", min_value=2, max_value=10, value=5, step=1)

# --- Time index slider for manual control ---
time_idx = st.sidebar.slider("Time Index", min_value=0, max_value=num_times-1, value=0)

# --- Display selected time ---
st.sidebar.info(f"Selected Time: {time_formatted[time_idx].strftime('%Y-%m-%d %H:%M:%S')}")

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
if current_memory > 1500 and not st.session_state.maintenance_mode:
    if st.sidebar.button("⚠️ Memory High - Click to Reduce"):
        st.session_state.maintenance_mode = True
        perform_memory_cleanup(clear_streamlit_cache=True)
        st.rerun()

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
    mag_results = []
    phase_results = []
    
    for i, (fiber_mag, fiber_phase) in enumerate(zip(fibers_mags, fibers_phases)):
        if fiber_mag is None or fiber_phase is None:
            mag_results.append(0)
            phase_results.append(0)
            continue
            
        try:
            indices = np.where((freqs >= freq_target - window) & (freqs <= freq_target + window))[0]
            if indices.size > 0:
                # Get magnitude values at the frequency range
                mag_vals = fiber_mag[indices, time_idx]
                max_idx = np.argmax(mag_vals)
                mag_results.append(mag_vals[max_idx])
                
                # Get the corresponding phase value at the same index
                phase_vals = fiber_phase[indices, time_idx]
                phase_results.append(phase_vals[max_idx])
            else:
                mag_results.append(0)
                phase_results.append(0)
        except Exception as e:
            print(f"Error in find_peak_magnitude_cached: {str(e)}")
            mag_results.append(0)
            phase_results.append(0)
            
    return mag_results, phase_results

def find_peak_magnitude(fft_data_mag, fft_data_phase, time_idx, freq_target, window=0.2):
    if fft_data_mag is None or fft_data_phase is None:
        return 0, 0
        
    try:
        indices = np.where((freqs >= freq_target - window) & (freqs <= freq_target + window))[0]
        if indices.size > 0:
            mag_vals = fft_data_mag[indices, time_idx]
            max_idx = np.argmax(mag_vals)
            phase_val = fft_data_phase[indices[max_idx], time_idx]
            return mag_vals[max_idx], phase_val
    except Exception as e:
        print(f"Error in find_peak_magnitude: {str(e)}")
    
    return 0, 0

# --- FFT Plot Function ---
@st.cache_data(ttl=60, max_entries=10)
def plot_fft_cached(time_idx, freq_selected):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    fig = go.Figure()
    for i, fiber in enumerate(fibers_mags):
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

# --- Magnitude History Plot Function ---
@st.cache_data(ttl=60, max_entries=5)
def get_magnitude_history(freq_selected):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    data = []
    for i, fiber_mag in enumerate(fibers_mags):
        fiber_phase = fibers_phases[i]
        mags = []
        for t in range(num_times):
            mag, _ = find_peak_magnitude(fiber_mag, fiber_phase, t, freq_selected)
            mags.append(mag)
            
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

# --- Polar Fiber Angles (from original script) ---
fiber_angles_deg = [315, 360, 45, 90, 135, 180, 225, 270]
fiber_angles_rad = np.radians(fiber_angles_deg)

# --- Animated Bearing Plot ---
@st.cache_data(ttl=60, max_entries=10)
def plot_bearing_animated_cached(time_idx, freq_selected, fps=30):
    # Get magnitude and phase values at the selected frequency and time
    magnitudes, phases = find_peak_magnitude_cached(freq_selected, time_idx)
    
    # Get timestamp for the title
    timestamp = time_formatted[time_idx].strftime('%Y-%m-%d %H:%M:%S')
    
    # Create the animated bearing plot
    fig = create_animated_bearing(magnitudes, phases, freq_selected, fps=fps, timestamp=timestamp)
    
    return fig

def create_animated_bearing(magnitudes, phases, freq_selected, fps=5, timestamp=None):
    """
    Create an animated bearing plot where each fiber location 
    shows a sinusoidal signal x(t) = A sin(wt+phi) over one cycle period T = 1/f
    """
    # Create base figure
    fig = go.Figure()
    
    # Calculate period T = 1/f for one complete cycle
    T = 1.0 / freq_selected
    
    # Create time vector with 100 points over one period
    num_points = 200
    tvec = np.linspace(0, 5*T, num_points)
    
    # Create frames for the animation - we'll show the full cycle regardless of fps
    frames = []
    for frame_idx, t in enumerate(tvec):
        # Calculate current radial values based on sine wave formula
        radii = []
        for mag, phase in zip(magnitudes, phases):
            # x(t) = A sin(wt + phi) where t goes from 0 to T
            val = mag * np.sin(2 * np.pi * freq_selected * t + phase)
            # Scale to ensure positive values and normalize with max magnitude
            max_mag = max(magnitudes) or 1.0
            scaled_val = 1 + ((val + mag) / (2 * max_mag)) * 1.5
            radii.append(scaled_val)
        
        # Create frame data
        frame_data = []
        
        # Add bearing circle
        frame_data.append(
            go.Scatterpolar(
                r=[1]*361,
                theta=list(range(361)),
                mode='lines',
                line=dict(color='gray', dash='dot'),
                showlegend=False
            )
        )
        
        # Add sensor locations with smaller blue dots
        frame_data.append(
            go.Scatterpolar(
                r=[1]*len(fiber_angles_deg),
                theta=fiber_angles_deg,
                mode='markers+text',
                marker=dict(size=8, color='blue'),
                text=[f"{name}" for name in fiber_names],
                textposition='top center',
                showlegend=False
            )
        )
        
        # Add magnitude lines in green
        for i in range(len(fiber_angles_rad)):
            frame_data.append(
                go.Scatterpolar(
                    r=[1, radii[i]],
                    theta=[np.degrees(fiber_angles_rad[i])]*2,
                    mode='lines',
                    line=dict(color='green', width=2),
                    showlegend=False
                )
            )
        
        # Add direct lines connecting the points in order (dotted blue line)
        ordered_thetas = fiber_angles_deg.copy()
        ordered_rs = radii.copy()
        # Add first point at the end to close the loop
        ordered_thetas.append(fiber_angles_deg[0])
        ordered_rs.append(radii[0])
        
        frame_data.append(
            go.Scatterpolar(
                r=ordered_rs,
                theta=ordered_thetas,
                mode='lines',
                line=dict(color='blue', width=2, dash='dot'),
                showlegend=False
            )
        )
        
        # Convert points to cartesian for interpolation
        points = np.array([np.cos(fiber_angles_rad) * radii, 
                          np.sin(fiber_angles_rad) * radii]).T
        
        # Only interpolate if we have enough unique points and valid data
        if len(np.unique(points, axis=0)) >= 3 and not np.isnan(points).any():
            try:
                # Add the first point to the end of the points array for a closed curve
                points_closed = np.vstack((points, points[0]))
                
                # Use periodic boundary condition for a closed curve
                tck, u = splprep([points_closed[:, 0], points_closed[:, 1]], s=0, per=True)
                
                # Use more points for smoother interpolation
                interp_points = 200
                t_interp = np.linspace(0, 1, interp_points)
                xi, yi = splev(t_interp, tck)
                
                # Convert back to polar coordinates
                r_interp = np.sqrt(xi**2 + yi**2)
                theta_interp_deg = np.degrees(np.arctan2(yi, xi)) % 360
                
                # Sort by theta for proper line drawing
                sort_idx = np.argsort(theta_interp_deg)
                theta_interp_deg = theta_interp_deg[sort_idx]
                r_interp = r_interp[sort_idx]
                
                # Add interpolated line with fill
                frame_data.append(
                    go.Scatterpolar(
                        r=r_interp,
                        theta=theta_interp_deg,
                        mode='lines',
                        line=dict(color='rgba(255,0,0,0.5)', width=3),
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.1)',
                        showlegend=False
                    )
                )
                
                # Add red dots at interpolation points
                sample_indices = np.linspace(0, interp_points-1, len(fiber_angles_deg), dtype=int)
                sample_r = [r_interp[i] for i in sample_indices]
                sample_theta = [theta_interp_deg[i] for i in sample_indices]
                
                frame_data.append(
                    go.Scatterpolar(
                        r=sample_r,
                        theta=sample_theta,
                        mode='markers',
                        marker=dict(size=8, color='red'),
                        showlegend=False
                    )
                )
            except Exception as e:
                print(f"Interpolation skipped: {str(e)}")
                pass
        
        # Add current time indicator showing progress through the cycle
        frame_data.append(
            go.Scatter(
                x=[0.05],
                y=[0.05],
                mode='text',
                text=[f"t = {t:.3f}s (T={T:.3f}s)"],  # Show current time and period
                textposition="bottom right",
                showlegend=False
            )
        )
        
        # Create the frame
        frames.append(go.Frame(
            data=frame_data,
            name=f"frame{frame_idx}"
        ))
    
    # Add initial data (first frame) - reuse the first frame's data
    if frames:
        for trace in frames[0].data:
            fig.add_trace(trace)
    
    # Configure animation
    fig.frames = frames
    
    # Calculate frame duration to complete one cycle based on fps
    frame_duration = (T * 1000) / len(frames)  # Convert to milliseconds
    
    animation_settings = dict(
        frame=dict(duration=frame_duration, redraw=True),
        fromcurrent=True,
        mode="immediate"
    )
    
    # Add play and pause buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, animation_settings]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")]
                    )
                ],
                direction="left",
                pad=dict(r=10, t=10),
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top"
            )
        ],
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 3]),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        margin=dict(t=50, l=0, r=0, b=0),
        title=dict(
            text=f"Ball Bearing @ {freq_selected} Hz (T={T:.3f}s)" + 
                (f"<br>{timestamp}" if timestamp else ""),
            font=dict(size=16)
        ),
        height=600,
        width=600
    )
    
    return fig

# --- Create layout ---
st.subheader(f"Visualization for {freq_selected} Hz at {time_formatted[time_idx].strftime('%Y-%m-%d %H:%M:%S')}")

# Create two columns
col_bearing, col_fft = st.columns(2)

# Show the animated bearing plot
with col_bearing:
    st.subheader("Bearing Visualization (Animated)")
    animated_fig = plot_bearing_animated_cached(time_idx, freq_selected, fps=animation_fps)
    st.plotly_chart(animated_fig, use_container_width=True)

# Show FFT spectrum
with col_fft:
    st.subheader("FFT Spectrum")
    fig_fft = plot_fft_cached(time_idx, freq_selected)
    st.plotly_chart(fig_fft, use_container_width=True)

# Show magnitude history
st.subheader("Peak Magnitude History")
fig_hist = plot_magnitude_history(time_idx, freq_selected)
st.plotly_chart(fig_hist, use_container_width=True)

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

# --- Add memory safeguards throughout the app ---
def check_memory_critical():
    """Check if memory usage is critical and we need to recover"""
    try:
        memory = get_memory_usage()
        if memory > 3000:  # Critical threshold - 3GB
            # For extreme memory conditions, use out_of_memory mode
            if memory > 3500:  # Extreme critical condition
                st.session_state.out_of_memory = True
                # Force reload
                st.rerun()
                return True
            
            # For high but not extreme, use maintenance mode
            st.warning("⚠️ Critical memory usage detected. Entering Low Memory Mode to prevent crashes.")
            # Enable maintenance mode
            st.session_state.maintenance_mode = True
            # Clear all caches and perform cleanup
            perform_memory_cleanup(clear_streamlit_cache=True)
            # Force reload
            st.rerun()
            return True
        elif memory > 2000 and not st.session_state.maintenance_mode:
            # Perform normal cleanup at high (but not critical) memory usage
            perform_memory_cleanup(clear_streamlit_cache=False)
            print(f"Automatic cleanup due to high memory usage: {memory:.2f}MB")
            return False
        return False
    except Exception as e:
        print(f"Error in check_memory_critical: {str(e)}")
        return False

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
            except Exception as e:
                print(f"Error deleting session state key {key}: {str(e)}")
                pass
        
        st.session_state.data = None
        st.session_state.loaded_fiber_ids = []
        
        # Clear all caches
        try:
            st.cache_data.clear()
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Reset the flag
        st.session_state.out_of_memory = False
        
        st.info("Memory has been cleared. Please refresh the page to restart the application.")
        st.stop()

# Call this check in key parts of the application
check_memory_critical()

# Cleanup temporary files when app is closed
# Note: This might not always run in Streamlit cloud environment
def cleanup():
    try:
        if os.path.exists(st.session_state.temp_dir):
            import shutil
            shutil.rmtree(st.session_state.temp_dir)
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

import atexit
atexit.register(cleanup)
