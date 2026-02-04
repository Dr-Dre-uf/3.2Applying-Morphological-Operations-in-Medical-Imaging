import streamlit as st
import numpy as np
import cv2
from skimage.morphology import closing, disk
from PIL import Image
import psutil
import os

# --- MONITORING UTILITY ---
def display_performance_metrics():
    """Calculates and displays CPU and Memory usage in the sidebar."""
    process = psutil.Process(os.getpid())
    # Memory usage in MB
    mem_mb = process.memory_info().rss / (1024 * 1024)
    # CPU usage (short interval for responsiveness)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 System Performance")
    col_cpu, col_mem = st.sidebar.columns(2)
    col_cpu.metric("CPU", f"{cpu_percent}%")
    col_mem.metric("RAM", f"{mem_mb:.1f} MB")

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Morphological Closing in Medical Imaging", layout="wide")

st.title("Microskill 3.2: Applying Morphological Operations in Medical Imaging")

st.warning("⚠️ Do not upload sensitive or personal data. Images are processed locally in this demo app.")

# Function for morphological closing
def apply_morphological_closing(image, radius=5):
    selem = disk(radius)
    closed_image = closing(image, selem)
    return closed_image

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Image Selection")
use_uploaded = st.sidebar.checkbox("Upload your own image")

uploaded_img = None
if use_uploaded:
    uploaded_img = st.sidebar.file_uploader("Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])
else:
    # ✅ Correct sample path
    sample_path = "data/breast_US.png"

# Structuring element radius
radius = st.sidebar.slider(
    "Structuring Element Radius", 1, 15, 5, step=1,
    help="Controls the size of the disk used for morphological closing."
)

# Call the performance monitor
display_performance_metrics()

# --- IMAGE PROCESSING ---
# Load image
if use_uploaded and uploaded_img is not None:
    img = np.array(Image.open(uploaded_img).convert("L"))  # grayscale
elif not use_uploaded:
    # Attempt to load sample; handle case where file might be missing
    img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
else:
    img = None

# Process and display
if img is not None:
    closed_img = apply_morphological_closing(img, radius=radius)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Original Ultrasound", use_container_width=True, clamp=True)
    col2.image(closed_img, caption="Morphologically Closed", use_container_width=True, clamp=True)

    st.markdown("""
    ### Interpretation
    - **Benefit:** Morphological closing can smooth small holes, reduce speckle noise, and make structures (e.g., tumors, lesions) more continuous.
    - **Drawback:** It may also remove fine details that are diagnostically important, potentially hiding microcalcifications or subtle boundaries.
    """)
    
    
    
else:
    st.info("Please upload an image or use the provided sample (ensure 'data/breast_US.png' exists).")