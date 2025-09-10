import streamlit as st
import numpy as np
import cv2
from skimage.morphology import closing, disk
from PIL import Image

st.set_page_config(page_title="Morphological Closing in Medical Imaging", layout="wide")

st.title("Microskill 3.2: Applying Morphological Operations in Medical Imaging")

st.warning("⚠️ Do not upload sensitive or personal data. Images are processed locally in this demo app.")

# Function for morphological closing
def apply_morphological_closing(image, radius=5):
    selem = disk(radius)
    closed_image = closing(image, selem)
    return closed_image

# Sidebar controls
st.sidebar.header("Image Selection")
use_uploaded = st.sidebar.checkbox("Upload your own image")

uploaded_img = None
if use_uploaded:
    uploaded_img = st.sidebar.file_uploader("Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])
else:
    # ✅ Correct sample path
    sample_path = "data/breast_US.png"

# Structuring element radius
radius = st.sidebar.slider("Structuring Element Radius", 1, 15, 5, step=1,
                           help="Controls the size of the disk used for morphological closing.")

# Load image
if use_uploaded and uploaded_img is not None:
    img = np.array(Image.open(uploaded_img).convert("L"))  # grayscale
elif not use_uploaded:
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
    st.info("Please upload an image or use the provided sample.")
