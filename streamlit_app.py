import os
os.environ['OPENCV_AV_LOG_LEVEL'] = 'essential'
import streamlit as st
import numpy as np
import cv2
from skimage.morphology import closing, disk
import matplotlib.pyplot as plt

def apply_morphological_closing(image, selem_size):
    """Applies morphological closing to an image with a specified selem size."""
    selem = disk(selem_size)
    closed_image = closing(image, selem)
    return closed_image

# --- Initialize session state ---
if 'image' not in st.session_state:
    st.session_state.image = cv2.imread('./data/breast_US.png', 0)

# --- Sidebar for Image Selection and Selem Size ---
st.sidebar.header("Settings")

# Image selection
image_option = st.sidebar.radio("Select Image:", ("Default", "Upload"))

if image_option == "Upload":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.session_state.image = cv2.imread(uploaded_file, 0)  # Load as grayscale
else:
    st.session_state.image = cv2.imread('./data/breast_US.png', 0)

# Selem size slider
selem_size = st.sidebar.slider("Selem Size", 1.0, 15.0, 5.0, 1.0)

# --- Load Image ---
image = st.session_state.image

# --- Apply Morphological Closing ---
closed_img = apply_morphological_closing(image, selem_size)

# --- Display Images (Aligned) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    fig1, ax1 = plt.subplots()
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')
    st.pyplot(fig1)
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)  # Add padding

with col2:
    st.subheader("Morphologically Closed Image")
    fig2, ax2 = plt.subplots()
    ax2.imshow(closed_img, cmap='gray')
    ax2.axis('off')
    st.pyplot(fig2)