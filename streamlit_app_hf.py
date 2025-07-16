#!/usr/bin/env python3
"""
🚛 PUP TRAILER DETECTOR STREAMLIT APPLICATION 🚛
Streamlit web application for pup trailer detection using the breakthrough model
WITH HUGGING FACE MODEL DOWNLOAD
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import io
import base64
import json
from datetime import datetime
import uuid
import requests
import logging
from huggingface_hub import hf_hub_download

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    tf = None
    load_model = None
    st.error("❌ TensorFlow is not available in this environment.")
    st.error(f"Import error: {str(e)}")
    st.info("**Solutions:**")
    st.info("1. Make sure requirements.txt includes 'tensorflow-cpu'")
    st.info("2. Try redeploying the app")
    st.info("3. Check the app logs for more details")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_REPO_ID = "Jackaiuser/pup_detect"
MODEL_FILENAME = "final_breakthrough_model.h5"
MODEL_PATH = 'final_breakthrough_model.h5'
IMG_HEIGHT = 224
IMG_WIDTH = 224
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Page configuration
st.set_page_config(
    page_title="🚛 Pup Trailer Detector",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
    .pup-positive {
        border-color: #4CAF50;
        background-color: #f8fff8;
    }
    .pup-negative {
        border-color: #f44336;
        background-color: #fff8f8;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_breakthrough_model():
    """Load the breakthrough model from Hugging Face Hub with caching."""
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow is not available. Cannot load model.")
        return None
    
    try:
        # Check if model exists locally first
        if not os.path.exists(MODEL_PATH):
            st.info("⏳ Downloading model from Hugging Face Hub... This may take a moment.")
            
            # Create a progress placeholder
            progress_placeholder = st.empty()
            
            try:
                # Download from Hugging Face Hub
                with progress_placeholder.container():
                    st.write("📥 Connecting to Hugging Face Hub...")
                    
                downloaded_path = hf_hub_download(
                    repo_id=MODEL_REPO_ID,
                    filename=MODEL_FILENAME,
                    cache_dir="./hf_cache",
                    local_dir="./",
                    local_dir_use_symlinks=False
                )
                
                progress_placeholder.empty()
                st.success("✅ Model downloaded successfully from Hugging Face!")
                logger.info(f"Model downloaded to: {downloaded_path}")
                
            except Exception as download_error:
                progress_placeholder.empty()
                st.error(f"❌ Failed to download model from Hugging Face: {str(download_error)}")
                st.info("Please check:")
                st.info("1. Your internet connection")
                st.info("2. The Hugging Face repository is public and accessible")
                st.info(f"3. Repository: {MODEL_REPO_ID}")
                st.info(f"4. Filename: {MODEL_FILENAME}")
                return None
        
        # Load the model
        logger.info(f"Loading breakthrough model: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        logger.info("✅ Breakthrough model loaded successfully!")
        
        # Display model info
        st.sidebar.success(f"📦 Model loaded from: {MODEL_REPO_ID}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        st.info("**Troubleshooting:**")
        st.info("1. Check if TensorFlow is properly installed")
        st.info("2. Verify the model file is not corrupted")
        st.info("3. Try redeploying the application")
        return None

# ...existing code...
