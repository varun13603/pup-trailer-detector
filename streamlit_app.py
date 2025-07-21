#!/usr/bin/env python3
"""
🚛 PUP TRAILER DETECTOR STREAMLIT APPLICATION 🚛
Streamlit web application for pup trailer detection using the breakthrough model
WITH HUGGING FACE MODEL DOWNLOAD
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image  # Updated to use TensorFlow image processing
import streamlit as st
from PIL import Image  # Still used for display purposes, but not for preprocessing
import io
import base64
import json
from datetime import datetime
import uuid
import requests
import logging
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_REPO_ID = "Jackaiuser/pup_detect"
MODEL_FILENAME = "final_breakthrough_model.h5"
# Get Hugging Face token from environment or Streamlit secrets
try:
    HF_TOKEN = st.secrets.get('HF_TOKEN', None) or os.getenv('HF_TOKEN')
except Exception:
    HF_TOKEN = os.getenv('HF_TOKEN')
IMG_HEIGHT = 224
IMG_WIDTH = 224
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Page configuration
st.set_page_config(
    page_title="Pup Trailer Detector",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, modern design
def get_custom_css(dark_mode=False):
    if dark_mode:
        return """
<style>
    /* Dark mode theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460) !important;
        color: #e8e8e8 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .stApp .main .block-container {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460) !important;
        color: #e8e8e8 !important;
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 1rem;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        color: #e8e8e8 !important;
        background: rgba(30, 30, 45, 0.8) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        transform: translateY(-3px);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .pup-positive {
        background: rgba(76, 175, 80, 0.1) !important;
        border-left: 4px solid #81c784;
        border-color: #81c784 !important;
    }
    
    .pup-positive h3 {
        color: #81c784 !important;
        text-shadow: 0 0 10px rgba(129, 199, 132, 0.5);
    }
    
    .pup-positive p, .pup-positive strong {
        color: #a5d6a7 !important;
    }
    
    .pup-negative {
        background: rgba(255, 152, 0, 0.1) !important;
        border-left: 4px solid #ffb74d;
        border-color: #ffb74d !important;
    }
    
    .pup-negative h3 {
        color: #ffb74d !important;
        text-shadow: 0 0 10px rgba(255, 183, 77, 0.5);
    }
    
    .pup-negative p, .pup-negative strong {
        color: #ffcc02 !important;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #1a1a2e, #16213e) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 30, 45, 0.8) !important;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #b0b0b0 !important;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        margin: 0 2px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(30, 30, 45, 0.6) !important;
        color: #e8e8e8 !important;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stFileUploader {
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        background: rgba(30, 30, 45, 0.3) !important;
        color: #e8e8e8 !important;
        text-align: center;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stFileUploader:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.1) !important;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stTextInput input {
        background: rgba(30, 30, 45, 0.8) !important;
        color: #e8e8e8 !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
        outline: none !important;
    }
    
    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        height: 12px !important;
    }
    
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: rgba(30, 30, 45, 0.6) !important;
        color: #e8e8e8 !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stMetric label {
        color: #b0b0b0 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3) !important;
    }
    
    .streamlit-expanderHeader {
        background: rgba(30, 30, 45, 0.8) !important;
        color: #e8e8e8 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.1) !important;
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 30, 45, 0.6) !important;
        color: #e8e8e8 !important;
        padding: 1.5rem !important;
        border-radius: 0 0 12px 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-top: none !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSuccess {
        background: rgba(76, 175, 80, 0.1) !important;
        color: #81c784 !important;
        border-color: #81c784 !important;
    }
    
    .stInfo {
        background: rgba(102, 126, 234, 0.1) !important;
        color: #667eea !important;
        border-color: #667eea !important;
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.1) !important;
        color: #ffb74d !important;
        border-color: #ffb74d !important;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.1) !important;
        color: #e57373 !important;
        border-color: #e57373 !important;
    }
    
    .image-container {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .image-container:hover {
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4) !important;
        transform: translateY(-3px) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
    }
    
    .sidebar-section {
        background: rgba(30, 30, 45, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .sidebar-section h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        font-size: 1.1rem !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3) !important;
    }
    
    .dark-footer {
        background: rgba(30, 30, 45, 0.8) !important;
        color: #b0b0b0 !important;
        padding: 2rem !important;
        border-radius: 15px !important;
        text-align: center !important;
        margin-top: 3rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .dark-footer h3 {
        color: #667eea !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 45, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
"""
    else:
        return """
<style>
    /* Clean, modern light theme */
    .stApp {
        background: #ffffff !important;
        color: #2d3748 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .stApp .main .block-container {
        background: #ffffff !important;
        color: #2d3748 !important;
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #1a202c !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        border-bottom: 3px solid #3182ce;
        padding-bottom: 1rem;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        color: #2d3748 !important;
        background: #f7fafc !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }
    
    .pup-positive {
        background: #f0fff4 !important;
        border-left: 4px solid #38a169;
        border-color: #38a169 !important;
    }
    
    .pup-positive h3 {
        color: #38a169 !important;
    }
    
    .pup-positive p, .pup-positive strong {
        color: #2f855a !important;
    }
    
    .pup-negative {
        background: #fffaf0 !important;
        border-left: 4px solid #ed8936;
        border-color: #ed8936 !important;
    }
    
    .pup-negative h3 {
        color: #ed8936 !important;
    }
    
    .pup-negative p, .pup-negative strong {
        color: #c05621 !important;
    }
    
    .stSidebar {
        background: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #edf2f7 !important;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #cbd5e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #4a5568 !important;
        font-weight: 500;
        padding: 0.5rem 1rem;
        margin: 0 2px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #3182ce !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff !important;
        color: #2d3748 !important;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }
    
    .stButton button {
        background: #3182ce !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    .stButton button:hover {
        background: #2c5282 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stFileUploader {
        border: 2px dashed #cbd5e0 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background: #f8fafc !important;
        color: #2d3748 !important;
        text-align: center;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: #3182ce !important;
        background: #ebf8ff !important;
    }
    
    .stTextInput input {
        background: #ffffff !important;
        color: #2d3748 !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #3182ce !important;
        box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1) !important;
        outline: none !important;
    }
    
    .stProgress > div > div {
        background: #e2e8f0 !important;
        border-radius: 10px !important;
        height: 10px !important;
    }
    
    .stProgress .st-bo {
        background: #3182ce !important;
        border-radius: 10px !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: #f8fafc !important;
        color: #2d3748 !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
    }
    
    .stMetric label {
        color: #4a5568 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #1a202c !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        color: #2d3748 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #ebf8ff !important;
        border-color: #3182ce !important;
    }
    
    .streamlit-expanderContent {
        background: #ffffff !important;
        color: #2d3748 !important;
        padding: 1.5rem !important;
        border-radius: 0 0 8px 8px !important;
        border: 1px solid #e2e8f0 !important;
        border-top: none !important;
    }
    
    .stAlert {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    
    .stSuccess {
        background: #f0fff4 !important;
        color: #2f855a !important;
        border-color: #38a169 !important;
    }
    
    .stInfo {
        background: #ebf8ff !important;
        color: #2c5282 !important;
        border-color: #3182ce !important;
    }
    
    .stWarning {
        background: #fffaf0 !important;
        color: #c05621 !important;
        border-color: #ed8936 !important;
    }
    
    .stError {
        background: #fed7d7 !important;
        color: #c53030 !important;
        border-color: #e53e3e !important;
    }
    
    .image-container {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    .image-container:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        transform: translateY(-2px) !important;
    }
    
    .sidebar-section {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
    }
    
    .sidebar-section h4 {
        color: #1a202c !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        font-size: 1.1rem !important;
    }
    
    .clean-footer {
        background: #f8fafc !important;
        color: #4a5568 !important;
        padding: 2rem !important;
        border-radius: 12px !important;
        text-align: center !important;
        margin-top: 3rem !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .clean-footer h3 {
        color: #3182ce !important;
        margin-bottom: 1rem !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }
</style>
"""

st.markdown(get_custom_css(), unsafe_allow_html=True)

@st.cache_resource
def load_breakthrough_model():
    try:
        # Check if we have a token for private model
        if HF_TOKEN is None:
            st.error("⚠️ No Hugging Face token found. Private model access requires a token.")
            st.info("Add your HF_TOKEN to Streamlit secrets or environment variables.")
            return None
        
        # Create placeholders for loading messages
        token_placeholder = st.empty()
        download_placeholder = st.empty()
        progress_placeholder = st.empty()
        success_placeholder = st.empty()
        
        # Show loading messages
        token_placeholder.info(f"🔑 Token found: {HF_TOKEN[:10]}...")
        download_placeholder.info("⏳ Downloading model from Hugging Face Hub... This may take a moment.")
        
        try:
            # Download from Hugging Face Hub with token
            with progress_placeholder.container():
                st.write("📥 Connecting to Hugging Face Hub...")
                st.write(f"Repository: {MODEL_REPO_ID}")
                st.write(f"Filename: {MODEL_FILENAME}")
                
            model_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_FILENAME,
                token=HF_TOKEN,
                cache_dir="./hf_cache"
            )
            
            # Clear download messages
            progress_placeholder.empty()
            success_placeholder.success("✅ Model downloaded successfully from Hugging Face!")
            logger.info(f"Model downloaded to: {model_path}")
            
        except Exception as download_error:
            # Clear loading messages on error
            token_placeholder.empty()
            download_placeholder.empty()
            progress_placeholder.empty()
            success_placeholder.empty()
            
            error_msg = str(download_error)
            st.error(f"❌ Failed to download model from Hugging Face: {error_msg}")
            
            # Provide specific error guidance
            if "401" in error_msg or "Unauthorized" in error_msg:
                st.error("🔒 Authentication failed. Please check your token.")
            elif "404" in error_msg or "Not Found" in error_msg:
                st.error("📁 Model repository or file not found.")
            elif "403" in error_msg or "Forbidden" in error_msg:
                st.error("🚫 Access denied. Check if you have permission to access this model.")
            
            st.info("Please check:")
            st.info("1. Your internet connection")
            st.info("2. The Hugging Face repository exists and you have access")
            st.info("3. Your Hugging Face token has the correct permissions")
            st.info(f"4. Repository: {MODEL_REPO_ID}")
            st.info(f"5. Filename: {MODEL_FILENAME}")
            st.info(f"6. Token starts with: {HF_TOKEN[:10] if HF_TOKEN else 'None'}...")
            return None
        
        # Load the model
        logger.info(f"Loading breakthrough model: {model_path}")
        with st.spinner("Loading TensorFlow model..."):
            model = load_model(model_path)
        logger.info("✅ Breakthrough model loaded successfully!")
        
        # Clear all loading messages once model is loaded
        token_placeholder.empty()
        download_placeholder.empty()
        progress_placeholder.empty()
        success_placeholder.empty()
        
        # Display model info in sidebar only
        st.sidebar.success(f"📦 Model loaded from: {MODEL_REPO_ID}")
        
        return model
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error loading model: {error_msg}")
        st.error(f"❌ Failed to load model: {error_msg}")
        
        # Show debug info
        st.error("Debug information:")
        st.error(f"- Repository: {MODEL_REPO_ID}")
        st.error(f"- Filename: {MODEL_FILENAME}")
        st.error(f"- Token available: {HF_TOKEN is not None}")
        st.error(f"- Error type: {type(e).__name__}")
        
        return None

def preprocess_image(image_path):
    """Preprocess image for prediction using TensorFlow methods to match local script."""
    try:
        # Load and resize image using TensorFlow (matches test_images.py)
        img = tf_image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf_image.img_to_array(img)
        # Expand dimensions and normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, image):
    """Make prediction on uploaded image."""
    if model is None:
        return None, "Model not loaded"
    try:
        # Save uploaded image to temporary path for TensorFlow loading
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")  # Save as PNG to avoid format issues
            buffer.seek(0)
            temp_path = "temp_image.png"
            with open(temp_path, "wb") as f:
                f.write(buffer.getvalue())

        # Preprocess using the temporary path
        processed_image = preprocess_image(temp_path)
        os.remove(temp_path)  # Clean up temp file

        if processed_image is None:
            return None, "Error processing image"

        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])

        # Determine class
        is_pup = probability > 0.5
        confidence = probability if is_pup else 1 - probability
        result = {
            'is_pup': is_pup,
            'probability': probability,
            'confidence': confidence,
            'class': 'Pup Trailer' if is_pup else 'Not a Pup Trailer',
            'confidence_percentage': f"{confidence * 100:.2f}%"
        }
        return result, None
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None, f"Prediction error: {str(e)}"

def save_prediction_to_session(result, image_name):
    """Save prediction to session state."""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction_record = {
        'timestamp': datetime.now().isoformat(),
        'filename': image_name,
        'result': result,
        'id': str(uuid.uuid4())
    }
    
    st.session_state.prediction_history.append(prediction_record)
    
    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]

def main():
    """Main application."""
    # Initialize dark mode in session state if not exists
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True

    # Header with enhanced styling
    st.markdown('<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;"><h1>🚛 Pup Trailer Detector</h1><p>🤖 Powered by Deep Learning • 🎯 95%+ Accuracy • ⚡ Real-time Processing</p></div>', unsafe_allow_html=True)

    # Load model
    model = load_breakthrough_model()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.experimental_rerun()

        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            avg_conf = np.mean([p['result']['confidence'] for p in st.session_state.prediction_history])
            st.metric("Average Confidence", f"{avg_conf * 100:.1f}%")

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload Image", "🔗 URL Prediction", "📜 History", "ℹ️ About", "🛠️ Help"])

    with tab1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Select an image from your device to detect pup trailers", type=list(ALLOWED_EXTENSIONS))
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing..."):
                    result, error = predict_image(model, image)
                if error:
                    st.error(error)
                else:
                    st.success(f"**Prediction:** {result['class']}")
                    st.info(f"**Confidence:** {result['confidence_percentage']}")
                    st.info(f"**Probability:** {result['probability']:.4f}")
                    st.info(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    save_prediction_to_session(result, uploaded_file.name)
        else:
            st.info("Upload an image to get started")

    with tab2:
        st.header("Predict from URL")
        image_url = st.text_input("Enter an image URL to analyze remotely hosted images")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.image(image, caption="Image from URL", use_column_width=True)
                if st.button("🔍 Analyze URL Image"):
                    with st.spinner("Analyzing..."):
                        result, error = predict_image(model, image)
                    if error:
                        st.error(error)
                    else:
                        st.success(f"**Prediction:** {result['class']}")
                        st.info(f"**Confidence:** {result['confidence_percentage']}")
                        st.info(f"**Probability:** {result['probability']:.4f}")
                        st.info(f"**Source:** URL")
                        save_prediction_to_session(result, image_url)
            except Exception as e:
                st.error(f"Error loading image from URL: {str(e)}")

    with tab3:
        st.header("Prediction History")
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            for pred in reversed(st.session_state.prediction_history):
                with st.expander(f"{pred['timestamp']} - {pred['filename']}"):
                    st.write(f"**Prediction:** {pred['result']['class']}")
                    st.write(f"**Confidence:** {pred['result']['confidence_percentage']}")
                    st.write(f"**Probability:** {pred['result']['probability']:.4f}")
        else:
            st.info("No predictions yet. Make one to see history!")

    with tab4:
        st.header("About Pup Trailer Detector")
        st.write("A pup trailer is a short semi-trailer that is typically pulled behind a truck or another trailer. It's commonly used in logistics to increase cargo capacity while maintaining maneuverability.")
        st.write("Our AI model uses advanced computer vision techniques based on ResNet50V2 architecture to analyze images and classify whether they contain pup trailers or not.")

    with tab5:
        st.header("Help & Tips")
        st.write("- Ensure images are clear and focused on the trailer.")
        st.write("- For best results, use images with good lighting.")
        st.write("- If predictions seem off, check model loading logs in the sidebar.")
        st.write("- Contact support if issues persist.")

    # Footer
    st.markdown('<div style="text-align: center; padding: 10px; color: gray;">© 2025 Pup Trailer Detector. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
