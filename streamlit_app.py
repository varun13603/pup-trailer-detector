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
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
    page_title="🚛 Pup Trailer Detector",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header with animated gradient */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 300% 300%;
        animation: gradient-shift 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Animated loading spinner */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced prediction box with animations */
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        color: #333 !important;
        background: #ffffff !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .prediction-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .pup-positive {
        border-left: 4px solid #27AE60;
        background: linear-gradient(135deg, #E8F8F5, #D5F4E6) !important;
        color: #1E8449 !important;
    }
    
    .pup-positive h3 {
        color: #1E8449 !important;
    }
    
    .pup-positive p {
        color: #148F77 !important;
    }
    
    .pup-positive strong {
        color: #1E8449 !important;
    }
    
    .pup-negative {
        border-left: 4px solid #E74C3C;
        background: linear-gradient(135deg, #FDEDEC, #FADBD8) !important;
        color: #C0392B !important;
    }
    
    .pup-negative h3 {
        color: #C0392B !important;
    }
    
    .pup-negative p {
        color: #A93226 !important;
    }
    
    .pup-negative strong {
        color: #C0392B !important;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: #ffffff !important;
        color: #333 !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
        border: 1px solid #e9ecef;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Animated progress bar */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        animation: progress-glow 2s ease infinite;
    }
    
    @keyframes progress-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(76,175,80,0.5); }
        50% { box-shadow: 0 0 20px rgba(76,175,80,0.8); }
    }
    
    /* Fade in animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced buttons */
    .stButton button {
        background: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,123,255,0.2);
    }
    
    .stButton button:hover {
        background: #0056b3;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,123,255,0.3);
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #f8f9fa !important;
        border-radius: 10px;
        padding: 0.5rem;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        transition: all 0.3s ease;
        color: #495057 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e9ecef !important;
        color: #212529 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #007bff !important;
        color: #ffffff !important;
    }
    
    /* Enhanced file uploader */
    .stFileUploader {
        border: 2px dashed #007bff;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
        background: #f8f9ff !important;
        color: #333 !important;
    }
    
    .stFileUploader:hover {
        border-color: #0056b3;
        background: #e6f3ff !important;
        transform: translateY(-2px);
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: #f8f9fa !important;
        color: #333 !important;
        border-right: 1px solid #dee2e6;
    }
    
    /* General text visibility fixes */
    .stMarkdown, .stText {
        color: #333 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #212529 !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #495057 !important;
    }
    
    .stMarkdown strong, .stMarkdown b {
        color: #212529 !important;
    }
    
    /* Ensure all text in containers is visible */
    div[data-testid="stMarkdownContainer"] {
        color: #333 !important;
        background: #ffffff !important;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #f0f0f0;
    }
    
    div[data-testid="stMarkdownContainer"] p {
        color: #495057 !important;
    }
    
    div[data-testid="stMarkdownContainer"] strong {
        color: #212529 !important;
    }
    
    /* Fix text in custom styled divs */
    .stMarkdown div[style*="background"] {
        color: #333 !important;
    }
    
    .stMarkdown div[style*="background"] p {
        color: #495057 !important;
    }
    
    .stMarkdown div[style*="background"] strong {
        color: #212529 !important;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(90deg, #2c3e50, #3498db);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
    }
    
    /* Success/Error message animations */
    .stAlert {
        animation: slideInRight 0.5s ease-out;
        border-radius: 10px;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Enhanced expander */
    .streamlit-expanderHeader {
        background: #f8f9fa !important;
        color: #333 !important;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #e9ecef !important;
        transform: translateX(3px);
    }
    
    .streamlit-expanderContent {
        background: #ffffff !important;
        color: #333 !important;
        padding: 1rem;
        border-radius: 0 0 8px 8px;
        border: 1px solid #dee2e6;
        border-top: none;
    }
    
    /* Image container enhancement */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
    }
    
    /* Stats cards animation */
    .stats-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #4ECDC4, #FF6B6B);
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0,0,0,0.8);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        white-space: nowrap;
        font-size: 0.8rem;
        z-index: 1000;
    }
    
    /* Additional text visibility fixes */
    .stSelectbox label, .stTextInput label, .stFileUploader label {
        color: #333 !important;
        font-weight: 500 !important;
    }
    
    .stMetric label {
        color: #495057 !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: #ffffff !important;
        color: #333 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Fix for expander text */
    .streamlit-expanderHeader p {
        color: #333 !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        color: #333 !important;
    }
    
    /* Fix for info boxes */
    .stInfo {
        background: #e7f3ff !important;
        color: #0056b3 !important;
        border-left: 4px solid #007bff !important;
    }
    
    .stSuccess {
        background: #e8f5e9 !important;
        color: #2e7d32 !important;
        border-left: 4px solid #4caf50 !important;
    }
    
    .stWarning {
        background: #fff3cd !important;
        color: #856404 !important;
        border-left: 4px solid #ffc107 !important;
    }
    
    .stError {
        background: #f8d7da !important;
        color: #721c24 !important;
        border-left: 4px solid #dc3545 !important;
    }
    
    /* Fix for tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #495057 !important;
        font-weight: 500 !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff !important;
        color: #333 !important;
        padding: 1.5rem;
        border-radius: 0 0 10px 10px;
        border: 1px solid #dee2e6;
        border-top: none;
    }
    
    /* Fix for all text elements */
    * {
        color: inherit;
    }
    
    /* Ensure readability on all backgrounds */
    .stApp {
        background: #f8f9fa !important;
        color: #333 !important;
    }
    
    .stApp .main .block-container {
        background: #ffffff !important;
        color: #333 !important;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_breakthrough_model():
    """Load the breakthrough model from Hugging Face Hub with caching."""
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

def preprocess_image(image):
    """Preprocess image for prediction."""
    try:
        # Resize image
        img = image.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Normalize and expand dimensions
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
        # Preprocess image
        processed_image = preprocess_image(image)
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
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🚛 Pup Trailer Detector</h1>', unsafe_allow_html=True)
    
    # Add subtitle with animation
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; animation: fadeInUp 0.7s ease-out;">
        <h3 style="color: #666; font-weight: 300; margin-bottom: 1rem;">
            Advanced AI-Powered Trailer Classification System
        </h3>
        <p style="color: #888; font-size: 1.1rem;">
            🤖 Powered by Deep Learning • 🎯 90%+ Accuracy • ⚡ Real-time Processing
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model first
    model = load_breakthrough_model()
    
    if model is None:
        st.error("❌ Failed to load the breakthrough model from Hugging Face Hub.")
        # Show debug information only on error
        with st.expander("🔧 Debug Information"):
            st.write(f"**Repository:** {MODEL_REPO_ID}")
            st.write(f"**Filename:** {MODEL_FILENAME}")
            st.write(f"**Token available:** {HF_TOKEN is not None}")
            if HF_TOKEN:
                st.write(f"**Token prefix:** {HF_TOKEN[:10]}...")
            st.write("**Environment variables:**")
            st.write(f"- HF_TOKEN in env: {'HF_TOKEN' in os.environ}")
            st.write("**Streamlit secrets:**")
            try:
                st.write(f"- HF_TOKEN in secrets: {'HF_TOKEN' in st.secrets}")
            except Exception as e:
                st.write(f"- Secrets error: {e}")
        st.info("Please check the debug information above and try refreshing the page.")
        st.stop()
    
    # Success message with animation
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; animation: slideInRight 0.5s ease-out;">
        <div style="background: linear-gradient(90deg, #4CAF50, #45a049); color: white; padding: 1rem; border-radius: 10px; display: inline-block;">
            ✅ Model loaded successfully! Ready for predictions
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar with better organization
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="margin: 0; font-size: 1.5rem;">📊 Dashboard</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Control Panel</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model source info in sidebar
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, #FF6B6B, #4ECDC4); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <strong>🤗 Model Source:</strong><br>
        <a href="https://huggingface.co/{MODEL_REPO_ID}" target="_blank" style="color: white; text-decoration: underline;">
            Hugging Face Hub
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Optional debug info (collapsed by default)
    if st.sidebar.checkbox("🔧 Show Debug Info", value=False):
        with st.expander("🔧 Debug Information"):
            st.write(f"**Repository:** {MODEL_REPO_ID}")
            st.write(f"**Filename:** {MODEL_FILENAME}")
            st.write(f"**Token available:** {HF_TOKEN is not None}")
            if HF_TOKEN:
                st.write(f"**Token prefix:** {HF_TOKEN[:10]}...")
            st.write("**Environment variables:**")
            st.write(f"- HF_TOKEN in env: {'HF_TOKEN' in os.environ}")
            st.write("**Streamlit secrets:**")
            try:
                st.write(f"- HF_TOKEN in secrets: {'HF_TOKEN' in st.secrets}")
            except Exception as e:
                st.write(f"- Secrets error: {e}")
    
    # Enhanced model info in sidebar
    with st.sidebar.expander("🔍 Model Information", expanded=True):
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <strong>🧠 Model Details:</strong><br>
            • <strong>Name:</strong> Breakthrough Pup Trailer Detector<br>
            • <strong>Architecture:</strong> ResNet50V2 + Custom Head<br>
            • <strong>Input:</strong> 224×224×3 RGB Images<br>
            • <strong>Training:</strong> 2-Phase Strategy<br>
            • <strong>Classes:</strong> Pup / Non-Pup Trailer<br>
            • <strong>Accuracy:</strong> 90%+
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced statistics in sidebar
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        with st.sidebar.expander("📈 Statistics", expanded=True):
            history = st.session_state.prediction_history
            pup_count = sum(1 for p in history if p['result']['is_pup'])
            total_predictions = len(history)
            
            # Enhanced metrics with colors
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem;">
                <div style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; font-size: 1.5rem;">{total_predictions}</h3>
                    <p style="margin: 0; font-size: 0.8rem;">Total Predictions</p>
                </div>
                <div style="background: linear-gradient(135deg, #27ae60, #229954); color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; font-size: 1.5rem;">{pup_count}</h3>
                    <p style="margin: 0; font-size: 0.8rem;">Pup Detected</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; font-size: 1.5rem;">{total_predictions - pup_count}</h3>
                <p style="margin: 0; font-size: 0.8rem;">Non-Pup Detected</p>
            </div>
            """, unsafe_allow_html=True)
            
            if total_predictions > 0:
                avg_confidence = np.mean([p['result']['confidence'] for p in history])
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; font-size: 1.5rem;">{avg_confidence * 100:.1f}%</h3>
                    <p style="margin: 0; font-size: 0.8rem;">Avg Confidence</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Quick actions in sidebar
    st.sidebar.markdown("### 🚀 Quick Actions")
    if st.sidebar.button("📊 Export Statistics", help="Export prediction statistics as JSON"):
        if 'prediction_history' in st.session_state:
            stats = {
                'total_predictions': len(st.session_state.prediction_history),
                'pup_count': sum(1 for p in st.session_state.prediction_history if p['result']['is_pup']),
                'export_time': datetime.now().isoformat()
            }
            st.sidebar.download_button(
                label="💾 Download Stats",
                data=json.dumps(stats, indent=2),
                file_name=f"pup_detector_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    if st.sidebar.button("🧹 Clear All History", help="Clear all prediction history"):
        st.session_state.prediction_history = []
        st.sidebar.success("History cleared!")
        st.experimental_rerun()
    
    # Enhanced main content with better tabs
    st.markdown("""
    <div style="margin: 2rem 0; text-align: center;">
        <h2 style="color: #333; font-weight: 400;">Choose Your Prediction Method</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Upload Image", "🌐 URL Prediction", "📊 History", "ℹ️ About"])
    
    with tab1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #4ECDC4;">📷 Upload Image for Analysis</h3>
            <p style="color: #666;">Select an image from your device to detect pup trailers</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            help="Upload an image of a trailer to detect if it's a pup trailer"
        )
        
        if uploaded_file is not None:
            # Display image with enhanced styling
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <h4 style="color: #FF6B6B;">📸 Uploaded Image</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Image info
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong>📋 Image Details:</strong><br>
                    • <strong>Size:</strong> {image.size[0]} × {image.size[1]} pixels<br>
                    • <strong>Format:</strong> {image.format}<br>
                    • <strong>Mode:</strong> {image.mode}<br>
                    • <strong>File Size:</strong> {len(uploaded_file.getvalue())/1024:.1f} KB
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <h4 style="color: #4ECDC4;">🔍 Prediction Result</h4>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("🤖 Analyzing image..."):
                    result, error = predict_image(model, image)
                
                if error:
                    st.error(f"❌ {error}")
                elif result:
                    # Save to history
                    save_prediction_to_session(result, uploaded_file.name)
                    
                    # Display enhanced result
                    css_class = "pup-positive" if result['is_pup'] else "pup-negative"
                    emoji = "🚛" if result['is_pup'] else "🚚"
                    
                    st.markdown(f"""
                    <div class="prediction-box {css_class}">
                        <h3>{emoji} {result['class']}</h3>
                        <p><strong>Confidence:</strong> {result['confidence_percentage']}</p>
                        <p><strong>Probability:</strong> {result['probability']:.4f}</p>
                        <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced progress bar
                    st.markdown("**Confidence Level:**")
                    st.progress(result['confidence'])
                    
                    # Enhanced metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🎯 Confidence", result['confidence_percentage'], 
                                delta=f"{result['confidence']:.3f}" if result['confidence'] > 0.8 else None)
                    with col_b:
                        st.metric("📊 Classification", result['class'])
                    
                    # Confidence interpretation
                    if result['confidence'] > 0.9:
                        st.success("🎯 Very High Confidence - Excellent prediction!")
                    elif result['confidence'] > 0.7:
                        st.info("✅ Good Confidence - Reliable prediction")
                    else:
                        st.warning("⚠️ Low Confidence - Consider trying another image")
    
    with tab2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #4ECDC4;">🌐 Predict from URL</h3>
            <p style="color: #666;">Enter an image URL to analyze remotely hosted images</p>
        </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input("🔗 Enter image URL:", placeholder="https://example.com/image.jpg")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if url:
                st.markdown(f"**Preview URL:** {url}")
        with col2:
            predict_button = st.button("🔍 Predict from URL", disabled=not url)
        
        if predict_button and url:
            try:
                with st.spinner("🌐 Downloading and analyzing image..."):
                    # Download image
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    # Open image
                    image = Image.open(io.BytesIO(response.content))
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("""
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <h4 style="color: #FF6B6B;">📸 Downloaded Image</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(image, caption="🌐 Downloaded from URL", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <h4 style="color: #4ECDC4;">🔍 Prediction Result</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        result, error = predict_image(model, image)
                        
                        if error:
                            st.error(f"❌ {error}")
                        elif result:
                            # Save to history
                            save_prediction_to_session(result, f"URL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                            
                            # Display result
                            css_class = "pup-positive" if result['is_pup'] else "pup-negative"
                            emoji = "🚛" if result['is_pup'] else "🚚"
                            
                            st.markdown(f"""
                            <div class="prediction-box {css_class}">
                                <h3>{emoji} {result['class']}</h3>
                                <p><strong>Confidence:</strong> {result['confidence_percentage']}</p>
                                <p><strong>Probability:</strong> {result['probability']:.4f}</p>
                                <p><strong>Source:</strong> URL</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(result['confidence'])
                            
            except Exception as e:
                st.error(f"❌ Error processing URL: {str(e)}")
                st.info("💡 **Tips:**\n- Make sure the URL is accessible\n- URL should point directly to an image file\n- Try a different image URL")
    
    with tab3:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #4ECDC4;">📊 Prediction History</h3>
            <p style="color: #666;">View and manage your prediction history</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            history = st.session_state.prediction_history
            
            # Enhanced stats summary
            pup_count = sum(1 for p in history if p['result']['is_pup'])
            total_predictions = len(history)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total", total_predictions)
            with col2:
                st.metric("🚛 Pup", pup_count)
            with col3:
                st.metric("🚚 Non-Pup", total_predictions - pup_count)
            with col4:
                if total_predictions > 0:
                    avg_conf = np.mean([p['result']['confidence'] for p in history])
                    st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")
            
            # Clear history button
            if st.button("🗑️ Clear History", help="Clear all prediction history"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
            
            st.markdown("---")
            
            # Enhanced history display
            for i, prediction in enumerate(reversed(history[-10:])):  # Show last 10
                result = prediction['result']
                emoji = "🚛" if result['is_pup'] else "🚚"
                
                with st.expander(f"{emoji} Prediction {len(history) - i}: {result['class']} ({result['confidence_percentage']})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        **📁 File:** {prediction['filename']}  
                        **🕒 Time:** {prediction['timestamp'][:19]}  
                        **🆔 ID:** {prediction['id'][:8]}...
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **🎯 Class:** {result['class']}  
                        **📊 Confidence:** {result['confidence_percentage']}  
                        **🔢 Probability:** {result['probability']:.4f}
                        """)
                        st.progress(result['confidence'])
        else:
            st.info("📝 No predictions yet. Upload an image to get started!")
            
            # Call to action
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <p style="color: #666; font-size: 1.1rem;">
                    Ready to make your first prediction? 🚀
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #4ECDC4;">ℹ️ About This Application</h3>
            <p style="color: #666;">Learn more about the Pup Trailer Detector</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced about section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h4 style="color: #333; margin-bottom: 1rem;">🎯 What is a Pup Trailer?</h4>
            <p style="color: #666; line-height: 1.6;">
                A pup trailer is a short semi-trailer that is typically pulled behind a truck or another trailer. 
                It's commonly used in logistics to increase cargo capacity while maintaining maneuverability.
            </p>
            
            <h4 style="color: #333; margin: 1.5rem 0 1rem 0;">🤖 How It Works</h4>
            <p style="color: #666; line-height: 1.6;">
                Our AI model uses advanced computer vision techniques based on ResNet50V2 architecture to analyze 
                images and classify whether they contain pup trailers or not.
            </p>
            
            <h4 style="color: #333; margin: 1.5rem 0 1rem 0;">🔧 Technical Details</h4>
            <ul style="color: #666; line-height: 1.6;">
                <li><strong>Model:</strong> ResNet50V2 with custom classification head</li>
                <li><strong>Training:</strong> 2-phase training strategy for optimal performance</li>
                <li><strong>Accuracy:</strong> 90%+ on validation data</li>
                <li><strong>Input:</strong> 224×224 pixel RGB images</li>
                <li><strong>Deployment:</strong> Hugging Face Hub integration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF6B6B, #ff5252); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 1rem 0;">🚀 Features</h4>
                <ul style="margin: 0; padding-left: 1.5rem;">
                    <li>Real-time image analysis</li>
                    <li>High accuracy predictions</li>
                    <li>URL-based image processing</li>
                    <li>Prediction history tracking</li>
                    <li>Statistics and analytics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ECDC4, #26c6da); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 1rem 0;">💡 Tips for Best Results</h4>
                <ul style="margin: 0; padding-left: 1.5rem;">
                    <li>Use clear, well-lit images</li>
                    <li>Include the entire trailer in view</li>
                    <li>Avoid heavily distorted images</li>
                    <li>Higher resolution images work better</li>
                    <li>Try different angles if unsure</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("""
    <div class="footer">
        <h3 style="margin: 0 0 1rem 0;">🚛 Pup Trailer Detector</h3>
        <p style="margin: 0 0 1rem 0; opacity: 0.9;">
            Built with ❤️ using Streamlit & TensorFlow
        </p>
        <p style="margin: 0; opacity: 0.8;">
            Model powered by 🤗 <a href="https://huggingface.co/{MODEL_REPO_ID}" target="_blank" style="color: #FFD700;">Hugging Face</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
