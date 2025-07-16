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
        st.sidebar.info(f"📦 Model loaded from: {MODEL_REPO_ID}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
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
    # Header
    st.markdown('<h1 class="main-header">🚛 Pup Trailer Detector</h1>', unsafe_allow_html=True)
    
    # Show Hugging Face info
    st.info(f"🤗 This app uses a model from Hugging Face Hub: [{MODEL_REPO_ID}](https://huggingface.co/{MODEL_REPO_ID})")
    
    # Load model
    model = load_breakthrough_model()
    
    if model is None:
        st.error("❌ Failed to load the breakthrough model from Hugging Face Hub.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Dashboard")
    
    # Model info in sidebar
    with st.sidebar.expander("🔍 Model Information"):
        st.write("**Model Name**: Breakthrough Pup Trailer Detection Model")
        st.write("**Architecture**: ResNet50V2 + Custom Head")
        st.write("**Input Shape**: 224x224x3")
        st.write("**Training Strategy**: 2-Phase Training")
        st.write("**Classes**: Non-Pup Trailer, Pup Trailer")
        st.write(f"**Source**: [Hugging Face Hub]({MODEL_REPO_ID})")
    
    # Statistics in sidebar
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        with st.sidebar.expander("📈 Statistics"):
            history = st.session_state.prediction_history
            pup_count = sum(1 for p in history if p['result']['is_pup'])
            total_predictions = len(history)
            
            st.metric("Total Predictions", total_predictions)
            st.metric("Pup Detections", pup_count)
            st.metric("Non-Pup Detections", total_predictions - pup_count)
            
            if total_predictions > 0:
                avg_confidence = np.mean([p['result']['confidence'] for p in history])
                st.metric("Average Confidence", f"{avg_confidence * 100:.1f}%")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Upload Image", "🌐 URL Prediction", "📊 History"])
    
    with tab1:
        st.header("Upload Image for Prediction")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            help="Upload an image of a trailer to detect if it's a pup trailer"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Uploaded Image")
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            
            with col2:
                st.subheader("🔍 Prediction Result")
                
                with st.spinner("Analyzing image..."):
                    result, error = predict_image(model, image)
                
                if error:
                    st.error(f"❌ {error}")
                elif result:
                    # Save to history
                    save_prediction_to_session(result, uploaded_file.name)
                    
                    # Display result
                    css_class = "pup-positive" if result['is_pup'] else "pup-negative"
                    
                    st.markdown(f"""
                    <div class="prediction-box {css_class}">
                        <h3>🎯 {result['class']}</h3>
                        <p><strong>Confidence:</strong> {result['confidence_percentage']}</p>
                        <p><strong>Probability:</strong> {result['probability']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.progress(result['confidence'])
                    
                    # Detailed metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confidence", result['confidence_percentage'])
                    with col_b:
                        st.metric("Classification", result['class'])
    
    with tab2:
        st.header("Predict from URL")
        
        url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        
        if st.button("🔍 Predict from URL") and url:
            try:
                with st.spinner("Downloading and analyzing image..."):
                    # Download image
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    # Open image
                    image = Image.open(io.BytesIO(response.content))
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📸 Downloaded Image")
                        st.image(image, caption="Downloaded from URL", use_container_width=True)
                    
                    with col2:
                        st.subheader("🔍 Prediction Result")
                        
                        result, error = predict_image(model, image)
                        
                        if error:
                            st.error(f"❌ {error}")
                        elif result:
                            # Save to history
                            save_prediction_to_session(result, f"URL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                            
                            # Display result
                            css_class = "pup-positive" if result['is_pup'] else "pup-negative"
                            
                            st.markdown(f"""
                            <div class="prediction-box {css_class}">
                                <h3>🎯 {result['class']}</h3>
                                <p><strong>Confidence:</strong> {result['confidence_percentage']}</p>
                                <p><strong>Probability:</strong> {result['probability']:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(result['confidence'])
                            
            except Exception as e:
                st.error(f"❌ Error processing URL: {str(e)}")
    
    with tab3:
        st.header("📊 Prediction History")
        
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            history = st.session_state.prediction_history
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
            
            # Display history
            for i, prediction in enumerate(reversed(history[-20:])):  # Show last 20
                with st.expander(f"Prediction {len(history) - i}: {prediction['result']['class']} ({prediction['result']['confidence_percentage']})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**File:** {prediction['filename']}")
                        st.write(f"**Time:** {prediction['timestamp']}")
                        st.write(f"**ID:** {prediction['id'][:8]}...")
                    
                    with col2:
                        result = prediction['result']
                        st.write(f"**Class:** {result['class']}")
                        st.write(f"**Confidence:** {result['confidence_percentage']}")
                        st.write(f"**Probability:** {result['probability']:.4f}")
                        st.progress(result['confidence'])
        else:
            st.info("No predictions yet. Upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🚛 Pup Trailer Detector | Built with Streamlit & TensorFlow | Model from 🤗 <a href="https://huggingface.co/{MODEL_REPO_ID}" target="_blank">Hugging Face</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
