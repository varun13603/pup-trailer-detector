#!/usr/bin/env python3
"""
🚛 PUP TRAILER DETECTOR - DEPLOYMENT VERSION 🚛
Streamlit application optimized for deployment compatibility
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import io
import requests
import logging
from huggingface_hub import hf_hub_download
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_REPO_ID = "Jackaiuser/pup_detect"
MODEL_FILENAME = "final_breakthrough_model.h5"
HF_TOKEN = os.getenv('HF_TOKEN') or st.secrets.get('HF_TOKEN', None)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Page configuration
st.set_page_config(
    page_title="🚛 Pup Trailer Detector",
    page_icon="🚛",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #FF6B6B;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .pup-positive {
        background: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    
    .pup-negative {
        background: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_deployment():
    """Load the model for deployment."""
    try:
        # Download model
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
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
        elif len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Normalize and expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image):
    """Make prediction on image."""
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
        logger.error(f"Error making prediction: {e}")
        return None, f"Prediction error: {str(e)}"

def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚛 Pup Trailer Detector</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model_deployment()
    
    if model is None:
        st.error("❌ Failed to load model. Please try again later.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
        help="Upload an image of a trailer to detect if it's a pup trailer"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption=f"File: {uploaded_file.name}")
        
        with col2:
            st.subheader("🔍 Prediction")
            
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    result, error = predict_image(model, image)
                
                if error:
                    st.error(f"❌ {error}")
                elif result:
                    # Display result
                    css_class = "pup-positive" if result['is_pup'] else "pup-negative"
                    
                    st.markdown(f"""
                    <div class="prediction-box {css_class}">
                        <h3>🎯 {result['class']}</h3>
                        <p><strong>Confidence:</strong> {result['confidence_percentage']}</p>
                        <p><strong>Probability:</strong> {result['probability']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(result['confidence'])
                    
                    # Metrics
                    st.metric("Confidence", result['confidence_percentage'])
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit & TensorFlow")
    st.markdown(f"Model: {MODEL_REPO_ID}")

if __name__ == "__main__":
    main()
