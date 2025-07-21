#!/usr/bin/env python3
"""
🚛 PUP TRAILER DETECTOR STREAMLIT APPLICATION 🚛
Streamlit web application for pup trailer detection using the breakthrough model WITH HUGGING FACE MODEL DOWNLOAD
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
        return """ """
    else:
        return """ """

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
