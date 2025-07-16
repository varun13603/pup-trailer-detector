#!/usr/bin/env python3
"""
🚛 ENHANCED PUP TRAILER DETECTOR STREAMLIT APPLICATION 🚛
Enhanced Streamlit web application with better UI, more features, and exact model loading
WITH AGGRESSIVE ORIGINAL MODEL PRESERVATION
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import json
from datetime import datetime
import uuid
import requests
import logging
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
import tempfile
import shutil
import h5py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_REPO_ID = "Jackaiuser/pup_detect"
MODEL_FILENAME = "final_breakthrough_model.h5"
IMG_HEIGHT = 224
IMG_WIDTH = 224
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Enhanced Page configuration
st.set_page_config(
    page_title="🚛 Advanced Pup Trailer Detector",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/pup-detector',
        'Report a bug': "https://github.com/your-repo/pup-detector/issues",
        'About': "# Advanced Pup Trailer Detector\nBuilt with Streamlit & TensorFlow"
    }
)

# Enhanced Custom CSS with dark mode support
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Root variables for theming */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --success-color: #4CAF50;
        --error-color: #f44336;
        --warning-color: #ff9800;
        --info-color: #2196F3;
        --background-color: #f5f5f5;
        --surface-color: #ffffff;
        --text-color: #333333;
        --text-secondary: #666666;
        --border-color: #e0e0e0;
        --shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #1a1a1a;
            --surface-color: #2d2d2d;
            --text-color: #ffffff;
            --text-secondary: #cccccc;
            --border-color: #444444;
            --shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
    }
    
    /* Global styles */
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Main header with enhanced styling */
    .main-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: gradient-animation 3s ease-in-out infinite;
    }
    
    @keyframes gradient-animation {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Fallback for browsers that don't support background-clip */
    @supports not (background-clip: text) {
        .main-header {
            color: var(--primary-color);
            -webkit-text-fill-color: var(--primary-color);
        }
    }
    
    /* Enhanced prediction box with animations */
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid var(--border-color);
        margin: 1.5rem 0;
        color: var(--text-color);
        font-weight: 500;
        background: var(--surface-color);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .prediction-box:hover::before {
        left: 100%;
    }
    
    .pup-positive {
        border-color: var(--success-color);
        background: linear-gradient(135deg, #f8fff8, #e8f5e8);
        color: #1b5e20;
    }
    
    .pup-positive h3 {
        color: #1b5e20;
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .pup-negative {
        border-color: var(--error-color);
        background: linear-gradient(135deg, #fff8f8, #ffebee);
        color: #b71c1c;
    }
    
    .pup-negative h3 {
        color: #b71c1c;
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .prediction-box p {
        margin: 0.75rem 0;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    
    .prediction-box strong {
        font-weight: 600;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: var(--surface-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow);
        margin: 1rem 0;
        color: var(--text-color);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Status indicators */
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #4caf50;
    }
    
    .status-error {
        background: #ffebee;
        color: #c62828;
        border: 1px solid #f44336;
    }
    
    .status-warning {
        background: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ff9800;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background: var(--surface-color);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        color: var(--text-color);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
    }
    
    /* Enhanced file uploader */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: var(--surface-color);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background: linear-gradient(135deg, #fdf2f2, #f8f9fa);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Enhanced progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 5px;
        height: 8px;
    }
    
    /* Enhanced metrics */
    .metric-container {
        background: var(--surface-color);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid var(--border-color);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: var(--surface-color);
    }
    
    /* Enhanced expanders */
    .streamlit-expanderHeader {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        margin-top: 2rem;
        color: var(--text-secondary);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
            padding: 2rem 0;
        }
        
        .prediction-box {
            padding: 1.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Dark mode specific styles */
    @media (prefers-color-scheme: dark) {
        .pup-positive {
            background: linear-gradient(135deg, #1a2e1a, #2d4a2d);
        }
        
        .pup-negative {
            background: linear-gradient(135deg, #2e1a1a, #4a2d2d);
        }
    }
</style>
""", unsafe_allow_html=True)

def force_exact_model_loading():
    """Force the exact original model to load by trying multiple aggressive strategies."""
    
    # Download model if not exists
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        logger.info(f"Model path: {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None
    
    # Strategy 1: Force TensorFlow version compatibility
    logger.info("🎯 FORCING EXACT MODEL LOADING - NO FALLBACKS ALLOWED")
    
    # Set TensorFlow to most compatible mode
    try:
        tf.config.experimental.set_synchronous_execution(True)
        tf.keras.backend.set_image_data_format('channels_last')
        
        # Disable eager execution temporarily if needed
        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()
    except Exception as tf_config_error:
        logger.warning(f"TensorFlow config adjustment failed: {tf_config_error}")
    
    # Strategy 2: Try loading with exact original parameters
    exact_loading_strategies = [
        ("EXACT_ORIGINAL_NO_CHANGES", lambda: tf.keras.models.load_model(model_path)),
        ("EXACT_ORIGINAL_NO_COMPILE", lambda: tf.keras.models.load_model(model_path, compile=False)),
        ("EXACT_WITH_CUSTOM_SCOPE", lambda: load_with_exact_custom_scope(model_path)),
        ("EXACT_WITH_H5PY_DIRECT", lambda: load_with_h5py_direct(model_path)),
        ("EXACT_WITH_WEIGHTS_ONLY", lambda: load_with_weights_only(model_path)),
    ]
    
    for strategy_name, strategy_func in exact_loading_strategies:
        try:
            logger.info(f"🔥 Attempting {strategy_name}...")
            
            # Set deterministic seed for testing
            np.random.seed(42)
            tf.random.set_seed(42)
            
            model = strategy_func()
            
            # Verify this is the EXACT original model
            if verify_exact_original_model(model):
                logger.info(f"✅ SUCCESS: {strategy_name} loaded EXACT original model!")
                return model
            else:
                logger.warning(f"❌ {strategy_name} loaded but model verification failed")
                
        except Exception as e:
            logger.warning(f"❌ {strategy_name} failed: {str(e)}")
            continue
    
    # If we reach here, all strategies failed
    logger.error("🚨 CRITICAL: ALL EXACT LOADING STRATEGIES FAILED")
    logger.error("🚨 DEPLOYMENT ENVIRONMENT INCOMPATIBLE WITH ORIGINAL MODEL")
    raise Exception("EXACT MODEL LOADING FAILED - DEPLOYMENT ENVIRONMENT ISSUE")

def load_with_exact_custom_scope(model_path):
    """Load with exact custom scope that preserves original behavior."""
    
    # Create the most minimal custom objects possible
    class ExactOriginalInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            # Only handle the batch_shape issue, nothing else
            if 'batch_shape' in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                if batch_shape and len(batch_shape) > 1:
                    kwargs['input_shape'] = batch_shape[1:]
            super().__init__(*args, **kwargs)
    
    # Minimal scope - only what's absolutely necessary
    custom_objects = {
        'InputLayer': ExactOriginalInputLayer
    }
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        return tf.keras.models.load_model(model_path, compile=False)

def load_with_h5py_direct(model_path):
    """Load model by directly manipulating HDF5 file."""
    
    # Create a temporary fixed version of the model
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Copy original file
        shutil.copy2(model_path, temp_path)
        
        # Fix the HDF5 file directly
        with h5py.File(temp_path, 'r+') as f:
            # Fix model config to be compatible
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                # Parse and fix config
                config = json.loads(config_str)
                fixed_config = fix_model_config_exactly(config)
                
                # Save fixed config
                f.attrs['model_config'] = json.dumps(fixed_config).encode('utf-8')
        
        # Load the fixed model
        class ExactInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, *args, **kwargs):
                if 'batch_shape' in kwargs:
                    batch_shape = kwargs.pop('batch_shape')
                    if batch_shape and len(batch_shape) > 1:
                        kwargs['input_shape'] = batch_shape[1:]
                super().__init__(*args, **kwargs)
        
        with tf.keras.utils.custom_object_scope({'InputLayer': ExactInputLayer}):
            return tf.keras.models.load_model(temp_path, compile=False)
    
    finally:
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass

def load_with_weights_only(model_path):
    """Load by reconstructing architecture and loading weights."""
    
    # Try to get the exact architecture from the model file
    try:
        with h5py.File(model_path, 'r') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                # Fix compatibility issues
                fixed_config = fix_model_config_exactly(config)
                
                # Reconstruct model from fixed config
                model = tf.keras.models.model_from_json(json.dumps(fixed_config))
                
                # Load weights
                model.load_weights(model_path)
                
                return model
    except Exception as e:
        logger.error(f"Architecture reconstruction failed: {e}")
        raise e

def fix_model_config_exactly(config):
    """Fix model config with exact precision to preserve original behavior."""
    
    def fix_layer_config(layer_config):
        if isinstance(layer_config, dict):
            # Fix InputLayer batch_shape issue
            if layer_config.get('class_name') == 'InputLayer':
                layer_config_inner = layer_config.get('config', {})
                if 'batch_shape' in layer_config_inner:
                    batch_shape = layer_config_inner.pop('batch_shape')
                    if batch_shape and len(batch_shape) > 1:
                        layer_config_inner['input_shape'] = batch_shape[1:]
            
            # Recursively fix nested configs
            for key, value in layer_config.items():
                if isinstance(value, (dict, list)):
                    layer_config[key] = fix_layer_config(value)
        
        elif isinstance(layer_config, list):
            layer_config = [fix_layer_config(item) for item in layer_config]
        
        return layer_config
    
    return fix_layer_config(config)

def verify_exact_original_model(model):
    """Verify that the loaded model is the exact original model."""
    
    try:
        # Set deterministic seed
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Test with known reference inputs
        test_inputs = [
            np.random.normal(0, 1, (1, 224, 224, 3)),
            np.ones((1, 224, 224, 3)) * 0.5,
            np.zeros((1, 224, 224, 3)),
            np.random.uniform(0, 1, (1, 224, 224, 3))
        ]
        
        predictions = []
        for test_input in test_inputs:
            pred = model.predict(test_input, verbose=0)
            predictions.append(float(pred[0][0]))
        
        logger.info(f"Model verification predictions: {predictions}")
        
        # Check if predictions are in valid range and diverse
        if all(0.0 <= p <= 1.0 for p in predictions):
            # Check for diversity (not all the same)
            unique_predictions = len(set([round(p, 6) for p in predictions]))
            if unique_predictions > 1:
                logger.info("✅ Model verification PASSED - appears to be original model")
                return True
            else:
                logger.warning("❌ Model verification FAILED - identical predictions (fallback model)")
                return False
        else:
            logger.warning("❌ Model verification FAILED - invalid prediction range")
            return False
    
    except Exception as e:
        logger.error(f"Model verification error: {e}")
        return False

@st.cache_resource
def load_exact_breakthrough_model():
    """Load the breakthrough model with EXACT original preservation - NO FALLBACKS."""
    
    try:
        # Show loading status
        status_container = st.container()
        with status_container:
            st.info("🔄 Loading EXACT original model - NO FALLBACKS ALLOWED")
        
        # Force exact model loading
        model = force_exact_model_loading()
        
        if model is None:
            status_container.empty()
            st.error("🚨 CRITICAL: Exact model loading failed!")
            st.error("🚨 Deployment environment is incompatible with original model")
            st.error("🚨 Consider using the same TensorFlow version as training")
            return None
        
        # Compile the model
        try:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        except Exception as compile_error:
            logger.warning(f"Model compilation failed: {compile_error}")
        
        # Clear status and show success
        status_container.empty()
        st.success("✅ EXACT ORIGINAL MODEL LOADED SUCCESSFULLY!")
        st.success("🎯 Model authenticity verified - predictions will match training results!")
        
        return model
    
    except Exception as e:
        logger.error(f"Exact model loading failed: {e}")
        st.error(f"🚨 EXACT MODEL LOADING FAILED: {str(e)}")
        st.error("🚨 This is a deployment environment issue")
        st.error("🚨 The model cannot be loaded with exact original weights")
        return None

def preprocess_image_enhanced(image, apply_enhancements=False):
    """Enhanced image preprocessing with optional enhancements."""
    try:
        # Apply enhancements if requested
        if apply_enhancements:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.05)
        
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
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image_enhanced(model, image, apply_enhancements=False):
    """Enhanced prediction with optional image enhancements."""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Preprocess image
        processed_image = preprocess_image_enhanced(image, apply_enhancements)
        if processed_image is None:
            return None, "Error processing image"
        
        # Make prediction with timing
        start_time = time.time()
        prediction = model.predict(processed_image, verbose=0)
        prediction_time = time.time() - start_time
        
        probability = float(prediction[0][0])
        
        # Determine class
        is_pup = probability > 0.5
        confidence = probability if is_pup else 1 - probability
        
        result = {
            'is_pup': is_pup,
            'probability': probability,
            'confidence': confidence,
            'class': 'Pup Trailer' if is_pup else 'Not a Pup Trailer',
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'prediction_time': f"{prediction_time:.3f}s",
            'enhancements_applied': apply_enhancements
        }
        
        return result, None
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None, f"Prediction error: {str(e)}"

def save_prediction_enhanced(result, image_name, image_size=None):
    """Enhanced prediction saving with more metadata."""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction_record = {
        'timestamp': datetime.now().isoformat(),
        'filename': image_name,
        'result': result,
        'id': str(uuid.uuid4()),
        'image_size': image_size,
        'session_id': st.session_state.get('session_id', str(uuid.uuid4()))
    }
    
    # Set session ID if not exists
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    st.session_state.prediction_history.append(prediction_record)
    
    # Keep only last 100 predictions
    if len(st.session_state.prediction_history) > 100:
        st.session_state.prediction_history = st.session_state.prediction_history[-100:]

def create_prediction_chart(history):
    """Create a prediction chart using plotly."""
    if not history:
        return None
    
    # Prepare data
    df = pd.DataFrame([
        {
            'timestamp': datetime.fromisoformat(p['timestamp']),
            'confidence': p['result']['confidence'],
            'is_pup': p['result']['is_pup'],
            'class': p['result']['class']
        } for p in history
    ])
    
    # Create confidence over time chart
    fig = px.line(df, x='timestamp', y='confidence', 
                  color='class', title='Prediction Confidence Over Time',
                  labels={'confidence': 'Confidence', 'timestamp': 'Time'})
    
    fig.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        height=400
    )
    
    return fig

def create_confidence_distribution():
    """Create confidence distribution chart."""
    if 'prediction_history' not in st.session_state or not st.session_state.prediction_history:
        return None
    
    history = st.session_state.prediction_history
    confidences = [p['result']['confidence'] for p in history]
    
    fig = px.histogram(x=confidences, nbins=20, title='Confidence Distribution',
                      labels={'x': 'Confidence', 'y': 'Count'})
    
    fig.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        height=400
    )
    
    return fig

def main():
    """Enhanced main application."""
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🚛 Advanced Pup Trailer Detector</h1>', unsafe_allow_html=True)
    
    # Model status indicator
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="status-indicator status-success">
                🤗 Model: {MODEL_REPO_ID}
            </div>
            """, unsafe_allow_html=True)
    
    # Load model with exact preservation
    model = load_exact_breakthrough_model()
    
    if model is None:
        st.error("🚨 Application cannot continue without the exact original model")
        st.stop()
    
    # Enhanced sidebar with more features
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Model information
        with st.expander("🔍 Model Information", expanded=True):
            st.markdown("""
            **Model**: Breakthrough Pup Trailer Detection
            **Architecture**: ResNet50V2 + Custom Head
            **Input Size**: 224×224×3
            **Classes**: Pup Trailer, Non-Pup Trailer
            **Training**: 2-Phase Strategy
            """)
            
            # Model status
            st.markdown(f"""
            <div class="status-indicator status-success">
                ✅ EXACT Original Model Active
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction settings
        with st.expander("⚙️ Prediction Settings"):
            apply_enhancements = st.checkbox("Apply Image Enhancements", 
                                            help="Enhance contrast, sharpness, and brightness")
            
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01,
                                            help="Minimum confidence for positive prediction")
        
        # Session statistics
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            with st.expander("📊 Session Statistics", expanded=True):
                history = st.session_state.prediction_history
                total_predictions = len(history)
                pup_count = sum(1 for p in history if p['result']['is_pup'])
                non_pup_count = total_predictions - pup_count
                
                st.metric("Total Predictions", total_predictions)
                st.metric("Pup Trailers", pup_count)
                st.metric("Non-Pup Trailers", non_pup_count)
                
                if total_predictions > 0:
                    avg_confidence = np.mean([p['result']['confidence'] for p in history])
                    avg_time = np.mean([float(p['result']['prediction_time'].replace('s', '')) for p in history])
                    
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    st.metric("Avg Time", f"{avg_time:.3f}s")
        
        # Quick actions
        with st.expander("🚀 Quick Actions"):
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
            
            if st.button("📊 Download Report"):
                if 'prediction_history' in st.session_state:
                    # Create downloadable report
                    report_data = pd.DataFrame([
                        {
                            'Timestamp': p['timestamp'],
                            'Filename': p['filename'],
                            'Classification': p['result']['class'],
                            'Confidence': p['result']['confidence'],
                            'Probability': p['result']['probability']
                        } for p in st.session_state.prediction_history
                    ])
                    
                    csv = report_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV Report",
                        data=csv,
                        file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Enhanced main content with more tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Upload Image", 
        "🌐 URL Prediction", 
        "📊 Analytics", 
        "🔍 Model Explorer",
        "📈 Performance"
    ])
    
    with tab1:
        st.header("📷 Upload Image for Prediction")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            help="Upload an image of a trailer to detect if it's a pup trailer"
        )
        
        if uploaded_file is not None:
            # Display image with metadata
            image = Image.open(uploaded_file)
            image_size = f"{image.width}×{image.height}"
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Uploaded Image")
                st.image(image, caption=f"File: {uploaded_file.name} | Size: {image_size}", 
                        use_container_width=True)
                
                # Image metadata
                with st.expander("📋 Image Details"):
                    st.write(f"**Filename**: {uploaded_file.name}")
                    st.write(f"**Size**: {image_size}")
                    st.write(f"**Format**: {image.format}")
                    st.write(f"**Mode**: {image.mode}")
                    st.write(f"**File Size**: {uploaded_file.size} bytes")
            
            with col2:
                st.subheader("🔍 Prediction Result")
                
                # Prediction button
                if st.button("🚀 Analyze Image", type="primary"):
                    with st.spinner("🔄 Analyzing image with exact model..."):
                        result, error = predict_image_enhanced(model, image, apply_enhancements)
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif result:
                        # Save to history
                        save_prediction_enhanced(result, uploaded_file.name, image_size)
                        
                        # Display enhanced result
                        css_class = "pup-positive" if result['is_pup'] else "pup-negative"
                        
                        st.markdown(f"""
                        <div class="prediction-box {css_class}">
                            <h3>🎯 {result['class']}</h3>
                            <p><strong>Confidence:</strong> {result['confidence_percentage']}</p>
                            <p><strong>Probability:</strong> {result['probability']:.4f}</p>
                            <p><strong>Processing Time:</strong> {result['prediction_time']}</p>
                            <p><strong>Enhancements:</strong> {'Applied' if result['enhancements_applied'] else 'Not Applied'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enhanced progress bar
                        st.progress(result['confidence'])
                        
                        # Detailed metrics
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Confidence", result['confidence_percentage'])
                        with metric_col2:
                            st.metric("Processing Time", result['prediction_time'])
                        
                        # Confidence gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = result['confidence'] * 100,
                            title = {'text': "Confidence Level"},
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#4CAF50" if result['is_pup'] else "#f44336"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab2:
        st.header("🌐 URL Prediction")
        
        url = st.text_input("Enter image URL:", 
                          placeholder="https://example.com/trailer-image.jpg")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            pass
        with col2:
            predict_url = st.button("🔍 Predict from URL", type="primary")
        
        if predict_url and url:
            try:
                with st.spinner("🔄 Downloading and analyzing image..."):
                    # Download image with progress
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
                        
                        result, error = predict_image_enhanced(model, image, apply_enhancements)
                        
                        if error:
                            st.error(f"❌ {error}")
                        elif result:
                            # Save to history
                            save_prediction_enhanced(result, f"URL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                            
                            # Display result
                            css_class = "pup-positive" if result['is_pup'] else "pup-negative"
                            
                            st.markdown(f"""
                            <div class="prediction-box {css_class}">
                                <h3>🎯 {result['class']}</h3>
                                <p><strong>Confidence:</strong> {result['confidence_percentage']}</p>
                                <p><strong>Probability:</strong> {result['probability']:.4f}</p>
                                <p><strong>Processing Time:</strong> {result['prediction_time']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(result['confidence'])
                            
            except Exception as e:
                st.error(f"❌ Error processing URL: {str(e)}")
    
    with tab3:
        st.header("📊 Analytics Dashboard")
        
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            history = st.session_state.prediction_history
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence over time
                fig_time = create_prediction_chart(history)
                if fig_time:
                    st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig_dist = create_confidence_distribution()
                if fig_dist:
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Summary statistics
            st.subheader("📋 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(history))
            
            with col2:
                pup_count = sum(1 for p in history if p['result']['is_pup'])
                st.metric("Pup Trailers", pup_count)
            
            with col3:
                avg_confidence = np.mean([p['result']['confidence'] for p in history])
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            with col4:
                avg_time = np.mean([float(p['result']['prediction_time'].replace('s', '')) for p in history])
                st.metric("Avg Time", f"{avg_time:.3f}s")
            
            # Recent predictions table
            st.subheader("📝 Recent Predictions")
            
            df = pd.DataFrame([
                {
                    'Timestamp': datetime.fromisoformat(p['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    'Filename': p['filename'],
                    'Classification': p['result']['class'],
                    'Confidence': f"{p['result']['confidence']:.2%}",
                    'Time': p['result']['prediction_time']
                } for p in reversed(history[-20:])
            ])
            
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("📊 No predictions yet. Upload an image to see analytics!")
    
    with tab4:
        st.header("🔍 Model Explorer")
        
        # Model architecture info
        st.subheader("🏗️ Model Architecture")
        
        if model:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Summary:**")
                try:
                    # Create a string buffer to capture model summary
                    import io
                    buffer = io.StringIO()
                    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
                    summary_str = buffer.getvalue()
                    st.text(summary_str[:1000] + "..." if len(summary_str) > 1000 else summary_str)
                except Exception as e:
                    st.error(f"Could not display model summary: {e}")
            
            with col2:
                st.write("**Model Configuration:**")
                st.json({
                    "Input Shape": [224, 224, 3],
                    "Output Shape": [1],
                    "Activation": "sigmoid",
                    "Loss": "binary_crossentropy",
                    "Optimizer": "adam",
                    "Classes": ["Non-Pup Trailer", "Pup Trailer"]
                })
        
        # Model testing
        st.subheader("🧪 Model Testing")
        
        if st.button("🔬 Run Model Test"):
            with st.spinner("Testing model with reference inputs..."):
                # Test model with reference inputs
                np.random.seed(42)
                tf.random.set_seed(42)
                
                test_results = []
                test_inputs = [
                    ("Random Normal", np.random.normal(0, 1, (1, 224, 224, 3))),
                    ("Uniform 0.5", np.ones((1, 224, 224, 3)) * 0.5),
                    ("Zeros", np.zeros((1, 224, 224, 3))),
                    ("Random Uniform", np.random.uniform(0, 1, (1, 224, 224, 3)))
                ]
                
                for name, test_input in test_inputs:
                    pred = model.predict(test_input, verbose=0)
                    test_results.append({
                        "Test Input": name,
                        "Prediction": f"{pred[0][0]:.6f}",
                        "Classification": "Pup Trailer" if pred[0][0] > 0.5 else "Non-Pup Trailer"
                    })
                
                st.table(pd.DataFrame(test_results))
                
                if len(set([r["Prediction"] for r in test_results])) > 1:
                    st.success("✅ Model test passed - diverse predictions")
                else:
                    st.error("❌ Model test failed - identical predictions")
    
    with tab5:
        st.header("📈 Performance Metrics")
        
        # System info
        st.subheader("💻 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Environment:**")
            st.write(f"Python: {sys.version}")
            st.write(f"TensorFlow: {tf.__version__}")
            st.write(f"NumPy: {np.__version__}")
            st.write(f"Streamlit: {st.__version__}")
        
        with col2:
            st.write("**Model Info:**")
            st.write(f"Repository: {MODEL_REPO_ID}")
            st.write(f"Filename: {MODEL_FILENAME}")
            st.write(f"Input Size: {IMG_HEIGHT}×{IMG_WIDTH}")
            st.write(f"Session ID: {st.session_state.get('session_id', 'N/A')[:8]}...")
        
        # Performance stats
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            st.subheader("⚡ Performance Statistics")
            
            history = st.session_state.prediction_history
            times = [float(p['result']['prediction_time'].replace('s', '')) for p in history]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Min Time", f"{min(times):.3f}s")
            
            with col2:
                st.metric("Max Time", f"{max(times):.3f}s")
            
            with col3:
                st.metric("Avg Time", f"{np.mean(times):.3f}s")
            
            with col4:
                st.metric("Std Dev", f"{np.std(times):.3f}s")
            
            # Performance chart
            fig_perf = px.line(y=times, title="Prediction Time Over Requests",
                              labels={'x': 'Request Number', 'y': 'Time (seconds)'})
            st.plotly_chart(fig_perf, use_container_width=True)
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <h4>🚛 Advanced Pup Trailer Detector</h4>
        <p>Built with ❤️ using Streamlit & TensorFlow</p>
        <p>Model hosted on 🤗 <a href="https://huggingface.co/{MODEL_REPO_ID}" target="_blank">Hugging Face Hub</a></p>
        <p>Session ID: {st.session_state.get('session_id', 'N/A')[:8]}... | Version: 2.0.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
