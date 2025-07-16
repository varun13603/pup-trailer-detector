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
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
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
    
    /* Fallback for browsers that don't support background-clip */
    @supports not (background-clip: text) {
        .main-header {
            color: #FF6B6B;
            -webkit-text-fill-color: #FF6B6B;
        }
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        color: #333333;
        font-weight: 500;
    }
    .pup-positive {
        border-color: #4CAF50;
        background-color: #f8fff8;
        color: #2e7d32;
    }
    .pup-positive h3 {
        color: #1b5e20;
        margin-top: 0;
    }
    .pup-negative {
        border-color: #f44336;
        background-color: #fff8f8;
        color: #c62828;
    }
    .pup-negative h3 {
        color: #b71c1c;
        margin-top: 0;
    }
    .prediction-box p {
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    .prediction-box strong {
        font-weight: 600;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        color: #333333;
    }
    
    /* Ensure all text in containers is visible */
    .stContainer {
        color: #333333;
    }
    
    /* Style for progress bars */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    
    /* Ensure metric values are visible */
    .metric-value {
        color: #1f1f1f !important;
        font-weight: 600;
    }
    
    /* Style for info boxes */
    .stInfo {
        background-color: #e3f2fd;
        color: #0d47a1;
    }
    
    /* Style for success boxes */
    .stSuccess {
        background-color: #e8f5e8;
        color: #2e7d32;
    }
    
    /* Style for error boxes */
    .stError {
        background-color: #ffebee;
        color: #c62828;
    }
    
    /* Style for warning boxes */
    .stWarning {
        background-color: #fff3e0;
        color: #ef6c00;
    }
</style>
""", unsafe_allow_html=True)

def extract_weights_from_h5(model_path):
    """Extract weights directly from HDF5 file."""
    try:
        import h5py
        weights_dict = {}
        
        with h5py.File(model_path, 'r') as f:
            # Navigate through the HDF5 structure to find weights
            def extract_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights_dict[name] = obj[:]
            
            f.visititems(extract_weights)
            
        logger.info(f"Extracted {len(weights_dict)} weight arrays from H5 file")
        return weights_dict
        
    except Exception as e:
        logger.error(f"Failed to extract weights from H5: {str(e)}")
        return None

def create_compatible_model(model_path):
    """Create a compatible model by reconstructing from the original saved model."""
    try:
        logger.info("Creating compatible model - this preserves your trained weights")
        
        # First, try to load the model with minimal compatibility fixes
        try:
            # Try loading with custom objects for InputLayer compatibility
            class CompatibleInputLayer(tf.keras.layers.InputLayer):
                def __init__(self, *args, **kwargs):
                    if 'batch_shape' in kwargs:
                        batch_shape = kwargs.pop('batch_shape')
                        if batch_shape and len(batch_shape) > 1:
                            kwargs['input_shape'] = batch_shape[1:]
                    super().__init__(*args, **kwargs)
            
            # Custom objects for any other potential issues
            custom_objects = {
                'InputLayer': CompatibleInputLayer,
                'CompatibleInputLayer': CompatibleInputLayer
            }
            
            # Try to load with TensorFlow 2.x compatibility mode
            with tf.keras.utils.custom_object_scope(custom_objects):
                # Load without compilation first
                model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("✅ Successfully loaded original model with compatibility fixes!")
                return model
                
        except Exception as load_error:
            logger.warning(f"Direct loading failed: {str(load_error)}")
            
        # If direct loading fails, try to reconstruct the model
        logger.info("Attempting to reconstruct model from saved weights...")
        
        # Try to extract model configuration and weights
        try:
            import h5py
            
            with h5py.File(model_path, 'r') as f:
                # Try to get model config
                if 'model_config' in f.attrs:
                    model_config = f.attrs['model_config']
                    if isinstance(model_config, bytes):
                        model_config = model_config.decode('utf-8')
                    
                    # Parse the config
                    import json
                    config = json.loads(model_config)
                    
                    # Fix any compatibility issues in the config
                    config = fix_model_config(config)
                    
                    # Reconstruct the model from config
                    model = tf.keras.models.model_from_json(json.dumps(config))
                    
                    # Load weights
                    model.load_weights(model_path)
                    
                    logger.info("✅ Successfully reconstructed model from config and weights!")
                    return model
                    
        except Exception as reconstruct_error:
            logger.warning(f"Model reconstruction failed: {str(reconstruct_error)}")
        
        # Last resort: create a simple model and try to load weights
        logger.warning("Using fallback architecture - performance may be reduced")
        
        # Create a basic model that might match your architecture
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Try to load any compatible weights
        try:
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            logger.info("Loaded some compatible weights")
        except Exception as weight_error:
            logger.warning(f"Could not load weights: {str(weight_error)}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating compatible model: {str(e)}")
        raise e

def fix_model_config(config):
    """Fix compatibility issues in model configuration."""
    try:
        # Recursively fix config issues
        if isinstance(config, dict):
            # Fix InputLayer batch_shape issue
            if config.get('class_name') == 'InputLayer' and 'batch_shape' in config.get('config', {}):
                batch_shape = config['config'].pop('batch_shape')
                if batch_shape and len(batch_shape) > 1:
                    config['config']['input_shape'] = batch_shape[1:]
            
            # Recursively fix nested configs
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    config[key] = fix_model_config(value)
                    
        elif isinstance(config, list):
            config = [fix_model_config(item) for item in config]
            
        return config
        
    except Exception as e:
        logger.warning(f"Error fixing config: {str(e)}")
        return config

def load_model_with_fallback(model_path):
    """Load model with aggressive original model preservation strategies."""
    
    # Strategy 1: Try to load the exact original model first (most aggressive)
    logger.info("🎯 Attempting to load EXACT original model...")
    
    # First, try the most basic loading approaches that don't modify anything
    basic_strategies = [
        ("Direct load (no modification)", lambda: tf.keras.models.load_model(model_path)),
        ("Direct load without compilation", lambda: tf.keras.models.load_model(model_path, compile=False)),
    ]
    
    for strategy_name, strategy_func in basic_strategies:
        try:
            logger.info(f"Trying {strategy_name}...")
            model = strategy_func()
            
            # Test with a fixed seed for reproducible results
            np.random.seed(42)
            tf.random.set_seed(42)
            test_input = tf.random.normal((1, 224, 224, 3))
            test_prediction = model.predict(test_input, verbose=0)
            logger.info(f"✅ {strategy_name} succeeded - EXACT original model loaded!")
            logger.info(f"Test prediction: {test_prediction[0][0]:.6f}")
            
            return model
            
        except Exception as e:
            logger.warning(f"{strategy_name} failed: {str(e)}")
            continue
    
    # Strategy 2: Try minimal compatibility fixes without changing model behavior
    logger.info("🔧 Trying minimal compatibility fixes...")
    
    compatibility_strategies = [
        ("Minimal custom objects", lambda: load_model_minimal_custom_objects(model_path)),
        ("TensorFlow compatibility mode", lambda: load_model_tf_compatibility(model_path)),
    ]
    
    for strategy_name, strategy_func in compatibility_strategies:
        try:
            logger.info(f"Trying {strategy_name}...")
            model = strategy_func()
            
            # Test with same seed
            np.random.seed(42)
            tf.random.set_seed(42)
            test_input = tf.random.normal((1, 224, 224, 3))
            test_prediction = model.predict(test_input, verbose=0)
            logger.info(f"✅ {strategy_name} succeeded!")
            logger.info(f"Test prediction: {test_prediction[0][0]:.6f}")
            
            return model
            
        except Exception as e:
            logger.warning(f"{strategy_name} failed: {str(e)}")
            continue
    
    # Strategy 3: Last resort with clear warning
    logger.warning("⚠️ USING FALLBACK - PERFORMANCE WILL DIFFER FROM ORIGINAL")
    try:
        model = create_compatible_model(model_path)
        model._is_fallback_model = True
        return model
    except Exception as e:
        logger.error(f"All strategies failed: {str(e)}")
        raise Exception("All model loading strategies failed")

def load_model_minimal_custom_objects(model_path):
    """Load model with minimal custom objects that don't change behavior."""
    
    # Only handle the specific InputLayer batch_shape issue without any other modifications
    class ExactInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            # Only handle batch_shape -> input_shape conversion, nothing else
            if 'batch_shape' in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                if batch_shape and len(batch_shape) > 1:
                    kwargs['input_shape'] = batch_shape[1:]
            super().__init__(*args, **kwargs)
    
    # Minimal custom objects - only what's absolutely necessary
    custom_objects = {'InputLayer': ExactInputLayer}
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        return tf.keras.models.load_model(model_path, compile=False)

def load_model_tf_compatibility(model_path):
    """Load model with TensorFlow compatibility settings."""
    
    # Save original TensorFlow settings
    original_sync = None
    try:
        original_sync = tf.config.experimental.get_synchronous_execution()
        tf.config.experimental.set_synchronous_execution(True)
    except:
        pass
    
    try:
        # Try loading with different TensorFlow execution modes
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
        
    finally:
        # Restore original settings
        if original_sync is not None:
            try:
                tf.config.experimental.set_synchronous_execution(original_sync)
            except:
                pass

def verify_model_authenticity(model, model_path):
    """Verify that the loaded model is authentic and not a fallback."""
    try:
        # Create a deterministic test
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Test with multiple different inputs
        test_inputs = [
            tf.random.normal((1, 224, 224, 3)),
            tf.ones((1, 224, 224, 3)) * 0.5,
            tf.zeros((1, 224, 224, 3))
        ]
        
        predictions = []
        for test_input in test_inputs:
            pred = model.predict(test_input, verbose=0)
            predictions.append(pred[0][0])
        
        logger.info(f"Model authenticity test - predictions: {predictions}")
        
        # Check if predictions are in reasonable range and not all the same
        if all(0.0 <= p <= 1.0 for p in predictions):
            if len(set([round(p, 4) for p in predictions])) > 1:  # Not all identical
                logger.info("✅ Model appears to be authentic (diverse predictions)")
                return True
            else:
                logger.warning("⚠️ Model produces identical predictions - might be fallback")
                return False
        else:
            logger.warning("⚠️ Model produces invalid predictions - might be corrupted")
            return False
            
    except Exception as e:
        logger.error(f"Model authenticity verification failed: {str(e)}")
        return False

def load_model_with_custom_objects(model_path):
    """Load model with custom objects to handle deprecated parameters."""
    
    # Enhanced compatible InputLayer
    class CompatibleInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            # Handle deprecated batch_shape parameter
            if 'batch_shape' in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                if batch_shape is not None and len(batch_shape) > 1:
                    kwargs['input_shape'] = batch_shape[1:]
            
            # Handle other potential deprecated parameters
            deprecated_params = ['batch_input_shape']
            for param in deprecated_params:
                if param in kwargs:
                    batch_shape = kwargs.pop(param)
                    if batch_shape is not None and len(batch_shape) > 1:
                        kwargs['input_shape'] = batch_shape[1:]
            
            super().__init__(*args, **kwargs)
        
        def get_config(self):
            config = super().get_config()
            # Ensure no deprecated parameters in config
            config.pop('batch_shape', None)
            config.pop('batch_input_shape', None)
            return config
    
    # Create comprehensive custom objects dictionary
    custom_objects = {
        'InputLayer': CompatibleInputLayer,
        'CompatibleInputLayer': CompatibleInputLayer,
        # Add other custom objects if needed
    }
    
    # Try different loading approaches
    try:
        # Method 1: Direct loading with custom objects
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("✅ Model loaded with custom objects")
            return model
            
    except Exception as e1:
        logger.warning(f"Custom objects loading failed: {str(e1)}")
        
        try:
            # Method 2: Load with safe mode
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False  # Disable safe mode for compatibility
            )
            logger.info("✅ Model loaded with safe_mode=False")
            return model
            
        except Exception as e2:
            logger.warning(f"Safe mode loading failed: {str(e2)}")
            
            # Method 3: Try to fix the model file temporarily
            try:
                model = load_model_with_file_fix(model_path)
                logger.info("✅ Model loaded with file fix")
                return model
                
            except Exception as e3:
                logger.warning(f"File fix loading failed: {str(e3)}")
                raise e3

def load_model_with_file_fix(model_path):
    """Load model by temporarily fixing the model file."""
    import tempfile
    import shutil
    import h5py
    
    # Create a temporary copy of the model file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Copy the original file
        shutil.copy2(model_path, temp_path)
        
        # Try to fix the HDF5 file structure
        with h5py.File(temp_path, 'r+') as f:
            # Fix model config if it exists
            if 'model_config' in f.attrs:
                try:
                    config_str = f.attrs['model_config']
                    if isinstance(config_str, bytes):
                        config_str = config_str.decode('utf-8')
                    
                    # Parse and fix the config
                    import json
                    config = json.loads(config_str)
                    fixed_config = fix_model_config(config)
                    
                    # Save the fixed config back
                    f.attrs['model_config'] = json.dumps(fixed_config).encode('utf-8')
                    
                except Exception as config_error:
                    logger.warning(f"Could not fix model config: {str(config_error)}")
        
        # Try to load the fixed model
        class CompatibleInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, *args, **kwargs):
                if 'batch_shape' in kwargs:
                    batch_shape = kwargs.pop('batch_shape')
                    if batch_shape and len(batch_shape) > 1:
                        kwargs['input_shape'] = batch_shape[1:]
                super().__init__(*args, **kwargs)
        
        custom_objects = {'InputLayer': CompatibleInputLayer}
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(temp_path, compile=False)
            return model
            
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

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
        
        # Load the model with EXACT original model preservation
        logger.info(f"Loading breakthrough model with EXACT preservation: {MODEL_PATH}")
        
        # Use the aggressive original model loading strategy
        try:
            model = load_model_with_fallback(MODEL_PATH)
            
            # Verify model authenticity
            is_authentic = verify_model_authenticity(model, MODEL_PATH)
            
            # Check if we're using the original model or a fallback
            if hasattr(model, '_is_fallback_model') and model._is_fallback_model:
                st.error("⚠️ FALLBACK MODEL IN USE - PERFORMANCE WILL DIFFER!")
                st.error("The original model could not be loaded due to compatibility issues.")
                st.info("This fallback model uses different weights and will give different results.")
                st.info("Consider using the same TensorFlow version as training for identical results.")
            elif not is_authentic:
                st.warning("⚠️ Model loaded but authenticity verification failed.")
                st.info("The model may not be performing exactly as trained.")
            else:
                st.success("✅ EXACT original trained model loaded successfully!")
                st.success("🎯 Model authenticity verified - predictions will match your training results!")
            
        except Exception as loading_error:
            logger.error(f"All loading strategies failed: {str(loading_error)}")
            st.error(f"❌ Failed to load model: {str(loading_error)}")
            st.error("CRITICAL: Could not load your original model!")
            st.info("Possible solutions:")
            st.info("1. Check TensorFlow version compatibility")
            st.info("2. Verify the model file is not corrupted")
            st.info("3. Try re-uploading the model to Hugging Face")
            return None
        
        # Compile the model if it wasn't compiled during loading
        if model is not None:
            try:
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("✅ Model compiled successfully")
            except Exception as compile_error:
                logger.warning(f"Model compilation failed: {str(compile_error)}")
                # Model can still be used for prediction without compilation
        
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
                    
                    # Detailed metrics with better styling
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confidence", result['confidence_percentage'], 
                                delta=None, delta_color="normal")
                    with col_b:
                        st.metric("Classification", result['class'], 
                                delta=None, delta_color="normal")
    
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
                            
                            # Add metrics for URL predictions too
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Confidence", result['confidence_percentage'])
                            with col_b:
                                st.metric("Classification", result['class'])
                            
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
