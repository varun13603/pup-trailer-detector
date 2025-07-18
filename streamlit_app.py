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

# API endpoint handling
def handle_api_request():
    """Handle API requests for programmatic image uploads"""
    # Check if this is an API request
    query_params = st.query_params
    
    if 'api' in query_params and query_params['api'] == 'predict':
        # This is an API request
        st.markdown("# 🤖 API Prediction Endpoint")
        
        # Check for base64 image data in query params
        if 'image_data' in query_params:
            try:
                image_data = query_params['image_data']
                
                # Decode base64 image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Load model and make prediction
                model = load_breakthrough_model()
                if model:
                    result, error = predict_image(model, image)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                        return True
                    
                    # Convert result format for compatibility
                    api_result = {
                        'prediction': 'pup' if result['is_pup'] else 'not_pup',
                        'confidence': result['confidence'] * 100,  # Convert to percentage
                        'probability': result['probability'],
                        'class': result['class'],
                        'filename': 'tampermonkey_screenshot.png'
                    }
                    
                    # Display result
                    st.success(f"✅ Prediction completed!")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    with col2:
                        prediction_class = "🚛 PUP TRAILER" if api_result['prediction'] == 'pup' else "🚚 NOT PUP"
                        confidence = api_result['confidence']
                        
                        st.markdown(f"""
                        <div style="background: {'#e6fffa' if api_result['prediction'] == 'pup' else '#fff5f5'}; 
                                    border: 2px solid {'#38a169' if api_result['prediction'] == 'pup' else '#e53e3e'}; 
                                    border-radius: 12px; padding: 1.5rem; text-align: center;">
                            <h3 style="color: {'#2f855a' if api_result['prediction'] == 'pup' else '#c53030'}; margin-bottom: 1rem;">
                                {prediction_class}
                            </h3>
                            <p style="font-size: 1.2rem; font-weight: bold; color: #4a5568;">
                                Confidence: {confidence:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Return JSON response for programmatic access
                    st.json(api_result)
                    return True
                else:
                    st.error("❌ Model failed to load")
                    return True
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                return True
        else:
            st.info("📝 Send POST request with 'image_data' parameter containing base64 encoded image")
            return True
    
    # Check if this is from Tampermonkey (localStorage method)
    if 'source' in query_params and query_params['source'] == 'tampermonkey':
        st.markdown("# 🤖 Tampermonkey Image Prediction")
        
        # Check if chunked data transfer is being used
        is_chunked = 'chunks' in query_params and query_params['chunks'] == 'true'
        
        # Check if blob URL method is being used
        is_blob = 'method' in query_params and query_params['method'] == 'blob'
        direct_blob = 'blob' in query_params
        
        # Add JavaScript to check localStorage with enhanced chunked data support and blob handling
        html_content = """
        <script>
        console.log('🔄 Tampermonkey integration script started');
        let processingComplete = false;
        
        function processImageData() {
            if (processingComplete) {
                console.log('⏭️ Processing already completed');
                return;
            }
            
            console.log('🔍 Checking for image data...');
            
            // Check for blob URL in query params first
            const urlParams = new URLSearchParams(window.location.search);
            const blobUrl = urlParams.get('blob');
            if (blobUrl) {
                console.log('📎 Found blob URL in query params');
                processBlobUrl(decodeURIComponent(blobUrl));
                return;
            }
            
            // Check for blob reference in localStorage
            try {
                const blobData = localStorage.getItem('pupBlobTransfer');
                if (blobData) {
                    console.log('📎 Found blob transfer data');
                    const data = JSON.parse(blobData);
                    console.log('📊 Blob data:', data.filename, data.size + 'KB');
                    
                    processingComplete = true;
                    localStorage.removeItem('pupBlobTransfer');
                    
                    processBlobUrl(data.blobUrl);
                    return;
                }
            } catch (error) {
                console.error('❌ Error processing blob data:', error);
                showStatus('❌ Error Processing Blob Data: ' + error.message, '#ffebee');
            }
            
            // Check for chunked data
            try {
                const chunkedData = localStorage.getItem('pupPredictionImageChunks');
                if (chunkedData) {
                    console.log('📦 Found chunked data');
                    const chunkInfo = JSON.parse(chunkedData);
                    console.log('📊 Chunk info:', chunkInfo.totalChunks, 'chunks');
                    
                    // Reconstruct the full image from chunks
                    const fullImageData = chunkInfo.chunks.join('');
                    console.log('🔗 Reconstructed image size:', Math.round(fullImageData.length / 1024), 'KB');
                    
                    processingComplete = true;
                    localStorage.removeItem('pupPredictionImageChunks');
                    
                    // Redirect to API with the reconstructed data
                    redirectToAPI(fullImageData);
                    return;
                }
            } catch (error) {
                console.error('❌ Error processing chunked data:', error);
                showStatus('❌ Error Processing Chunked Data: ' + error.message, '#ffebee');
                return;
            }
            
            // Check for regular image data
            try {
                const imageData = localStorage.getItem('pupPredictionImage');
                if (imageData) {
                    console.log('🖼️ Found regular image data');
                    const data = JSON.parse(imageData);
                    
                    const sizeKB = Math.round(data.data.length / 1024);
                    console.log('📏 Image size:', sizeKB, 'KB, compressed:', data.compressed);
                    
                    processingComplete = true;
                    localStorage.removeItem('pupPredictionImage');
                    
                    // Redirect to API with the image data
                    redirectToAPI(data.data);
                    return;
                }
            } catch (error) {
                console.error('❌ Error processing image data:', error);
                showStatus('❌ Error Processing Image Data: ' + error.message, '#ffebee');
                return;
            }
            
            // No data found
            console.log('📭 No image data found in localStorage');
            showStatus(
                '📷 Waiting for image from Tampermonkey...<br>' +
                'Take a screenshot using the Tampermonkey script and it will appear here automatically.<br>' +
                '<small>✅ Integration is working correctly. Supports images of any size.</small><br>' +
                '<small>🌐 Production URL: https://pup-test.streamlit.app/</small>',
                '#e3f2fd'
            );
        }
        
        function processBlobUrl(blobUrl) {
            console.log('📎 Processing blob URL:', blobUrl.substring(0, 50) + '...');
            showStatus('📎 Processing blob image data...', '#fff3cd');
            
            fetch(blobUrl)
                .then(response => response.blob())
                .then(blob => {
                    console.log('📦 Blob loaded, size:', Math.round(blob.size / 1024), 'KB');
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const base64Data = e.target.result.split(',')[1];
                        console.log('🔄 Converted blob to base64, redirecting...');
                        redirectToAPI(base64Data);
                    };
                    reader.readAsDataURL(blob);
                })
                .catch(error => {
                    console.error('❌ Error processing blob URL:', error);
                    showStatus('❌ Error Processing Blob: ' + error.message, '#ffebee');
                });
        }
        
        function redirectToAPI(imageData) {
            console.log('🚀 Redirecting to API endpoint...');
            showStatus('🚀 Redirecting to prediction API...', '#fff3cd');
            
            // Build the redirect URL
            const baseUrl = window.location.origin + window.location.pathname;
            const apiUrl = baseUrl + '?api=predict&image_data=' + encodeURIComponent(imageData) + '&image_loaded=true';
            
            console.log('🔗 API URL length:', apiUrl.length);
            
            // Try to redirect
            try {
                window.location.href = apiUrl;
            } catch (error) {
                console.error('❌ Redirect failed:', error);
                showStatus('❌ Redirect failed: ' + error.message, '#ffebee');
            }
        }
        
        function showStatus(message, backgroundColor) {
            const statusDiv = document.getElementById('status');
            if (statusDiv) {
                statusDiv.innerHTML = 
                    '<div style="text-align: center; padding: 20px; background: ' + backgroundColor + '; border: 1px solid #ddd; border-radius: 8px;">' +
                    '<h3>' + message + '</h3>' +
                    '</div>';
            }
        }
        
        // Start processing immediately
        console.log('🎬 Starting image processing...');
        processImageData();
        
        // Also check every 2 seconds for 30 seconds as fallback
        let checkCount = 0;
        const maxChecks = 15;
        
        const intervalId = setInterval(() => {
            checkCount++;
            console.log('🔄 Periodic check', checkCount, '/', maxChecks);
            
            if (processingComplete || checkCount >= maxChecks) {
                clearInterval(intervalId);
                console.log('⏹️ Stopping periodic checks');
                return;
            }
            
            processImageData();
        }, 2000);
        
        console.log('✅ Tampermonkey integration script initialized');
        </script>
        
        <div id="status">
            <div style="text-align: center; padding: 20px;">
                <h3>🔄 Initializing...</h3>
            </div>
        </div>
        """
        
        st.components.v1.html(html_content, height=200)
        
        return True
    
    return False

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
    # Check if this is an API request first
    if handle_api_request():
        return
    
    # Initialize dark mode in session state if not exists
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🚛 Pup Trailer Detector</h1>', unsafe_allow_html=True)
    
    # Add clean subtitle
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; padding: 1.5rem; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
        <h3 style="color: #4a5568; font-weight: 500; margin-bottom: 1rem; font-size: 1.3rem;">
            Advanced AI-Powered Trailer Classification System
        </h3>
        <p style="color: #718096; font-size: 1rem; line-height: 1.6; margin: 0;">
            🤖 Powered by Deep Learning • 🎯 95%+ Accuracy • ⚡ Real-time Processing
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
    
    # Success message
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <div style="background: #f0fff4; color: #2f855a; padding: 1.5rem; border-radius: 12px; display: inline-block; border: 1px solid #38a169; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="margin: 0; font-weight: 600; font-size: 1.1rem;">✅ Model loaded successfully! Ready for predictions</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar with better organization
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h4>📊 Control Panel</h4>
        <p style="margin: 0; color: #718096; font-size: 0.9rem;">AI Dashboard & Settings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dark mode toggle
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h4>🌙 Theme Settings</h4>
    </div>
    """, unsafe_allow_html=True)
    
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, help="Switch between light and dark themes")
    
    # Update session state and apply CSS
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    # Apply the appropriate CSS based on dark mode setting
    st.markdown(get_custom_css(dark_mode), unsafe_allow_html=True)
    
    # Model source info in sidebar
    st.sidebar.markdown(f"""
    <div class="sidebar-section">
        <h4>🤗 Model Source</h4>
        <a href="https://huggingface.co/{MODEL_REPO_ID}" target="_blank" style="color: #3182ce; text-decoration: none; font-weight: 500;">
            {MODEL_REPO_ID}
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Optional debug info (collapsed by default)
    if st.sidebar.checkbox("🔧 Debug Info", value=False, help="Show technical debugging information"):
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <h4>🔧 Debug Information</h4>
            <div style="font-size: 0.9rem; color: #4a5568; line-height: 1.6;">
                <div style="margin-bottom: 0.5rem;"><strong>Repository:</strong> {MODEL_REPO_ID}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Filename:</strong> {MODEL_FILENAME}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Token:</strong> {"Available" if HF_TOKEN else "Missing"}</div>
                <div><strong>Status:</strong> <span style="color: #38a169;">Connected</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced model info in sidebar
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h4>🎯 Model Details</h4>
        <div style="color: #4a5568; font-size: 0.9rem; line-height: 1.6;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Architecture:</span>
                <span style="color: #3182ce; font-weight: 500;">ResNet50V2</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Classes:</span>
                <span style="color: #3182ce; font-weight: 500;">PUP, Non-PUP</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Input Size:</span>
                <span style="color: #3182ce; font-weight: 500;">224x224</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Status:</span>
                <span style="color: #38a169; font-weight: 500;">✓ Ready</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced statistics in sidebar
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.sidebar.markdown("""
        <div style="background: rgba(30, 30, 45, 0.6); color: #e8e8e8; padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; border: 1px solid rgba(33, 150, 243, 0.3); backdrop-filter: blur(10px);">
            <h4 style="color: #64b5f6; margin-bottom: 1rem; font-weight: 600;">� Performance</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                <div style="background: rgba(76, 175, 80, 0.1); padding: 0.75rem; border-radius: 10px; text-align: center;">
                    <div style="color: #81c784; font-weight: bold; font-size: 1.1rem;">95.2%</div>
                    <div style="color: #b0b0b0; font-size: 0.8rem;">Accuracy</div>
                </div>
                <div style="background: rgba(103, 126, 234, 0.1); padding: 0.75rem; border-radius: 10px; text-align: center;">
                    <div style="color: #667eea; font-weight: bold; font-size: 1.1rem;">0.94</div>
                    <div style="color: #b0b0b0; font-size: 0.8rem;">F1-Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        history = st.session_state.prediction_history
        pup_count = sum(1 for p in history if p['result']['is_pup'])
        total_predictions = len(history)
        
        st.sidebar.markdown(f"""
        <div style="background: rgba(30, 30, 45, 0.6); color: #e8e8e8; padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; border: 1px solid rgba(118, 75, 162, 0.3); backdrop-filter: blur(10px);">
            <h4 style="color: #ba68c8; margin-bottom: 1rem; font-weight: 600;">� Session Stats</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.75rem;">
                <div style="background: rgba(33, 150, 243, 0.1); padding: 0.75rem; border-radius: 10px; text-align: center;">
                    <div style="color: #64b5f6; font-weight: bold; font-size: 1.2rem;">{total_predictions}</div>
                    <div style="color: #b0b0b0; font-size: 0.8rem;">Total</div>
                </div>
                <div style="background: rgba(76, 175, 80, 0.1); padding: 0.75rem; border-radius: 10px; text-align: center;">
                    <div style="color: #81c784; font-weight: bold; font-size: 1.2rem;">{pup_count}</div>
                    <div style="color: #b0b0b0; font-size: 0.8rem;">PUP</div>
                </div>
            </div>
            <div style="background: rgba(244, 67, 54, 0.1); padding: 0.75rem; border-radius: 10px; text-align: center;">
                <div style="color: #e57373; font-weight: bold; font-size: 1.2rem;">{total_predictions - pup_count}</div>
                <div style="color: #b0b0b0; font-size: 0.8rem;">Non-PUP</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if total_predictions > 0:
            avg_confidence = np.mean([p['result']['confidence'] for p in history])
            st.sidebar.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; text-align: center; box-shadow: 0 4px 20px rgba(103, 126, 234, 0.3); backdrop-filter: blur(10px);">
                <h4 style="margin: 0; font-size: 1.8rem; font-weight: 600;">{avg_confidence * 100:.1f}%</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">Average Confidence</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Export statistics button
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        if st.sidebar.button("📊 Export Statistics", help="Export prediction statistics as JSON"):
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
        
        if st.sidebar.button("🧹 Clear History", help="Clear all prediction history"):
            st.session_state.prediction_history = []
            st.sidebar.success("History cleared!")
            st.experimental_rerun()
    else:
        st.sidebar.markdown("""
        <div style="background: #ffffff; color: #2d3748; padding: 1.5rem; border-radius: 15px; text-align: center; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-size: 0.9rem;">📝 No predictions yet</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;">Upload an image to get started</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced main content with clean tabs
    st.markdown("""
    <div style="margin: 2rem 0; text-align: center;">
        <h2 style="color: #2d3748; font-weight: 600; font-size: 1.8rem; margin-bottom: 2rem;">Choose Your Prediction Method</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Upload Image", "🌐 URL Prediction", "📊 History", "ℹ️ About"])
    
    with tab1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #3182ce; font-weight: 600; font-size: 1.5rem;">📷 Upload Image for Analysis</h3>
            <p style="color: #4a5568; font-size: 1rem;">Select an image from your device to detect pup trailers</p>
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
                    <h4 style="color: #38a169; font-weight: 600;">📸 Uploaded Image</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Image info with clean theme
                st.markdown(f"""
                <div class="sidebar-section">
                    <h4>📋 Image Details</h4>
                    <div style="color: #4a5568; line-height: 1.6; font-size: 0.9rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>📐 Dimensions:</span>
                            <span style="color: #3182ce; font-weight: 500;">{image.size[0]} × {image.size[1]} px</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>🎨 Format:</span>
                            <span style="color: #38a169; font-weight: 500;">{image.format}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>🌈 Mode:</span>
                            <span style="color: #ed8936; font-weight: 500;">{image.mode}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>💾 File Size:</span>
                            <span style="color: #9f7aea; font-weight: 500;">{len(uploaded_file.getvalue())/1024:.1f} KB</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <h4 style="color: #ed8936; font-weight: 600;">🔍 Prediction Result</h4>
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
            <h3 style="color: #3182ce; font-weight: 600; font-size: 1.5rem;">🌐 Predict from URL</h3>
            <p style="color: #4a5568; font-size: 1rem;">Enter an image URL to analyze remotely hosted images</p>
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
                            <h4 style="color: #00ff88; text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);">📸 Downloaded Image</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(image, caption="🌐 Downloaded from URL", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <h4 style="color: #ff00ff; text-shadow: 0 0 10px rgba(255, 0, 255, 0.5);">🔍 Prediction Result</h4>
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
            <h3 style="color: #3182ce; font-weight: 600; font-size: 1.5rem;">📊 Prediction History</h3>
            <p style="color: #4a5568; font-size: 1rem;">View and manage your prediction history</p>
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
            <h3 style="color: #3182ce; font-weight: 600; font-size: 1.5rem;">ℹ️ About This Application</h3>
            <p style="color: #4a5568; font-size: 1rem;">Learn more about the Pup Trailer Detector</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced about section with modern theme
        st.markdown("""
        <div style="background: rgba(30, 30, 45, 0.6); color: #e8e8e8; padding: 2rem; border-radius: 20px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px);">
            <h4 style="color: #667eea; margin-bottom: 1rem; font-weight: 600; font-size: 1.3rem;">🎯 What is a Pup Trailer?</h4>
            <p style="color: #b0b0b0; line-height: 1.8; font-size: 1rem;">
                A pup trailer is a short semi-trailer that is typically pulled behind a truck or another trailer. 
                It's commonly used in logistics to increase cargo capacity while maintaining maneuverability.
            </p>
            
            <h4 style="color: #81c784; margin: 1.5rem 0 1rem 0; font-weight: 600; font-size: 1.3rem;">🤖 How It Works</h4>
            <p style="color: #b0b0b0; line-height: 1.8; font-size: 1rem;">
                Our AI model uses advanced computer vision techniques based on ResNet50V2 architecture to analyze 
                images and classify whether they contain pup trailers or not.
            </p>
            
            <h4 style="color: #ba68c8; margin: 1.5rem 0 1rem 0; font-weight: 600; font-size: 1.3rem;">🔧 Technical Details</h4>
            <div style="color: #b0b0b0; line-height: 1.8; font-size: 1rem;">
                <div style="margin-bottom: 0.5rem;"><strong style="color: #667eea;">Model:</strong> ResNet50V2 with custom classification head</div>
                <div style="margin-bottom: 0.5rem;"><strong style="color: #667eea;">Training:</strong> 2-phase training strategy for optimal performance</div>
                <div style="margin-bottom: 0.5rem;"><strong style="color: #667eea;">Accuracy:</strong> 95%+ on validation data</div>
                <div style="margin-bottom: 0.5rem;"><strong style="color: #667eea;">Input:</strong> 224×224 pixel RGB images</div>
                <div><strong style="color: #667eea;">Deployment:</strong> Hugging Face Hub integration</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 2rem; border-radius: 20px; margin-bottom: 1rem; box-shadow: 0 8px 32px rgba(103, 126, 234, 0.3); backdrop-filter: blur(10px);">
                <h4 style="margin: 0 0 1rem 0; font-weight: 600; font-size: 1.2rem;">🚀 Features</h4>
                <div style="line-height: 1.8; font-size: 1rem;">
                    <div style="margin-bottom: 0.5rem;">• Real-time image analysis</div>
                    <div style="margin-bottom: 0.5rem;">• High accuracy predictions</div>
                    <div style="margin-bottom: 0.5rem;">• URL-based image processing</div>
                    <div style="margin-bottom: 0.5rem;">• Prediction history tracking</div>
                    <div>• Statistics and analytics</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #764ba2, #667eea); color: white; padding: 2rem; border-radius: 20px; margin-bottom: 1rem; box-shadow: 0 8px 32px rgba(118, 75, 162, 0.3); backdrop-filter: blur(10px);">
                <h4 style="margin: 0 0 1rem 0; font-weight: 600; font-size: 1.2rem;">💡 Tips for Best Results</h4>
                <div style="line-height: 1.8; font-size: 1rem;">
                    <div style="margin-bottom: 0.5rem;">• Use clear, well-lit images</div>
                    <div style="margin-bottom: 0.5rem;">• Include the entire trailer in view</div>
                    <div style="margin-bottom: 0.5rem;">• Avoid heavily distorted images</div>
                    <div style="margin-bottom: 0.5rem;">• Higher resolution images work better</div>
                    <div>• Try different angles if unsure</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced footer
    footer_class = "dark-footer" if dark_mode else "clean-footer"
    st.markdown(f"""
    <div class="{footer_class}">
        <h3>🚛 Pup Trailer Detector</h3>
        <p style="margin: 0 0 1rem 0;">
            Built with ❤️ using Streamlit & TensorFlow
        </p>
        <div style="font-size: 0.9rem;">
            <span>🤗 Hugging Face Hub</span> • 
            <span>🧠 ResNet50V2</span> • 
            <span>⚡ Real-time AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
