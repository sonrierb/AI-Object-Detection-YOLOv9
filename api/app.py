import streamlit as st
import requests
from PIL import Image
import io
import tempfile
import time
import json
import subprocess
import sys
import os

# Page configuration
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .detection-container {
        border: 2px solid #e6e6e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔍 AI Object Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>Powered by YOLOv8</strong> - Real-time object detection for images and videos. 
    Upload your media files and get instant detection results with bounding boxes and confidence scores.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150/1f77b4/ffffff?text=AI", width=150)
    st.title("Settings")
    st.markdown("---")
    
    # Model configuration
    st.subheader("Model Configuration")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5,
        help="Adjust the minimum confidence level for detections"
    )
    
    st.markdown("---")
    st.subheader("About")
    st.info("""
    This system uses YOLOv8 for object detection:
    - **Images**: Instant processing
    - **Videos**: Real-time processing with live preview
    - **Output**: Annotated files saved locally
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">📁 Upload Media</div>', unsafe_allow_html=True)
    
    # File type selection
    file_type = st.radio(
        "Select input type:",
        ["Image", "Video"],
        horizontal=True,
        help="Choose between image or video detection"
    )
    
    # File upload
    uploaded_file = None
    if file_type == "Image":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a video", 
            type=["mp4", "mov", "avi", "mkv"],
            help="Supported formats: MP4, MOV, AVI, MKV"
        )

    # File preview
    if uploaded_file:
        st.markdown("###Preview")
        if file_type == "Image":
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            # Save uploaded video temporarily for preview
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.video(tmp_file.name)

with col2:
    st.markdown('<div class="sub-header">🎯 Detection Results</div>', unsafe_allow_html=True)
    
# app.py - Only the main detection part (rest remains same)

# In the detection section, replace with this:
    if uploaded_file and st.button("🚀 Start Detection", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Processing... Please wait"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                if file_type == "Image":
                    # Image detection
                    response = requests.post(
                        "http://127.0.0.1:8000/detect", 
                        files=files,
                        params={"confidence": confidence_threshold},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        # Display annotated image
                        annotated_image = Image.open(io.BytesIO(response.content))
                        
                        # Get metadata
                        meta = response.headers.get("X-Meta", "{}")
                        try:
                            meta = json.loads(meta)
                        except:
                            meta = {"detections": "Available"}
                        
                        st.success("✅ Detection Complete!")
                        st.image(annotated_image, caption="Annotated Image", use_column_width=True)
                        
                        # Display metadata
                        with st.expander("📊 Detection Details"):
                            st.json(meta)
                        
                        # Download button
                        img_bytes = io.BytesIO()
                        annotated_image.save(img_bytes, format="JPEG")
                        st.download_button(
                            label="📥 Download Result",
                            data=img_bytes.getvalue(),
                            file_name="detected_image.jpg",
                            mime="image/jpeg"
                        )
                        
                    else:
                        st.error(f"❌ Detection failed: {response.status_code}")
                        st.text(f"Error: {response.text}")
                
                else:
                    # Video detection
                    response = requests.post(
                        "http://127.0.0.1:8000/detect_video", 
                        files=files,
                        params={"confidence": confidence_threshold},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        task_data = response.json()
                        task_id = task_data["task_id"]
                        
                        st.success("🎬 Video processing started! Tracking progress...")
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        info_text = st.empty()
                        
                        # Poll for completion
                        max_wait_time = 300  # 5 minutes
                        start_time = time.time()
                        
                        while time.time() - start_time < max_wait_time:
                            try:
                                status_response = requests.get(f"http://127.0.0.1:8000/video_status/{task_id}")
                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    progress = status_data.get("progress", 0)
                                    current_status = status_data.get("status", "")
                                    
                                    progress_bar.progress(progress)
                                    status_text.text(f"Status: {current_status} - {progress}%")
                                    
                                    if current_status == "done":
                                        st.success("✅ Video processing completed!")
                                        
                                        # Download link
                                        download_url = f"http://127.0.0.1:8000/video_result/{task_id}"
                                        st.markdown(f"""
                                        <div style="background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745;">
                                            <h4>🎉 Processing Complete!</h4>
                                            <p>Video has been processed and saved.</p>
                                            <a href="{download_url}" target="_blank">
                                                <button style="background-color: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                                                    📥 Download Processed Video
                                                </button>
                                            </a>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Show detection summary
                                        if "detection_summary" in status_data:
                                            summary = status_data["detection_summary"]
                                            st.info(f"Processed {summary['total_frames_processed']} frames with {summary['total_detections']} total detections")
                                        
                                        break
                                    elif current_status == "error":
                                        error_msg = status_data.get("error", "Unknown error")
                                        st.error(f"❌ Video processing failed: {error_msg}")
                                        break
                                    elif current_status == "processing":
                                        info_text.info("⏳ Processing video frames... This may take a few minutes for longer videos.")
                                
                            except requests.exceptions.RequestException as e:
                                st.warning(f"⚠️ Connection issue: {e}")
                            
                            time.sleep(2)
                        
                        else:
                            st.warning("⏰ Processing time exceeded. Please check back later.")
                            
                    else:
                        st.error(f"❌ Failed to start video processing: {response.status_code}")
                        st.text(f"Error: {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"🔌 Connection error: Please make sure the FastAPI server is running on http://127.0.0.1:8000")
                st.code("uvicorn api:app --reload", language="bash")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

    else:
        # Placeholder when no file is uploaded
        st.info("📤 Please upload a file to start detection")
        st.image("https://via.placeholder.com/400x300/e6e6e6/666666?text=Upload+File+to+See+Results", 
                use_column_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "AI Object Detection System • Powered by YOLOv8 • Built with Streamlit & FastAPI"
    "</div>", 
    unsafe_allow_html=True
)