import os
import io
import cv2
import tempfile
import numpy as np
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid
import asyncio
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your utilities
from utils import get_model, annotate_image_and_extract, bgr_to_jpeg_bytes, save_output_image

# ------------------------------------------------------------
# Initialize FastAPI app
# ------------------------------------------------------------
app = FastAPI(
    title="YOLOv9 Detection API",
    description="Real-time object detection for images and videos",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ------------------------------------------------------------
# Load YOLO model
# ------------------------------------------------------------
try:
    model = get_model()
    if model is None:
        logger.error("Failed to load YOLO model")
    else:
        logger.info("YOLOv9 model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# ------------------------------------------------------------
# Global state for task management
# ------------------------------------------------------------
TASKS: dict[str, dict] = {}

# ------------------------------------------------------------
# IMAGE DETECTION + SAVE TO OUTPUT FOLDER
# ------------------------------------------------------------
@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=1.0)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    annotated, detections = annotate_image_and_extract(model, img, confidence)
    img_bytes = bgr_to_jpeg_bytes(annotated)

    # SAVE IMAGE TO outputs/
    saved_path = save_output_image(annotated, file.filename)

    meta = {
        "filename": file.filename,
        "saved_path": saved_path,
        "detections": detections,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/jpeg",
        headers={"X-Meta": json.dumps(meta)}
    )

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    try:
        temp_dir = tempfile.gettempdir()
        file_ext = Path(upload_file.filename).suffix if upload_file.filename else '.tmp'
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
        
        with open(file_path, "wb") as buffer:
            content = upload_file.file.read()
            buffer.write(content)
        
        logger.info(f"File saved temporarily: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise

def process_video_detection(task_id: str, input_path: str, confidence: float = 0.5):
    """Process video detection in background"""
    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["start_time"] = datetime.now().isoformat()
        
        # Output path
        output_filename = f"detected_{task_id}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            TASKS[task_id]["status"] = "error"
            TASKS[task_id]["error"] = "Cannot open video file"
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        detection_data = []
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            try:
                annotated_frame, detections = annotate_image_and_extract(model, frame, confidence)
                
                # Filter detections by confidence
                filtered_detections = [
                    det for det in detections 
                    if det.get('confidence', 0) >= confidence
                ]
                
                # Save frame detection data
                detection_data.append({
                    "frame": frame_count,
                    "detections": filtered_detections,
                    "detections_count": len(filtered_detections)
                })
                
                # Write annotated frame
                out.write(annotated_frame)
                
                frame_count += 1
                
                # Update progress
                if total_frames > 0:
                    progress = min(100, int((frame_count / total_frames) * 100))
                    TASKS[task_id]["progress"] = progress
                    logger.info(f"Frame {frame_count}/{total_frames} - Progress: {progress}%")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                # Continue with next frame
                continue
        
        # Release resources
        cap.release()
        out.release()
        
        # Update task status
        TASKS[task_id].update({
            "status": "done",
            "progress": 100,
            "output_path": str(output_path),
            "output_url": f"/outputs/{output_filename}",
            "end_time": datetime.now().isoformat(),
            "total_frames": frame_count,
            "detection_summary": {
                "total_frames_processed": frame_count,
                "total_detections": sum(len(frame_det["detections"]) for frame_det in detection_data),
                "detection_data": detection_data[:100]  # First 100 frames for preview
            }
        })
        
        logger.info(f"Video processing completed: {output_path}")
        
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)
            
    except Exception as e:
        logger.error(f"Error in video processing: {e}")
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)
        
        # Cleanup on error
        if os.path.exists(input_path):
            os.remove(input_path)

# ------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "YOLOv8 Detection API", "status": "running", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    model_status = "loaded" if model is not None else "failed"
    return {"status": "healthy", "model": model_status}

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=1.0)
):
    """Detect objects in image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Processing image: {file.filename}, Size: {img.shape}")
        
        # Perform detection
        annotated, detections = annotate_image_and_extract(model, img, confidence)
        
        if annotated is None:
            raise HTTPException(status_code=500, detail="Image processing failed")
        
        # Filter by confidence
        filtered_detections = [
            det for det in detections 
            if det.get('confidence', 0) >= confidence
        ]
        
        # Convert to bytes
        img_bytes = bgr_to_jpeg_bytes(annotated)
        if img_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        # Prepare metadata
        meta = {
            "filename": file.filename,
            "detections_count": len(filtered_detections),
            "detections": filtered_detections,
            "confidence_threshold": confidence,
            "image_size": f"{img.shape[1]}x{img.shape[0]}",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Detection completed: {len(filtered_detections)} objects found")
        
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg",
            headers={"X-Meta": json.dumps(meta)}
        )
        
    except Exception as e:
        logger.error(f"Image detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect_video")
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=1.0)
):
    """Start video detection task"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded file
        input_path = save_uploaded_file(file)
        
        # Create task
        task_id = uuid.uuid4().hex
        TASKS[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "filename": file.filename,
            "confidence": confidence,
            "created_at": datetime.now().isoformat()
        }
        
        # Start background task
        background_tasks.add_task(
            process_video_detection, 
            task_id, 
            input_path, 
            confidence
        )
        
        logger.info(f"Video task started: {task_id} for {file.filename}")
        
        return JSONResponse({
            "task_id": task_id,
            "status": "queued",
            "message": "Video processing started",
            "confidence_threshold": confidence
        })
        
    except Exception as e:
        logger.error(f"Video upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.get("/video_status/{task_id}")
async def video_status(task_id: str):
    """Get video processing status"""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return JSONResponse(task)

@app.get("/video_result/{task_id}")
async def video_result(task_id: str):
    """Download processed video"""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] != "done":
        raise HTTPException(status_code=400, detail="Video processing not complete")
    
    output_path = task.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    filename = f"detected_{task['filename']}"
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=filename
    )

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return JSONResponse({
        "total_tasks": len(TASKS),
        "tasks": list(TASKS.keys())
    })

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")