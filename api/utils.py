import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path


def get_model():
    """Load YOLO model"""
    try:
        model = YOLO("yolov9c.pt")  # or yolov8s.pt, yolov8m.pt based on your needs
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def annotate_image_and_extract(model, image, confidence_threshold=0.5):
    """Annotate image and extract detection data"""
    try:
        # Run inference
        results = model(image, conf=confidence_threshold)
        
        # Extract detections
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]
                    
                    # Filter by confidence
                    if conf >= confidence_threshold:
                        detections.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': {
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2),
                                'y2': float(y2)
                            }
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_image, label,
                                  (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_image, detections
        
    except Exception as e:
        print(f"Error in annotation: {e}")
        return image, []

def bgr_to_jpeg_bytes(image):
    """Convert BGR image to JPEG bytes"""
    try:
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            return encoded_image.tobytes()
        else:
            raise Exception("Failed to encode image")
    except Exception as e:
        print(f"Error converting image to bytes: {e}")
        return None
    
def save_output_image(annotated_img, filename):
    output_dir = Path("/Users/muskanbansal/Downloads/object_detection/api/outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"detected_{filename}"
    cv2.imwrite(str(output_path), annotated_img)
    return str(output_path)    