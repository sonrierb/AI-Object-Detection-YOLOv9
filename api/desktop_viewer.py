import cv2
import requests
import tempfile
import os
import sys
from pathlib import Path

def view_detection_result(task_id: str):
    """Open desktop window to view detection results"""
    
    # Get video status
    try:
        status_response = requests.get(f"http://127.0.0.1:8000/video_status/{task_id}")
        if status_response.status_code != 200:
            print("Error: Could not get task status")
            return
        
        status_data = status_response.json()
        
        if status_data.get("status") != "done":
            print("Video is still processing...")
            return
            
        output_path = status_data.get("output_path")
        if not output_path or not os.path.exists(output_path):
            print("Output file not found")
            return
        
        # Open video in window
        cap = cv2.VideoCapture(output_path)
        
        print("Press 'q' to quit, 'p' to pause")
        
        paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow(f"YOLOv8 Detection - {task_id}", frame)
            
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):  # restart
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python desktop_viewer.py <task_id>")
        sys.exit(1)
    
    task_id = sys.argv[1]
    view_detection_result(task_id)