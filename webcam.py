from ultralytics import YOLO
import cv2

model = YOLO("yolov9c.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, conf=0.5, iou=0.5, imgsz=960, agnostic_nms=True)

    annotated = results[0].plot()
    cv2.imshow("YOLOv8 Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



