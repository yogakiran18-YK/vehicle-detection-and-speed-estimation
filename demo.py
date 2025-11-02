import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np


VIDEO_PATH = "C:\\Users\\cvaru\\OneDrive\\Desktop\\WhatsApp Video 2025-11-02 at 20.11.03_357114e8.mp4"   
OUTPUT_PATH = "output_speed.avi" 
MODEL_NAME = "yolov8s.pt"         


METERS_PER_PIXEL = 0.05  


model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))


tracker = sv.ByteTrack()  
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


last_positions = {}

print("🚗 Processing video... Please wait...")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    
    labels = [model.model.names[class_id] for class_id in detections.class_id]
    vehicle_indices = [
        i for i, label in enumerate(labels)
        if label in ["car", "motorbike", "bus", "truck"]
    ]
    detections = detections[vehicle_indices]

   
    detections = tracker.update_with_detections(detections)

   
    speeds = {}
    for i, tracker_id in enumerate(detections.tracker_id):
        box = detections.xyxy[i]
        x_center = int((box[0] + box[2]) / 2)
        y_center = int((box[1] + box[3]) / 2)

        if tracker_id in last_positions:
            prev_x, prev_y = last_positions[tracker_id]
            pixel_distance = np.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
         
            speed_mps = (pixel_distance * METERS_PER_PIXEL) * fps
            speed_kmph = speed_mps * 3.6
            speeds[tracker_id] = round(speed_kmph, 1)
        else:
            speeds[tracker_id] = 0.0

        last_positions[tracker_id] = (x_center, y_center)

    
    labels = [
        f"ID:{tracker_id} Speed:{speeds[tracker_id]} km/h"
        for tracker_id in detections.tracker_id
    ]
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    out.write(annotated)
    cv2.imshow("Vehicle Speed Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Processing complete! Output saved as:", OUTPUT_PATH)

