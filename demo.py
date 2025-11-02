import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

# === Step 1: Configuration ===
VIDEO_PATH = "C:\\Users\\cvaru\\OneDrive\\Desktop\\WhatsApp Video 2025-11-02 at 20.11.03_357114e8.mp4"   # <-- Change this to your video path
OUTPUT_PATH = "output_speed.avi"  # Output video file
MODEL_NAME = "yolov8s.pt"         # Pretrained YOLOv8 model

# Scale conversion — tune this to real-world scale (meters per pixel)
# Example: if 1 pixel = 0.05 meters, adjust based on camera view
METERS_PER_PIXEL = 0.05  

# === Step 2: Load YOLO model and video ===
model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# === Step 3: Initialize Supervision tracker and annotator ===
tracker = sv.ByteTrack()  # Built-in tracker for consistent IDs
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# To store previous positions of each object
last_positions = {}

print("🚗 Processing video... Please wait...")

# === Step 4: Process video frame by frame ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter only vehicles
    labels = [model.model.names[class_id] for class_id in detections.class_id]
    vehicle_indices = [
        i for i, label in enumerate(labels)
        if label in ["car", "motorbike", "bus", "truck"]
    ]
    detections = detections[vehicle_indices]

    # Track objects
    detections = tracker.update_with_detections(detections)

    # Compute speed for each object
    speeds = {}
    for i, tracker_id in enumerate(detections.tracker_id):
        box = detections.xyxy[i]
        x_center = int((box[0] + box[2]) / 2)
        y_center = int((box[1] + box[3]) / 2)

        if tracker_id in last_positions:
            prev_x, prev_y = last_positions[tracker_id]
            pixel_distance = np.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
            # Distance (in meters) per second
            speed_mps = (pixel_distance * METERS_PER_PIXEL) * fps
            speed_kmph = speed_mps * 3.6
            speeds[tracker_id] = round(speed_kmph, 1)
        else:
            speeds[tracker_id] = 0.0

        last_positions[tracker_id] = (x_center, y_center)

    # Annotate frames
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

# === Step 5: Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Processing complete! Output saved as:", OUTPUT_PATH)
