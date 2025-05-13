import torch
import cv2
import numpy as np

# Load YOLOv5x model (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', trust_repo=True)
model.conf = 0.5  # Confidence threshold (higher = more accurate, but fewer detections)
model.classes = None  # Detect all classes

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame (optional for speed)
    resized_frame = cv2.resize(frame, (640, 480))

    # Inference
    results = model(resized_frame)

    # Extract predictions
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    for i in range(len(labels)):
        row = cords[i]
        if row[4] >= model.conf:
            x1, y1, x2, y2 = int(row[0]*frame.shape[1]), int(row[1]*frame.shape[0]), int(row[2]*frame.shape[1]), int(row[3]*frame.shape[0])
            label = model.names[int(labels[i])]
            confidence = row[4].item()

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display
    cv2.imshow('YOLOv5x Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
