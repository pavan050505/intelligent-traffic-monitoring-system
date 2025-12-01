import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Video
cap = cv2.VideoCapture(r"C:\vehicle detection and category wise counting\video (1).mp4")

# Counting Line
count_line_position = 550
offset = 10

# Counters
counter = {"car": 0, "truck": 0, "motorbike": 0}
vehicle_centers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Preprocessing for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.forward(net.getUnconnectedOutLayersNames())

    detections = []
    for output in output_layers:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["car", "truck", "motorbike"]:
                cx, cy, w, h = detection[0:4] * np.array([width, height, width, height])
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                detections.append([x, y, int(w), int(h), classes[class_id]])

    for x, y, w, h, label in detections:
        x_center = int((x + x + w) / 2)
        y_center = int((y + y + h) / 2)

        # Draw bounding box & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.circle(frame, (x_center, y_center), 4, (0,0,255), -1)

        # Count only if crossing the line
        if y_center < count_line_position + offset and y_center > count_line_position - offset:
            counter[label] += 1

    # Counting output
    cv2.putText(frame, f"Car: {counter['car']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Truck: {counter['truck']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Bike: {counter['motorbike']}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Draw line
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255,127,0), 3)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
