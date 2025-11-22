from flask import Flask, render_template, request, redirect, send_file, url_for
import cv2
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure upload and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLO model
net = cv2.dnn.readNet(r"C:\vehicle detection and category wise counting\yolov3.weights",  r"C:\vehicle detection and category wise counting\yolov3.cfg")
with open(r"C:\vehicle detection and category wise counting\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Vehicle detection function
def detect_vehicles(video_path):
    # Read the input video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video path and writer
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.avi')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Initialize vehicle count dictionary
    vehicle_counts = {"car": 0, "truck": 0, "bus": 0, "motorbike": 0}
    green_line_y = int(height * 0.5)  # Adjust as needed for line position

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to blob and pass through YOLO model
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = classes[class_ids[i]]
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if y < green_line_y < y + h:  # Vehicle crosses line
                vehicle_counts[label] += 1

        # Draw line and counts on the frame
        cv2.line(frame, (0, green_line_y), (width, green_line_y), (0, 255, 0), 2)
        y_offset = 30
        for vehicle, count in vehicle_counts.items():
            cv2.putText(frame, f"{vehicle.capitalize()} Count: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 30

        # Write the frame to output video
        out.write(frame)

    cap.release()
    out.release()
    return output_video_path  # Return path to the saved output video

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            output_path = detect_vehicles(file_path)
            return send_file(output_path, as_attachment=True)  # Download processed video
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
