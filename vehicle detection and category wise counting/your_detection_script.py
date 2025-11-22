import cv2
import numpy as np

# Load YOLO pre-trained weights and config file
net = cv2.dnn.readNet(r"C:\vehicle detection and category wise counting\yolov3 (2).weights", r"C:\vehicle detection and category wise counting\yolov3 (2).cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names (COCO dataset contains vehicle classes like car, truck, bus)
with open(r"C:\vehicle detection and category wise counting\coco (2).names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define green line position (a horizontal line across the frame)
green_line_y = 500  # Adjust this value as per your frame's height

# Function to process and count vehicles
def detect_vehicles(image, vehicle_crossed, vehicle_counts):
    # Get image dimensions
    height, width = image.shape[:2]

    # Convert image to blob (input for the neural network)
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Run forward pass through the network
    layer_outputs = net.forward(output_layers)

    # Variables to hold detected bounding boxes, confidences, and class ids
    boxes = []
    confidences = []
    class_ids = []

    # Iterate over the outputs and process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]  # We are interested in class scores
            class_id = np.argmax(scores)  # Get the class with the highest score
            confidence = scores[class_id]  # Confidence of the detected object
            if confidence > 0.5:  # Confidence threshold
                # Get the bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression (NMS) to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Variables to hold the count of vehicles and IDs of vehicles that crossed
    vehicle_crossed_this_frame = set()

    if len(indexes) > 0:
        for i in indexes.flatten():
            if classes[class_ids[i]] in ["car", "truck", "bus", "motorbike"]:
                x, y, w, h = boxes[i]
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, classes[class_ids[i]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Check if the vehicle's bounding box crosses the green line
                if (y + h > green_line_y > y) and i not in vehicle_crossed:
                    vehicle_crossed.add(i)  # Mark the vehicle as crossed this frame
                    vehicle_crossed_this_frame.add(i)  # Only count once in this frame

                    # Update the count for the vehicle's category
                    vehicle_counts[classes[class_ids[i]]] += 1

    # Return the processed frame with category counts and updated crossing info
    return image, vehicle_counts, vehicle_crossed


# Video processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    vehicle_crossed = set()  # Set to track which vehicles have crossed previously
    vehicle_counts = {"car": 0, "truck": 0, "bus": 0, "motorbike": 0}  # Dictionary to store counts for each category

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the green line on the frame
        cv2.line(frame, (0, green_line_y), (frame.shape[1], green_line_y), (0, 255, 0), 2)

        # Detect vehicles in each frame and count by category
        processed_frame, vehicle_counts, vehicle_crossed = detect_vehicles(frame, vehicle_crossed, vehicle_counts)

        # Display the processed frame with detected vehicles and category-wise count
        cv2.putText(processed_frame, f"Car Count: {vehicle_counts['car']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Truck Count: {vehicle_counts['truck']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Bus Count: {vehicle_counts['bus']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Motorbike Count: {vehicle_counts['motorbike']}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Vehicle Detection", processed_frame)

        # Press 'q' to exit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ask the user for a video path and process it
if __name__ == "__main__":
    video_path = input("Enter the path of the video file: ")
    process_video(video_path)

