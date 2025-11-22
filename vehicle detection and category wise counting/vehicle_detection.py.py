import cv2
import numpy as np
def center_handle(x, y, w, h):
    x_center = int((x + x + w) / 2)
    y_center = int((y + y + h) / 2)
    return x_center, y_center
cap = cv2.VideoCapture("C:\vehicle detection and category wise counting\video (1).mp4")  
count_line_position = 550
offset = 6  
min_width_react = 80 
min_height_react = 80 
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

current_vehicles = []
counter = {"Car": 0, "Truck": 0, "Bike": 0}
while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    current_vehicles = []
    for c in counterShape:
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)

        if not validate_counter:
            continue

        aspect_ratio = float(w) / h
        vehicle_type = "Unknown"

        if 1.0 < aspect_ratio < 2.0 and w > 20 and h > 20 and h / w < 1.5:
            vehicle_type = "Car"
        elif aspect_ratio >= 2.0 and w > 20 and h >20 :
            vehicle_type = "Truck"
        elif 0.0 <= aspect_ratio <= 1.0 and w > 20 and h > 10:
            vehicle_type = "Bike"

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = center_handle(x, y, w, h)
        current_vehicles.append({"center": center, "type": vehicle_type, "counted": False})

    for vehicle in current_vehicles:
        x, y = vehicle["center"]
        cv2.circle(frame1, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(frame1, f"Type: {vehicle['type']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if not vehicle["counted"] and y < (count_line_position + offset) and y > (count_line_position - offset):
            counter[vehicle["type"]] += 1
            vehicle["counted"] = True

    cv2.putText(frame1, "Car Count:" + str(counter["Car"]), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(frame1, "Truck Count:" + str(counter["Truck"]), (450, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(frame1, "Bike Count:" + str(counter["Bike"]), (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv2.imshow('video original', frame1)
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
     