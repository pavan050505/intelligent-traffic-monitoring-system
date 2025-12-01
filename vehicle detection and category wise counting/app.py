from flask import Flask, render_template, request, redirect, url_for, Response
import cv2, os, numpy as np, threading, torch
from ultralytics import YOLO

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
RESIZE_SCALE = 0.6        # slightly higher for better accuracy
METERS_PER_PIXEL = 0.05   # rough estimate for speed

processing_done = False
output_data = {}
latest_frame = None

# --------------- FLASK APP --------------
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------- YOLO MODEL ---------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov5n.pt")
model.to(DEVICE)

VALID_CLASSES = ["car", "truck", "bus", "motorbike"]


def normalize_class(cls_name: str) -> str:
    """Map YOLO class names to our 4 standard categories."""
    if cls_name.lower() in ["motorcycle", "motorbike"]:
        return "motorbike"
    return cls_name.lower()


# ------------- VEHICLE DETECTION ----------------
def detect_vehicles(video_path):
    global processing_done, output_data, latest_frame
    processing_done = False

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    output_path = os.path.join(OUTPUT_FOLDER, "output.mp4")
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # Zone rectangle (ROI) – adjust if needed
    zx1, zy1 = int(width * 0.15), int(height * 0.35)
    zx2, zy2 = int(width * 0.85), int(height * 0.9)

    frame_skip = max(1, int(fps / 15))  # a bit denser for accuracy

    # Tracking state
    tracks = {}       # id -> dict(cx, cy, label, last_seen, in_zone)
    prev_centers = {} # id -> (cx, cy) for speed
    speeds = {}       # id -> km/h
    next_track_id = 0
    frame_idx = 0
    MAX_MISSED = 10   # frames before track is dropped

    # Init global metrics
    output_data = {
        "output_path": output_path,
        "counts": {c: 0 for c in VALID_CLASSES},  # live in-zone counts per category
        "total": 0,
        "zone_count": 0,
        "avg_speed": 0,
        "risk_percent": 0,
        "density_label": "Low"
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Draw zone rectangle (only visual aid)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 255), 2)

        # Skip some frames for performance, still stream last processed
        run_detection = (frame_idx % frame_skip == 0)

        if run_detection:
            # -------- YOLO Inference ----------
            sw, sh = int(width * RESIZE_SCALE), int(height * RESIZE_SCALE)
            resized = cv2.resize(frame, (sw, sh))
            results = model(resized, verbose=False, conf=0.5, iou=0.5)[0]  # tighter for accuracy

            detections = []  # list of (label, x1, y1, x2, y2, cx, cy)

            for box in results.boxes:
                raw_cls = model.names[int(box.cls)]
                label = normalize_class(raw_cls)
                if label not in VALID_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Filter tiny boxes (likely noise)
                if (x2 - x1) * (y2 - y1) < 900:
                    continue

                # Scale back to original frame coordinates
                x1 = int(x1 / RESIZE_SCALE)
                y1 = int(y1 / RESIZE_SCALE)
                x2 = int(x2 / RESIZE_SCALE)
                y2 = int(y2 / RESIZE_SCALE)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append((label, x1, y1, x2, y2, cx, cy))

            # -------- Assign detections to tracks ----------
            for label, x1, y1, x2, y2, cx, cy in detections:
                assigned_id = None
                min_dist = 1e9

                # simple nearest-centroid matching
                for tid, t in tracks.items():
                    px, py = t["cx"], t["cy"]
                    dist = np.hypot(cx - px, cy - py)
                    if dist < 50 and dist < min_dist:
                        min_dist = dist
                        assigned_id = tid

                if assigned_id is None:
                    next_track_id += 1
                    assigned_id = next_track_id

                # speed estimation
                if assigned_id in prev_centers:
                    px, py = prev_centers[assigned_id]
                    dist_pixels = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    dist_meters = dist_pixels * METERS_PER_PIXEL
                    dt = frame_skip / float(fps)
                    speed_mps = dist_meters / dt if dt > 0 else 0
                    speeds[assigned_id] = round(speed_mps * 3.6, 1)  # km/h

                prev_centers[assigned_id] = (cx, cy)

                tracks[assigned_id] = {
                    "cx": cx,
                    "cy": cy,
                    "label": label,
                    "last_seen": frame_idx
                }

            # Remove stale tracks
            to_delete = [tid for tid, t in tracks.items()
                         if frame_idx - t["last_seen"] > MAX_MISSED]
            for tid in to_delete:
                tracks.pop(tid, None)
                prev_centers.pop(tid, None)
                speeds.pop(tid, None)

            # -------- Compute in-zone membership & live counts ----------
            live_counts = {c: 0 for c in VALID_CLASSES}
            zone_ids = []

            for tid, t in tracks.items():
                cx, cy, label = t["cx"], t["cy"], t["label"]
                inside = (zx1 < cx < zx2) and (zy1 < cy < zy2)
                tracks[tid]["in_zone"] = inside

                if inside:
                    live_counts[label] += 1
                    zone_ids.append(tid)

                # Draw box + label + speed ONLY (no red text overlays)
                # Find an approximate box around center? We have original x1,y1,x2,y2 in detections,
                # so reuse by recomputing bbox here would be complex. Instead, draw based on center.
                # Better: get bbox from detections again – simpler: use radius box.

            # We still need actual bboxes for drawing. Re-loop detections:
            for label, x1, y1, x2, y2, cx, cy in detections:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # show label + speed if available
                speed_str = ""
                # find which track this center belongs to (again nearest)
                best_id = None
                best_d = 1e9
                for tid, t in tracks.items():
                    d = np.hypot(cx - t["cx"], cy - t["cy"])
                    if d < best_d:
                        best_d = d
                        best_id = tid
                if best_id in speeds:
                    speed_str = f" {int(speeds[best_id])} km/h"

                cv2.putText(frame, f"{label}{speed_str}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ----- Metrics for dashboard -----
            zone_count = sum(live_counts.values())
            zone_speeds = [speeds[tid] for tid in tracks if tracks[tid].get("in_zone") and tid in speeds]
            avg_speed = int(sum(zone_speeds) / len(zone_speeds)) if zone_speeds else 0

            density_norm = min(zone_count / 20.0, 1.0)
            speed_norm = min(avg_speed / 80.0, 1.0)
            risk_percent = int((0.6 * density_norm + 0.4 * speed_norm) * 100)

            if zone_count < 5:
                density_label = "Low"
            elif zone_count < 12:
                density_label = "Medium"
            else:
                density_label = "High"

            output_data["counts"] = live_counts
            output_data["total"] = zone_count
            output_data["zone_count"] = zone_count
            output_data["avg_speed"] = avg_speed
            output_data["risk_percent"] = risk_percent
            output_data["density_label"] = density_label

        # Save & stream this (possibly annotated) frame
        out.write(frame)
        latest_frame = cv2.imencode('.jpg', frame)[1].tobytes()

    cap.release()
    out.release()
    processing_done = True


# --------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file or file.filename == "":
            return redirect("/")
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        threading.Thread(target=detect_vehicles, args=(filepath,), daemon=True).start()
        return redirect("/result")
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/status_data")
def status_data():
    counts = output_data.get("counts", {c: 0 for c in VALID_CLASSES})
    return {
        "counts": counts,
        "vehicles_in_zone": output_data.get("zone_count", 0),
        "avg_speed": output_data.get("avg_speed", 0),
        "risk_percent": output_data.get("risk_percent", 0),
        "density": output_data.get("density_label", "Low"),
        "weather": "No weather data"
    }


@app.route("/result")
def result():
    return render_template("result.html", live_url="/video_feed")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

#python "d:\OG projects\FYP\vehicle detection and category wise counting\app.py"