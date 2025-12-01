# 🚦 AI-Powered Intelligent Traffic Monitoring System
Real-time **vehicle detection, tracking, speed estimation, density analysis, and accident risk prediction** using Deep Learning (YOLO), Flask, and Computer Vision.

🎥 **Demo Video:** https://www.youtube.com/watch?v=gb8st7U4UGQ

---

## ⭐ Project Overview

The **AI-Powered Intelligent Traffic Monitoring System** is a web-based application that performs **real-time traffic video analysis** using state-of-the-art computer vision models.  
After uploading a traffic video, the system provides:

- 🚗 Vehicle detection & classification (Car, Truck, Bus, Bike)
- 🎯 Zone-based monitoring (entering, exiting, inside tracking)
- ⚡ Speed estimation (km/h)
- 🔥 Accident risk calculation (0–100%)
- 📈 Traffic density level (Low / Medium / High)
- 🧠 AI-powered live dashboard
- 🎨 Modern animated UI

Ideal for **smart city automation**, **traffic surveillance**, and **road safety analytics**.

---

## 🌟 Key Features

| Feature | Status |
|--------|--------|
| YOLO-based vehicle detection | ✅ |
| Multi-class classification | ✅ |
| Real-time vehicle tracking | ✅ |
| Zone-based counting | ✅ |
| Speed estimation (km/h) | ✅ |
| Accident risk scoring | ✅ |
| Density analysis | ✅ |
| Live dashboard with charts | ✅ |
| Modern HTML/CSS animated UI | ✅ |
| Video overlay output | ✅ |

---

## 📂 Project Structure

```
FYP/
│── app.py
│── requirements.txt
│── yolov5n.pt
│
├── templates/
│   ├── index.html
│   ├── processing.html
│   └── result.html
│
├── static/
│   └── style.css
│
├── uploads/        # Auto-created to store user-input videos
├── output/         # Auto-generated output videos
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/pavan050505/intelligent-traffic-monitoring-system.git
cd intelligent-traffic-monitoring-system
```

### 2️⃣ Create & activate virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add YOLO model

Ensure `yolov5n.pt` exists in the project root.

### 5️⃣ Run the web application

```bash
python app.py
```

Open in browser:

👉 http://127.0.0.1:5000

Upload a video — the dashboard will show **live AI analysis**.

---

## 🎬 System Workflow

1. User uploads a traffic video  
2. Flask backend starts AI processing  
3. YOLO detects vehicles frame-by-frame  
4. Tracking algorithm assigns unique IDs  
5. Speed estimated using pixel displacement & calibration  
6. Accident risk = speed + density + zone occupancy  
7. Dashboard updates every second via JSON API  
8. Processed video is displayed with overlay  

---

## 📊 Dashboard Insights

| Metric | Description |
|--------|-------------|
| **Vehicle Count** | Vehicles currently inside defined zone |
| **Average Speed** | Real-time speed of all tracked vehicles |
| **Accident Risk Score** | 0–100% probability |
| **Density Level** | Low / Medium / High |
| **Category Count Chart** | Car / Truck / Bus / Bike distribution |
| **Live Video Stream** | AI processed output |

---

## 🔧 Customizable Parameters

| Variable | Description |
|----------|-------------|
| `RESIZE_SCALE` | Controls accuracy vs speed |
| `frame_skip` | Faster runtime vs precision |
| `zone_coordinates` | Region of interest |
| `METERS_PER_PIXEL` | Speed calibration |

---

## 🚀 Future Enhancements

- Automatic Number Plate Recognition (ANPR)
- Lane violation detection
- Wrong-way driving alerts
- Historical analytics & reports
- Cloud-based video processing
- SMS/Email alert system
- Weather-aware risk predictions

---

## 🪪 License
This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🙌 Credits

| Component | Technology |
|----------|------------|
| Object Detection | YOLOv5 |
| Dashboard Charts | Chart.js |
| UI Animations | Particles.js |
| Backend | Flask |

---

## 📽 Demo Video
🎥 Watch how the system detects, tracks & analyzes traffic in real-time:  
👉 https://www.youtube.com/watch?v=gb8st7U4UGQ

---

## ⭐ Support
If this project helped you, please **⭐ star the repository** — it motivates future updates!


