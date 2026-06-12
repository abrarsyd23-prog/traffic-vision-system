# 🚦 Traffic Vision System
**CSCI435 — Computer Vision Algorithms and Systems**  
University of Wollongong in Dubai · Spring 2026

Real-time traffic scene analysis integrating four computer vision tasks into a unified pipeline.

---

## CV Capabilities
| # | Task | Method | Purpose |
|---|------|---------|---------|
| 1 | **Object Detection** | YOLOv8n (pretrained + fine-tuned) | Detect vehicles, pedestrians, traffic signs |
| 2 | **Edge Detection** | OpenCV Canny | Map lane markings and road boundaries |
| 3 | **Face Detection** | MediaPipe Face Detection | Detect pedestrian and driver faces |
| 4 | **Object Tracking** | OpenCV CSRT | Track a selected vehicle across frames |

---

## Project Structure
```
traffic-vision-system/
├── app.py                      # Streamlit frontend — entry point
├── requirements.txt            # Pinned dependencies
├── core/
│   ├── config.py               # All constants and thresholds
│   ├── pipeline.py             # Unified 4-task pipeline
│   ├── edge_detection.py       # Canny edge detection
│   ├── object_detection.py     # YOLOv8 object detection
│   ├── face_detection.py       # MediaPipe face detection
│   └── object_tracking.py     # CSRT object tracking
├── utils/
│   └── helpers.py              # Frame conversion and video utilities
├── models/
│   └── yolov8n_traffic.pt      # Fine-tuned weights (add after training)
├── data/
│   ├── sample_images/
│   └── sample_videos/
└── notebooks/
    └── finetune_yolov8.ipynb   # YOLOv8 fine-tuning on Google Colab
```

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/traffic-vision-system.git
cd traffic-vision-system

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Problem Statement
Traffic authorities and road safety analysts need automated tools to monitor road activity in real time — detecting and counting vehicles, tracking movement, identifying pedestrians, and mapping lane boundaries — without expensive hardware. This system provides comprehensive traffic scene analysis on a standard camera feed using only commodity hardware and open-source models.

## User Story
A traffic monitoring operator uploads a camera feed. The system instantly detects and labels all vehicles and pedestrians, maps road and lane edges, counts faces at a checkpoint, and tracks a selected vehicle across frames — giving the operator a complete real-time scene understanding.

---

## References
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- MediaPipe: https://mediapipe.dev
- OpenCV: https://opencv.org
- Roboflow: https://roboflow.com
