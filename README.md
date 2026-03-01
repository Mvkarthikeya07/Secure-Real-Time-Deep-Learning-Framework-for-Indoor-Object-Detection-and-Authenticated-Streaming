🔐 Secure Real-Time Indoor Object Detection & Authenticated Streaming
A Production-Oriented AI Deployment System Using YOLOv8m

📌 Overview

This project implements a secure, low-latency, real-time indoor object detection system integrated with authenticated browser-based video streaming.

Unlike standard YOLO demonstrations, this system focuses on end-to-end AI deployment engineering, integrating:

Real-time deep learning inference

Secure session-controlled access

REST-based detection services

Structured backend modularization

Performance-aware optimization

The objective is to bridge the gap between computer vision research prototypes and deployable AI web systems.

📸 System Demonstration

🔑 1. Secure Login Session

<img width="1366" height="768" alt="Screenshot (199)" src="https://github.com/user-attachments/assets/f9df24c7-4f20-4095-8530-f1f04d420a80" />

Authenticated access ensures that only authorized users can access real-time video streaming and detection endpoints.

✔ Credential validation
✔ Session creation
✔ Protected route redirection

📡 2. Authenticated Dashboard

<img width="1366" height="768" alt="Screenshot (200)" src="https://github.com/user-attachments/assets/3724c7d4-9a9a-4a16-89ee-c85050ee6598" />

Upon successful login, users are redirected to a protected dashboard that serves the live detection stream.

✔ Session-protected /home route
✔ Embedded MJPEG streaming
✔ Continuous frame delivery

🎯 3. Real-Time Detection Output (Live Session)

<img width="1366" height="768" alt="Screenshot (202)" src="https://github.com/user-attachments/assets/580792e1-0cfa-4a8d-b997-414b3ca3006a" />

The system performs live YOLOv8m inference with bounding box visualization and confidence scoring.

Detected objects in sample session:

Person (0.81)

Cell Phone (0.95)

Bottle (0.62)

✔ Stable bounding boxes
✔ IoU-based duplicate suppression
✔ Indoor-aware class filtering
✔ Confidence-based thresholding

🎯 Key Contributions

Secure session-based authentication with protected routes

Authenticated MJPEG real-time streaming

Optimized YOLOv8m inference pipeline (GPU/CPU adaptive)

Indoor-aware class filtering to suppress irrelevant COCO classes

Tuned Non-Maximum Suppression (IoU-based duplicate reduction)

REST API for external detection integration

Threaded streaming architecture for uninterrupted frame serving

Performance evaluation under controlled hardware conditions

🧠 System Architecture
Client Browser
      │
      ▼
Flask Web Server
      │
      ├── Authentication Layer
      ├── YOLOv8 Inference Engine
      ├── Indoor Class Filtering Module
      ├── Frame Annotation Pipeline
      ├── REST API Endpoints
      │
      ▼
OpenCV Video Capture
      │
      ▼
MJPEG Streaming Response → Browser
Architectural Design Principles

Separation of concerns (auth, inference, streaming decoupled)

Threaded execution for uninterrupted frame streaming

Minimal shared global state

Modular inference wrapper for future model replacement

⚙️ Technical Stack

Component	Technology
Detection Model	YOLOv8m (Ultralytics)
Deep Learning Backend	PyTorch
Web Framework	Flask
Image Processing	OpenCV
Streaming Protocol	MJPEG
Authentication	Session-based (Flask sessions)
Execution	GPU/CPU adaptive
🔬 Performance Evaluation
Benchmark Environment

GPU: NVIDIA RTX 3050 (4GB)

CPU: Intel i5 12th Gen

RAM: 16GB

Resolution: 640×480

Camera: 30 FPS USB Webcam

Lighting: Controlled indoor conditions

Observed Inference Performance
Metric	GPU	CPU
Model Inference	~25–35 FPS	~8–15 FPS
End-to-End Stream Latency	Low (real-time capable)	Moderate
Frame Stability	Stable	Slight frame drops under load

🏠 Indoor-Aware Class Filtering

Standard COCO models detect 80 object classes, many irrelevant indoors.

This system filters detections to a curated indoor subset such as:

Person

Cell Phone

Laptop

Bottle

Chair

Monitor

Impact

Reduced false positives

Improved bounding box stability

Lower visual clutter

Increased reliability in office/classroom environments

🔐 Security Design

The system integrates structured access control:

Credential validation on login

Protected /home route

Protected /video_feed endpoint

Authenticated /detect API

Secure session management

Logout session invalidation

Threat Model Scope

Prevents unauthorized stream access

Prevents unauthenticated API usage

Designed for controlled LAN / prototype environments

Note: Production deployment should include HTTPS, CSRF protection, and hardened credential storage.

🌐 REST API
GET /detect

Returns system readiness and configuration.

POST /detect

Accepts image file input and returns:

{
  "detections": [
    {
      "class": "person",
      "confidence": 0.81,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "count": 1
}
📁 Project Structure
secure-real-time-indoor-object-detection/
│
├── app.py
├── models/
│   └── yolov8m.pt
├── templates/
│   ├── login.html
│   └── index.html
├── requirements.txt
└── README.md

🚀 Installation
Clone Repository
git clone https://github.com/yourusername/secure-real-time-indoor-object-detection.git
cd secure-real-time-indoor-object-detection
Install Dependencies
pip install -r requirements.txt
Run Application
python app.py

Access:

http://127.0.0.1:5000

Default Credentials:

Username: demo
Password: password
🌍 Applications

Smart indoor surveillance systems

Secure office monitoring

AI-enabled classrooms

Industrial workspace monitoring

Edge AI deployment prototypes

⚠ Limitations

Performance dependent on hardware

MJPEG bandwidth overhead

Uses pretrained COCO dataset

Single-camera support

🔮 Future Work

Multi-camera support

Object tracking (DeepSORT)

Role-based access control

Docker containerization

HTTPS + hardened security

Custom indoor dataset training

WebRTC-based streaming optimization

🔬 Research & Engineering Significance

This project demonstrates:

Transitioning deep learning models to secure web deployment

Real-time inference optimization under hardware constraints

Integration of authentication mechanisms with AI pipelines

Systems-level design thinking in applied computer vision

It reflects a deployment-focused approach to modern AI engineering rather than a standalone model experiment.

👨‍💻 Author

M V Karthikeya
B.Tech – Computer Science (AI & ML)

📜 License

This project is licensed under the MIT License.
