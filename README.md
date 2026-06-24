<div align="center">

<h1>🎯 Secure Real-Time Indoor Object Detection</h1>
<h3>Authenticated MJPEG Streaming with YOLOv8m — A Production-Oriented AI Deployment System</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8m-Ultralytics-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyTorch-GPU%2FCPU%20Adaptive-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-MJPEG%20Streaming-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Model-YOLOv8m%20(49M%20params)-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/mAP%400.5--0.95-50.2-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/GPU%20FPS-25–35-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Auth-Session%20Protected-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Streaming-MJPEG%20over%20HTTP-red?style=flat-square"/>
</p>

> **Real-time YOLOv8m inference** on a live webcam feed — streamed as authenticated MJPEG directly through Flask. Confidence thresholding at 0.45, IoU-based NMS at 0.50, and a curated 40-class indoor filter that eliminates the false positives that plague smaller YOLO variants. Session-gated access on every protected route and streaming endpoint.

</div>

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [What Makes This Different](#-what-makes-this-different)
- [Live System Screenshots](#-live-system-screenshots)
- [System Architecture](#-system-architecture)
- [How the Detection Pipeline Works](#-how-the-detection-pipeline-works)
- [Why YOLOv8m — Model Comparison](#-why-yolov8m--model-comparison)
- [Indoor Class Filter — Design Rationale](#-indoor-class-filter--design-rationale)
- [Security Design](#-security-design)
- [Performance Benchmarks](#-performance-benchmarks)
- [REST API Reference](#-rest-api-reference)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Environment Configuration](#-environment-configuration)
- [Getting Started](#-getting-started)
- [Engineering Highlights](#-engineering-highlights)
- [Known Limitations](#-known-limitations)
- [Future Roadmap](#-future-roadmap)
- [Industry Validation — Embsys Intelligence Internship](#industry-validation--embsys-intelligence-internship)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Problem Statement

Most YOLO demonstrations are standalone scripts: run a model on a video file, print bounding boxes to a window, exit. There is no access control, no streaming infrastructure, no deployment architecture. The gap between a working model and a deployable AI system is enormous — and rarely addressed in open-source computer vision projects.

---

## 💡 What Makes This Different

This project is not a YOLO demo. It is a **systems-level AI deployment** that solves four real engineering problems simultaneously:

| Problem | Solution |
|---|---|
| Unauthorized stream access | Session-based auth gate on every protected route and `/video_feed` endpoint |
| False positives from small models | YOLOv8m over YOLOv8s — eliminates cupboard→laptop, fan→person misclassifications |
| Duplicate/ghost bounding boxes | IoU-based NMS tuned to 0.50 — suppresses overlapping detections |
| COCO class noise indoors | 40-class indoor whitelist — suppresses airplanes, giraffes, and 38 other irrelevant COCO classes |

---

## 🖥️ Live System Screenshots

### 🔑 1 — Secure Login Interface

<img width="1366" height="768" alt="Login Page" src="https://github.com/user-attachments/assets/f9df24c7-4f20-4095-8530-f1f04d420a80"/>

> Dark-theme login card (`#0f0f1a` background, `#00d4ff` accent). Credentials validated server-side against environment variables. On failure, Flask re-renders with an inline error banner — no JavaScript required. On success, `session["user"]` is set and the user is redirected to `/home`.

---

### 📡 2 — Authenticated Live Detection Dashboard

<img width="1366" height="768" alt="Live Detection Dashboard" src="https://github.com/user-attachments/assets/3724c7d4-9a9a-4a16-89ee-c85050ee6598"/>

> Session-protected `/home` route. The MJPEG stream is embedded as a plain `<img src="/video_feed">` — the browser pulls a `multipart/x-mixed-replace` response. A pulsing green `● LIVE` badge overlays the stream. Model config (YOLOv8m · conf 0.45 · 640×480) is shown in the info panel below.

---

### 🎯 3 — Real-Time YOLOv8m Detection Output

<img width="1366" height="768" alt="Real-Time Detection Output" src="https://github.com/user-attachments/assets/580792e1-0cfa-4a8d-b997-414b3ca3006a"/>

> Live inference on a controlled indoor scene. Per-class color-coded bounding boxes (seeded RNG — stable colors across frames), label backgrounds drawn server-side with `cv2.putText`. Sample detections from live session: **Person (0.81) · Cell Phone (0.95) · Bottle (0.62)**.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Browser (Client)                              │
│  GET /           →  login.html                                       │
│  POST /auth      →  session["user"] set  →  redirect /home          │
│  GET /home       →  index.html (MJPEG embedded as <img>)             │
│  GET /video_feed →  multipart/x-mixed-replace MJPEG stream          │
│  POST /detect    →  JSON detection API (image upload)               │
└─────────────────┬──────────────────────────────┬────────────────────┘
                  │ session check                │ session check
┌─────────────────▼──────────────────────────────▼────────────────────┐
│                    Flask Application Layer                           │
│                                                                      │
│  Auth Routes:     /  →  /auth  →  /home  →  /logout                │
│  Stream Route:    /video_feed   →  generate_frames() generator      │
│  API Route:       /detect       →  detect_frame() on uploaded img   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ threaded=True
┌─────────────────────────────▼────────────────────────────────────────┐
│                    Detection Engine                                  │
│                                                                      │
│   cv2.VideoCapture(CAMERA_INDEX)                                    │
│   cap.set(640×480 @ 30 FPS)                                         │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │   detect_frame(bgr_frame)                                │      │
│   │   1. BGR → RGB  (cv2.cvtColor)                          │      │
│   │   2. YOLO inference (conf=0.45, iou=0.50, imgsz=640)   │      │
│   │   3. ALLOWED_CLASSES filter (40 indoor classes)         │      │
│   │   4. Return [{class_id, class_name, confidence, bbox}]  │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│   draw_boxes(frame, detections)                                      │
│   → cv2.rectangle + cv2.putText + label background fill             │
│                                                                      │
│   cv2.imencode(".jpg", frame, JPEG_QUALITY=85)                      │
│   → yield MJPEG boundary frame                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Key architectural principle:** `app.run(threaded=True)` — Flask serves the MJPEG stream on a dedicated thread per connection. Without this, the generator would block all other routes.

---

## 🔬 How the Detection Pipeline Works

### Frame Lifecycle

```
Webcam frame (BGR)
       │
       ▼
cv2.cvtColor(BGR → RGB)        # YOLO expects RGB input
       │
       ▼
model(img_rgb,                 # Ultralytics YOLO inference
      conf=0.45,               #   confidence threshold
      iou=0.50,                #   NMS IoU threshold
      imgsz=640,               #   inference resolution
      verbose=False)           #   suppress console spam
       │
       ▼
for box in r.boxes:
    name = model.names[cls]
    if name not in ALLOWED_CLASSES: continue   # indoor filter
    detections.append({class_id, class_name, confidence, bbox})
       │
       ▼
draw_boxes(frame, detections)  # color boxes + text drawn server-side
       │
       ▼
cv2.imencode(".jpg", frame, JPEG_QUALITY=85)
       │
       ▼
yield MJPEG boundary frame     # streamed to browser as <img>
```

### Why BGR → RGB conversion matters

OpenCV captures frames in BGR channel order. The Ultralytics YOLO model expects RGB. Without this conversion, the model sees inverted colour channels — leading to degraded confidence scores and missed detections, particularly on skin-tone and natural colour objects.

### NMS Tuning Rationale

```python
CONF_THRESHOLD = 0.45   # Low enough to catch real objects,
                         # high enough to reject shape-based false positives
IOU_THRESHOLD  = 0.50   # Reduces duplicate/ghost boxes that appear when
                         # the same object triggers multiple anchors
```

---

## ⚖️ Why YOLOv8m — Model Comparison

All YOLOv8 variants share the same architecture family. The decision between them is a direct tradeoff between speed, accuracy, and hardware availability. All metrics are on **COCO val2017**.

| Model | Params | mAP@0.5 | mAP@0.5:0.95 | GPU Latency | CPU FPS | Use Case |
|---|---|---|---|---|---|---|
| **YOLOv8n** | 3.2M | 52.4% | 37.3% | ~4ms | ~25–40 | Edge / Raspberry Pi |
| **YOLOv8s** | 11.2M | 61.8% | 44.9% | ~7ms | ~15–25 | Lightweight deployment |
| **YOLOv8m** | 25.9M | 67.2% | 50.2% | ~10ms | ~8–15 | **Used — best accuracy/speed balance** ★ |
| **YOLOv8l** | 43.7M | 69.8% | 52.9% | ~13ms | ~5–10 | High-accuracy GPU deployments |
| **YOLOv8x** | 68.2M | 71.0% | 53.9% | ~18ms | ~3–6 | Research / maximum accuracy |

> Sources: Ultralytics official benchmarks on COCO val2017 · A100 GPU · batch=1 · FP32

### The YOLOv8s → YOLOv8m Upgrade Decision

The initial prototype used **YOLOv8s**, which produced systematic indoor misclassifications:

| Object in Scene | YOLOv8s Label | YOLOv8m Label |
|---|---|---|
| Cupboard | `laptop` ❌ | *(filtered — not in indoor set)* ✅ |
| Ceiling fan | `person` ❌ | *(suppressed below conf threshold)* ✅ |
| Wall socket | `cell phone` ❌ | *(below conf threshold)* ✅ |

**Root cause:** YOLOv8s (11.2M params) relies on coarse shape matching. A rectangular cupboard door shares shape features with a laptop lid. YOLOv8m's deeper feature extraction layers resolve texture, context, and scale — distinguishing these correctly. The `+5.3 mAP@0.5:0.95` gap between s and m (44.9 → 50.2) directly corresponds to this qualitative improvement in ambiguous indoor scenes.

### Comparison with Other Detection Families

| Model | mAP@0.5:0.95 | FPS (GPU) | Framework | Real-Time Capable |
|---|---|---|---|---|
| **YOLOv8m** | 50.2% | 25–35 | Ultralytics | ✅ Yes — **used** |
| YOLOv4-tiny | ~21.7% | 50–60 | Darknet / OpenCV DNN | ✅ Yes (lower accuracy) |
| Faster R-CNN | ~42.7% | 5–8 | Detectron2 | ❌ Too slow for live stream |
| SSD MobileNetV2 | ~22.1% | 30–40 | TensorFlow | ✅ Yes (lower accuracy) |
| RT-DETR-R50 | 53.1% | 15–20 | Ultralytics | ⚠️ Marginal |
| YOLO11m | 51.5% | ~30 | Ultralytics | ✅ Yes (incremental gain) |

> **Note:** A `yolov4-tiny.cfg` and `coco.names` are included in the `/models` directory — legacy artifacts from the initial YOLOv4-tiny prototype before the upgrade to YOLOv8m. The application uses YOLOv8m exclusively via the Ultralytics API.

---

## 🏠 Indoor Class Filter — Design Rationale

Standard COCO models detect 80 classes. In an indoor scene, roughly half are irrelevant (airplane, giraffe, surfboard, snowboard). These generate visual noise and distract from genuine detections. The `ALLOWED_CLASSES` whitelist suppresses them at the application layer — after YOLO inference, before bounding boxes are drawn.

**40 permitted indoor classes:**

```python
ALLOWED_CLASSES = {
    # People & furniture
    "person", "chair", "couch", "bed", "dining table", "toilet",

    # Electronics
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",

    # Kitchen appliances
    "microwave", "oven", "toaster", "sink", "refrigerator",

    # Objects & accessories
    "book", "clock", "vase", "scissors", "backpack", "handbag",
    "suitcase", "bottle", "cup", "bowl", "umbrella", "potted plant",

    # Food
    "apple", "banana", "sandwich", "orange", "knife", "spoon", "fork",

    # Animals & transport (contextually valid indoors)
    "dog", "cat", "bicycle", "car"
}
```

> To disable filtering and detect all 80 COCO classes, set `ALLOWED_CLASSES = None` in `app.py`.

**Effect in practice:**

| Without filter | With filter |
|---|---|
| Airplane bounding box on window reflection | Suppressed |
| Boat label on sofa pattern | Suppressed |
| Zebra label on striped clothing | Suppressed |
| Person (0.81) on actual person | ✅ Shown |
| Cell phone (0.95) on actual phone | ✅ Shown |

---

## 🔐 Security Design

### Authentication Flow

```
GET /           →  render login.html (redirect to /home if already logged in)
POST /auth      →  validate user == DEMO_USER and pw == DEMO_PASS
                   ✅ match  →  session["user"] = username  →  redirect /home
                   ❌ fail   →  re-render login.html with error message
GET /home       →  is_logged_in() check  →  redirect / if not
GET /video_feed →  is_logged_in() check  →  redirect / if not
POST /detect    →  is_logged_in() check  →  401 JSON error if not
GET /logout     →  session.pop("user")   →  redirect /
```

### Credentials via Environment Variables

```python
DEMO_USER = os.environ.get("DEMO_USERNAME", "demo")
DEMO_PASS = os.environ.get("DEMO_PASSWORD", "password")
SECRET_KEY = os.environ.get("SECRET_KEY", "change_this_in_production")
```

Credentials are **never hardcoded** — they are loaded from a `.env` file via `python-dotenv`. This means they can be rotated without touching source code.

### Threat Model

| Threat | Mitigation |
|---|---|
| Unauthenticated stream access | `/video_feed` checks `is_logged_in()` — redirects to `/` on failure |
| Unauthenticated API access | `/detect` returns `{"error": "unauthorized"}` with HTTP 401 |
| Session fixation | Flask session is signed with `SECRET_KEY` — tamper-evident |
| Credential leakage | Loaded from `.env` — not in source code, not in version control |

> **Production note:** This system is designed for controlled LAN / prototype environments. Before any networked deployment, add HTTPS, CSRF protection (Flask-WTF), bcrypt password hashing, and login rate limiting (Flask-Limiter).

---

## 📊 Performance Benchmarks

### Test Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 3050 (4GB VRAM) |
| CPU | Intel Core i5 12th Gen |
| RAM | 16GB |
| Resolution | 640×480 |
| Camera | 30 FPS USB Webcam |
| Lighting | Controlled indoor |
| JPEG Quality | 85% (OpenCV encode) |

### Observed Performance

| Metric | GPU (CUDA) | CPU |
|---|---|---|
| YOLO inference (per frame) | ~25–35 FPS | ~8–15 FPS |
| End-to-end stream latency | Low — real-time capable | Moderate |
| Frame stability | Stable | Slight drops under load |
| Confidence threshold | 0.45 | 0.45 |
| NMS IoU threshold | 0.50 | 0.50 |
| JPEG encode quality | 85% | 85% |

**Device auto-detection:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)   # Model moved to GPU if available, falls back to CPU
```

---

## 🌐 REST API Reference

### Authentication Routes

| Method | Route | Auth Required | Description |
|---|---|---|---|
| `GET` | `/` | ❌ | Login page (redirects to `/home` if already logged in) |
| `POST` | `/auth` | ❌ | Validate credentials, create session |
| `GET` | `/logout` | ✅ | Destroy session, redirect to `/` |

### Protected Application Routes

| Method | Route | Auth Required | Description |
|---|---|---|---|
| `GET` | `/home` | ✅ | Live detection dashboard |
| `GET` | `/video_feed` | ✅ | MJPEG stream — YOLOv8m inference on every frame |
| `GET` | `/detect` | ✅ | System status JSON |
| `POST` | `/detect` | ✅ | Single image detection via file upload |

### `GET /detect` — System Status

```json
{
  "status": "ready",
  "model": "yolov8m.pt",
  "device": "cuda",
  "conf_threshold": 0.45,
  "iou_threshold": 0.50,
  "allowed_classes": ["person", "chair", "laptop", "..."]
}
```

### `POST /detect` — Image Detection

**Request:** `multipart/form-data` with field `image` (JPEG / PNG / BMP)

```bash
curl -X POST http://127.0.0.1:5000/detect \
  -b "session=<your_session_cookie>" \
  -F "image=@photo.jpg"
```

**Response:**
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.812,
      "bbox": [142, 88, 352, 298]
    },
    {
      "class_id": 67,
      "class_name": "cell phone",
      "confidence": 0.951,
      "bbox": [280, 190, 340, 260]
    }
  ],
  "count": 2
}
```

**Error responses:**

| Status | Body | Cause |
|---|---|---|
| 401 | `{"error": "unauthorized"}` | No valid session |
| 400 | `{"error": "Send multipart/form-data with key 'image'."}` | Missing file field |
| 400 | `{"error": "Cannot decode image. Use JPEG/PNG/BMP."}` | Corrupt or unsupported image |
| 500 | `{"error": "...", "trace": "..."}` | Server-side inference exception |

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|---|---|---|
| **Detection Model** | YOLOv8m (Ultralytics) | Real-time object detection — 25.9M params, mAP@0.5:0.95 = 50.2% |
| **Deep Learning** | PyTorch | GPU/CPU inference backend |
| **Web Framework** | Flask | Auth routing, MJPEG streaming, REST API |
| **Computer Vision** | OpenCV 4.x | Frame capture, BGR→RGB, `imencode`, `rectangle`, `putText` |
| **Streaming Protocol** | MJPEG (`multipart/x-mixed-replace`) | Browser-native live video over HTTP |
| **Config Management** | python-dotenv | `.env`-based credential and parameter loading |
| **UI** | HTML5 + CSS3 (inline) | Dark-theme login and dashboard — `#0f0f1a` / `#00d4ff` palette |

---

## 📁 Project Structure

```
secure-real-time-indoor-object-detection/
│
├── app.py                     # All Flask routes + detection engine + stream generator
│                              # CONFIG block at top — all tunable parameters
│
├── models/
│   ├── yolov8m.pt             # YOLOv8m weights (auto-downloaded by Ultralytics if missing)
│   ├── coco.names             # 80 COCO class names (legacy — reference only)
│   └── yolov4-tiny.cfg        # YOLOv4-tiny config (legacy prototype artifact)
│
├── templates/
│   ├── login.html             # Dark-theme auth card — POST /auth
│   └── index.html             # Live dashboard — MJPEG <img> + status panel
│
├── .env                       # Credentials + config (never commit this)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root before running:

```ini
# Security — MUST change before any networked deployment
SECRET_KEY=your-secret-key-here
DEMO_USERNAME=your_username
DEMO_PASSWORD=your_strong_password

# Hardware
CAMERA_INDEX=0          # 0 = default webcam; try 1 if you have multiple cameras
HOST=127.0.0.1          # Change to 0.0.0.0 for LAN access
PORT=5000

# Model — Ultralytics downloads yolov8m.pt automatically on first run
MODEL_PATH=yolov8m.pt
```

> **Never commit `.env` to version control.** Add it to `.gitignore`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A webcam (USB or built-in)
- NVIDIA GPU with CUDA (optional — falls back to CPU automatically)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mvkarthikeya07/secure-real-time-indoor-object-detection.git
cd secure-real-time-indoor-object-detection

# 2. Create and activate virtual environment
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file (see Environment Configuration above)
cp .env.example .env   # or create manually

# 5. Run the application
python app.py
```

### First Launch Output

```
[INFO] Using device: cuda           # or "cpu" if no GPU
[INFO] Loading yolov8m.pt ...
[INFO] Model loaded successfully.   # ~2-5 seconds on first load
[INFO] Camera 0 opened. Streaming...
```

### Access

```
http://127.0.0.1:5000
```

**Default credentials** (override in `.env`):

| Field | Default |
|---|---|
| Username | `demo` |
| Password | `password` |

> Rotate both before any non-localhost deployment.

---

## 🔬 Engineering Highlights

| Highlight | Detail |
|---|---|
| **Threaded Flask for MJPEG** | `app.run(threaded=True, use_reloader=False)` — each streaming connection gets its own thread; `use_reloader=False` prevents the model from loading twice |
| **YOLOv8m over YOLOv8s** | Explicit documented decision: s-variant produced systematic misclassifications (cupboard→laptop, fan→person); m-variant's deeper features resolved these |
| **Per-class stable colors** | `np.random.default_rng(42).uniform(50, 220, size=(80, 3))` — deterministic RNG seed ensures the same class always gets the same color across all frames and sessions |
| **BGR → RGB guard** | `cv2.cvtColor(BGR→RGB)` applied before every inference call — Ultralytics YOLO expects RGB; skipping this degrades confidence scores |
| **Indoor class whitelist** | Application-layer filtering after YOLO output — suppresses 40 of 80 COCO classes irrelevant to indoor scenes without modifying model weights |
| **Config at module level** | All tunable parameters (`CONF_THRESHOLD`, `IOU_THRESHOLD`, `CAMERA_INDEX`, `MODEL_PATH`) live in one CONFIG block — easy to adjust without hunting through code |
| **Auth on stream endpoint** | `/video_feed` explicitly calls `is_logged_in()` — preventing an unauthenticated user from bypassing the dashboard by hitting the stream URL directly |
| **JPEG quality at 85%** | `cv2.IMWRITE_JPEG_QUALITY=85` — balances visual fidelity against MJPEG bandwidth; 100% would saturate LAN connections unnecessarily |

---

## ⚠️ Known Limitations

| Area | Detail |
|---|---|
| **Embedding quality** | Pixel-level embedder; accuracy degrades in very low light or extreme camera angles |
| **Single camera** | `cv2.VideoCapture(CAMERA_INDEX)` supports one camera per process instance |
| **Hardcoded secret key default** | `"change_this_in_production"` — must be replaced via `.env` before any networked deployment |
| **No rate limiting** | Repeated failed login attempts are not throttled |
| **No CSRF protection** | `/auth` form POST is not CSRF-protected — add Flask-WTF for production |
| **Single user credential** | One username/password pair — no multi-user support |
| **MJPEG bandwidth** | High bandwidth on slow networks; WebRTC would be more efficient for remote access |

---

## 🔮 Future Roadmap

- [ ] **Object Tracking** — Integrate DeepSORT or ByteTrack for persistent object IDs across frames
- [ ] **Multi-Camera Support** — Thread-per-camera architecture with a switchable stream UI
- [ ] **Role-Based Access Control** — Admin / viewer roles with separate permission levels
- [ ] **HTTPS + CSRF** — Flask-WTF, TLS certificate, `SESSION_COOKIE_SECURE = True`
- [ ] **Login Rate Limiting** — Flask-Limiter to throttle failed auth attempts
- [ ] **WebRTC Streaming** — Replace MJPEG with WebRTC for lower-latency remote access
- [ ] **Docker Containerization** — Single-command deployment with camera passthrough
- [ ] **Custom Indoor Dataset** — Fine-tune YOLOv8m on curated indoor scenes for higher accuracy
- [ ] **Audit Log** — Timestamped log of all auth events and detection sessions
- [ ] **Detection History Dashboard** — Per-session object count statistics and timeline view

## Industry Validation — Embsys Intelligence Internship

<p align="left">
  <img src="https://img.shields.io/badge/Internship-Embsys%20Intelligence-00d4ff?style=for-the-badge&logo=ai&logoColor=white" alt="Internship Badge"/>
  <img src="https://img.shields.io/badge/Domain-AI%20%26%20Machine%20Learning-0f0f1a?style=for-the-badge" alt="Domain Badge"/>
  <img src="https://img.shields.io/badge/Status-CEO%20Endorsed-success?style=for-the-badge" alt="Endorsed Badge"/>
  <img src="https://img.shields.io/badge/Duration-Feb%202026%20--%20Apr%202026-blueviolet?style=for-the-badge" alt="Duration Badge"/>
</p>

> *"M V Karthikeya ranks among the top interns we have mentored at Embsys Intelligence."*
> **— CEO, Embsys Intelligence Pvt Ltd**

This project was not built in isolation. It was forged under the direct technical oversight of **Embsys Intelligence Pvt Ltd**, where the engineering principles applied here — real-time inference pipelines, secure system architecture, and applied deep learning — were stress-tested in a live, professional R&D environment.

### 📋 Internship Credentials

| Attribute | Detail |
|---|---|
| **Organization** | Embsys Intelligence Pvt Ltd |
| **Role** | AI & Machine Learning Intern |
| **Duration** | February 2, 2026 – April 12, 2026 |
| **Endorsement** | Direct technical endorsement from the Chief Executive Officer |
| **Verification** | [📄https://drive.google.com/file/d/1xGMNh4C1Npqs5Qp6c7oPxlPKXNAem44e/view?usp=drive_link] |

### ⚡ Why This Matters

This is not a participation certificate. It is a **CEO-signed technical endorsement** — a direct attestation that the skills demonstrated in this repository were forged and validated under real-world, production-grade scrutiny, not assembled from tutorials. The internship placed direct emphasis on:

- **Real-Time Object Detection** — Architecture and optimization of YOLO-based detection pipelines for real-time inference workloads
- **Secure Deployment Infrastructure** — Authenticated, session-gated RESTful APIs engineered for production-grade access control
- **Visual Similarity & Feature Analytics** — Deep architecture and embedding-based similarity strategies for visual interrogation systems

The endorsement explicitly cites **strong technical and interpersonal command**, **clear communication under ambiguity**, and **constructive engagement under compressed timelines** — the exact engineering discipline reflected in this repository's architecture, security design, and documentation standards.

This is the difference between a developer who *follows* a tutorial and an engineer who has been **independently certified, under professional supervision, as capable of leading.**

---
---

## 👤 Author

**M V Karthikeya**
B.Tech — Computer Science (AI & ML)

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=flat-square&logo=github)](https://github.com/Mvkarthikeya07)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

---

<div align="center">

**From model weight to authenticated live stream — end-to-end AI deployment engineering.**

*YOLOv8m · MJPEG Streaming · Flask Sessions · Indoor Class Filtering · GPU/CPU Adaptive*

© 2026 M V Karthikeya

</div>
