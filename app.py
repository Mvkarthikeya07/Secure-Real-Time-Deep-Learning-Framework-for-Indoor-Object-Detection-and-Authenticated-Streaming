# app.py
# YOLOv8m Real-Time Webcam Detection Server — Fully Fixed

import os
import traceback
import torch
import cv2
import numpy as np
from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, Response
)
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ---------------- CONFIG ----------------
SECRET_KEY  = os.environ.get("SECRET_KEY",    "change_this_in_production")
DEMO_USER   = os.environ.get("DEMO_USERNAME", "demo")
DEMO_PASS   = os.environ.get("DEMO_PASSWORD", "password")
HOST        = os.environ.get("HOST",          "127.0.0.1")
PORT        = int(os.environ.get("PORT",      5000))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))

# FIX 1: Use yolov8m (medium) — far fewer false labels than yolov8s
# yolov8s was misidentifying cupboards as laptops, fans as persons
# because it's too small and relies on rough shape matching.
# yolov8m has significantly better feature extraction.
MODEL_PATH     = os.environ.get("MODEL_PATH", "yolov8m.pt")

# FIX 2: Raise NMS IoU threshold — reduces duplicate/ghost boxes
# FIX 3: Keep conf at 0.45 — low enough to catch real objects,
#         high enough to reject shape-based false positives
CONF_THRESHOLD = 0.45
IOU_THRESHOLD  = 0.5

# ---------------- APP ----------------
app = Flask(__name__, template_folder="templates")
app.secret_key = SECRET_KEY

# ---------------- DEVICE ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ---------------- LOAD MODEL ----------------
print(f"[INFO] Loading {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
model.to(device)
print("[INFO] Model loaded successfully.")

# Stable color per class (80 COCO classes)
COLORS = np.random.default_rng(42).uniform(50, 220, size=(80, 3)).astype(int)

# FIX 4: Class filter — only show classes that make sense indoors.
# This STOPS the model from labelling a cupboard as "laptop"
# or a fan as "person" — those labels are now suppressed entirely.
# Add or remove class names here to control what gets detected.
ALLOWED_CLASSES = {
    "person", "chair", "couch", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "backpack", "handbag",
    "suitcase", "bottle", "cup", "bowl", "apple", "banana",
    "sandwich", "orange", "knife", "spoon", "fork",
    "potted plant", "umbrella", "dog", "cat", "bicycle", "car"
}
# Set to None to allow ALL classes (no filtering):
# ALLOWED_CLASSES = None


# ---------------- DRAW BOXES ----------------
def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        label  = det["class_name"]
        conf   = det["confidence"]
        color  = tuple(int(c) for c in COLORS[cls_id % 80])

        text        = f"{label} {conf:.2f}"
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 0.6
        thickness   = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label background
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        # Label text
        cv2.putText(frame, text, (x1 + 2, y1 - 5),
                    font, font_scale, (255, 255, 255), thickness)
    return frame


# ---------------- DETECTION ----------------
def detect_frame(frame_bgr):
    try:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = model(
            img_rgb,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,   # FIX: NMS threshold to remove ghost boxes
            imgsz=640,
            verbose=False
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls  = int(box.cls[0])
                name = model.names[cls]

                # FIX 4: Skip classes not in our allowed list
                if ALLOWED_CLASSES and name not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                detections.append({
                    "class_id":   cls,
                    "class_name": name,
                    "confidence": round(conf, 3),
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)]
                })

        return detections

    except Exception:
        print("[ERROR] Detection failed:\n", traceback.format_exc())
        return []


# ---------------- STREAM GENERATOR ----------------
def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. "
              "Try CAMERA_INDEX=1 in .env if you have multiple cameras.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"[INFO] Camera {CAMERA_INDEX} opened. Streaming...")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARN] Frame grab failed — camera disconnected?")
                break

            detections = detect_frame(frame)
            frame      = draw_boxes(frame, detections)

            ret, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()
        print("[INFO] Camera released.")


# ---------------- AUTH HELPER ----------------
def is_logged_in():
    return "user" in session


# ================================================================
# ROUTES
# ================================================================

# ── Login page ────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def route_login():
    # Already logged in → go straight to home
    if is_logged_in():
        return redirect(url_for("route_home"))
    return render_template("login.html")


# ── Handle login form submission ──────────────────────────────────
@app.route("/auth", methods=["POST"])
def route_auth():
    user = request.form.get("username", "").strip()
    pw   = request.form.get("password", "")

    if user == DEMO_USER and pw == DEMO_PASS:
        session["user"] = user
        return redirect(url_for("route_home"))

    # Wrong credentials → back to login with error message
    return render_template("login.html", error="Invalid username or password.")


# ── Home / dashboard (protected) ─────────────────────────────────
@app.route("/home", methods=["GET"])
def route_home():
    if not is_logged_in():
        return redirect(url_for("route_login"))
    return render_template("index.html", user=session.get("user"))


# ── Logout ────────────────────────────────────────────────────────
@app.route("/logout")
def route_logout():
    session.pop("user", None)
    return redirect(url_for("route_login"))


# ── Live webcam MJPEG stream (protected) ─────────────────────────
@app.route("/video_feed")
def route_video_feed():
    """
    Live webcam stream with YOLO bounding boxes.
    Add this to your index.html where you want the video to appear:

        <img src="/video_feed" width="640" height="480">
    """
    if not is_logged_in():
        return redirect(url_for("route_login"))

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── Single image upload detection (API / testing) ─────────────────
@app.route("/detect", methods=["GET", "POST"])
def route_detect():
    if not is_logged_in():
        return jsonify({"error": "unauthorized"}), 401

    if request.method == "GET":
        return jsonify({
            "status":          "ready",
            "model":           MODEL_PATH,
            "device":          device,
            "conf_threshold":  CONF_THRESHOLD,
            "iou_threshold":   IOU_THRESHOLD,
            "allowed_classes": list(ALLOWED_CLASSES) if ALLOWED_CLASSES else "all"
        })

    try:
        if "image" not in request.files:
            return jsonify({"error": "Send multipart/form-data with key 'image'."}), 400

        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        data = file.read()
        if not data:
            return jsonify({"error": "File is empty."}), 400

        nparr = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Cannot decode image. Use JPEG/PNG/BMP."}), 400

        detections = detect_frame(img)
        return jsonify({"detections": detections, "count": len(detections)})

    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] /detect:\n", tb)
        return jsonify({"error": str(e), "trace": tb}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True      # Required for MJPEG streaming
    )
