from flask import Flask, render_template, request, jsonify, Response
import serial
import time
import glob
import cv2
import joblib
import requests
import numpy as np
import json
import os
from datetime import datetime
from threading import Lock

app = Flask(__name__)
SERIAL_BAUD = 9600
ser = None
last_command = None

# ================= CAMERA SETUP =================
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
if not camera.isOpened():
    print("Camera 0 failed, trying camera 1...")
    camera = cv2.VideoCapture(1, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
if not camera.isOpened():
    print("Camera failed to open.")
else:
    print("Camera opened successfully.")

# ================= SERIAL SETUP =================
def find_teensy_port():
    ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    if not ports:
        return None
    return ports[0]

def connect_serial():
    global ser
    if ser is not None and ser.is_open:
        return True
    port = find_teensy_port()
    if port is None:
        print("No Teensy serial port found.")
        return False
    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        print(f"Connected to Teensy on {port}")
        return True
    except Exception as e:
        print(f"Serial connection error: {e}")
        return False

def send_command(cmd):
    global last_command
    allowed = ["f", "b", "l", "r", "s", "p", "m", "n", "v"]
    if cmd not in allowed:
        return False, "Invalid command"
    if not connect_serial():
        return False, "Serial not connected"
    try:
        if cmd in ["f", "b", "l", "r", "s"]:
            if cmd == last_command:
                return True, "Same command ignored"
            last_command = cmd
        ser.write(cmd.encode())
        print(f"Sent: {cmd}")
        return True, f"Sent {cmd}"
    except Exception as e:
        return False, str(e)

# ================= ML + WEATHER SETUP =================
MODEL_PATH = "water_model.pkl"
FEEDBACK_LOG = "feedback_log.jsonl"
DEFICIT_STATE_FILE = "deficit_state.json"
LATITUDE = 36.17           # Las Vegas
LONGITUDE = -115.14
WEATHER_CACHE_SECONDS = 600
CROP_COEFFICIENT = 0.7     # match what you used in build_dataset.py

try:
    bundle = joblib.load(MODEL_PATH)
    water_model = bundle["model"]
    MODEL_FEATURES = bundle["feature_cols"]
    print(f"Loaded ML model. Features: {MODEL_FEATURES}")
except Exception as e:
    water_model = None
    MODEL_FEATURES = []
    print(f"Could not load model: {e}")

_weather_cache = {"data": None, "timestamp": None}
_last_recommendation = {"data": None, "timestamp": None}
_feedback_lock = Lock()
_deficit_lock = Lock()


def update_deficit(weather_today):
    """
    Maintain a running soil water deficit across Flask restarts.
    Updates once per calendar day. Returns the current deficit in mm.
    Only relevant if 'water_deficit_mm' is in MODEL_FEATURES.
    """
    with _deficit_lock:
        if os.path.exists(DEFICIT_STATE_FILE):
            try:
                with open(DEFICIT_STATE_FILE) as f:
                    state = json.load(f)
            except Exception:
                state = {"deficit": 0.0, "last_date": None}
        else:
            state = {"deficit": 0.0, "last_date": None}

        today = datetime.now().date().isoformat()
        if state["last_date"] != today:
            et = weather_today.get("et0_fao_evapotranspiration", 0) or 0
            rain = weather_today.get("rain_last_24h", 0) or 0
            state["deficit"] = max(0.0, state["deficit"] + et * CROP_COEFFICIENT - rain)
            state["last_date"] = today
            with open(DEFICIT_STATE_FILE, "w") as f:
                json.dump(state, f)
        return state["deficit"]


def reset_deficit():
    """Call this when the user actually waters the plants."""
    with _deficit_lock:
        state = {"deficit": 0.0, "last_date": datetime.now().date().isoformat()}
        with open(DEFICIT_STATE_FILE, "w") as f:
            json.dump(state, f)


def fetch_weather():
    """Pull current + last 7 days of daily weather from Open-Meteo via wlan1."""
    now = datetime.now()
    cached = _weather_cache["data"]
    cached_time = _weather_cache["timestamp"]
    if cached and cached_time and (now - cached_time).total_seconds() < WEATHER_CACHE_SECONDS:
        return cached

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        "&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        "relative_humidity_2m_mean,wind_speed_10m_max,"
        "shortwave_radiation_sum,precipitation_sum,et0_fao_evapotranspiration"
        "&past_days=7&forecast_days=1&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        daily = r.json()["daily"]
        today = 7  # past_days=7 means index 7 is today

        precip = daily["precipitation_sum"]
        et0_series = daily["et0_fao_evapotranspiration"]

        rain_last_24h = (precip[today] or 0.0)
        rain_last_3d  = sum((p or 0) for p in precip[today-2:today+1])
        rain_last_7d  = sum((p or 0) for p in precip[today-6:today+1])
        et0_last_7d   = sum((e or 0) for e in et0_series[today-6:today+1])

        days_since_rain = 0
        for p in reversed(precip[:today+1]):
            if p and p > 1.0:
                break
            days_since_rain += 1

        weather = {
            "temperature_2m_mean":         daily["temperature_2m_mean"][today],
            "temperature_2m_max":          daily["temperature_2m_max"][today],
            "temperature_2m_min":          daily["temperature_2m_min"][today],
            "relative_humidity_2m_mean":   daily["relative_humidity_2m_mean"][today],
            "wind_speed_10m_max":          daily["wind_speed_10m_max"][today],
            "shortwave_radiation_sum":     daily["shortwave_radiation_sum"][today],
            "et0_fao_evapotranspiration":  et0_series[today],
            "rain_last_24h":               rain_last_24h,
            "rain_last_3d":                rain_last_3d,
            "rain_last_7d":                rain_last_7d,
            "days_since_rain":             days_since_rain,
            "et0_last_7d":                 et0_last_7d,
        }

        # Add deficit only if model expects it
        if "water_deficit_mm" in MODEL_FEATURES:
            weather["water_deficit_mm"] = update_deficit(weather)

        _weather_cache["data"] = weather
        _weather_cache["timestamp"] = now
        return weather
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return cached


def get_water_recommendation():
    if water_model is None:
        return {"error": "Model not loaded"}
    weather = fetch_weather()
    if weather is None:
        return {"error": "Could not fetch weather"}

    missing = [f for f in MODEL_FEATURES if f not in weather]
    if missing:
        return {"error": f"Weather payload missing features: {missing}"}

    feature_values = [weather[f] for f in MODEL_FEATURES]
    features = np.array([feature_values])

    pred = int(water_model.predict(features)[0])
    proba = float(water_model.predict_proba(features)[0][1])

    importances = water_model.feature_importances_
    top_factors = sorted(
        zip(MODEL_FEATURES, feature_values, importances),
        key=lambda x: -x[2]
    )[:4]

    result = {
        "recommendation": "water" if pred == 1 else "skip",
        "confidence": round(proba, 3),
        "weather": weather,
        "top_factors": [
            {"feature": f, "value": round(float(v), 2),
             "importance": round(float(i), 3)}
            for f, v, i in top_factors
        ],
        "timestamp": datetime.now().isoformat(),
    }
    _last_recommendation["data"] = result
    _last_recommendation["timestamp"] = datetime.now()
    return result


def log_feedback(user_label, recommendation_snapshot, notes=""):
    record = {
        "timestamp": datetime.now().isoformat(),
        "user_label": user_label,
        "recommendation": recommendation_snapshot.get("recommendation"),
        "confidence": recommendation_snapshot.get("confidence"),
        "weather": recommendation_snapshot.get("weather"),
        "notes": notes,
    }
    with _feedback_lock:
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    return record


# ================= CAMERA STREAM =================
def generate_camera_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_camera_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/command", methods=["POST"])
def command():
    data = request.get_json()
    cmd = data.get("cmd")
    success, message = send_command(cmd)
    return jsonify({"success": success, "message": message, "cmd": cmd})

@app.route("/status")
def status():
    connected = connect_serial()
    port = ser.port if ser and ser.is_open else None
    return jsonify({"connected": connected, "port": port})

# --- ML routes ---
@app.route("/water_recommendation")
def water_recommendation():
    return jsonify(get_water_recommendation())

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json() or {}
    label = data.get("label")
    notes = data.get("notes", "")
    if label not in ("good", "bad"):
        return jsonify({"success": False, "error": "label must be 'good' or 'bad'"}), 400
    snapshot = _last_recommendation["data"]
    if snapshot is None:
        return jsonify({"success": False, "error": "No recommendation yet"}), 400
    record = log_feedback(label, snapshot, notes)
    # If user said the "water" recommendation was good and they actually watered,
    # reset the deficit so the running state stays honest
    if (label == "good" and snapshot.get("recommendation") == "water"
            and "water_deficit_mm" in MODEL_FEATURES):
        reset_deficit()
    return jsonify({"success": True, "logged": record})

@app.route("/feedback_stats")
def feedback_stats():
    if not os.path.exists(FEEDBACK_LOG):
        return jsonify({"total": 0, "good": 0, "bad": 0})
    good = bad = 0
    with open(FEEDBACK_LOG) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("user_label") == "good": good += 1
                elif rec.get("user_label") == "bad": bad += 1
            except json.JSONDecodeError:
                continue
    return jsonify({"total": good + bad, "good": good, "bad": bad})

@app.route("/reset_deficit", methods=["POST"])
def reset_deficit_route():
    """Manual reset endpoint — call when you've actually watered."""
    reset_deficit()
    return jsonify({"success": True})


if __name__ == "__main__":
    connect_serial()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
