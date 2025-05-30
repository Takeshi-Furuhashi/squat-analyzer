from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import mediapipe as mp
import uuid
import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def analyze_squat(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hip_ys, speeds, times, timestamps = [], [], [], []
    prev_hip_y = None
    hip_y_at_depth = knee_y_at_depth = None

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]

            hip_ys.append(hip.y)
            timestamps.append(i / fps)

            if hip_y_at_depth is None or hip.y > hip_y_at_depth:
                hip_y_at_depth = hip.y
                knee_y_at_depth = knee.y

            if prev_hip_y is not None:
                dy = hip.y - prev_hip_y
                speeds.append(dy * fps)
                times.append(i / fps)
            prev_hip_y = hip.y

    cap.release()

    if not speeds:
        return "解析失敗", {}, None

    bottom_idx = hip_ys.index(max(hip_ys))
    downward_time = timestamps[bottom_idx] - timestamps[0]
    upward_time = timestamps[-1] - timestamps[bottom_idx]
    total_time = timestamps[-1]

    max_down = min(speeds[:bottom_idx]) if bottom_idx > 0 else 0
    max_up = max(speeds[bottom_idx:]) if bottom_idx < len(speeds) else 0

    depth_ratio = hip_y_at_depth / knee_y_at_depth if knee_y_at_depth else 0
    is_good_lift = depth_ratio >= 1.00
    status = "Good Lift" if is_good_lift else "No Lift"
    color = "green" if is_good_lift else "red"

    result_data = {
        "color": color,
        "depth_ratio": f"{depth_ratio:.3f}",
        "total_time": f"{total_time:.2f}",
        "down_time": f"{downward_time:.2f}",
        "up_time": f"{upward_time:.2f}",
        "max_down_speed": f"{max_down:.3f}",
        "max_up_speed": f"{max_up:.3f}"
    }

    return status, result_data, result_data

def save_result_to_txt(filename_base, status, metrics):
    txt_filename = f"{filename_base}.txt"
    txt_path = os.path.join(UPLOAD_FOLDER, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Generated at: {datetime.datetime.now()}\n")
        f.write(f"Squat judgment: {status}\n")
        f.write(f"Hip/Knee ratio: {metrics['depth_ratio']}\n")
        f.write(f"Total duration: {metrics['total_time']} s\n")
        f.write(f"Downward phase: {metrics['down_time']} s\n")
        f.write(f"Upward phase: {metrics['up_time']} s\n")
        f.write(f"Max downward speed: {metrics['max_down_speed']} px/s\n")
        f.write(f"Max upward speed: {metrics['max_up_speed']} px/s\n")
    return txt_filename

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    txt_filename = None

    if request.method == "POST":
        file = request.files["video"]
        if file:
            uid = uuid.uuid4().hex
            filename = f"{uid}.mp4"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            status, metrics, result_data = analyze_squat(filepath)
            txt_filename = save_result_to_txt(uid, status, metrics)
            os.remove(filepath)

            result = {"status": status, "txt_filename": txt_filename, **metrics}

    return render_template("index.html", result=result)

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
