from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from datetime import datetime
import sqlite3
import random
from scipy.spatial import distance

app = Flask(__name__)

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# EAR threshold for blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

# Database setup
conn = sqlite3.connect(':memory:', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS blinks
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp DATETIME,
              duration REAL)''')
conn.commit()

# Per-person blink tracking
blink_data = {}  # face_id -> {counter, total_blinks, last_seen, blink_history, centroid}
next_face_id = 0


def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


def assign_face_id(current_centroid, known_faces, threshold=0.05):
    global next_face_id
    for face_id, data in known_faces.items():
        dist = distance.euclidean(current_centroid, data.get('centroid'))
        if dist < threshold:
            data['centroid'] = current_centroid
            data['last_seen'] = time.time()
            return face_id

    # Assign new ID
    new_id = next_face_id
    known_faces[new_id] = {
        'counter': 0,
        'total_blinks': 0,
        'blink_history': deque(maxlen=10),
        'centroid': current_centroid,
        'last_seen': time.time()
    }
    next_face_id += 1
    return new_id


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Estimate face "centroid" using outer eye corners
                centroid_x = (landmarks[33].x + landmarks[263].x) / 2
                centroid_y = (landmarks[33].y + landmarks[263].y) / 2
                face_id = assign_face_id((centroid_x, centroid_y), blink_data)

                left_eye = np.array([
                    (landmarks[33].x, landmarks[33].y),
                    (landmarks[160].x, landmarks[160].y),
                    (landmarks[158].x, landmarks[158].y),
                    (landmarks[133].x, landmarks[133].y),
                    (landmarks[153].x, landmarks[153].y),
                    (landmarks[144].x, landmarks[144].y)
                ])
                right_eye = np.array([
                    (landmarks[362].x, landmarks[362].y),
                    (landmarks[385].x, landmarks[385].y),
                    (landmarks[387].x, landmarks[387].y),
                    (landmarks[263].x, landmarks[263].y),
                    (landmarks[373].x, landmarks[373].y),
                    (landmarks[380].x, landmarks[380].y)
                ])

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                person = blink_data[face_id]

                if ear < EYE_AR_THRESH:
                    person['counter'] += 1
                else:
                    if person['counter'] >= EYE_AR_CONSEC_FRAMES:
                        person['total_blinks'] += 1
                        duration = person['counter'] / 30.0
                        person['blink_history'].append(time.time())
                        c.execute("INSERT INTO blinks (timestamp, duration) VALUES (?, ?)",
                                  (datetime.now(), duration))
                        conn.commit()
                    person['counter'] = 0

                # Display info
                x = int(centroid_x * frame.shape[1])
                y = int(centroid_y * frame.shape[0])
                text = f"ID {face_id} | Blinks: {person['total_blinks']}"
                cv2.putText(frame, text, (x - 50, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Cleanup: remove inactive faces
        now = time.time()
        inactive_ids = [fid for fid, data in blink_data.items() if now - data['last_seen'] > 10]
        for fid in inactive_ids:
            del blink_data[fid]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_stats')
def get_stats():
    # Return total blinks per person
    person_stats = {str(face_id): data['total_blinks'] for face_id, data in blink_data.items()}
    return jsonify({
        'persons': person_stats,
        'active_faces': len(blink_data)
    })


if __name__ == '__main__':
    app.run(debug=True)
