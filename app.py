from flask import Flask, request, jsonify, render_template, send_from_directory, Response
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import base64
import time
import os
from flask_cors import CORS
import pygame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(
    __name__,
    template_folder=os.environ.get('TEMPLATE_FOLDER'),
    static_folder=os.environ.get('STATIC_FOLDER')
)
CORS(app)

# Initialize Pygame mixer for audio playback
pygame.mixer.init()

# Path to the buzzer sounds
BUZZER_SOUND_PATH = os.path.join(app.static_folder, os.environ.get('BUZZER_SOUND'))
SECONDARY_SOUND_PATH = os.path.join(app.static_folder, os.environ.get('SECONDARY_BUZZER'))

def play_buzzer():
    pygame.mixer.music.load(BUZZER_SOUND_PATH)
    pygame.mixer.music.play()

def play_secondary_buzzer():
    pygame.mixer.music.load(SECONDARY_SOUND_PATH)
    pygame.mixer.music.play()

# Load the pre-trained model
MODEL_PATH = os.environ.get('MODEL_PATH')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define mouth and eye landmarks (MediaPipe's 468 landmarks)
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Default eye threshold
EYE_THRESHOLD = 0.25

# Timers for drowsiness detection
EYES_CLOSED_START_TIME = None
YAWNING_START_TIME = None
EYES_CLOSED_DURATION = 1 # seconds
YAWNING_DURATION = 3  # seconds
drowsiness_timestamps = []

# Initialize webcam (change index to 1 for external webcam)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(f"Failed to open camera {1}, trying another index...")
    cap = cv2.VideoCapture(0)  # Fallback to default camera if external fails



def extract_mouth(frame, landmarks, w, h):
    points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_LANDMARKS])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    margin = 10  # Add some margin88
    x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
    x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)
    mouth_img = frame[y_min:y_max, x_min:x_max]
    if mouth_img.size == 0:
        return None
    mouth_img = cv2.resize(mouth_img, (224, 224)) / 255.0
    return np.expand_dims(mouth_img, axis=0)




def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return float((A + B) / (2.0 * C))  # Convert to Python float


def convert_to_native_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    else:
        return obj


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'API is running',
        'model_loaded': bool(model_loaded),
        'endpoints': {
            'calibrate': '/api/calibrate',
            'detect': '/api/detect'
        }
    })


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global EYE_THRESHOLD
    ear_values = []

    if 'frames' not in request.json:
        return jsonify({'error': 'No frames provided'}), 400

    frames = request.json['frames']
    for frame_data in frames:
        try:
            encoded_data = frame_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = np.array(
                        [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE])
                    right_eye = np.array(
                        [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])
                    ear_values.append(float(
                        (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0))
        except Exception as e:
            print(f"Error processing frame: {str(e)}")

    if len(ear_values) > 0:
        EYE_THRESHOLD = float(np.mean(ear_values) + 0.01)
        return jsonify({'threshold': float(EYE_THRESHOLD)})
    else:
        return jsonify({'error': 'No face detected in calibration frames'}), 400


def check_drowsiness_time_window():
    global drowsiness_timestamps

    # Remove timestamps older than 20 seconds
    current_time = time.time()
    drowsiness_timestamps = [timestamp for timestamp in drowsiness_timestamps if current_time - timestamp <= 20]

    # If there are 4 or more detections within 20 seconds, play the secondary sound
    if len(drowsiness_timestamps) >= 5:
        play_secondary_buzzer()  # Play secondary buzzer
        # Optionally, reset the timestamps if you want to clear them after the secondary sound is played
        # drowsiness_timestamps = []


@app.route('/api/detect', methods=['POST'])
def detect_drowsiness():
    global EYES_CLOSED_START_TIME, YAWNING_START_TIME

    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        encoded_data = request.json['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        response = {'drowsy': False, 'yawning': False, 'eyes_closed': False, 'face_detected': False}

        if results.multi_face_landmarks:
            response['face_detected'] = True
            for face_landmarks in results.multi_face_landmarks:
                mouth_img = extract_mouth(frame, face_landmarks.landmark, w, h)
                is_yawning = False

                if mouth_img is not None:
                    yawning_prob = model.predict(mouth_img, verbose=0)[0][0]
                    is_yawning = bool(yawning_prob > 0.5)

                    if is_yawning:
                        if YAWNING_START_TIME is None:
                            YAWNING_START_TIME = time.time()
                        elif time.time() - YAWNING_START_TIME >= YAWNING_DURATION:
                            response['drowsy'] = True
                            if not pygame.mixer.music.get_busy():
                                play_buzzer()  # Play buzzer when drowsiness is detected
                                # Track the timestamp of drowsiness detection
                                drowsiness_timestamps.append(time.time())
                                check_drowsiness_time_window()  # Check if secondary buzzer should play
                    else:
                        YAWNING_START_TIME = None

                    response['yawning'] = is_yawning
                    response['yawn_probability'] = float(yawning_prob)

                left_eye = np.array(
                    [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE])
                right_eye = np.array(
                    [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])
                avg_ear = float((eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0)
                response['eye_aspect_ratio'] = float(avg_ear)
                response['eyes_closed'] = avg_ear < EYE_THRESHOLD

                if response['eyes_closed']:
                    if EYES_CLOSED_START_TIME is None:
                        EYES_CLOSED_START_TIME = time.time()
                    elif time.time() - EYES_CLOSED_START_TIME >= EYES_CLOSED_DURATION:
                        response['drowsy'] = True
                        if not pygame.mixer.music.get_busy():  # Check if sound is playing
                            play_buzzer()  # Play buzzer when drowsiness is detected
                            # Track the timestamp of drowsiness detection
                            drowsiness_timestamps.append(time.time())
                            check_drowsiness_time_window()  # Check if secondary buzzer should play
                else:
                    EYES_CLOSED_START_TIME = None

        return jsonify(convert_to_native_types(response))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)