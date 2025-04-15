# Driver Drowsiness Detection System

A real-time drowsiness detection system built with TensorFlow, MediaPipe, and Flask that monitors for signs of driver fatigue through facial analysis.
Features

Real-time Drowsiness Detection: Monitors eye closure and yawning patterns to detect driver fatigue
Facial Landmark Tracking: Uses MediaPipe Face Mesh to accurately track facial features
Audio Alerts: Plays warning sounds when drowsiness is detected
Customizable Sensitivity: Includes calibration feature to adjust detection thresholds to individual users
Escalating Alerts: Secondary alarm triggers when multiple drowsiness events occur within a short time period
Web Interface: Easy-to-use browser-based monitoring interface

How It Works
The system uses a combination of techniques to detect drowsiness:

Eye Aspect Ratio (EAR): Measures the openness of the eyes using facial landmarks
Yawn Detection: Uses a pre-trained TensorFlow model to detect yawning
Time-Based Analysis: Monitors duration of eye closure and yawning to identify potential drowsiness
Frequency Analysis: Tracks multiple drowsiness events within a time window (20 seconds) to identify persistent fatigue

Getting Started
Prerequisites

Python 3.7+
Webcam or camera device
The following Python packages:

TensorFlow
OpenCV
MediaPipe
Flask
NumPy
Pygame
python-dotenv
Flask-CORS



Installation

Clone the repository:
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection

Install dependencies:
pip install -r requirements.txt

Set up environment variables by creating a .env file:
TEMPLATE_FOLDER=path/to/templates
STATIC_FOLDER=path/to/static
BUZZER_SOUND=sounds/buzzer.wav
SECONDARY_BUZZER=sounds/secondary_buzzer.wav
MODEL_PATH=models/yawn_detection_model.h5


Running the Application

Start the Flask server:
python app.py

Open your web browser and navigate to:
http://localhost:5000

Follow on-screen instructions to calibrate the system for your face

API Endpoints

GET / - Web interface for the drowsiness detection system
GET /video_feed - Streaming endpoint for camera feed
GET /api/status - Check system status and model availability
POST /api/calibrate - Calibrate eye aspect ratio threshold
POST /api/detect - Process a single frame for drowsiness detection

Customization
Adjusting Detection Parameters
You can modify the following variables in app.py to adjust sensitivity:

EYE_THRESHOLD - Default threshold for eye aspect ratio
EYES_CLOSED_DURATION - Time in seconds eyes must be closed to trigger alert (default: 1)
YAWNING_DURATION - Time in seconds yawning must be detected to trigger alert (default: 3)

Adding Custom Alert Sounds
Replace the sound files in your static folder and update the corresponding paths in your .env file.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

MediaPipe team for the face mesh solution
TensorFlow team for the machine learning framework
