# app.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask import Flask, render_template, Response, jsonify
import time
import os

# --- Global Flags and Objects ---
# Use a simple dictionary for application state
app_state = {
    "is_running": False
}
cap = None # Global VideoCapture object
model = None # Global model object

# Load Haar-cascades for face and eye detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Model parameters
MODEL_PATH = "vgg16_transfer_model.h5"
CLASS_NAMES = ["closed", "no_yawn", "open", "yawn"]
IMG_SIZE = (224, 224)

app = Flask(__name__)

# --- Initialization Function ---
def load_resources():
    """Loads the Keras model and sets up TensorFlow/Keras environment."""
    global model
    if model is None:
        try:
            # Setting the environment for Keras in a multi-threaded Flask app
            # Keras model loading should be done once globally.
            print("Loading Keras model...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Keras model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Ensure the model file exists in the correct directory!
            exit()

# Load the model when the application starts
load_resources()

# --- Core Detection Logic (Generator) ---
def generate_frames():
    """Reads frames from the camera, processes them, and yields JPEG bytes."""
    global cap
    
    # Initialize camera if not already open
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        app_state["is_running"] = False
        return

    app_state["is_running"] = True
    print("Camera stream started.")

    while app_state["is_running"]:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        # --- Your Existing Detection Logic ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
        
        label_to_display = ""
        for (x, y, w, h) in faces:
            # Face ROI for image processing
            face_roi_color = frame[y:y+h, x:x+w]
            
            # 1. Face/Yawn Prediction (to decide whether to check eyes)
            face_input = cv2.resize(face_roi_color, IMG_SIZE)
            face_input = preprocess_input(np.expand_dims(face_input, axis=0))
            pred = model.predict(face_input, verbose=0) # Use verbose=0 to suppress output
            idx = np.argmax(pred)
            label = CLASS_NAMES[idx]

            # Determine the main classification (Eye state or Yawn state)
            is_eye_state = label in ("open", "closed")
            
            if is_eye_state:
                # 2. Eye State Check
                # Detect eyes within the face region of the grayscale image
                eyes = EYE_CASCADE.detectMultiScale(gray[y:y+h, x:x+w], 1.1, 4)
                
                eye_detected = False
                for (ex, ey, ew, eh) in eyes:
                    # Eye ROI relative to the face ROI
                    eye_roi_color = face_roi_color[ey:ey+eh, ex:ex+ew]
                    if eye_roi_color.size == 0:
                        continue
                        
                    # Predict eye state
                    eye_input = cv2.resize(eye_roi_color, IMG_SIZE)
                    eye_input = preprocess_input(np.expand_dims(eye_input, axis=0))
                    pred_eye = model.predict(eye_input, verbose=0)
                    idx_eye = np.argmax(pred_eye)
                    label_eye = CLASS_NAMES[idx_eye]
                    
                    label_to_display = f"Eye: {label_eye} ({pred_eye[0][idx_eye]:.2f})"
                    # Draw rectangle on the eye (relative to face_roi_color)
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    eye_detected = True
                    break # Only use the first detected eye

                if not eye_detected:
                    # Fallback if no eye is detected but face prediction suggests eye state
                    label_to_display = f"Face: {label} (No Eye Found)"

            else:
                # 3. Yawn State Check
                label_to_display = f"Face: {label} ({pred[0][idx]:.2f})"

            # Draw rectangle on the face (relative to the whole frame)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label_to_display, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            break # Only handle the first detected face

        # --- Stream the Frame ---
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    # Cleanup when streaming loop stops
    if cap:
        cap.release()
    cap = None
    print("Camera stream stopped and released.")

# --- Flask Routes ---

@app.route('/')
def index():
    """Render the main front-end page."""
    return render_template('index.html', is_running=app_state["is_running"])

@app.route('/video_feed')
def video_feed():
    """Video streaming route. It continually serves processed JPEG frames."""
    # Use the generator function to serve a Multi-part encoded response
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """API endpoint to start the streaming."""
    if not app_state["is_running"]:
        # Simply setting the flag to True will let the video_feed route start the generator
        app_state["is_running"] = True
        return jsonify(success=True, message="Stream starting."), 200
    return jsonify(success=True, message="Stream already running."), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """API endpoint to stop the streaming."""
    if app_state["is_running"]:
        # Setting the flag to False will break the 'while' loop in generate_frames()
        app_state["is_running"] = False
        # Give a small delay for the generator to clean up resources
        time.sleep(0.5) 
        return jsonify(success=True, message="Stream stopping."), 200
    return jsonify(success=True, message="Stream already stopped."), 200

# --- Main Execution ---
if __name__ == '__main__':
    # Running in debug mode is fine for development, but for production,
    # consider setting threaded=True or using a production WSGI server.
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)