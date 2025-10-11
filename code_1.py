import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load Haar-cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load your trained Keras model
model = load_model("vgg16_transfer_model.h5")
class_names = ["closed", "no_yawn", "open", "yawn"]  # Change order if necessary
IMG_SIZE = (224, 224)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    label_to_display = ""
    for (x, y, w, h) in faces:
        face_roi_color = frame[y:y+h, x:x+w]
        face_input = cv2.resize(face_roi_color, IMG_SIZE)
        face_input = preprocess_input(np.expand_dims(face_input, axis=0))
        pred = model.predict(face_input)
        idx = np.argmax(pred)
        label = class_names[idx]
        
        if label in ("open", "closed"):
            # Eye state, run eye detection within face
            eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
            for (ex, ey, ew, eh) in eyes:
                eye_roi_color = face_roi_color[ey:ey+eh, ex:ex+ew]
                if eye_roi_color.size == 0:
                    continue
                eye_input = cv2.resize(eye_roi_color, IMG_SIZE)
                eye_input = preprocess_input(np.expand_dims(eye_input, axis=0))
                pred_eye = model.predict(eye_input)
                idx_eye = np.argmax(pred_eye)
                label_eye = class_names[idx_eye]
                label_to_display = f"Eye: {label_eye} ({pred_eye[0][idx_eye]:.2f})"
                cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                break  # Only use the first detected eye for simplicity
        else:
            # Yawn or no_yawn, use face region
            label_to_display = f"Face: {label} ({pred[0][idx]:.2f})"

        # Draw rectangle on face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label_to_display, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        break  # Only handle the first detected face for real-time efficiency

    cv2.imshow('Drowsiness & Yawn Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
