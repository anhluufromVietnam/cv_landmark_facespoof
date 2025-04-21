import cv2
import joblib
import numpy as np
import mediapipe as mp

# Load the trained model
clf = joblib.load('models/liveness_model.pkl')

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Extract all landmarks (468 points)
def extract_all_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    return np.array([[lm[i].x, lm[i].y, lm[i].z] for i in range(len(lm))]).flatten()

# Compute relative distances between important landmarks
def compute_distances(landmarks):
    eye_dist = np.linalg.norm(landmarks[33] - landmarks[263])  # Left-right eye distance
    mouth_dist = np.linalg.norm(landmarks[61] - landmarks[291])  # Mouth left-right distance
    return np.array([eye_dist, mouth_dist])

# Confidence threshold for classification
confidence_threshold = 0.6  # Adjust based on your model's behavior

# Real-time prediction loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_all_landmarks(frame)
    if landmarks is not None:
        distances = compute_distances(landmarks)
        features = np.concatenate([landmarks, distances])
        probs = clf.predict_proba([features])[0]

        # Apply threshold for prediction
        confidence = probs[1] if probs[1] > probs[0] else probs[0]
        label = "REAL" if probs[1] > probs[0] else "FAKE"

        # Display confidence threshold
        if confidence > confidence_threshold:
            text = f"{label}: {int(confidence * 100)}%"
        else:
            text = "Uncertain"

        color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
