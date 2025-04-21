import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

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
    # Eye distance: left-right eye (indices 33, 263)
    eye_dist = np.linalg.norm(landmarks[33] - landmarks[263])
    # Mouth distance: left-right mouth (indices 61, 291)
    mouth_dist = np.linalg.norm(landmarks[61] - landmarks[291])
    return np.array([eye_dist, mouth_dist])

# Load data from 'real' and 'fake' folders
def load_data(folder, label):
    X, y = [], []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue
        landmarks = extract_all_landmarks(img)
        if landmarks is not None:
            distances = compute_distances(landmarks)
            X.append(np.concatenate([landmarks, distances]))  # Combine landmarks and distances
            y.append(label)
    return X, y

X_real, y_real = load_data('data/real', 1)
X_fake, y_fake = load_data('data/fake', 0)

X = np.array(X_real + X_fake)
y = np.array(y_real + y_fake)

# Train SVM with class weights (giving more importance to fake faces)
clf = SVC(kernel='rbf', class_weight={0: 2, 1: 1}, probability=True)
clf.fit(X, y)

# Evaluate model performance
print(classification_report(y, clf.predict(X)))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/liveness_model.pkl')
print("Model saved.")
