```markdown
# Liveness Detection Pipeline

This project uses OpenCV, MediaPipe, and SVM to build a real-time liveness detection system capable of distinguishing between **real (3D)** and **fake (2D)** faces. It captures facial data, trains a machine learning model, and uses the trained model for live face detection via a webcam.

## 📦 **Dependencies**

Make sure you have the following Python libraries installed:

```bash
pip install opencv-python mediapipe scikit-learn numpy joblib
```

## 🧑‍💻 **Folder Structure**

```
liveness_project/
├── data/
│   ├── real/               # Folder for real face images
│   └── fake/               # Folder for fake face images
├── models/                 # Folder to save the trained model
├── collect_data.py         # Script for collecting real/fake face data
├── train_liveness_model.py # Script for training the liveness detection model
├── detect_live_face.py     # Script for real-time liveness detection using webcam
├── README.md               # Project description and instructions
```

## 🚀 **Steps to Run the Pipeline**

### 1. **Collect Data (Real and Fake Faces)**

To collect **real** or **fake** face data, use the `collect_data.py` script. This will capture images from your webcam and save them in the respective folders (`real/` or `fake/`).

- Run the following command to collect **real** face data:
  ```bash
  python collect_data.py
  ```
  - **When prompted**, enter "real" to capture images of real faces.
  - Press `c` to capture an image and `q` to quit.

- Similarly, run the command to collect **fake** face data:
  ```bash
  python collect_data.py
  ```
  - **When prompted**, enter "fake" to capture images of fake faces.

### 2. **Train the Model**

After collecting enough data, use the `train_liveness_model.py` script to train the liveness detection model using **SVM**.

- Run the following command:
  ```bash
  python train_liveness_model.py
  ```
  - This will:
    - Load images from the `data/real/` and `data/fake/` folders.
    - Extract facial landmarks using **MediaPipe** and compute additional features like **eye and mouth distances**.
    - Train an **SVM model** to distinguish between real and fake faces.
    - Save the trained model to the `models/` directory as `liveness_model.pkl`.

### 3. **Real-time Liveness Detection (Webcam)**

After training the model, use the `detect_live_face.py` script to run **real-time liveness detection** on your webcam. It will classify the face as **REAL** or **FAKE** based on the trained model.

- Run the following command:
  ```bash
  python detect_live_face.py
  ```
  - This will:
    - Open your webcam, process the frames, and classify the face as either **REAL** or **FAKE**.
    - Display the result with the confidence percentage on the screen.
    - If the model is **uncertain**, it will show "Uncertain."

### 4. **Model Evaluation**

You can evaluate the model performance using the classification metrics (precision, recall, F1-score). After training the model, the script will output the classification report.

---

## 📝 **Additional Notes**

### **Features Extracted**:
- **468 face landmarks** from MediaPipe FaceMesh.
- **Relative distances** between key facial points like eyes and mouth.

### **Improvement Areas**:
- **Class imbalance**: The model might perform better with **more data** or **balanced classes**.
- **Hyperparameter tuning**: You can tune the SVM or try other classifiers (like Random Forest or neural networks).
- **Data augmentation**: For better real face detection, augment the real face dataset (e.g., with transformations like rotation, scaling).

---

## 🛠️ **Future Work**

- **Advanced models**: Integrate deep learning models like CNNs for better feature extraction and more accurate results.
- **Multi-class detection**: Implement uncertainty or "neutral" class for uncertain predictions.
- **Additional Features**: Implement **texture analysis**, **motion analysis**, or use **depth sensors** for more robust liveness detection.
```

---

This `README.md` is properly formatted for display on GitHub, with syntax highlighting for code blocks and clear sectioning for each part of the pipeline. You can copy this content directly into your `README.md` file.

Let me know if you'd like any further adjustments!
