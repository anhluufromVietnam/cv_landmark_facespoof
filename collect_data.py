import cv2
import os
import uuid

mode = input("Enter mode (real/fake): ").strip().lower()
save_path = f"data/{mode}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press 'c' to capture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(save_path, filename), frame)
        print(f"Saved: {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

