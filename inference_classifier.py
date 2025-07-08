# inference_classifier.py
import pickle
import cv2
import mediapipe as mp
import numpy as np
from typing import Any, cast
import sys

# ── Load the trained model and class names ───────────────────────────────
with open("model.p", "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]
class_names = model_dict["classes"]  # e.g., ['A', ..., 'Z', '0', ..., '9']

# ── Initialize camera safely ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)  # try 1 or 2 if 0 doesn't work (i have default laptop webcam i.e- 0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    sys.exit()

# ── Initialize MediaPipe Hands ───────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            continue

        H, W = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = cast(Any, hands.process(frame_rgb))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

            # Extract features
            x_, y_, data_aux = [], [], []

            for lm in results.multi_hand_landmarks[0].landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in results.multi_hand_landmarks[0].landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:  # Only proceed if valid input
                # Bounding box
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

                # Prediction
                predicted_idx = model.predict([np.asarray(data_aux)])[0]
                predicted_char = class_names[predicted_idx]

                # Display result
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(frame, predicted_char, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
                            cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
