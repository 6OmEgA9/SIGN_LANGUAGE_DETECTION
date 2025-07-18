import os
import pickle
import mediapipe as mp
import cv2
from typing import Any, cast
from sklearn.preprocessing import LabelEncoder

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for dir_ in sorted(os.listdir(DATA_DIR)):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_, y_ = [], []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = cast(Any, hands.process(img_rgb))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Encode string labels to numeric
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Save everything
with open('data.pickle', 'wb') as f:
    pickle.dump({
        'data': data,
        'labels': labels_encoded,
        'label_encoder': label_encoder.classes_.tolist()
    }, f)

print(f"[INFO] Saved dataset with {len(data)} samples and {len(set(labels))} classes.")
