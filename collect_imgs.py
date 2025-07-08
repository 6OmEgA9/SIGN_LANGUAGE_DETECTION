import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 🔡 List of classes: A-Z + 0-9
classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]
dataset_size = 100  # Increase if possible (150–300 per class gives better accuracy)

cap = cv2.VideoCapture(0)

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'[INFO] Collecting data for class "{class_name}"')
    print('[INFO] Show the gesture and press "q" to start collection.')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Class: {class_name} | Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collecting Data', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Collecting Data', frame)
        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        counter += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
