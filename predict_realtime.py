import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

MODEL_PATH = "best_model.h5"
LABELS = ['Cover', 'Pull', 'Straight', 'Sweep']
SEQUENCE_LENGTH = 30
PREPARE_FRAMES = 30 #Gives 30 Frames time to take stance, did'nt used time
CONFIDENCE_THRESHOLD = 0.80  # For Confidence >80% accepted only

model = load_model(MODEL_PATH)
print("Model loading done Successfully")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0]*33*4)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*468*3)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*21*3)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*21*3)
    return np.array(keypoints)

cap = cv2.VideoCapture(0)
sequence = []
predicted_label = ""
collecting = False
prepare_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    keypoints = extract_keypoints(results)

    if collecting:
        if prepare_count < PREPARE_FRAMES:
            prepare_count += 1
            cv2.putText(image, f"Get Ready... {PREPARE_FRAMES - prepare_count}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            sequence.append(keypoints)
            cv2.putText(image, f"Capturing: {len(sequence)}/{SEQUENCE_LENGTH}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]
                max_index = np.argmax(prediction)
                confidence = prediction[max_index]
                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_label = f"{LABELS[max_index]} ({confidence*100:.1f}%)"
                else:
                    predicted_label = f"No confident shot"
                collecting = False
                sequence = []
                prepare_count = 0

    # Display current predicted label
    cv2.rectangle(image, (0, 0), (380, 50), (0, 0, 0), -1)
    cv2.putText(image, f"Shot: {predicted_label}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Instructions on the screen
    cv2.putText(image, "Press 's' to start", (10, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("PlayCam Realtime", image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not collecting:
        collecting = True
        predicted_label = ""
        prepare_count = 0
        sequence = []

cap.release()
cv2.destroyAllWindows()
