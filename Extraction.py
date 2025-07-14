import os
import cv2
import numpy as np
import mediapipe as mp
SEQUENCE_LENGTH = 30
STEP = 5 #Sliding window for better generalization, because of less data
DATASET_PATH = 'dataset'
OUTPUT_PATH = 'MP_Data'
os.makedirs(OUTPUT_PATH, exist_ok=True)
# Mediapipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
#holistic because of full body landmarks detection
def extract_keypoints(results):
    keypoints = []
    # Pose - 33 points
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0]*33*4)

    # Face - 468 points
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*468*3)

    # Left hand - 21 points
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*21*3)

    # Right hand - 21 points
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*21*3)

    return np.array(keypoints)

# Main loop
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    video_file = os.path.join(label_path, f"{label}.mp4")
    out_dir = os.path.join(OUTPUT_PATH, label)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    frames = []
    success = True

    print(f"\nProcessing: {label}")

    while success:
        success, frame = cap.read()
        if not success:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        keypoints = extract_keypoints(results)
        frames.append(keypoints)
    cap.release()

    total_sequences = 0
    for start in range(0, len(frames) - SEQUENCE_LENGTH + 1, STEP):
        sequence = frames[start:start + SEQUENCE_LENGTH]
        if len(sequence) == SEQUENCE_LENGTH:
            npy_path = os.path.join(out_dir, f"{str(total_sequences).zfill(3)}.npy")
            np.save(npy_path, sequence)
            total_sequences += 1

print("\nExtraction Completed")
