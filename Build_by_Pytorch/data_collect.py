import cv2
import numpy as np
import os
import mediapipe as mp
import csv
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(65, 70, 80), thickness=3, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(123, 234, 123), thickness=3, circle_radius=2))


def extract_keypoints(results):
    if results.pose_landmarks:
        pose1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose1 = np.zeros(33 * 4)
    return pose1


cap = cv2.VideoCapture('E:/fall_dataset/fall_dataset/stand.mp4')

with mp_pose.Pose(static_image_mode=True, enable_segmentation=True, min_detection_confidence=0.3, model_complexity=2) as pose:
    with open("MY_Data/train98_2.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        while True:
            success, frame = cap.read()
            img = cv2.resize(frame, (720, 720))
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.putText(img, 'STARTING COLLECTION~~~~', (0, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (155, 0, 255), 4, cv2.LINE_AA)
            results = pose.process(imgRGB)
            draw_styled_landmarks(img, results)
            cv2.imshow('OpenCV Feed', img)
            keypoints = extract_keypoints(results)
            writer.writerow(keypoints)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()