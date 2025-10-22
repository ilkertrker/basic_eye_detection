#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 23:00:52 2025
@author: ilker
"""
import cv2
import os

# === Step 1: Define the video file path ===
video_path = "/Users/ilker/Desktop/basic_eye_detection/front_camera.mp4"  # ✅ Update if needed

# Check if video file exists
if not os.path.exists(video_path):
    print("Video not found. Check your file path!")
    exit()

# === Step 2: Load Haar Cascade Classifiers ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# === Step 3: Open the video ===
cap = cv2.VideoCapture(video_path)

# === Step 4: Process the video frame by frame ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face, detect eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Eye Detection from Video", frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# === Step 5: Cleanup ===
cap.release()
cv2.destroyAllWindows()
