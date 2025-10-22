#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye Direction Tracker + Data Recorder
Created on Oct 23, 2025
@author: ilker
"""
import cv2
import os
import csv
import time
import numpy as np

# === Step 1: Video file path ===
video_path = "/Users/ilker/Desktop/basic_eye_detection/front_camera.mp4"
output_csv = "/Users/ilker/Desktop/basic_eye_detection/eye_tracking_data.csv"

if not os.path.exists(video_path):
    print("Video not found. Check your file path!")
    exit()

# === Step 2: Load Haar cascades ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# === Step 3: Open video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# === Step 4: Prepare CSV file ===
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Frame", "Eye_X", "Eye_Y", "Direction"])

    frame_num = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        frame_num += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Eye region for pupil detection
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)
                threshold = cv2.medianBlur(threshold, 5)

                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    (cx, cy), _ = cv2.minEnclosingCircle(contour)
                    cv2.circle(roi_color, (ex + int(cx), ey + int(cy)), 3, (0, 0, 255), -1)

                    # Determine eye direction
                    if cx < ew / 3:
                        direction = "Left"
                    elif cx > 2 * ew / 3:
                        direction = "Right"
                    else:
                        direction = "Center"

                    timestamp = round(time.time() - start_time, 2)
                    writer.writerow([timestamp, frame_num, int(ex + cx), int(ey + cy), direction])

                    cv2.putText(frame, direction, (x + ex, y + ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Eye Direction Tracker", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Eye tracking completed. Data saved to {output_csv}")
