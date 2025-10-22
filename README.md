# Basic Eye Detection & Tracking

This project is a simple Python-based **eye tracking system** that processes a video recorded from a front camera, detects faces and eyes, tracks pupil positions, determines eye direction, and saves the data for analysis. Additionally, it includes a script to **analyze and visualize eye movement patterns** using the generated CSV.

---

## Project Structure

basic_eye_detection/
│
├─ eye_tracker_video.py # Main script to track eyes and save CSV data
├─ eye_tracking_analysis.py # Script to analyze and visualize eye tracking CSV
├─ eye_tracking_data.csv # Output CSV file with tracked positions & directions
├─ README.md # This file
└─ .gitignore # Optional: ignore temporary files

markdown


---

## Features

### Eye Tracker
- Detects **faces** and **eyes** using OpenCV Haar cascades.
- Tracks **pupil positions** in each detected eye.
- Estimates **eye direction**: Left, Center, Right.
- Saves the following information to a CSV file:
  - Timestamp (seconds)
  - Frame number
  - Eye X/Y coordinates
  - Eye direction

### Data Analysis
- Reads the generated CSV file.
- Visualizes **eye direction over time**.
- Creates a **heatmap of pupil positions**.
- Computes **statistics** like the percentage of time looking left, right, or center.

---

## Requirements

- Python 3.8+ (Recommended: 3.10)
- OpenCV (`opencv-python`)
- Pandas
- Matplotlib
- NumPy

Install the required packages:

```bash
pip install opencv-python pandas matplotlib numpy
Note for Mac users: If you encounter NumPy conflicts in Anaconda, create a new environment:

bash

conda create -n eye_env python=3.10 numpy=1.24 pandas matplotlib opencv
conda activate eye_env
Usage
1. Eye Tracking
Edit eye_tracker_video.py:

python

video_path = "/Users/ilker/Desktop/basic_eye_detection/front_camera.mp4"
output_csv = "/Users/ilker/Desktop/basic_eye_detection/eye_tracking_data.csv"
Run the script:

bash

python eye_tracker_video.py
Press q to quit early.

The video will show rectangles around faces and eyes, along with the eye direction.

The data is saved automatically to eye_tracking_data.csv.

2. Analysis
Edit eye_tracking_analysis.py:

python

df = pd.read_csv("/Users/ilker/Desktop/basic_eye_detection/eye_tracking_data.csv")
Run the script:

bash

python eye_tracking_analysis.py
Outputs:

Bar chart or percentage statistics of left/center/right eye direction.

Time series plot showing eye movement over video duration.

Heatmap of pupil positions.

Example Output
CSV File
Timestamp	Frame	Eye_X	Eye_Y	Direction
0.25	1	210	180	Left
0.28	1	300	185	Center

Visualization
Eye direction timeline over video.

Heatmap showing where the eyes looked most frequently.

Notes
Works with pre-recorded front camera videos, not live webcam streams.

Small project intended to demonstrate Python automation, OpenCV, and data analysis.

Can be extended to:

Track both eyes individually

Detect blinks

Overlay heatmap directly on video

Export video with annotated rectangles
