#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 00:38:45 2025
@author: ilker
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("/Users/ilker/Desktop/basic_eye_detection/eye_tracking_data.csv")


# Direction statistics
print(df['Direction'].value_counts(normalize=True) * 100)

# Map direction to numbers
direction_map = {'Left': -1, 'Center': 0, 'Right': 1}
df['Dir_Num'] = df['Direction'].map(direction_map)

# Plot eye direction over time
plt.figure(figsize=(12,4))
plt.plot(df['Timestamp'], df['Dir_Num'], marker='o', linestyle='-')
plt.yticks([-1,0,1], ['Left','Center','Right'])
plt.xlabel('Time (s)')
plt.ylabel('Eye Direction')
plt.title('Eye Direction Over Time')
plt.show()

# Heatmap of eye positions
plt.figure(figsize=(6,6))
plt.hexbin(df['Eye_X'], df['Eye_Y'], gridsize=50, cmap='Reds')
plt.gca().invert_yaxis()  # Images origin at top-left
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Pupil Position Heatmap')
plt.colorbar(label='Frequency')
plt.show()
