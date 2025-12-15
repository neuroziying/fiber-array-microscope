import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import time


"""
This file collects different fiber localization strategies
explored during early-stage experiments.

Some functions retain exploratory structures or alternative
pipelines for comparison purposes.
"""


from methods import (
    method_hough_circle,
    method_std_pipeline,
    method_std_weighted_pipeline
)

videos = glob.glob(r'C:\Users\Lenovo\Desktop\pycodes\box_0425_200-1000.avi') #demo

if not videos:
    print("no video found")

for video in videos:
    print(f"processing video: {video}")

    video_path = os.path.abspath(video)
    capture = cv2.VideoCapture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"fps={fps}, size=({width},{height}), frames={total_frames}")
    spatial_decompose = 2
    std_per5 = []   

   
    # Method 1: raw frame Hough
    method_hough_circle(capture, width, height, spatial_decompose)

    # Method 2: STD pipeline (primary)
    # method_std_pipeline(capture, width, height, spatial_decompose)

    # Method 3: STD weighted
    # method_std_weighted_pipeline(capture, width, height, std_per5, spatial_decompose)

    capture.release()
    cv2.destroyAllWindows()
