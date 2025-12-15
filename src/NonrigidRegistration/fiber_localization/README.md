# STD-based Fiber Localization Pipeline
This repository contains an std-basedvideo processing pipeline developed for automatic fiber localization and calibration in experimental imaging data. The method was designed to handle **non-rigid shifts**(caused by freely behaving mice), frame-to-frame noise, and low-contrast conditions that often cause conventional frame-wise detection methods to fail.

## Overview
The core idea of this pipeline is to exploit **temporal intensity fluctuations** across video frames:

* Static background regions exhibit low temporal variance
* Fiber cross-sections and moving structures exhibit higher temporal variance

By computing the **temporal standard deviation (STD)** over short frame windows and combining it with motion-aware masking, the fiber region can be robustly localized even under unstable imaging conditions.

## Pipeline Summary
The finalized pipeline follows these steps:
1. **Video loading and grayscale conversion**
2. **Spatial downsampling** (optional, for speed)
3. **Temporal binning** of frames (`bin_size`)
4. **Frame difference + STD analysis** to detect non-rigid shifts ("shake")
5. **Dynamic mask selection** based on detected stable frames
6. **STD-weighted ROI construction**
7. **Hough circle detection** on the enhanced ROI
8. **Result visualization and video export**

Two output videos are generated:

* `std_output.avi`: STD-enhanced grayscale frames
* `circles_output.avi`: Circle detection results overlaid on ROI

## Dependencies
* Python 3.x
* NumPy
* OpenCV (cv2)
* Matplotlib (optional, for visualization)
* scikit-learn (currently unused, reserved for future extensions)

## Notes
* Internal code comments are primarily written in **Chinese**, as this codebase was originally developed for in-lab use.
* A fully documented English version may be added in future updates.
* Video paths are currently hard-coded and should be modified by the user.
* You can find demo videos in the /demo_data folder, such as "box_0425_200_1000.avi".
