# Automatic Fiber Localization and Calibration

This folder contains a MATLAB-based pipeline for automatic fiber
localization and calibration in fiber-array microscopy experiments.

## Overview
The pipeline processes raw imaging data through background correction,
ROI confirmation, and multi-stage calibration, producing calibrated
fiber positions and summary outputs.

## Demo Data
The folders named `36-1-*` contain example input data and intermediate
results for a single demo run.

## Main Scripts
- `main_processing.m`: entry point of the pipeline
- `setup_folders.m`: initializes folder structure
- `get_processing_parameters.m`: loads and validates parameters
- `confirm_roi_parameters.m`: interactive ROI confirmation
- `FC_1_module.m` – `FC_4_module.m`: core calibration modules

## Notes
Internal code comments are primarily in Chinese, as this codebase was
originally developed for in-lab use. An English version of detailed
documentation will be added in future updates.
