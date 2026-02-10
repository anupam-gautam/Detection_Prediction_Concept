🖥️ On-Screen Time Detection System
Overview

This project implements an On-Screen Time Detection System that determines whether a user is actively engaged with their screen.

The system combines:

Eye gaze estimation

Face detection

Keyboard and mouse activity

By fusing these signals, the system avoids common false negatives (e.g., typing while looking down) while remaining responsive and efficient in real time.

System Architecture

The system consists of three main components:

Feature Extraction – Vision and input signals

Feature Fusion – Decision logic with temporal smoothing

Application Layer – Real-time execution and visualization

Feature Extraction (src/features/)
VisionDetector

Purpose
Handles all camera-based detection using MediaPipe FaceLandmarker.

Outputs

is_face_present

True if a face is detected in the frame

gaze_ratio

Normalized horizontal gaze direction (0.0 – 1.0)

is_looking_at_screen

True if the gaze ratio is within the calibrated range (0.46 – 0.54)

Key Improvements

Face detection and gaze estimation are combined into a single pipeline

Reduces redundant computation and simplifies downstream logic

InputMonitor

Purpose
Tracks keyboard and mouse activity in the background.

Outputs

is_active(threshold=5)

Returns True if any input occurred within the last threshold seconds

get_idle_time()

Returns the number of seconds since the last detected input

Feature Fusion (src/fusion/)
FusionEngine

Purpose
Aggregates vision and input signals to determine the final ON SCREEN / OFF SCREEN state.

Decision Logic

The user is considered ON SCREEN if any of the following conditions are met:

The user is looking at the screen

A face is detected and recent input activity is present

(Handles cases where the user is typing or looking down at notes)

Temporal Smoothing

Uses a 5-frame rolling history to prevent flickering or unstable state changes

Main Application (main.py)

Description

main.py runs the full system:

Initializes all detectors

Runs the fusion loop

Displays the camera feed with a real-time engagement status overlay

How to Run

Activate your virtual environment (if applicable)

Run the main script:

python main.py

Controls

Esc — Exit the application

Verification Results

The system was verified by running main.py end-to-end:

InputMonitor successfully attached keyboard and mouse listeners

VisionDetector loaded the MediaPipe model and initialized the camera

The main loop processed frames and updated engagement state in real time

All components initialized and operated correctly.