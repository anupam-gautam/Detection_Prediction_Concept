# Active Laptop Usage & Attention Inference System

## 1. Project Overview
This project implements a real-time, multi-modal **Active Usage and Attention Inference System** for laptop users. It continuously monitors user engagement by fusing signals from:
*   **Input Devices**: Keyboard and mouse activity.
*   **Computer Vision**: Face presence and eye gaze tracking via a webcam.

The system determines whether the user is:
*   **Active**: Intentionally using the laptop (Interactive or Focused Reading).
*   **Passive**: Present but disengaged (e.g., looking away).
*   **Inactive**: Absent or completely disengaged.

It is designed for robustness, dealing with missing signals (e.g., temporary camera failure) and prioritizing input activity (typing overrides lack of face detection).

---

## 2. Key Features
*   **Multi-Modal Fusion**: Combines input input (low latency) with computer vision (context).
*   **Robust Inference Logic**:
    *   **Input Dominance**: Typing/mousing is always considered "Active" regardless of gaze.
    *   **Visual Validation**: If input is idle, face + gaze on screen confirms "Active (Reading)".
*   **Temporal Smoothing**: Uses a history window to prevent flickering states.
*   **Session Analytics**: Automatically records session data and generates detailed reports (`session_report.txt`) and raw logs (`session_raw_log.txt`) upon exit.
*   **Privacy-Aware**: Runs locally; camera feed can be hidden/disabled via configuration.

---

## 3. Usage Guide

### Prerequisites
*   Python 3.8+
*   Webcam
*   Dependencies: `opencv-python`, `mediapipe`, `pynput`, `numpy`

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install opencv-python mediapipe pynput numpy
    ```

### Running the System
Execute the main script:
```bash
python main.py
```

*   **Exit**: Press `Esc` in the camera window or `Ctrl+C` in the console.
*   **Headless Mode**: Set `CAMERA_PREVIEW_ENABLED = False` in `main.py` to run without a video window.

### Output
*   **Real-time**:
    *   **Console**: Status updates / errors.
    *   **Window**: Live video feed with overlay showing State, Confidence, and Reasoning.
*   **Post-Session**:
    *   `session_report.txt`: Summary of time spent in Active/Passive/Inactive states.
    *   `session_raw_log.txt`: CSV-style log of every inference step.

---

## 4. System Architecture & Workflow

### A. High-Level Flow
1.  **Capture**: `InputMonitor` tracks HID events; `VisionDetector` captures frames and runs AI models.
2.  **Extract**: Raw signals (idle time, face landmarks, gaze vector) are extracted.
3.  **normalize**: Raw signals are converted to 0.0-1.0 scores.
4.  **Fuse & Infer**: `InferenceEngine` applies logic rules to determine the state.
5.  **Smooth**: State probabilities are averaged over time.
6.  **Act**: UI is updated, and metrics are logged.

### B. Directory Structure
```
.
├── main.py                  # Entry point. Orchestrates the loop.
├── models/                  # ML models (MediaPipe Face Landmarker)
├── src/
│   ├── features/
│   │   ├── input_monitor.py   # Tracks keyboard/mouse idle time via pynput.
│   │   └── vision_detector.py # MediaPipe wrapper for face & gaze detection.
│   └── fusion/
│       └── engine.py        # CORE LOGIC: InferenceEngine class.
└── tests/                   # Unit verification tests.
```

---

## 5. Detailed Concept & Algorithms

### A. Input Monitor (`src/features/input_monitor.py`)
*   **Concept**: Uses system hooks (pynput) to detect `on_move`, `on_click`, `on_press` events.
*   **Metric**: `idle_time` (seconds since last event).
*   **Logic**: If `idle_time < threshold`, user is interactively active.

### B. Vision Detector (`src/features/vision_detector.py`)
*   **Concept**: Uses Google MediaPipe Face Landmarker.
*   **Metrics**:
    *   `is_face_present`: Binary detection.
    *   `gaze_ratio`: Horizontal iris position relative to eye corners (0.0=Left, 1.0=Right, 0.5=Center).
    *   `is_looking_at_screen`: Thresholded `gaze_ratio` (0.46 - 0.54).

### C. Inference Engine (`src/fusion/engine.py`)
This is the "brain" of the system.

#### 1. Signal Normalization
*   **Input Score**: Exponential decay based on idle time.
    *   `math.exp(-0.5 * idle_time)` -> High score immediately, decays to ~0 near 5s.
*   **Gaze Score**: Linear mapping of deviation from center.
    *   Closer to 0.5 (center) -> Higher score (1.0).

#### 2. Decision Logic (The "Why")
The engine prioritizes signals based on reliability:
1.  **Level 1: Input Dominance**
    *   IF `input_score > 0.6` -> **ACTIVE (Interactive)**
    *   *Reasoning*: You cannot type without being there. Gaze is irrelevant (blind typing/glancing at notes).
2.  **Level 2: Visual Validation** (When Input is Idle)
    *   IF `Face` AND `Gaze on Screen` -> **ACTIVE (Non-Interactive)**
    *   *Reasoning*: User is reading or watching content.
3.  **Level 3: Presence but Distraction**
    *   IF `Face` BUT `Gaze Away` -> **PASSIVE**
    *   *Reasoning*: User is physically present but attention is elsewhere.
4.  **Level 4: Absence**
    *   IF `No Face` AND `No Input` -> **INACTIVE**
    *   *Reasoning*: No evidence of user presence.

#### 3. Confidence Scoring
A score (0.0 - 1.0) is calculated to represent certainty:
*   **High (0.9+)**: Strong input activity.
*   **Medium (0.6-0.8)**: Visual signals only (camera noise can affect this).
*   **Low (<0.5)**: Missing signals or conflicting data.

#### 4. Temporal Smoothing
*   A `deque` (sliding window) stores the last N probabilities.
*   The final output probability is the **average** of this window.
*   **Benefit**: Prevents the status from jumping "Active" -> "Inactive" -> "Active" instantly if the camera blips for 1 frame.

---

## 6. Analytics & Reporting
When the session ends, the `InferenceEngine` exports data.
*   **Duration Tracking**: Accumulates `dt` (time delta) for each state (Active, Passive, Inactive).
*   **Report Generation**: text file summary of % time spent in each state and dominance analysis.

---

## 7. Configuration
Key settings can be modified in `src/config.py` or defined constants in `main.py`:
*   `CAMERA_PREVIEW_ENABLED`: Toggle video window.
*   `smoothing_window`: Size of the average window (higher = more stable but more lag).
*   `threshold_seconds` (in InputMonitor): Time before input is considered "Idle".