# Gaze Tracking Implementation - File Index

## 📁 Project Structure

```
d:\Projects\Sign Language Detection\
│
├── 🔧 CORE IMPLEMENTATION
│   ├── GazeTracker.py                    [450+ lines] Main gaze tracker class
│   ├── config.py                         [300+ lines] Configuration parameters
│   └── Detection Project/
│       └── Initial.py                    [108 lines]  Integrated version
│
├── 🧪 TEST & EXAMPLES
│   ├── GazeTracker_Test.py               [120+ lines] Simple test script
│   └── GazeTracker_Examples.py           [350+ lines] Advanced examples
│
├── 📚 DOCUMENTATION
│   ├── README_GAZE.md                    [300+ lines] Feature overview
│   ├── QUICKSTART.md                     [400+ lines] Getting started
│   ├── IMPLEMENTATION_GUIDE.md           [500+ lines] Technical guide
│   ├── PROJECT_SUMMARY.md                [350+ lines] Project overview
│   ├── FILE_INDEX.md                     [This file]  File descriptions
│   └── readme.txt                        [Original]   Original readme
│
├── 🎯 MODELS & DATA
│   ├── face_landmarker.task              MediaPipe face model
│   ├── hand_landmarker.task              MediaPipe hand model
│   └── data/                             Dataset folder
│
└── 📝 OTHER
    ├── blaze_face_short_range.tflite     TensorFlow model
    ├── CameraInput.py                    [Original]   Camera utilities
    ├── create_dataset.py                 [Original]   Dataset creation
    ├── HandDetection_MediaPipe.py        [Original]   Hand detection
    └── ThumbDetectionRealtime.py         [Original]   Thumb detection
```

---

## 📄 File Descriptions

### 🔴 CORE IMPLEMENTATION FILES

#### `GazeTracker.py` (450+ lines)
**Purpose**: Main gaze tracking implementation

**Contains**:
- `GazeTracker` class - Main API
- `estimate_head_pose()` - Head pose estimation
- `track_gaze()` - Main tracking pipeline
- `compute_gaze_direction()` - Gaze computation
- `draw_gaze_visualization()` - Visualization
- Supporting helper methods

**Key Features**:
- 3D face model definition
- 6 key face landmarks extraction
- Head pose computation (solvePnP)
- Pupil 3D projection (estimateAffine3D)
- Gaze direction calculation
- Head movement compensation
- Real-time visualization

**Usage**:
```python
from GazeTracker import GazeTracker
tracker = GazeTracker()
gaze_2d, info = tracker.track_gaze(frame, face_landmarks, w, h)
```

---

#### `config.py` (300+ lines)
**Purpose**: Centralized configuration management

**Contains**:
- Camera parameters
- Gaze tracking parameters
- Visualization settings
- Performance options
- Preset configurations

**Key Features**:
- Easy parameter adjustment
- Camera matrix utilities
- Configuration validation
- Preset configurations (fast/balanced/quality)
- Debug configuration
- AOI (Area of Interest) settings

**Usage**:
```python
from config import DISTANCE_MAGIC_NUMBER, EYE_SELECTION
# or
import config
tracker = GazeTracker(camera_matrix=config.get_camera_matrix())
```

---

#### `Detection Project/Initial.py` (108 lines)
**Purpose**: Integration of gaze tracking with hand & face detection

**Modified to Include**:
- Import and initialize GazeTracker
- Call `track_gaze()` for each detected face
- Visualization of gaze points
- Status display

**Status**: ✅ Fully integrated

---

### 🟡 TEST & EXAMPLE FILES

#### `GazeTracker_Test.py` (120+ lines)
**Purpose**: Simple standalone test script

**Features**:
- Basic gaze tracking demo
- Eye selection switching ('s' key)
- FPS and success rate monitoring
- Real-time visualization
- Minimal dependencies

**Usage**:
```bash
python GazeTracker_Test.py
```

**Controls**:
- 'q' - Quit
- 's' - Switch eye selection

---

#### `GazeTracker_Examples.py` (350+ lines)
**Purpose**: Advanced examples and demonstrations

**Includes**:
1. **Example 1**: Basic gaze tracking
2. **Example 2**: Area of Interest (AOI) tracking
3. **Example 3**: Left vs Right vs Both eyes comparison

**GazeAnalyzer Class**:
- AOI tracking functionality
- Gaze point history
- Trail visualization
- Statistics collection

**Usage**:
```bash
python GazeTracker_Examples.py
```

**Features**:
- Interactive menu selection
- Real-time statistics
- Visual AOI regions
- Comparative analysis

---

### 🟢 DOCUMENTATION FILES

#### `README_GAZE.md` (300+ lines)
**Purpose**: Feature overview and methodology explanation

**Contents**:
- Project overview
- Key concepts explained
- 3D face model description
- Algorithm explanations
- Class and method reference
- Basic usage examples
- Accuracy improvements
- References and resources

**Audience**: Everyone (beginners to advanced)

---

#### `QUICKSTART.md` (400+ lines)
**Purpose**: Practical getting started guide

**Contents**:
- What was implemented
- Quick start instructions
- How it works (5-step pipeline)
- Key classes and methods
- Parameters you can tune
- Visualization guide
- Troubleshooting
- Camera calibration
- Performance tips
- Advanced usage
- Next steps

**Audience**: Users and developers

---

#### `IMPLEMENTATION_GUIDE.md` (500+ lines)
**Purpose**: Technical deep dive with mathematics

**Contents**:
- Architecture overview with diagrams
- Component details
- Mathematical foundations
- Pinhole camera model explanation
- Rotation vectors (Rodrigues formula)
- Affine transformations
- Data flow examples
- Performance characteristics
- Integration guidelines
- Tuning guide
- Advanced customization
- Validation techniques
- Future improvements
- References
- Support resources

**Audience**: Developers and researchers

---

#### `PROJECT_SUMMARY.md` (350+ lines)
**Purpose**: Project overview and status

**Contents**:
- What was delivered
- Key features checklist
- Architecture diagram
- How to use (3 methods)
- File structure
- Technical details
- Performance metrics
- Learning resources
- Customization options
- Key innovation points
- Troubleshooting table
- Use cases
- Next steps
- Implementation checklist
- Project status

**Audience**: Project managers and stakeholders

---

#### `FILE_INDEX.md` (This file)
**Purpose**: Navigate all project files

**Contents**:
- File structure overview
- Detailed file descriptions
- File purposes and contents
- Usage examples
- Quick reference guide

**Audience**: Everyone

---

### 🟣 ORIGINAL FILES (Preserved)

#### `readme.txt`
- Original project readme
- Preserved for reference

#### `CameraInput.py`
- Original camera input utilities
- Preserved for reference

#### `HandDetection_MediaPipe.py`
- Original hand detection code
- Preserved for reference

#### `ThumbDetectionRealtime.py`
- Original thumb detection code
- Preserved for reference

#### `create_dataset.py`
- Original dataset creation script
- Preserved for reference

#### `blaze_face_short_range.tflite`
- TensorFlow Lite model
- Preserved for reference

#### `data/`
- Dataset folder
- Preserved for reference

---

## 🚀 Quick Navigation Guide

### I want to...

#### ✅ Get started quickly
- Start with: `QUICKSTART.md`
- Run: `GazeTracker_Test.py`
- Integrate: `GazeTracker.py`

#### ✅ Understand the theory
- Read: `README_GAZE.md`
- Deep dive: `IMPLEMENTATION_GUIDE.md`
- Code: `GazeTracker.py` (with comments)

#### ✅ See advanced examples
- Run: `GazeTracker_Examples.py`
- Check: Menu-driven options
- Explore: Different eye selections

#### ✅ Integrate into my code
- Import: `from GazeTracker import GazeTracker`
- Configure: `import config`
- Use: See examples in test scripts

#### ✅ Tune parameters
- Edit: `config.py`
- Read: `QUICKSTART.md` (Parameters section)
- Test: `GazeTracker_Test.py`

#### ✅ Debug issues
- Check: `QUICKSTART.md` (Troubleshooting)
- Review: `debug_info` output
- Enable: `config.py` (DEBUG_MODE)

#### ✅ Understand the code
- Start: `README_GAZE.md`
- Then: `IMPLEMENTATION_GUIDE.md`
- Finally: Source code comments

---

## 📊 File Statistics

| File | Lines | Purpose |
|------|-------|---------|
| GazeTracker.py | 450+ | Core implementation |
| config.py | 300+ | Configuration |
| GazeTracker_Test.py | 120+ | Testing |
| GazeTracker_Examples.py | 350+ | Examples |
| README_GAZE.md | 300+ | Overview |
| QUICKSTART.md | 400+ | Getting started |
| IMPLEMENTATION_GUIDE.md | 500+ | Technical guide |
| PROJECT_SUMMARY.md | 350+ | Project info |
| FILE_INDEX.md | 200+ | Navigation |
| **TOTAL** | **3,000+** | Complete project |

---

## 🔗 Cross-References

### Reading Order (Recommended)
1. This file (FILE_INDEX.md) - Get oriented
2. README_GAZE.md - Understand features
3. QUICKSTART.md - Get started
4. Run GazeTracker_Test.py - See it working
5. IMPLEMENTATION_GUIDE.md - Understand internals
6. Review GazeTracker.py - Study code

### By Audience

**🔵 Beginners**
1. FILE_INDEX.md (this file)
2. QUICKSTART.md
3. GazeTracker_Test.py
4. README_GAZE.md

**🟡 Developers**
1. QUICKSTART.md
2. GazeTracker.py (code)
3. IMPLEMENTATION_GUIDE.md
4. config.py

**🔴 Researchers**
1. README_GAZE.md
2. IMPLEMENTATION_GUIDE.md
3. GazeTracker.py (source)
4. Academic references in docs

**🟣 DevOps/Integration**
1. QUICKSTART.md
2. config.py
3. GazeTracker.py
4. Initial.py (integration example)

---

## 🎯 File Dependencies

```
GazeTracker.py
    ├── Requires: numpy, cv2, mediapipe
    └── Used by: All other Python files

config.py
    ├── Requires: numpy
    └── Optional: Used by any file for configuration

GazeTracker_Test.py
    ├── Requires: GazeTracker.py, config.py, mediapipe
    └── Standalone: Can run independently

GazeTracker_Examples.py
    ├── Requires: GazeTracker.py, mediapipe
    └── Optional: Advanced features

Initial.py
    ├── Requires: GazeTracker.py, mediapipe
    └── Integration: Full pipeline

Documentation
    ├── No dependencies
    └── Reference only
```

---

## 💾 Storage Organization

### By Category

**📥 Input**
- face_landmarker.task
- hand_landmarker.task
- blaze_face_short_range.tflite
- Camera (webcam)
- Data folder

**⚙️ Processing**
- GazeTracker.py
- config.py
- Initial.py

**📤 Output**
- Video display
- Gaze points
- Debug information

**📚 Reference**
- All .md files
- Original Python files

---

## 🔐 Important Notes

### File Modification Status
- ✅ GazeTracker.py - **NEW** (complete implementation)
- ✅ config.py - **NEW** (configuration file)
- ✅ Detection Project/Initial.py - **MODIFIED** (gaze integration)
- ✅ GazeTracker_Test.py - **NEW** (test script)
- ✅ GazeTracker_Examples.py - **NEW** (examples)
- ℹ️ All .md files - **NEW** (documentation)
- ℹ️ Original files - **PRESERVED** (unchanged)

### Backward Compatibility
- ✅ Original code preserved
- ✅ New code in separate files
- ✅ Easy to revert if needed
- ✅ No breaking changes

---

## 🎓 Learning Resources by File

### Computer Vision Concepts
- `IMPLEMENTATION_GUIDE.md` - Pinhole camera, 3D-2D projection
- `GazeTracker.py` - Algorithm implementation
- `README_GAZE.md` - Conceptual overview

### Algorithm Details
- `IMPLEMENTATION_GUIDE.md` - Mathematics and formulas
- `GazeTracker.py` - Source code with comments
- `config.py` - Parameter settings

### Practical Usage
- `QUICKSTART.md` - Getting started
- `GazeTracker_Test.py` - Runnable example
- `GazeTracker_Examples.py` - Advanced examples

### Integration Patterns
- `Initial.py` - Real-world integration
- `config.py` - Configuration management
- `QUICKSTART.md` - Integration guide

---

## ✅ Checklist for Getting Started

- [ ] Read FILE_INDEX.md (this file)
- [ ] Read README_GAZE.md
- [ ] Read QUICKSTART.md
- [ ] Run GazeTracker_Test.py
- [ ] Review GazeTracker.py code
- [ ] Try GazeTracker_Examples.py
- [ ] Integrate into your project
- [ ] Adjust config.py as needed
- [ ] Read IMPLEMENTATION_GUIDE.md for deeper understanding
- [ ] Customize for your use case

---

## 📞 Need Help?

1. **Getting Started?** → Read QUICKSTART.md
2. **Technical Questions?** → Check IMPLEMENTATION_GUIDE.md
3. **How to Use?** → See GazeTracker_Test.py
4. **Code Issues?** → Review inline comments in GazeTracker.py
5. **Configuration?** → Edit config.py and review comments
6. **Advanced Usage?** → Run GazeTracker_Examples.py

---

**Last Updated**: February 8, 2026
**Project Status**: ✅ Complete
**Documentation Status**: ✅ Comprehensive

🚀 **Happy coding!** 🚀
