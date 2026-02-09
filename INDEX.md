# Project Index & Navigation Guide

## 📚 Documentation Map

### 🚀 Start Here
1. **[README.md](README.md)** ← Start with this!
   - Project overview
   - Features list
   - Quick start instructions
   - API reference
   - Troubleshooting

### ⚙️ Setup Instructions
2. **[SETUP.md](SETUP.md)**
   - Prerequisites
   - Installation steps
   - Model downloading
   - Configuration
   - Verification

### ⚡ Quick Reference
3. **[QUICK_START.md](QUICK_START.md)**
   - Common tasks
   - Code examples
   - Configuration tips
   - API cheat sheet
   - Keyboard shortcuts

### 📋 Information
4. **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)**
   - What changed
   - New architecture
   - Removed items
   - Code improvements

5. **[COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md)**
   - Task completion status
   - Code quality metrics
   - Performance info
   - Support resources

---

## 📁 File Structure

```
Sign Language Detection/
│
├── 📄 Documentation Files
│   ├── README.md                    # Main guide (START HERE)
│   ├── SETUP.md                     # Installation guide
│   ├── QUICK_START.md               # Quick reference
│   ├── REFACTORING_SUMMARY.md       # What changed
│   ├── COMPLETION_CHECKLIST.md      # Status & checklist
│   └── INDEX.md                     # This file
│
├── 🐍 Application
│   └── main.py                      # Run this to start
│
├── ⚙️ Configuration
│   ├── requirements.txt             # Python dependencies
│   └── src/config.py                # Settings & parameters
│
├── 📦 Source Code (src/)
│   ├── __init__.py
│   ├── config.py                    # Configuration
│   ├── detectors/                   # Detection modules
│   │   ├── __init__.py
│   │   ├── hand_detector.py        # Hand detection
│   │   ├── face_detector.py        # Face detection
│   │   └── gaze_tracker.py         # Gaze tracking
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── visualization.py        # Drawing & display
│       └── file_utils.py           # File management
│
├── 📊 Data Directories
│   ├── models/                      # ML models (download)
│   ├── sample_data/                 # Test images/videos
│   └── data/                        # General data
│
└── 📦 Backup
    └── _old_files_backup/           # Old files (for reference)
```

---

## 🎯 Common Tasks

### I want to...

**Run the application**
→ See: [README.md - Quick Start](README.md#-quick-start)
```bash
python main.py
```

**Install dependencies**
→ See: [SETUP.md - Installation](SETUP.md#step-3-install-dependencies)
```bash
pip install -r requirements.txt
```

**Download models**
→ See: [SETUP.md - Download Models](SETUP.md#4-download-mediapipe-models)

**Customize settings**
→ Edit: `src/config.py`
→ See: [QUICK_START.md - Common Tasks](QUICK_START.md#-common-tasks)

**Use the API directly**
→ See: [README.md - API Reference](README.md#-api-reference)
→ See: [QUICK_START.md - API Examples](QUICK_START.md#-api-quick-examples)

**Troubleshoot issues**
→ See: [SETUP.md - Troubleshooting](SETUP.md#troubleshooting)
→ See: [QUICK_START.md - Troubleshooting](QUICK_START.md#-troubleshooting)

**Understand what changed**
→ See: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)

**Check completion status**
→ See: [COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md)

---

## 🔍 Finding Things

### By Topic

**Installation & Setup**
- [SETUP.md](SETUP.md) - Complete setup guide

**Usage & Examples**
- [README.md](README.md) - Full documentation
- [QUICK_START.md](QUICK_START.md) - Code examples

**Code & Architecture**
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Architecture changes
- `src/` - Source code with inline comments

**Configuration**
- [QUICK_START.md](QUICK_START.md#-common-tasks) - How to customize
- `src/config.py` - All settings

**Troubleshooting**
- [SETUP.md](SETUP.md#troubleshooting) - Setup issues
- [QUICK_START.md](QUICK_START.md#-troubleshooting) - Common issues
- [README.md](README.md#-troubleshooting) - Detailed solutions

---

### By Component

**Hand Detection**
- Module: `src/detectors/hand_detector.py`
- API Doc: [README.md - HandDetector](README.md#handdetector)
- Example: [QUICK_START.md](QUICK_START.md#using-handdetector)

**Face Detection**
- Module: `src/detectors/face_detector.py`
- API Doc: [README.md - FaceDetector](README.md#facedetector)
- Example: [QUICK_START.md](QUICK_START.md#using-facedetector)

**Gaze Tracking**
- Module: `src/detectors/gaze_tracker.py`
- API Doc: [README.md - GazeTracker](README.md#gazetracker)
- Example: [QUICK_START.md](QUICK_START.md#using-gazetracker)

**Main Application**
- File: `main.py`
- Usage: [README.md - Usage](README.md#-usage-examples)

**Configuration**
- File: `src/config.py`
- Customization: [QUICK_START.md](QUICK_START.md#customize-colors)

---

## 📖 Reading Order

### For Quick Start (5 minutes)
1. [QUICK_START.md](QUICK_START.md)
2. Run `python main.py`

### For Complete Understanding (15 minutes)
1. [README.md](README.md) - Overview
2. [QUICK_START.md](QUICK_START.md) - Examples
3. `src/` - Code review

### For Installation (10 minutes)
1. [SETUP.md](SETUP.md)
2. Download models
3. Run application

### For Development (30+ minutes)
1. [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
2. [README.md](README.md) - API Reference
3. `src/detectors/` - Code review
4. Explore examples in [QUICK_START.md](QUICK_START.md)

---

## 🎓 Learning Path

### Beginner
- Read: [README.md](README.md)
- Do: Run `python main.py`
- Try: Adjust settings in [QUICK_START.md](QUICK_START.md)

### Intermediate
- Study: Code in `src/detectors/`
- Learn: API in [README.md](README.md#-api-reference)
- Build: Custom application using the API

### Advanced
- Review: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- Extend: Add new features to detectors
- Integrate: Use in your own projects

---

## 🆘 Need Help?

### Problem: I'm stuck
→ Check [QUICK_START.md](QUICK_START.md#-troubleshooting)

### Problem: Installation issues
→ See [SETUP.md](SETUP.md#troubleshooting)

### Problem: Want to understand the code
→ Review [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)

### Problem: Need API documentation
→ See [README.md](README.md#-api-reference)

### Problem: Want code examples
→ Check [QUICK_START.md](QUICK_START.md#-api-quick-examples)

---

## 📞 Documentation Summary

| Document | Purpose | Read Time |
|----------|---------|-----------|
| README.md | Complete guide | 10 min |
| SETUP.md | Installation | 10 min |
| QUICK_START.md | Quick reference | 5 min |
| REFACTORING_SUMMARY.md | What changed | 5 min |
| COMPLETION_CHECKLIST.md | Status | 5 min |
| INDEX.md | This file | 3 min |

---

## ✨ Key Files You'll Use

1. **main.py** - Run this to start the application
2. **src/config.py** - Edit this to customize settings
3. **src/detectors/** - Study this to understand detection
4. **README.md** - Reference this for API documentation

---

## 🚀 Getting Started Right Now

1. **Read**: [README.md](README.md#-quick-start)
2. **Install**: `pip install -r requirements.txt`
3. **Download**: Models to `models/`
4. **Run**: `python main.py`
5. **Customize**: Edit `src/config.py`

---

## 📚 Helpful Links

- **MediaPipe**: https://developers.google.com/mediapipe
- **OpenCV**: https://docs.opencv.org/
- **Python**: https://docs.python.org/3/

---

**Total Documentation**: ~5 markdown files, 100+ pages of content

**Status**: ✅ Complete and ready to use

**Last Updated**: February 9, 2026
