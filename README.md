# 😴 Driver Drowsiness Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00897B?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.x-FF6B6B?style=for-the-badge&logo=python&logoColor=white)

![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-MacOS%20%7C%20Windows%20%7C%20Linux-lightgrey?style=for-the-badge)
![GPU](https://img.shields.io/badge/GPU-Not%20Required-orange?style=for-the-badge)

**A real-time AI-powered system that detects driver drowsiness using facial landmark analysis and triggers instant alerts to prevent road accidents.**

</div>

---

## 🚨 The Problem

<div align="center">

> **Road accidents due to driver drowsiness cause thousands of deaths every year.**
> Traditional solutions are expensive, invasive, or require special hardware.
> This system works with just a standard webcam — no extra hardware needed.

</div>

---

## ✨ Key Features

<div align="center">

| Feature | Description |
|:---:|:---|
| 👁️ **Eye Closure Detection** | Detects prolonged eye closure using Eye Aspect Ratio (EAR) algorithm |
| 🥱 **Yawning Detection** | Detects yawning using Mouth Aspect Ratio (MAR) algorithm |
| 🎯 **Auto Calibration** | Adapts to each user's unique facial structure in 4 seconds |
| 🔊 **Instant Alert** | Audio-visual alert triggers within 0.5 seconds of drowsiness |
| 📝 **Event Logging** | All drowsiness events logged with timestamps automatically |
| 💻 **No GPU Needed** | Runs smoothly on any standard laptop webcam |

</div>

---

## 🧠 How It Works

### 🔬 Core Algorithm — Eye Aspect Ratio (EAR)

The EAR formula measures how open your eye is using 6 facial landmark points:

```
        p2   p3
p1                 p4
        p6   p5

EAR = (|p2 - p6| + |p3 - p5|) / (2 × |p1 - p4|)
```

```
Eye Open   →  EAR ≈ 0.30   ✅ AWAKE
Eye Closing →  EAR ≈ 0.20   ⚠️  WARNING  
Eye Closed  →  EAR ≈ 0.00   🚨 ALERT
```

> Alert triggers when EAR stays below threshold for **12 consecutive frames**

---

### 👄 Mouth Aspect Ratio (MAR) — Yawn Detection

Same geometric principle applied to mouth landmarks:

```
MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (2 × |p1-p5|)
```

```
Mouth Closed  →  MAR ≈ 0.30   ✅ NORMAL
Mouth Opening →  MAR ≈ 0.50   ⚠️  WATCH
Wide Yawn     →  MAR > 0.70   🚨 ALERT
```

> Alert triggers when MAR exceeds threshold for **8 consecutive frames**

---

### 🎯 Auto Calibration System

```
┌─────────────────────────────────────────────────┐
│           CALIBRATION PHASE (4 seconds)          │
│                                                   │
│  📸 Captures your natural resting EAR & MAR      │
│  📊 Calculates personalized thresholds           │
│  ✅ Eliminates false alerts completely            │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│              DETECTION PHASE                      │
│                                                   │
│  👁️  Monitor EAR every frame                     │
│  👄  Monitor MAR every frame                     │
│  🚨  Alert if threshold crossed consistently     │
│  📝  Log event with timestamp                    │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Version | Purpose |
|:---:|:---:|:---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.8+ | Core programming language |
| ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) | 4.x | Webcam feed & image processing |
| ![MediaPipe](https://img.shields.io/badge/-MediaPipe-00897B?logo=google&logoColor=white) | 0.10+ | 468-point face landmark detection |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | 2.x | EAR & MAR mathematical calculations |
| ![Pygame](https://img.shields.io/badge/-Pygame-FF6B6B?logo=python&logoColor=white) | 2.x | Audio alert generation |

</div>

---

## 📊 Performance

<div align="center">

| Metric | Value |
|:---:|:---:|
| ⚡ Detection Speed | < 0.5 seconds |
| 🎥 FPS | 20–25 on standard laptop |
| 🎯 False Alert Rate | Near Zero (with auto-calibration) |
| 💾 GPU Required | ❌ No |
| 📦 Dataset Required | ❌ No — uses live webcam |

</div>

---

## 🚀 Getting Started

### 📋 Prerequisites

```bash
Python 3.8 or higher
A working webcam
```

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/mishikabagrecha/driver-drowsiness-detection.git

# Navigate to project folder
cd driver-drowsiness-detection

# Install dependencies
pip install opencv-python mediapipe pygame numpy
```

### ▶️ Run the System

```bash
python drowsiness.py
```

### 🎮 How to Use

```
Step 1 → Run the program
Step 2 → Calibration starts (4 seconds) — keep eyes open, mouth closed
Step 3 → System says "Starting..." — detection begins
Step 4 → Drive safely — system monitors in real time
Step 5 → Press Q to quit
```

---

## 📁 Project Structure

```
driver-drowsiness-detection/
│
├── 📄 drowsiness.py          # Main detection script
├── 🤖 face_landmarker.task   # MediaPipe face model
├── 📝 drowsiness_log.txt     # Auto-generated event log
└── 📖 README.md              # Project documentation
```

---

## 📝 Sample Log Output

```
[2026-04-09 01:23:45] DROWSINESS DETECTED - Eye closure
[2026-04-09 01:25:12] YAWNING DETECTED
[2026-04-09 01:28:33] DROWSINESS DETECTED - Eye closure
```

---

## 🔮 Future Improvements

- [ ] 🌙 Night mode for low light conditions
- [ ] 🤯 Head nodding detection
- [ ] 📱 Mobile deployment
- [ ] 📊 Real-time dashboard for fleet monitoring
- [ ] 🔗 Integration with vehicle systems

---

## 🎓 What I Learned

> - Real-time computer vision pipeline development
> - Facial landmark detection using MediaPipe
> - Geometric algorithms (EAR & MAR) for drowsiness detection
> - Auto-calibration techniques for personalization
> - OpenCV for live video processing

---

## 👩‍💻 Author

<div align="center">

**Mishika Bagrecha**

B.Tech CSE | AI/ML & Computer Vision Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-mishikabagrecha-181717?style=for-the-badge&logo=github)](https://github.com/mishikabagrecha)

</div>

---

<div align="center">

⭐ **If you found this project useful, please give it a star!** ⭐

*Built with ❤️ for road safety*

</div>
