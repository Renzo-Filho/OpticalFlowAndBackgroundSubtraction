# Optical Flow & Background Subtraction: Interactive Exhibition System

An interactive computer vision system designed for public exhibitions and creative applications. It applies real-time fluid dynamics, temporal recursion, and geometric effects to human motion using advanced Optical Flow and AI-based Background Subtraction.

## 📋 Table of Contents

* [Overview](#overview)
* [System Architecture](#system-architecture)
* [Core Technologies](#core-technologies)
* [Visual Effects Gallery](#visual-effects-gallery)
* [🚀 Quick Start (1-Click Setup)](#-quick-start-1-click-setup)
* [⌨️ Controls](#️-controls)

---

## 🔍 Overview

This project isolates moving foregrounds (people) from static backgrounds to apply stunning visual effects in real-time. Designed specifically for totens, interactive displays, and live events, it features a highly scalable, automated setup process that requires zero terminal knowledge for the end-user (monitors/staff).

---

## 🏗 System Architecture

The project is built using a **Modular Object-Oriented (OOP)** design to ensure stability, memory safety, and easy expansion during long-running public exhibitions.

```text
├── assets/                  
│   ├── icons/
│   │   ├── input.png
│   │   └── icone.ico
│   └── models/
|       ├── pose_landmarker_lite.task
│       └── selfie_segmenter_landscape.tflite
├── prototypes/              
│   ├── backSubtr.py         
│   └── tests/
├── src/                     
│   ├── core/
│   ├── effects/
│   ├── utils/               
│   └── main.py
├── demo.bat
├── demo.sh
├── setup_win.bat
├── setup_linux.sh
├── requirements.txt
└── README.md
```

---

## ⚙ Core Technologies

The system supports hot-swapping between two primary masking engines on the fly, depending on the environment:

1. **AI Selfie Segmenter (MediaPipe):** Uses a lightweight neural network (`selfie_segmenter_landscape.tflite`) for robust semantic segmentation. Ideal for dynamic environments where the background might change or lighting is inconsistent.
2. **AI Pose Landmarker (MediaPipe):** Utilizes the `pose_landmarker_lite.task` model to asynchronously track the user's bone structure and joints in real-time. This provides the precise 2D spatial coordinates required to drive physics-based hand interactions and dynamic skeletal overlays.
3. **Static Background Subtraction (YCrCb + Otsu):** A classic, ultra-fast method that requires capturing an empty room first. Great for controlled studio lighting.
4. **Optical Flow (DIS):** Uses the Dense Inverse Search (DIS) algorithm on the `FAST` preset to calculate precise pixel-by-pixel motion vectors for the interactive effects.

---

## 🎨 Visual Effects Gallery

Thanks to the plugin architecture, each effect manages its own isolated memory and canvas:

* **Temporal Tunnels & Clones:** Features the `TimeTunnel` and `DrosteTunnel` effects, mapping historical frames into recursive visual depth.
* **Physics & Particles:** Real-time simulations including `FluidPaint` (color advection), `WaveEquation` (water ripples), `KineticParticles` (wind-blown dust), and `FlowBender` (hand-emitted energy).
* **Geometry & Vectors:** `GridWarp` deforms a virtual wireframe based on physical motion, `Arrows` visualizes raw mathematical vector fields, and `DelaunayConstellation` builds a living cybernetic mesh.
* **Pose & Anatomy:** Uses MediaPipe to track and draw glowing bone structures with `NeonSkeleton`.
* **Artistic Filters:** Real-time processing applying `Cartoon`, `Heatmap`, `Negative`, `CyberGlitch`, and `NeonSilhouette` aesthetics to the motion masks.
* **Overlays & Chroma Key:** High-quality scrolling LaTeX formulas composited seamlessly with the user via `MathChromaKey`.

---

## 🚀 Quick Start (1-Click Setup)

This project is built to be deployed seamlessly across different machines at events. You do not need to manually install dependencies or open terminals.

**Prerequisite:** Ensure **Python 3.8+** is installed on the machine.

### Step 1: Transfer

Clone this repository or copy the project folder to the target machine via USB drive.

### Step 2: Auto-Configuration

Navigate to the project folder and run the setup script for your operating system. This will automatically map the folder paths and create a ready-to-use shortcut on your Desktop.

* **Windows:** Double-click `setup_win.bat`
* **Linux (Ubuntu/Raspberry):** Right-click `setup_linux.sh` -> Properties -> Allow executing as a program. Then double-click it (or run `./setup_linux.sh` in the terminal).

### Step 3: Run the Exhibition

Go to your Desktop and double-click the newly created **"Demo Visão Computacional"** shortcut.

* *Note:* On the very first launch, it will take a few seconds to automatically download and isolate all required libraries (OpenCV, MediaPipe, etc.) in a virtual environment. Subsequent launches will be instant.

---

## ⌨️ Controls

The application is designed to be controlled via keyboard during an exhibition:

| Key | Action |
| --- | --- |
| `n` | **Next Effect:** Cycle through the visual styles playlist. |
| `l` | **Last Effect:** Go back to the previous effect. |
| `m` | **Toggle Mask Mode:** Switch between AI Selfie Segmenter and Static Subtraction. |
| `b` | **Capture Background:** Capture a new static room model (Only works in Static Mode). Step out of the frame first! |
| `r` | **Reset:** Clears the internal memory/canvas of the current effect. |
| `d` | **HUD:** Toggle on-screen debug information overlay. |
| `q` / `Esc` | **Quit:** Safely close the application. |