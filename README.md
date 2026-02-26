---

```markdown
# Optical Flow & Background Subtraction: Scientific Exhibition System

An interactive computer vision system designed for two purposes: an academic study of different **Optical Flow** algorithms and a creative application that applies real-time fluid dynamics, temporal recursion, and geometric effects to human motion.

## 📋 Table of Contents

* [Overview](#overview)
* [Scientific Objectives](#scientific-objectives)
* [System Architecture](#system-architecture)
* [Optical Flow Methods](#optical-flow-methods)
* [Visual Effects Gallery](#visual-effects-gallery)
* [Getting Started](#getting-started)
* [Controls](#controls)
* [Data Collection & Analysis](#data-collection--analysis)

---

## 🔍 Overview

This project explores the separation of moving foregrounds (people) from static backgrounds using various Optical Flow techniques. By moving from simple static background subtraction to dense motion estimation, the system achieves robust detection even in challenging exhibition environments with varying lighting, solving issues like the "Statue Problem" through mathematical motion accumulators.

## 🎓 Scientific Objectives

1. **Algorithm Comparison:** Evaluate the efficiency and accuracy of **Farneback**, **DIS**, and **Dual TV-L1** algorithms.
2. **Robustness Analysis:** Study the "Statue Problem" (disappearance of subjects during pauses) and solve it using temporal decay accumulators.
3. **Performance Benchmarking:** Quantify the trade-off between computational latency (ms) and motion sensitivity (Energy).

---

## 🏗 System Architecture

The project is built using a **Modular Object-Oriented (OOP)** design to ensure stability, memory safety, and easy expansion during long-running public exhibitions.

```text
├── main.py                # Application Controller & UI Loop
├── core/
│   ├── optFlow.py         # Optical Flow Engine (DIS, TV-L1, Farneback)
│   └── background.py      # Masking Logic (Static vs. Motion with Decay)
├── effects/
│   ├── baseEffect.py      # Abstract Blueprint for all plugins
│   ├── clones.py          # Solid temporal duplicates
│   ├── filters.py         # Cartoon, Heatmap, and Negative styling
│   ├── fluid.py           # Advection-based Fluid Simulation
│   ├── geometry.py        # Grid Warp and Arrow Vectors
│   ├── timeTunnel.py      # Droste and Temporal deep tunnels
│   ├── trails.py          # Ghosting and Motion Trails   
│   └── debug.py           # Binary foreground mask visualization
├── utils/
│   ├── hud.py             # User Interface Overlay
│   └── benchmarker.py     # CSV Logging & Math Metrics
└── analyze_results.py     # Data Analysis & Graphing Tool (External)

```

---

## ⚙ Optical Flow Methods

The system supports hot-swapping between three primary engines on the fly:

* **DIS (Dense Inverse Search):** Fast, modern CPU-based method. High temporal stability and excellent for real-time interaction.
* **Farneback:** Classical dense flow. Reliable and computationally balanced for standard hardware.
* **Dual TV-L1:** High-quality variational method. Exceptional smoothness, ideal for artistic effects (higher CPU cost).

---

## 🎨 Visual Effects Gallery

Thanks to the plugin architecture, each effect manages its own isolated memory and canvas:

* **Fluid Paint:** Real-time color advection. Motion "paints" the background with swirling colors while keeping the subject clean.
* **Temporal Tunnels & Clones:** Features the `TimeTunnel`, `DrosteTunnel`, and `SolidClone` effects, mapping historical frames into recursive visual depth.
* **Geometry & Vectors:** `GridWarp` deforms a virtual wireframe based on physical motion, while `Arrows` visualizes the raw mathematical vector fields.
* **Motion Trails:** Persistent `GhostTrails` and `MotionTrails` that track the path of the subject with adjustable decay.
* **Artistic Filters:** Real-time processing applying `Cartoon`, `Heatmap`, and `Negative` aesthetics to the motion masks.
* **Show Mask (Debug):** A scientific view illustrating the raw background subtraction logic for academic transparency.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* OpenCV (`opencv-contrib-python` required for advanced flow methods like DIS and TV-L1)
* Pandas & Matplotlib (for data analysis/graphing)

### Installation

```bash
git clone [https://github.com/yourusername/optical-flow-exhibition.git](https://github.com/yourusername/optical-flow-exhibition.git)
cd optical-flow-exhibition
pip install -r requirements.txt

```

### Running the System

```bash
python src/main.py

```

---

## ⌨ Controls

The application is designed to be controlled via keyboard during an exhibition:

| Key | Action |
| --- | --- |
| `n` | **Next Effect:** Cycle through the visual styles playlist. |
| `m` | **Toggle Mask:** Switch between Static Subtraction and Motion Masking. |
| `o` | **Swap Engine:** Cycle between DIS, TV-L1, and Farneback engines. |
| `b` | **Capture BG:** Capture a new static background model (Static mode only). |
| `r` | **Reset:** Clears the internal memory/canvas of the current effect. |
| `d` | **HUD:** Toggle on-screen information overlay. |
| `q` / `Esc` | **Quit:** Safely close the application. |

---

## 📊 Data Collection & Analysis

The system automatically logs performance data to `data/csv/exhibition_data.csv` every 0.5 seconds to avoid bottlenecking the framerate.

**Metrics Collected:**

* **Latency (ms):** Processing time per frame.
* **Energy:** Mean magnitude of motion vectors (Sensitivity).
* **Sparsity (%):** Percentage of moving pixels (Noise detection).

```
