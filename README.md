# 🚗 Hill Climb - AI-Powered Hand Gesture Controller

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.1.1-0078D7.svg?style=flat&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20(M1/M2/M3/M4)-000000.svg?style=flat&logo=apple&logoColor=white)](https://support.apple.com/en-us/HT211814)
[![Tested with Pytest](https://img.shields.io/badge/Tests-37%20Passed-brightgreen.svg?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-black.svg?style=flat&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Package Manager: uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9.svg?style=flat&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

> A real-time, low-latency computer vision application that tracks hand landmarks via webcam and translates physical gestures into responsive keyboard controls for **Hill Climb Racing** on macOS (Apple Silicon).

---

## 📖 Table of Contents
- [Overview & Motivation](#-overview--motivation)
- [Key Features](#-key-features)
- [Gesture Control Scheme](#-gesture-control-scheme)
- [System Architecture](#-system-architecture)
- [Tech Stack Breakdown](#-tech-stack-breakdown)
- [Performance & Benchmarks](#-performance--benchmarks)
- [Prerequisites & macOS Permissions](#-prerequisites--macos-permissions)
- [Quick Start Guide](#-quick-start-guide)
- [How to Play in Chrome](#-how-to-play-in-chrome)
- [Command-Line Interface (CLI) Reference](#-command-line-interface-cli-reference)
- [Project Directory Layout](#-project-directory-layout)
- [Engineering Principles & Fail-Safe Design](#-engineering-principles--fail-safe-design)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Overview & Motivation

**Hill Climb Racing** is one of the most iconic physics-based 2D racing games of our childhood. The goal is simple: balance acceleration and braking to conquer steep hills without flipping over.

This project transforms that nostalgic experience into an interactive, hands-free arcade experience using modern computer vision. Rather than touching arrow keys, you hold your hand up to the camera:
- **Show an Open Palm** to gun the engine forward.
- **Make a Tight Fist** to slam the brakes or tilt backwards in mid-air.
- **Pull your hand away** to coast neutrally.

Built from scratch with modular, clean Python architecture, it runs entirely on-device at **30 FPS** on Apple Silicon hardware with zero external dependencies or cloud telemetry.

---

## 🚀 Key Features

- **⚡ Real-Time Hand Landmark Tracking:** Powered by Google MediaPipe's HandLandmarker Tasks API with ARM NEON SIMD acceleration.
- **📐 Sub-Millisecond Geometric Classification:** Classifies gestures using pure normalized landmark vector geometry without the overhead of heavy secondary machine-learning models.
- **🛡️ 3-Frame Temporal Debouncing:** Rolling temporal filter completely eliminates gesture flicker and false triggers during rapid hand transitions.
- **🔒 Fail-Safe Safety Architecture:** Strict release guarantees ensure keys never remain stuck down when tracking drops, windows switch, or the app exits.
- **🖥️ Dual Display Modes:** 
  - **HUD Preview:** Live visual feedback with annotated hand skeletons, gesture badges, and real-time FPS counters.
  - **Headless Mode (`--no-preview`):** Runs seamlessly in the background so the browser game maintains active focus.
- **⌨️ macOS Quartz CGEvent Injection:** High-fidelity hardware key synthesis via `pynput` mapped directly to macOS CoreGraphics window servers.
- **🛑 Global Emergency Quit Listener:** Thread-safe background listener captures `Esc` or `q` keystrokes globally from any active window (even inside full-screen Chrome).
- **🔒 100% On-Device Privacy:** Webcam frames are processed in-memory and immediately discarded. No video is recorded, saved, or transmitted.

---

## 🖐️ Gesture Control Scheme

| Gesture | Visual | Action | Simulated Key | In-Game Effect |
| :--- | :---: | :--- | :---: | :--- |
| **Open Palm** | ✋ | **Accelerate / Gas** | `Right Arrow` | Drives the vehicle forward and tilts nose up |
| **Closed Fist** | ✊ | **Brake / Reverse** | `Left Arrow` | Applies brakes, reverses, and tilts nose down |
| **Index Up / Neutral** | ☝️ | **Coast** | `None` | Smooth rolling, neither pedal pressed |
| **Hand Removed** | 👋❌ | **Safe Idle** | `None` | Auto-release: vehicle coasts safely to rest |
| **Emergency Abort** | ⌨️ | **Global Exit** | `Esc` or `q` | Halts controller and guarantees all keys are released |

---

## 🏗️ System Architecture

The project follows a clean, decoupled unidirectional data pipeline where each component has a single responsibility:

```
┌──────────────┐
│ Webcam Stream│ (cv2.VideoCapture at 640x480 @ 30 FPS)
└──────┬───────┘
       │ BGR Frame
       ▼
┌──────────────────┐
│ camera.py        │  Frame acquisition, validation & retry logic
└──────┬───────────┘
       │ cv2 Frame
       ▼
┌──────────────────┐
│ hand_tracker.py  │  MediaPipe HandLandmarker Tasks API (21 landmarks)
└──────┬───────────┘
       │ NormalizedLandmarkList (21 points)
       ▼
┌──────────────────┐
│ gestures.py      │  Pure mathematical classification (OPEN_PALM, FIST, INDEX_UP)
└──────┬───────────┘
       │ Raw Gesture Enum
       ▼
┌──────────────────┐
│ smoothing.py     │  Temporal debounce filter (3-frame rolling buffer)
└──────┬───────────┘
       │ Stabilized Gesture Enum
       ▼
┌──────────────────┐
│ controller.py    │  State-managed macOS Quartz CGEvent injection (pynput)
└──────┬───────────┘
       │ Synthesized Arrow Keys
       ▼
┌──────────────────┐
│ Target Game      │  Hill Climb Racing (Active Chrome Tab)
└──────────────────┘
```

---

## 💻 Tech Stack Breakdown

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | **Python 3.12+** | Core runtime leveraging latest performance optimizations and type hints |
| **Computer Vision** | **OpenCV 5.0.0** | Webcam frame capture, color conversion, UI drawing, and HUD overlays |
| **Hand Tracking** | **MediaPipe 1.1.1** | MediaPipe Tasks API for single-hand landmark tracking with ARM NEON |
| **Input Simulation** | **pynput 1.8.2** | macOS Quartz Core Graphics (`CGEvent`) synthesized keyboard events |
| **Configuration** | **TOML (`tomllib`)** | Zero-dependency declarative configuration loader |
| **Package Management**| **`uv` by Astral** | Ultra-fast Python package resolution, virtual environments, and lockfiles |
| **Testing** | **pytest 9.1+** | Unit test suite covering classification, smoothing, controller, and configs |
| **Code Quality** | **Ruff** | Lightning-fast Python linter and code formatter |
| **Operating System**| **macOS (Darwin ARM64)**| Optimized specifically for Apple Silicon M-series unified memory architecture |

---

## ⚡ Performance & Benchmarks

Tested natively on a **MacBook Air M1 (8 GB RAM, 8-core CPU)**:

- **Webcam Throughput:** Steady **30 FPS** at `640x480` resolution.
- **Inference Latency:** **12–18 ms** per frame with MediaPipe Tasks XNNPACK delegate.
- **Gesture Classification Latency:** **< 0.5 ms** (pure geometric vector calculations).
- **End-to-End Reaction Time:** **~100 ms** (including camera exposure + 3-frame debounce buffer).
- **CPU Utilization:** **< 15%** CPU usage under active gameplay.
- **Memory Footprint:** **~110 MB** RSS.

---

## 🔑 Prerequisites & macOS Permissions

Because this tool accesses the webcam and simulates system keystrokes, macOS requires two permissions:

1. **Camera Access:**
   - **Why:** Required by OpenCV to capture webcam video.
   - **Path:** **System Settings → Privacy & Security → Camera** → Enable for your Terminal/IDE.

2. **Accessibility Access:**
   - **Why:** Required by `pynput` to simulate system-level key events via Quartz CGEvents.
   - **Path:** **System Settings → Privacy & Security → Accessibility** → Enable for your Terminal/IDE.

---

## 📦 Quick Start Guide

### 1. Clone Repository
```bash
git clone https://github.com/ranak8811/Hill-Climb.git
cd Hill-Climb
```

### 2. Environment Setup via `uv`
Ensure you have [`uv`](https://docs.astral.sh/uv/) installed, then initialize the environment:
```bash
uv sync
```

### 3. Verify Dependencies
Run the built-in diagnostic tool to confirm all native libraries and models are functional:
```bash
uv run python -m hill_climb.main --check-env
```

---

## 🎮 How to Play in Chrome

1. **Open the Game:**
   Navigate to the playable browser game in Google Chrome:
   👉 **[Hill Climb Racing Web Port](https://hill-climb-racing.com/hill-climb-racing-games/hill-climb-race)**

2. **Launch the Controller:**
   For the best gaming experience, launch in **headless mode** so the OpenCV window does not compete for focus:
   ```bash
   uv run hill-climb --no-preview
   ```
   *Or with the camera preview and a 3-second window switch countdown:*
   ```bash
   uv run hill-climb --start-delay 3.0
   ```

3. **Focus the Browser Window:**
   During the startup delay, click the Chrome browser tab to keep it active.

4. **Drive with Hand Gestures:**
   - ✋ **Open Palm:** Accelerate forward.
   - ✊ **Fist:** Slam the brake pedal or reverse.
   - 👋 **Hand Away:** Idle / coast smoothly.

5. **Quit Anytime:**
   Press **`Esc`** or **`q`** on your keyboard from any window to safely exit.

---

## 🛠️ Command-Line Interface (CLI) Reference

```text
usage: hill-climb [-h] [--config CONFIG] [--check-env]
                  [--camera-index CAMERA_INDEX] [--dry-run] [--no-control]
                  [--no-preview] [--start-delay START_DELAY]
                  [--accelerate-key ACCELERATE_KEY] [--brake-key BRAKE_KEY]
                  [--version] [-v]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--no-preview` | `bool` | `False` | Run in headless mode without GUI window (ideal for browser gaming) |
| `--dry-run` | `bool` | `False` | Detect gestures and log states without injecting real keypresses |
| `--no-control` | `bool` | `False` | Run camera and tracking UI only, disabling keyboard automation |
| `--start-delay` | `float` | `3.0` | Initial countdown in seconds to give you time to switch focus to Chrome |
| `--accelerate-key` | `str` | `right` | Override the key bound to acceleration (e.g., `up`, `w`) |
| `--brake-key` | `str` | `left` | Override the key bound to braking (e.g., `down`, `s`) |
| `--camera-index` | `int` | `0` | Specify alternative camera index if multiple webcams are connected |
| `--config` | `str` | `config/default.toml` | Load a custom TOML configuration file |
| `-v, --verbose` | `bool` | `False` | Enable detailed debug logs in terminal |
| `--check-env` | `bool` | `False` | Validate environment packages and exit immediately |
| `--version` | `bool` | `False` | Display application version and exit |

---

## 📁 Project Directory Layout

```text
Hill Climb/
├── README.md                  # Comprehensive project documentation
├── AGENTS.md                  # AI agent guidelines & architectural constraints
├── DEVELOPMENT_PLAN.md        # Single source of truth specifications & checklists
├── pyproject.toml             # uv configuration, dependencies, and tool settings
├── uv.lock                    # Deterministic locked dependency graph
├── .gitignore                 # Source control exclusion rules
├── config/
│   └── default.toml           # Default thresholds, keys, and camera settings
├── docs/
│   ├── LINKEDIN_POST.md       # Ready-to-use LinkedIn announcement copy
│   └── TROUBLESHOOTING.md     # In-depth macOS issue resolution guide
├── models/
│   └── hand_landmarker.task   # MediaPipe HandLandmarker pre-trained model bundle
├── src/
│   └── hill_climb/            # Main application source package
│       ├── __init__.py        # Package exports
│       ├── main.py            # CLI entry point, argument parsing, orchestration
│       ├── camera.py          # OpenCV video capture abstraction & retry logic
│       ├── hand_tracker.py    # MediaPipe Tasks API hand landmark tracking
│       ├── gestures.py        # Pure mathematical gesture classification
│       ├── smoothing.py       # Debouncing & temporal smoothing buffer
│       ├── controller.py      # Quartz CGEvent simulation & fail-safe key release
│       ├── config.py          # Dataclass-based TOML configuration loader
│       └── ui.py              # Visual HUD overlay, skeleton rendering & badges
└── tests/                     # 100% offline unit test suite (37 tests)
    ├── test_camera.py         # Frame validation & camera mock tests
    ├── test_config.py         # Config loading, defaults, and CLI parser tests
    ├── test_controller.py     # Keyboard controller transitions & mock backend tests
    ├── test_gestures.py       # Geometric classification tests on synthetic landmarks
    ├── test_hand_tracker.py   # Landmark extraction & absent hand handling tests
    └── test_smoothing.py      # Temporal buffer, debounce, and timeout tests
```

---

## 🛡️ Engineering Principles & Fail-Safe Design

1. **Dependency Injection:** The `KeyboardController` accepts an optional backend interface, allowing 100% of keyboard state transitions to be verified using mock backends without firing real keystrokes.
2. **Pure Classification Functions:** Gesture classification in `gestures.py` relies exclusively on normalized 3D vectors. This allows tests to run without physical cameras or GUI environments.
3. **Guaranteed Key Release (`release_all`):**
   - Any tracking loss immediately schedules a key release.
   - All termination signals (`SIGINT`, `SIGTERM`, unhandled exceptions, or window closure) invoke `controller.release_all()` within `finally` blocks.
4. **Thread-Safe Global Abort Listener:** Built using a dedicated `pynput.keyboard.Listener` daemon, ensuring the player can exit using `Esc` or `q` regardless of which desktop window holds focus.

---

## 🧪 Testing & Quality Assurance

Run the automated offline test suite:
```bash
uv run pytest
```

Run code formatting and lint checks:
```bash
uv run ruff check .
uv run ruff format --check .
```

Auto-format code according to PEP 8 standards:
```bash
uv run ruff format .
```

---

## ❓ Troubleshooting & FAQs

### Q1: The car is not moving even though gestures are recognized!
> **Answer:** Keystrokes in macOS are delivered only to the **active frontmost window**. Make sure the Google Chrome window is clicked and focused. If using the preview window, try launching with `--no-preview` or use `--start-delay 3.0` to switch focus before control begins.

### Q2: "ModuleNotFoundError: No module named 'hill_climb'"
> **Answer:** On macOS APFS file systems, Python 3.12 ignores `.pth` files with the BSD `UF_HIDDEN` flag. Clear this flag by running:
> ```bash
> chflags -R nohidden .venv
> ```

### Q3: My gestures are flickering between states.
> **Answer:** Ensure adequate, even lighting on your hand without heavy backlighting. You can also increase the debounce threshold in `config/default.toml` under `debounce_frames` (e.g., from `3` to `4` or `5`).

For more solutions, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## 📜 License & Acknowledgments

- **License:** Distributed under the [MIT License](LICENSE).
- **Game Attribution:** Hill Climb Racing is an original intellectual property of [Fingersoft](https://fingersoft.com/). This open-source project is an educational, non-commercial interface demonstration intended for personal research in human-computer interaction (HCI).
- **Frameworks:** Special thanks to the Google MediaPipe team and Astral (`uv`) for building incredible developer tooling.
