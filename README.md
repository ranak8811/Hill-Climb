# Hill Climb - AI-Powered Hand Gesture Controller

Real-time computer vision application that tracks one hand via webcam and translates hand gestures into keyboard controls for **Hill Climb Racing** on macOS (Apple Silicon).

## Prerequisites
- macOS on Apple Silicon (M1/M2/M3/M4)
- [`uv`](https://docs.astral.sh/uv/) (fast Python package and project manager)
- Google Chrome (or compatible browser) for running the target game

## Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ranak8811/Hill-Climb.git
   cd Hill-Climb
   ```

2. **Sync the environment and dependencies:**
   ```bash
   uv sync
   ```

3. **Verify environment and dependencies:**
   ```bash
   uv run python -m hill_climb.main --check-env
   ```

## Running Quality Checks

Run the automated test suite:
```bash
uv run pytest
```

Run code formatting and lint checks:
```bash
uv run ruff check .
uv run ruff format --check .
```

To auto-format code:
```bash
uv run ruff format .
```

## Running the Application

Launch the application entry point:
```bash
uv run python -m hill_climb.main
# or via console script:
uv run hill-climb
```

## macOS Permissions Notice

1. **Camera Permission**: When OpenCV accesses the webcam, macOS prompts for permission. If denied or missing, enable it in **System Settings → Privacy & Security → Camera**.
2. **Accessibility Permission**: Key simulation via `pynput` sends synthesized CGEvent keyboard events. The terminal or IDE running the app needs permission in **System Settings → Privacy & Security → Accessibility**.

## Game Target

- Playable browser game target: [Hill Climb Racing on WASM](https://wasm.rip/files/hcr/index.html) running in Google Chrome.
- Supported game controls: `W` / `Up Arrow` (Accelerate), `S` / `Down Arrow` (Brake/Reverse).
- **Note:** Keyboard simulation directs keys to the currently focused window. Keep the Chrome browser tab focused during gameplay.

## Project Structure

```text
Hill Climb/
├── README.md              # Project setup and usage instructions
├── AGENTS.md              # Agent development guidelines
├── DEVELOPMENT_PLAN.md    # Source of truth specification and progress tracker
├── pyproject.toml         # uv project configuration, dependencies, and tool settings
├── uv.lock                # Locked dependency tree
├── .gitignore             # Git ignore patterns
├── config/
│   └── default.toml       # Default configuration settings
├── docs/
│   ├── DEVELOPMENT_PLAN.md
│   └── TROUBLESHOOTING.md # Common macOS issues and solutions
├── src/
│   └── hill_climb/        # Core application source code
│       ├── __init__.py
│       ├── main.py        # CLI entry point
│       ├── camera.py      # OpenCV video capture
│       ├── hand_tracker.py# MediaPipe hand tracking interface
│       ├── gestures.py    # Gesture classification rules
│       ├── smoothing.py   # Debouncing & temporal filter
│       ├── controller.py  # Safe keyboard controller
│       ├── config.py      # Settings loading and validation
│       └── ui.py          # HUD preview overlay
└── tests/                 # Unit tests (run offline without camera)
    ├── test_config.py
    ├── test_controller.py
    ├── test_gestures.py
    └── test_smoothing.py
```
