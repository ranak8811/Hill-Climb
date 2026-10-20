# Agent Instructions & Conventions

This project builds an AI-powered hand gesture controller for Hill Climb Racing on macOS (Apple Silicon).

## Workflow Guidelines
- Development and task specifications follow `DEVELOPMENT_PLAN.md`.
- All development must proceed phase by phase. Never implement full features ahead of their designated phase.
- Use `uv` for package and environment management.
- Keep modules modular and decoupled (camera, hand tracking, gesture classification, temporal smoothing, keyboard control, and UI overlay).
- Classification helpers must remain pure functions so they can be unit-tested without real camera inputs or hardware.
- The keyboard controller must use dependency injection to allow mocking in automated tests.
- Safety first: Never leave simulated keys held down; ensure release on lost tracking, timeouts, exceptions, and shutdown.
- Lint and format using `ruff`. Run tests using `pytest`.
