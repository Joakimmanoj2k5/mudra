# MUDRA — Indian Sign Language Learning (Desktop)

<div align="center">

## 🐍 Contribution Snake

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Joakimmanoj2k5/mudra/output/github-contribution-grid-snake-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Joakimmanoj2k5/mudra/output/github-contribution-grid-snake.svg" />
  <img alt="github-snake" src="https://raw.githubusercontent.com/Joakimmanoj2k5/mudra/output/github-contribution-grid-snake-dark.svg" />
</picture>

</div>

---

MUDRA is an offline, privacy-first desktop application to learn and practice Indian Sign Language (ISL). It uses MediaPipe for hand landmark detection and PyTorch models for gesture recognition. The UI is written with PyQt5 and OpenCV is used for camera capture and overlays.

## Highlights
- Live practice with real-time hand tracking and recognition
- Study mode with rich text and image reference cards (124 gestures included)
- First-run automation that downloads MediaPipe task models, generates reference cards and creates placeholder model artifacts so the app is runnable out-of-the-box
- SQLite local database with seeded demo/admin accounts for testing

## Quick start (Recommended)
These commands assume you are in the repository root (`/path/to/mudra`). Use a Python 3.9+ interpreter.

macOS / Linux (zsh/bash)
```bash
# create and activate venv
python3 -m venv mudra_env
source mudra_env/bin/activate

# upgrade pip and install runtime deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run a quick preflight check (optional)
python3 scripts/preflight.py

# Start the GUI
python3 mudra_app.py
```

Windows (PowerShell)
```powershell
# create and activate venv
python -m venv .\mudra_env
.\mudra_env\Scripts\Activate.ps1

# upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# preflight (optional)
python .\scripts\preflight.py

# start the GUI
python mudra_app.py
```

Notes:
- On Apple Silicon (M1/M2), installing PyTorch may require selecting the right wheel. The project previously used the official CPU wheel index for macOS; if you face issues, follow the instructions on https://pytorch.org for your macOS architecture.
- MediaPipe is installed via pip; some systems may require extra build tools. The repository includes a first-run helper (`utils/first_run.py`) that downloads the MediaPipe task model automatically when missing.

## What this repo contains
- `mudra_app.py` — backward-compatible launcher (calls `ui.app.main()`)
- `ui/` — PyQt UI and screens (main window, dialogs)
- `inference/` — camera worker, mediapipe tracker, model predictors, preprocessing and overlay helpers
- `models/` — pretrained or placeholder artifacts. The first-run step will create placeholders if missing.
- `data/assets/gestures/isl_reference_data.json` — textual references and tips used to generate reference cards
- `data/assets/gestures/image_cache/` — generated reference PNG cards (created by `scripts/generate_reference_images.py` or on first run)
- `database/` — SQLite DB and seeding scripts (`database/seed/seed_database.py`)
- `scripts/` — utilities: preflight checks, asset generation, database migration helpers
- `requirements.txt` — pinned Python dependencies

## First-run behaviour
The application contains a convenience script which runs automatically on startup (`utils/first_run.py`) and will do the following if artifacts are missing:
- Download MediaPipe Hand Landmarker task model to `models/mediapipe/hand_landmarker.task`
- Create placeholder PyTorch state dicts for `models/static/*.pt` and `models/dynamic/*.pt` so that environment checks pass
- Generate reference PNG cards in `data/assets/gestures/image_cache/` from `data/assets/gestures/isl_reference_data.json`

This is intended to make the project runnable immediately for demo and development purposes. For production accuracy, replace placeholder models with trained weights.

## Running and debugging
- If the GUI opens but no camera feed appears: ensure the camera is not blocked by another app and that the OS has given camera permission to the terminal/IDE.
- If MediaPipe throws warnings or info logs related to TFLite, these are usually informational. A segmentation fault on process exit has been observed in some test runs due to native cleanup; the GUI typically runs fine.
- If models are not present, first-run will create placeholders. To train and replace them, see the `training/` folder and trainers.

Preflight checks
```bash
python3 scripts/preflight.py
```
This script will validate Python dependencies, model assets, camera availability, and other environment concerns. Use it to quickly triage issues.

## Training models
The repo includes training pipelines under `training/`:
- `training/trainers/train_static.py` — static gesture model training
- `training/trainers/train_dynamic.py` — dynamic gesture model training

Basic steps to train:
1. Prepare a dataset (see `training/datasets/`)
2. Run the feature extraction utilities (`training/features/extract_landmarks.py`)
3. Run the trainer scripts with the appropriate config/flags

Training requires significantly more data and compute. If you want help preparing a training recipe or example commands for your machine, open an issue or ask for a training walkthrough.

## Packaging for distribution
To create a standalone executable for end users, we recommend using PyInstaller. Typical steps:
```bash
# after creating venv and installing deps
pip install pyinstaller
pyinstaller --noconfirm --onefile --name MudraApp mudra_app.py
```
You may need to add extra-data entries for `models/`, `data/assets/`, and platform-specific handling for MediaPipe. Packaging MediaPipe can be tricky because of native libs; test thoroughly on each target OS.

## Troubleshooting tips
- Camera permission denied: macOS -> System Settings > Privacy & Security > Camera -> allow Terminal/IDE
- PyTorch installation problems: consult https://pytorch.org and pick the wheel for your OS and Python version
- MediaPipe incompatible API errors: ensure the installed mediapipe version matches code expectations (this repo uses the Tasks API and expects a `models/mediapipe/hand_landmarker.task` file)
- Missing requirements.txt: if your environment is missing packages, run `pip install mediapipe torch opencv-python PyYAML scikit-learn PyQt5` (the repo's `requirements.txt` is preferred)

## Developer notes
- Entry points:
  - GUI: `python3 mudra_app.py` (launches `ui.app.main()`)
  - API backend (if used): `python3 backend/run_api.py`
- Database: `database/mudra.db` (SQLite). Re-seeding utilities exist under `database/seed/`.
- Logging: application prints informational logs from MediaPipe/TFLite; these do not necessarily indicate a fatal error.

## Project roadmap / TODOs
- Replace placeholder models with trained models for higher accuracy
- Add optional remote-hosted reference videos (the app currently uses generated reference images)
- Provide platform-specific installers (Windows .exe, macOS .dmg)
- Add unit and integration tests for inference pipeline

## Contributing
PRs welcome. Please:
1. Fork the repo
2. Create a feature branch
3. Run `scripts/preflight.py` and ensure the app runs
4. Open a PR describing the change and a small test or reproducer

## License & Acknowledgements
This project is released under the MIT License. It relies on MediaPipe (Google), PyTorch, OpenCV and other OSS libraries — see `requirements.txt` for details.

## Contact
For questions, feature requests, or help reproducing results, open an issue or email the maintainer (check the repo settings for contact details).

---
Generated on 2026-02-25 — README created to make the project runnable for devs and contributors. If you want a shorter README (for GitHub front-page) or an expanded developer guide, tell me which sections to trim or expand.

# MUDRA - Interactive ISL Learning with Realtime Gesture Feedback

## What is implemented
- Modular architecture (UI, inference, training, backend API, database, tests)
- 26 alphabets + 100-word catalog seed-ready
- Realtime camera loop with MediaPipe landmarks and prediction smoothing
- Login/auth (local), practice, quiz, analytics, and admin reseed control
- Reproducible training/evaluation scripts with metrics and confusion matrix export
- Admin model registry table with active-version switching and predictor hot-reload
- Analytics confusion matrix heatmap view (per-user attempt data)
- Admin model upload/register workflow with version metadata, optional immediate activation, and rollback by model family

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m database.seed.seed_database
python mudra_app.py
```

## Demo credentials
- `demo@mudra.local / demo123`
- `admin@mudra.local / admin123`

## Run backend API
```bash
python -m backend.run_api
```

## Train static model
```bash
python -m training.features.extract_landmarks --input data/raw
python -m training.datasets.build_dataset
python -m training.trainers.train_static
python -m training.evaluation.evaluate
python -m training.trainers.train_dynamic
python -m training.evaluation.evaluate_dynamic
```

## Notes
- If MediaPipe/Torch model files are unavailable, app falls back to stable rule-based predictions for demo continuity.
- Dynamic model is used live for motion-based signs when selected target is `gesture_mode=dynamic`.
- Analytics page now includes a confusion-matrix view from recorded attempts.

## Phase 5 Deployment Ops
```bash
# preflight checks
python scripts/preflight.py

# start API + UI together
bash scripts/start_local.sh
```

### New API endpoints
- `GET /health`
- `GET /models`
- `POST /models/register` (admin)
- `POST /models/{model_version_id}/activate` (admin)
- `POST /models/{model_name}/rollback` (admin)

## Phase 6 Reproducible Env + CI/CD
```bash
# create .venv, install runtime/dev deps, seed DB, run preflight
bash scripts/bootstrap.sh

# deterministic local CI gate
make ci
```

CI workflow:
- GitHub Actions: [.github/workflows/ci.yml](/Users/joakimmanoj/Downloads/mudra/.github/workflows/ci.yml)
- Checks: `ruff`, DB seed, `scripts/ci_checks.py`, pytest suite

## Phase 7 Packaging + Release Engineering
```bash
# 1) migrate schema + release SQL patches
make migrate

# 2) backup current DB before release
make backup

# 3) build desktop package (requires pyinstaller in dev env)
make package

# 4) create signed-hash release bundle (artifacts + deploy configs)
make release TAG=v0.1.0
```

Deployment manifests:
- Docker API image: [Dockerfile.api](/Users/joakimmanoj/Downloads/mudra/deploy/Dockerfile.api)
- Docker Compose: [docker-compose.yml](/Users/joakimmanoj/Downloads/mudra/deploy/docker-compose.yml)
- systemd service: [mudra-api.service](/Users/joakimmanoj/Downloads/mudra/deploy/systemd/mudra-api.service)
- env template: [env.example](/Users/joakimmanoj/Downloads/mudra/deploy/env.example)

Database release safety:
- Migrations runner: [migrate.py](/Users/joakimmanoj/Downloads/mudra/scripts/migrate.py)
- Backup script: [backup_db.py](/Users/joakimmanoj/Downloads/mudra/scripts/backup_db.py)
- Restore script: [restore_db.py](/Users/joakimmanoj/Downloads/mudra/scripts/restore_db.py)

## Phase 8 Production Readiness + Launch
```bash
# security hardening checks
make security

# API load test (API should be running)
make loadtest

# health monitor loop (12 checks by default)
make monitor
```

Runbooks:
- Launch checklist: [launch_checklist.md](/Users/joakimmanoj/Downloads/mudra/docs/runbooks/launch_checklist.md)
- UAT test plan: [uat_test_plan.md](/Users/joakimmanoj/Downloads/mudra/docs/runbooks/uat_test_plan.md)

New API readiness endpoint:
- `GET /ready`

## Learning Runtime Upgrades
- Environment health checks: [environment_check.py](/Users/joakimmanoj/Downloads/mudra/utils/environment_check.py)
  - checks `mediapipe`, `torch`, camera access, and model artifact availability
  - practice start is automatically disabled if MediaPipe/camera is unavailable
- Gesture reference media mapping: [gesture_media_mapper.py](/Users/joakimmanoj/Downloads/mudra/utils/gesture_media_mapper.py)
  - asset root: `data/assets/gestures/`
  - expected examples:
    - `data/assets/gestures/alphabets/A.mp4`
    - `data/assets/gestures/words/hello.mp4`
- Dedicated Study Mode (separate from Practice)
  - left: gesture metadata (name/type/description/difficulty)
  - right: looping reference media
  - `Start Practice` button switches to Practice without auto-starting camera
- Side-by-side Practice layout
  - left: reference loop panel
  - right: live camera with target/model/fps/confidence overlay
- Stability filters
  - static: must remain same for 1.5s
  - dynamic: confidence must stay above threshold for consecutive frames
- FPS guard
  - inference processing capped to ~18 FPS
  - warning shown when measured FPS < 15
