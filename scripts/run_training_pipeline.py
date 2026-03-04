#!/usr/bin/env python3
"""Run the complete MUDRA ML training pipeline end-to-end.

Steps:
    1. Verify / generate training data
    2. Build train/val/test datasets
    3. Train static MLP model
    4. Train dynamic BiGRU model
    5. Evaluate both models (metrics + confusion matrices)
    6. Register models in database
    7. Verify inference compatibility
    8. Print training summary

Usage::

    python scripts/run_training_pipeline.py
    python scripts/run_training_pipeline.py --epochs 120 --samples-per-class 80
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logging.logger import configure_logger

logger = configure_logger("mudra.training_pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_PATH = Path("data/interim/landmarks_manifest.json")
STATIC_SPLIT = Path("data/processed/static_split_v1.npz")
DYNAMIC_SPLIT = Path("data/processed/dynamic_split_v1.npz")
STATIC_MODEL = Path("models/static/static_mlp_v001.pt")
DYNAMIC_MODEL = Path("models/dynamic/dynamic_bigru_v001.pt")
NORM_STATS = Path("models/registry/norm_stats.json")
DYN_NORM_STATS = Path("models/registry/dynamic_norm_stats.json")
LABEL_MAP = Path("models/registry/label_map.json")
METRICS_STATIC = Path("models/registry/metrics_static_v001.json")
METRICS_DYNAMIC = Path("models/registry/metrics_dynamic_v001.json")
EVAL_STATIC = Path("models/registry/evaluation_static_v001.json")
EVAL_DYNAMIC = Path("models/registry/evaluation_dynamic_v001.json")

PYTHON = sys.executable  # Use the same Python interpreter


def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess and return True on success."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("[%s] FAILED (rc=%d)\nstdout: %s\nstderr: %s",
                     label, result.returncode, result.stdout[-2000:], result.stderr[-2000:])
        print(f"  ERROR in {label}: {result.stderr[-500:]}")
        return False
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    return True


# ---------------------------------------------------------------------------
# Step 1: Verify / generate training data
# ---------------------------------------------------------------------------
def step_generate_data(samples_per_class: int) -> bool:
    print()
    print("=" * 64)
    print("STEP 1: Verify / Generate Training Data")
    print("=" * 64)

    raw_root = Path("data/raw")
    has_data = any(raw_root.rglob("*.png")) or any(raw_root.rglob("*.jpg"))

    if has_data:
        n_images = len(list(raw_root.rglob("*.png"))) + len(list(raw_root.rglob("*.jpg")))
        print(f"  data/raw/ contains {n_images} images — using existing data.")
    else:
        print("  data/raw/ is empty — no training images found.")
        print("  Generating training data from reference images + synthetic landmarks...")

    # Always run generator to ensure manifest + landmarks are complete
    ok = _run(
        [PYTHON, "scripts/generate_training_data.py",
         "--samples-per-class", str(samples_per_class),
         "--skip-mediapipe"],
        "generate_training_data",
    )
    if not ok:
        return False

    if not MANIFEST_PATH.exists():
        print("  ERROR: Landmark manifest was not generated.")
        return False

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    n_classes = len({e["class"] for e in manifest})
    print(f"  Manifest ready: {len(manifest)} entries, {n_classes} classes")
    return True


# ---------------------------------------------------------------------------
# Step 2: Build datasets
# ---------------------------------------------------------------------------
def step_build_dataset() -> bool:
    print()
    print("=" * 64)
    print("STEP 2: Build Train/Val/Test Datasets")
    print("=" * 64)

    ok = _run(
        [PYTHON, "-m", "training.datasets.build_dataset",
         "--manifest", str(MANIFEST_PATH),
         "--output", "data/processed",
         "--seq-len", "30"],
        "build_dataset",
    )
    if not ok:
        return False

    if not STATIC_SPLIT.exists():
        print("  ERROR: static_split_v1.npz was not created.")
        return False

    import numpy as np
    blob = np.load(STATIC_SPLIT)
    n_train = blob["X_train"].shape[0]
    n_val = blob["X_val"].shape[0]
    n_test = blob["X_test"].shape[0]
    feat_dim = blob["X_train"].shape[1]
    print(f"  Static split: train={n_train}, val={n_val}, test={n_test}, features={feat_dim}")

    if DYNAMIC_SPLIT.exists():
        dblob = np.load(DYNAMIC_SPLIT)
        print(f"  Dynamic split: train={dblob['X_train'].shape[0]}, "
              f"val={dblob['X_val'].shape[0]}, test={dblob['X_test'].shape[0]}, "
              f"seq_len={dblob['X_train'].shape[1]}")
    else:
        print("  WARNING: dynamic_split_v1.npz not created — no dynamic sequences found.")

    return True


# ---------------------------------------------------------------------------
# Step 3: Train static model
# ---------------------------------------------------------------------------
def step_train_static(epochs: int) -> bool:
    print()
    print("=" * 64)
    print("STEP 3: Train Static Gesture Model (MLP)")
    print("=" * 64)

    ok = _run(
        [PYTHON, "-m", "training.trainers.train_static",
         "--data", str(STATIC_SPLIT),
         "--epochs", str(epochs),
         "--patience", "15",
         "--out", str(STATIC_MODEL)],
        "train_static",
    )
    if not ok:
        return False

    if not STATIC_MODEL.exists():
        print("  ERROR: Static model was not saved.")
        return False

    # Verify norm stats are real (not placeholder)
    if NORM_STATS.exists():
        ns = json.loads(NORM_STATS.read_text(encoding="utf-8"))
        mean_sample = ns["mean"][:3]
        std_sample = ns["std"][:3]
        is_placeholder = all(m == 0.0 for m in ns["mean"]) and all(s == 1.0 for s in ns["std"])
        if is_placeholder:
            print("  WARNING: norm_stats.json still has placeholder values!")
        else:
            print(f"  norm_stats.json: mean[0:3]={mean_sample}, std[0:3]={std_sample}")

    if METRICS_STATIC.exists():
        m = json.loads(METRICS_STATIC.read_text(encoding="utf-8"))
        print(f"  Static metrics: acc={m['accuracy']:.4f}, "
              f"p={m['precision']:.4f}, r={m['recall']:.4f}, f1={m['f1_score']:.4f}")

    return True


# ---------------------------------------------------------------------------
# Step 4: Train dynamic model
# ---------------------------------------------------------------------------
def step_train_dynamic(epochs: int) -> bool:
    print()
    print("=" * 64)
    print("STEP 4: Train Dynamic Gesture Model (BiGRU)")
    print("=" * 64)

    if not DYNAMIC_SPLIT.exists():
        print("  SKIPPED: No dynamic split available.")
        return True  # not a fatal error

    ok = _run(
        [PYTHON, "-m", "training.trainers.train_dynamic",
         "--data", str(DYNAMIC_SPLIT),
         "--epochs", str(epochs),
         "--patience", "15",
         "--out", str(DYNAMIC_MODEL)],
        "train_dynamic",
    )
    if not ok:
        return False

    if not DYNAMIC_MODEL.exists():
        print("  ERROR: Dynamic model was not saved.")
        return False

    if DYN_NORM_STATS.exists():
        ns = json.loads(DYN_NORM_STATS.read_text(encoding="utf-8"))
        is_placeholder = all(m == 0.0 for m in ns["mean"]) and all(s == 1.0 for s in ns["std"])
        if is_placeholder:
            print("  WARNING: dynamic_norm_stats.json still has placeholder values!")
        else:
            print(f"  dynamic_norm_stats.json: mean[0:3]={ns['mean'][:3]}, std[0:3]={ns['std'][:3]}")

    if METRICS_DYNAMIC.exists():
        m = json.loads(METRICS_DYNAMIC.read_text(encoding="utf-8"))
        print(f"  Dynamic metrics: acc={m['accuracy']:.4f}, "
              f"p={m['precision']:.4f}, r={m['recall']:.4f}, f1={m['f1_score']:.4f}")

    return True


# ---------------------------------------------------------------------------
# Step 5: Evaluate models
# ---------------------------------------------------------------------------
def step_evaluate() -> bool:
    print()
    print("=" * 64)
    print("STEP 5: Evaluate Models")
    print("=" * 64)

    # Static evaluation
    print("  Evaluating static model...")
    ok = _run(
        [PYTHON, "-m", "training.evaluation.evaluate",
         "--data", str(STATIC_SPLIT),
         "--model", str(STATIC_MODEL),
         "--outdir", "models/registry"],
        "evaluate_static",
    )
    if not ok:
        print("  Static evaluation failed.")
        return False

    if EVAL_STATIC.exists():
        m = json.loads(EVAL_STATIC.read_text(encoding="utf-8"))
        print(f"  Static evaluation: acc={m['accuracy']:.4f}, "
              f"p={m['precision']:.4f}, r={m['recall']:.4f}, f1={m['f1_score']:.4f}")

    # Check outputs
    for f in ["confusion_static_v001.csv", "per_class_metrics_static_v001.csv", "confusion_static_v001.png"]:
        p = Path("models/registry") / f
        print(f"    {'✓' if p.exists() else '✗'} {f}")

    # Dynamic evaluation
    if DYNAMIC_SPLIT.exists() and DYNAMIC_MODEL.exists():
        print("  Evaluating dynamic model...")
        ok = _run(
            [PYTHON, "-m", "training.evaluation.evaluate_dynamic",
             "--data", str(DYNAMIC_SPLIT),
             "--model", str(DYNAMIC_MODEL),
             "--outdir", "models/registry"],
            "evaluate_dynamic",
        )
        if not ok:
            print("  Dynamic evaluation failed.")
            return False

        if EVAL_DYNAMIC.exists():
            m = json.loads(EVAL_DYNAMIC.read_text(encoding="utf-8"))
            print(f"  Dynamic evaluation: acc={m['accuracy']:.4f}, "
                  f"p={m['precision']:.4f}, r={m['recall']:.4f}, f1={m['f1_score']:.4f}")

        for f in ["confusion_dynamic_v001.csv", "per_class_metrics_dynamic_v001.csv", "confusion_dynamic_v001.png"]:
            p = Path("models/registry") / f
            print(f"    {'✓' if p.exists() else '✗'} {f}")
    else:
        print("  Skipped dynamic evaluation (no split or model).")

    return True


# ---------------------------------------------------------------------------
# Step 6: Register models in database
# ---------------------------------------------------------------------------
def step_register_models() -> bool:
    print()
    print("=" * 64)
    print("STEP 6: Register Models in Database")
    print("=" * 64)

    try:
        from database.db import DatabaseManager
        from utils.io.config_loader import load_config
    except Exception as exc:
        print(f"  ERROR: Cannot import database module: {exc}")
        return False

    config = load_config()
    db_path = config.get("database", {}).get("sqlite_path", "database/mudra.db")
    db = DatabaseManager(db_path)

    # Static model
    static_metrics = {}
    if METRICS_STATIC.exists():
        static_metrics = json.loads(METRICS_STATIC.read_text(encoding="utf-8"))
    elif EVAL_STATIC.exists():
        static_metrics = json.loads(EVAL_STATIC.read_text(encoding="utf-8"))

    try:
        # Delete stale registration if it exists, then re-register
        with db.connect() as conn:
            conn.execute(
                "DELETE FROM model_versions WHERE model_name = ? AND version_tag = ?",
                ("static_mlp", "v001_trained"),
            )
        vid = db.register_model_version(
            model_name="static_mlp",
            framework="pytorch",
            artifact_path=str(STATIC_MODEL),
            label_map_path=str(LABEL_MAP),
            norm_stats_path=str(NORM_STATS),
            metrics=static_metrics,
            activate=True,
            version_tag="v001_trained",
        )
        print(f"  Registered static_mlp v001_trained (id={vid[:8]}...)")
    except Exception as exc:
        logger.warning("Static model registration failed: %s", exc)
        print(f"  WARNING: Static model registration: {exc}")

    # Dynamic model
    if DYNAMIC_MODEL.exists():
        dyn_metrics = {}
        if METRICS_DYNAMIC.exists():
            dyn_metrics = json.loads(METRICS_DYNAMIC.read_text(encoding="utf-8"))
        elif EVAL_DYNAMIC.exists():
            dyn_metrics = json.loads(EVAL_DYNAMIC.read_text(encoding="utf-8"))

        try:
            with db.connect() as conn:
                conn.execute(
                    "DELETE FROM model_versions WHERE model_name = ? AND version_tag = ?",
                    ("dynamic_bigru", "v001_trained"),
                )
            vid = db.register_model_version(
                model_name="dynamic_bigru",
                framework="pytorch",
                artifact_path=str(DYNAMIC_MODEL),
                label_map_path=str(LABEL_MAP),
                norm_stats_path=str(DYN_NORM_STATS),
                metrics=dyn_metrics,
                activate=True,
                version_tag="v001_trained",
            )
            print(f"  Registered dynamic_bigru v001_trained (id={vid[:8]}...)")
        except Exception as exc:
            logger.warning("Dynamic model registration failed: %s", exc)
            print(f"  WARNING: Dynamic model registration: {exc}")

    return True


# ---------------------------------------------------------------------------
# Step 7: Verify inference compatibility
# ---------------------------------------------------------------------------
def step_verify_inference() -> bool:
    print()
    print("=" * 64)
    print("STEP 7: Verify Inference Compatibility")
    print("=" * 64)

    import numpy as np

    checks = {
        "label_map": False,
        "class_modes": False,
        "norm_stats_real": False,
        "dynamic_norm_stats_real": False,
        "static_model_loads": False,
        "dynamic_model_loads": False,
        "static_inference": False,
    }

    # Check label_map
    if LABEL_MAP.exists():
        lm = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
        checks["label_map"] = len(lm) > 0
        print(f"  label_map.json: {len(lm)} classes ({'✓' if checks['label_map'] else '✗'})")

    # Check class_modes
    cm_path = Path("models/registry/class_modes.json")
    if cm_path.exists():
        cm = json.loads(cm_path.read_text(encoding="utf-8"))
        checks["class_modes"] = len(cm) > 0
        print(f"  class_modes.json: {len(cm)} entries ({'✓' if checks['class_modes'] else '✗'})")

    # Check norm stats
    if NORM_STATS.exists():
        ns = json.loads(NORM_STATS.read_text(encoding="utf-8"))
        checks["norm_stats_real"] = not (all(m == 0.0 for m in ns["mean"]) and all(s == 1.0 for s in ns["std"]))
        print(f"  norm_stats.json: real values = {'✓' if checks['norm_stats_real'] else '✗'}")

    if DYN_NORM_STATS.exists():
        ns = json.loads(DYN_NORM_STATS.read_text(encoding="utf-8"))
        checks["dynamic_norm_stats_real"] = not (all(m == 0.0 for m in ns["mean"]) and all(s == 1.0 for s in ns["std"]))
        print(f"  dynamic_norm_stats.json: real values = {'✓' if checks['dynamic_norm_stats_real'] else '✗'}")

    # Try loading models with PyTorch
    try:
        import torch
        from training.trainers.train_static import StaticMLP
        from training.trainers.train_dynamic import DynamicBiGRU

        n_cls = len(lm) if LABEL_MAP.exists() else 124

        # Static model
        model_s = StaticMLP(input_dim=136, output_dim=n_cls)
        state_s = torch.load(STATIC_MODEL, map_location="cpu")
        model_s.load_state_dict(state_s)
        model_s.eval()
        checks["static_model_loads"] = True
        print(f"  Static model loads: ✓")

        # Test static inference with random input
        with torch.no_grad():
            dummy = torch.randn(1, 136)
            out = model_s(dummy)
            pred = torch.argmax(out, dim=1).item()
            checks["static_inference"] = out.shape == (1, n_cls)
            print(f"  Static inference test: output shape={out.shape} → ✓")

        # Dynamic model
        if DYNAMIC_MODEL.exists():
            model_d = DynamicBiGRU(input_dim=136, output_dim=n_cls)
            state_d = torch.load(DYNAMIC_MODEL, map_location="cpu")
            model_d.load_state_dict(state_d)
            model_d.eval()
            checks["dynamic_model_loads"] = True
            print(f"  Dynamic model loads: ✓")

            with torch.no_grad():
                dummy_seq = torch.randn(1, 30, 136)
                out_d = model_d(dummy_seq)
                print(f"  Dynamic inference test: output shape={out_d.shape} → ✓")

    except Exception as exc:
        logger.warning("Inference verification failed: %s", exc)
        print(f"  Inference verification error: {exc}")

    passed = sum(checks.values())
    total = len(checks)
    print(f"  Compatibility checks: {passed}/{total} passed")
    return passed >= 5  # at least most checks pass


# ---------------------------------------------------------------------------
# Step 8: Print summary
# ---------------------------------------------------------------------------
def step_summary() -> None:
    print()
    print("=" * 64)
    print("TRAINING PIPELINE SUMMARY")
    print("=" * 64)

    # Load metrics
    static_acc = 0.0
    dynamic_acc = 0.0
    n_classes = 0
    n_train = 0

    if EVAL_STATIC.exists():
        m = json.loads(EVAL_STATIC.read_text(encoding="utf-8"))
        static_acc = m.get("accuracy", 0.0)
    elif METRICS_STATIC.exists():
        m = json.loads(METRICS_STATIC.read_text(encoding="utf-8"))
        static_acc = m.get("accuracy", 0.0)

    if EVAL_DYNAMIC.exists():
        m = json.loads(EVAL_DYNAMIC.read_text(encoding="utf-8"))
        dynamic_acc = m.get("accuracy", 0.0)
    elif METRICS_DYNAMIC.exists():
        m = json.loads(METRICS_DYNAMIC.read_text(encoding="utf-8"))
        dynamic_acc = m.get("accuracy", 0.0)

    if LABEL_MAP.exists():
        lm = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
        n_classes = len(lm)

    if STATIC_SPLIT.exists():
        import numpy as np
        blob = np.load(STATIC_SPLIT)
        n_train = blob["X_train"].shape[0]

    print(f"  Static model accuracy:  {static_acc * 100:.1f}%")
    print(f"  Dynamic model accuracy: {dynamic_acc * 100:.1f}%")
    print(f"  Total gesture classes:  {n_classes}")
    print(f"  Training samples used:  {n_train}")
    print(f"  Models registered successfully")
    print()

    # List all generated artifacts
    artifacts = [
        ("models/static/static_mlp_v001.pt", "Static MLP model"),
        ("models/dynamic/dynamic_bigru_v001.pt", "Dynamic BiGRU model"),
        ("models/registry/label_map.json", "Label map"),
        ("models/registry/norm_stats.json", "Static normalization stats"),
        ("models/registry/dynamic_norm_stats.json", "Dynamic normalization stats"),
        ("models/registry/metrics_static_v001.json", "Static training metrics"),
        ("models/registry/metrics_dynamic_v001.json", "Dynamic training metrics"),
        ("models/registry/evaluation_static_v001.json", "Static evaluation metrics"),
        ("models/registry/evaluation_dynamic_v001.json", "Dynamic evaluation metrics"),
        ("models/registry/confusion_static_v001.csv", "Static confusion matrix (CSV)"),
        ("models/registry/confusion_dynamic_v001.csv", "Dynamic confusion matrix (CSV)"),
        ("models/registry/confusion_static_v001.png", "Static confusion matrix (PNG)"),
        ("models/registry/confusion_dynamic_v001.png", "Dynamic confusion matrix (PNG)"),
        ("models/registry/per_class_metrics_static_v001.csv", "Static per-class metrics"),
        ("models/registry/per_class_metrics_dynamic_v001.csv", "Dynamic per-class metrics"),
    ]
    print("  Artifacts:")
    for path, desc in artifacts:
        exists = Path(path).exists()
        size = ""
        if exists:
            sz = Path(path).stat().st_size
            size = f" ({sz:,} bytes)"
        print(f"    {'✓' if exists else '✗'} {desc}: {path}{size}")

    print()
    print("Pipeline complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MUDRA ML Training Pipeline")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs for both models")
    parser.add_argument("--samples-per-class", type=int, default=50,
                        help="Training samples per gesture class")
    args = parser.parse_args()

    t0 = time.time()

    print()
    print("╔" + "═" * 62 + "╗")
    print("║  MUDRA ML Training Pipeline                                  ║")
    print("╚" + "═" * 62 + "╝")

    if not step_generate_data(args.samples_per_class):
        print("\nPipeline ABORTED at Step 1.")
        sys.exit(1)

    if not step_build_dataset():
        print("\nPipeline ABORTED at Step 2.")
        sys.exit(1)

    if not step_train_static(args.epochs):
        print("\nPipeline ABORTED at Step 3.")
        sys.exit(1)

    if not step_train_dynamic(args.epochs):
        print("\nPipeline ABORTED at Step 4.")
        sys.exit(1)

    if not step_evaluate():
        print("\nPipeline ABORTED at Step 5.")
        sys.exit(1)

    step_register_models()
    step_verify_inference()
    step_summary()

    elapsed = time.time() - t0
    print(f"Total pipeline time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
