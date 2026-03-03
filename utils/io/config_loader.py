from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "app_name": "MUDRA",
    "window_title": "MUDRA - Interactive ISL Learning",
    "model": {
        "static_model_path": "models/static/static_mlp_v001.pt",
        "dynamic_model_path": "models/dynamic/dynamic_bigru_v001.pt",
        "label_map_path": "models/registry/label_map.json",
        "norm_stats_path": "models/registry/norm_stats.json",
    },
    "inference": {
        "static_threshold": 0.70,
        "dynamic_threshold": 0.65,
        "smoothing_alpha": 0.6,
        "confirmation_frames": 3,
    },
    "database": {"sqlite_path": "database/mudra.db"},
}


def load_config(path: str = "config/app.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_CONFIG)
    if yaml is None:
        return dict(DEFAULT_CONFIG)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or dict(DEFAULT_CONFIG)


def save_config(config: Dict[str, Any], path: str = "config/app.yaml") -> None:
    if yaml is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
