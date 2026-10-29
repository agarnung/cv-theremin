"""YAML configuration with hot-reload support."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULTS = {
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "fps_limit": 30,
        "mirror": True,
    },
    "audio": {
        "output_device": None,
        "sample_rate": 48000,
        "channels": 2,
        "buffer_size": 256,
        "duplex": 0,
        "initial_frequency": 440,
        "initial_volume": 0.0,
    },
    "pitch": {
        "mode": "depth_anything",
        "min_frequency": 200,
        "max_frequency": 600,
    },
    "volume": {
        "source": "left",
        "fallback": 0.5,
        "min": 0.0,
        "max": 1.0,
        "gamma": 1.5,
    },
    "depth": {
        "model": "LiheYoung/depth-anything-small-hf",
        "input_size": 336,
        "device": "auto",
        "half_precision": True,
        "inference_ttl": 2.0,
        "patch_scale": 0.25,
    },
    "filter": {
        "enabled": True,
        "frequency_min_cutoff": 1.0,
        "frequency_beta": 0.05,
        "volume_min_cutoff": 2.0,
        "volume_beta": 0.5,
        "d_cutoff": 1.0,
    },
    "hand_tracking": {
        "static_mode": False,
        "max_hands": 2,
        "model_complexity": 1,
        "detection_confidence": 0.5,
        "tracking_confidence": 0.5,
        "flip_type": False,
    },
    "display": {
        "mode": "hud",
    },
    "reload": {
        "enabled": True,
        "check_interval": 1.0,
    },
    "debug": {
        "show_hud": True,
        "hud_svg": "assets/theremin_pov.svg",
        "hud_scale": 0.75,
        "hud_opacity": 0.85,
        "hud_position": "bottom_center",
    },
}


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Load ``config.yaml`` and reload it when the file changes."""

    def __init__(self, path="config.yaml"):
        self.path = Path(path)
        self._mtime = None
        self.data = _merge(DEFAULTS, {})
        self.reload(force=True)

    def reload(self, force=False):
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            if force:
                logger.warning("Config file %s not found; using defaults.", self.path)
            return False

        if not force and mtime == self._mtime:
            return False

        self._mtime = mtime
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Cannot read %s: %s", self.path, exc)
            return False

        self.data = _merge(DEFAULTS, loaded)
        return True
