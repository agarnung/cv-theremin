"""Monocular depth estimation with Depth Anything for the theremin.

Adapted from /opt/smart-screensaver/smart_screensaver/depth_estimator.py

Depth Anything returns relative depth (larger values mean closer to the
camera).  The model runs in a background thread so the camera loop and
MediaPipe tracking never wait for inference.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "LiheYoung/depth-anything-small-hf"


def resolve_device(device: str = "auto"):
    import torch

    if device != "auto":
        return 0 if device == "cuda" else device
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class DepthAnythingEstimator:
    """Relative depth map of a frame. Larger values mean closer."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        input_size: int = 336,
        device: str = "auto",
        half_precision: bool = True,
    ):
        import torch

        self._torch = torch
        self.device = resolve_device(device)
        self.input_size = int(input_size)
        on_cuda = self.device == 0 or str(self.device).startswith("cuda")
        self.dtype = torch.float16 if (half_precision and on_cuda) else torch.float32

        if on_cuda:
            torch.backends.cudnn.benchmark = True

        logger.info(
            "Loading depth model %s on device=%s dtype=%s input_size=%d",
            model, self.device, self.dtype, self.input_size,
        )
        self._pipe = self._load_pipeline(model)
        if not on_cuda:
            logger.warning(
                "Depth Anything is running on %s. Expect slow inference; use a "
                "CUDA build of torch or the pinhole depth backend on CPU-only setups.",
                self.device,
            )
        self._warmup()

    def _load_pipeline(self, model: str):
        from transformers import pipeline

        last_error = None
        for dtype_key in ("dtype", "torch_dtype"):
            for local_only in (True, False, None):
                kwargs = {
                    "task": "depth-estimation",
                    "model": model,
                    "device": self.device,
                    dtype_key: self.dtype,
                }
                if local_only is not None:
                    kwargs["local_files_only"] = local_only
                try:
                    return pipeline(**kwargs)
                except (OSError, TypeError, ValueError) as exc:
                    last_error = exc
        raise last_error

    def _warmup(self) -> None:
        try:
            self.estimate(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
        except Exception as exc:
            logger.warning("Depth Anything warmup failed: %s", exc)

    def estimate(self, frame: np.ndarray, out_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Return a raw relative depth map resized to ``out_size`` (width, height)."""
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

        from PIL import Image

        with self._torch.inference_mode():
            prediction = self._pipe(Image.fromarray(resized))

        raw = prediction["predicted_depth"].detach().float().cpu().numpy()
        if raw.ndim == 3:
            raw = raw[0]
        if out_size is None:
            out_size = (w, h)
        depth = cv2.resize(raw, out_size, interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)


class DepthAnythingTheremin:
    """Threaded Depth Anything backend that maps hand depth to frequency."""

    def __init__(
        self,
        min_frequency: float = 200,
        max_frequency: float = 600,
        model: str = DEFAULT_MODEL,
        input_size: int = 336,
        device: str = "auto",
        half_precision: bool = True,
        inference_ttl: float = 2.0,
        patch_scale: float = 0.25,
    ):
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.inference_ttl = float(inference_ttl)
        self.patch_scale = float(patch_scale)

        self.estimator = DepthAnythingEstimator(
            model=model,
            input_size=input_size,
            device=device,
            half_precision=half_precision,
        )
        on_cuda = self.estimator.device == 0 or str(self.estimator.device).startswith("cuda")
        if not on_cuda:
            self.inference_ttl = max(self.inference_ttl, 5.0)

        self._lock = threading.Lock()
        self._job = None
        self._job_ready = threading.Event()
        self._stop = threading.Event()
        self._depth = None
        self._depth_time = 0.0
        self._busy = False
        self._worker = None

    def start(self):
        if self._worker is None or not self._worker.is_alive():
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._worker_loop, name="depth-anything", daemon=True
            )
            self._worker.start()
        return self

    def stop(self):
        self._stop.set()
        self._job_ready.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True
            self._job = frame.copy()
        self._job_ready.set()

    def _worker_loop(self):
        while not self._stop.is_set():
            if not self._job_ready.wait(0.1):
                continue
            self._job_ready.clear()
            with self._lock:
                frame, self._job = self._job, None
            if frame is None:
                continue

            depth = None
            try:
                depth = self.estimator.estimate(
                    frame, out_size=(frame.shape[1], frame.shape[0])
                )
            except Exception as exc:
                logger.warning("Depth Anything inference failed: %s", exc)

            with self._lock:
                self._busy = False
                if depth is not None:
                    self._depth = depth
                    self._depth_time = time.monotonic()

    def _latest_depth(self):
        with self._lock:
            if self._depth is None:
                return None
            if time.monotonic() - self._depth_time > self.inference_ttl:
                return None
            return self._depth

    @staticmethod
    def _patch_median(depth: np.ndarray, cx: int, cy: int, radius: int) -> float:
        h, w = depth.shape[:2]
        x0 = max(0, int(cx) - radius)
        x1 = min(w, int(cx) + radius + 1)
        y0 = max(0, int(cy) - radius)
        y1 = min(h, int(cy) + radius + 1)
        patch = depth[y0:y1, x0:x1]
        if patch.size == 0:
            return float("nan")
        return float(np.median(patch))

    def update_config(
        self,
        min_frequency=None,
        max_frequency=None,
        inference_ttl=None,
        patch_scale=None,
        **kwargs,
    ):
        """Apply tuning changes without reloading the depth model."""
        if min_frequency is not None:
            self.min_frequency = float(min_frequency)
        if max_frequency is not None:
            self.max_frequency = float(max_frequency)
        if inference_ttl is not None:
            self.inference_ttl = float(inference_ttl)
        if patch_scale is not None:
            self.patch_scale = float(patch_scale)

    def compute_depth_value(self, hand_bbox, hand_center=None):
        """Return the hand depth normalised to 0 (far) - 1 (near), or ``None``."""
        depth = self._latest_depth()
        if depth is None:
            return None

        if hand_center is None:
            x, y, bw, bh = hand_bbox
            hand_center = (x + bw // 2, y + bh // 2)
        cx, cy = hand_center
        bw, bh = hand_bbox[2], hand_bbox[3]

        radius = int(max(4, min(bw, bh) * self.patch_scale))
        hand_depth = self._patch_median(depth, cx, cy, radius)
        d_min, d_max = float(depth.min()), float(depth.max())
        if not np.isfinite(hand_depth) or d_max - d_min <= 1e-6:
            return None
        value = (hand_depth - d_min) / (d_max - d_min)
        return float(np.clip(value, 0.0, 1.0))

    def compute_tone_depth(self, hand_bbox, hand_center=None):
        """Return ``(frequency, depth_value)`` for a hand, or ``(None, None)``."""
        value = self.compute_depth_value(hand_bbox, hand_center)
        if value is None:
            return None, None
        frequency = self.min_frequency + value * (self.max_frequency - self.min_frequency)
        return float(frequency), float(value)
