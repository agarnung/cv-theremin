"""SVG theremin HUD rendered in perspective over the camera frame."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Hud:
    """Render an SVG theremin and overlay live pitch/volume indicators."""

    def __init__(
        self,
        svg_path="assets/theremin_pov.svg",
        scale=0.62,
        opacity=0.85,
        position="bottom_center",
    ):
        self.svg_path = Path(svg_path)
        self.scale = float(scale)
        self.opacity = float(opacity)
        self.position = position
        self._base = None
        self._frame_size = None

    def _render_base(self, width, height):
        import cairosvg

        target_w = max(240, int(width * self.scale))
        target_h = max(180, int(target_w * 0.75))
        png = cairosvg.svg2png(
            url=str(self.svg_path),
            output_width=target_w,
            output_height=target_h,
        )
        image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot render HUD SVG {self.svg_path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        return image

    def _origin(self, frame_w, frame_h, overlay_w, overlay_h):
        margin = max(8, int(frame_w * 0.015))
        if self.position == "bottom_left":
            return margin, frame_h - overlay_h - margin
        if self.position == "bottom_right":
            return frame_w - overlay_w - margin, frame_h - overlay_h - margin
        if self.position == "center":
            return (frame_w - overlay_w) // 2, (frame_h - overlay_h) // 2
        if self.position == "top_center":
            return (frame_w - overlay_w) // 2, margin
        return (frame_w - overlay_w) // 2, frame_h - overlay_h - margin

    def _draw_fallback(self, frame, pitch, volume, frequency, rms):
        cv2.putText(frame, f"Freq {frequency:6.1f} Hz", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vol {volume * 100:5.1f} %", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Out {rms * 100:5.1f} %", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def draw(self, frame, pitch, volume, frequency, rms, hands=None):
        pitch = float(np.clip(pitch, 0.0, 1.0))
        volume = float(np.clip(volume, 0.0, 1.0))
        height, width = frame.shape[:2]

        if self._base is None or self._frame_size != (width, height):
            try:
                self._base = self._render_base(width, height)
                self._frame_size = (width, height)
            except Exception as exc:
                logger.warning("HUD SVG unavailable: %s", exc)
                return self._draw_fallback(frame, pitch, volume, frequency, rms)

        overlay = self._base.copy()
        overlay_h, overlay_w = overlay.shape[:2]
        sx, sy = overlay_w / 800.0, overlay_h / 600.0

        pitch_x = int(565 * sx)
        pitch_base = int(395 * sy)
        pitch_y = int((395 - pitch * 235) * sy)
        cv2.line(overlay, (pitch_x, pitch_base), (pitch_x, pitch_y),
                 (255, 255, 255, 220), max(2, int(4 * sx)), cv2.LINE_AA)
        cv2.circle(overlay, (pitch_x, pitch_y), max(5, int(9 * sx)),
                   (255, 255, 255, 255), -1, cv2.LINE_AA)

        volume_x = int(230 * sx)
        volume_base = int(426 * sy)
        volume_y = int((426 - volume * 106) * sy)
        cv2.line(overlay, (volume_x, volume_base), (volume_x, volume_y),
                 (255, 255, 255, 220), max(2, int(4 * sx)), cv2.LINE_AA)
        cv2.circle(overlay, (volume_x, volume_y), max(5, int(9 * sx)),
                   (255, 255, 255, 255), -1, cv2.LINE_AA)

        x, y = self._origin(width, height, overlay_w, overlay_h)
        x = max(0, min(x, width - overlay_w))
        y = max(0, min(y, height - overlay_h))
        roi = frame[y:y + overlay_h, x:x + overlay_w].astype(np.float32)
        rgb = overlay[:, :, :3].astype(np.float32)
        alpha = (overlay[:, :, 3:4].astype(np.float32) / 255.0) * self.opacity
        frame[y:y + overlay_h, x:x + overlay_w] = (
            roi * (1.0 - alpha) + rgb * alpha
        ).astype(np.uint8)

        for hand in hands or []:
            cx, cy = hand["center"]
            pitch_hand = hand["type"] == "Right"
            color = (255, 246, 57) if pitch_hand else (216, 79, 255)
            label = "P" if pitch_hand else "V"
            cv2.circle(frame, (cx, cy), 18, color, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (cx + 22, cy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        self._draw_panel(frame, frequency, volume, rms)
        return frame

    @staticmethod
    def _draw_panel(frame, frequency, volume, rms):
        height, width = frame.shape[:2]
        panel_w = max(190, int(width * 0.32))
        panel_h = max(92, int(height * 0.23))
        x, y = 16, 16
        roi = frame[y:y + panel_h, x:x + panel_w]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, 0.68, roi, 0.32, 0, roi)
        cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h),
                      (255, 255, 255), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.65, width / 1100.0))
        line_h = max(20, int(panel_h / 3.2))
        text_x = x + max(10, int(panel_w * 0.06))
        text_y = y + line_h
        cv2.putText(frame, f"FREQ {frequency:5.0f} Hz", (text_x, text_y),
                    font, scale, (185, 253, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"VOL  {volume * 100:5.0f} %", (text_x, text_y + line_h),
                    font, scale, (255, 214, 246), 2, cv2.LINE_AA)
        cv2.putText(frame, f"OUT  {rms * 100:5.0f} %", (text_x, text_y + 2 * line_h),
                    font, scale, (109, 255, 138), 2, cv2.LINE_AA)
