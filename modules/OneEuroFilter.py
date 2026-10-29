"""One Euro filter for smoothing theremin control signals.

Adapted from /opt/visual_control_robot/src/vision/one_euro.py

Reduces jitter without adding lag during fast movements:
high speed -> high cutoff (reactive); low speed -> low cutoff (smooth).

Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems", CHI 2012.
"""

import math


class OneEuroFilter:
    """Scalar 1€ filter for a single control signal."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._x = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-9))

    def filter(self, x, dt):
        x = float(x)
        if self._x is None:
            self._x = x
            return x
        dt = max(float(dt), 1e-6)
        a_d = self._alpha(self._d_cutoff, dt)
        dx = (x - self._x) / dt
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self._min_cutoff + self._beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x

    def reset(self):
        self._x = None
        self._dx = 0.0
