"""Seguimiento del balón: un candidato por fotograma, rechazo de saltos imposibles e interpolación de huecos."""

from typing import Optional, Tuple

import numpy as np
import supervision as sv
from sports.common.ball import BallTracker

DETECTED = "detected"
INTERPOLATED = "interpolated"
LOST = "lost"

BALL_TRACKER_BUFFER = 20


class BallFilter:
    """Elige el candidato más coherente (BallTracker del repo) y descarta saltos imposibles."""

    def __init__(self, jump_max_px: float, gap_max: int):
        self.tracker = BallTracker(buffer_size=BALL_TRACKER_BUFFER)
        self.jump_max_px = jump_max_px
        self.gap_max = gap_max
        self.last_xy: Optional[np.ndarray] = None
        self.frames_since_last = 0
        self.n_rejected = 0

    def update(self, candidates: sv.Detections) -> np.ndarray:
        """Devuelve el centro del balón aceptado en este fotograma o (nan, nan)."""
        chosen = self.tracker.update(candidates)
        self.frames_since_last += 1
        if len(chosen) == 0:
            return np.full(2, np.nan)
        xy = chosen.get_anchors_coordinates(sv.Position.CENTER)[0].astype(float)
        if self._is_jump(xy):
            self.n_rejected += 1
            return np.full(2, np.nan)
        self.last_xy = xy
        self.frames_since_last = 0
        return xy

    def _is_jump(self, xy: np.ndarray) -> bool:
        """Salto imposible: más de jump_max_px por fotograma transcurrido desde la última posición."""
        if self.last_xy is None or self.frames_since_last > self.gap_max:
            return False
        allowed = self.jump_max_px * self.frames_since_last
        return float(np.linalg.norm(xy - self.last_xy)) > allowed


def interpolate_ball(xy: np.ndarray, gap_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """Interpola linealmente huecos de hasta gap_max fotogramas entre dos detecciones."""
    out = xy.astype(float).copy()
    detected = np.isfinite(out).all(axis=1)
    states = np.where(detected, DETECTED, LOST).astype(object)
    idx = np.flatnonzero(detected)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= gap_max:
            w = (np.arange(1, gap + 1) / (b - a))[:, None]
            out[a + 1:b] = (1 - w) * out[a] + w * out[b]
            states[a + 1:b] = INTERPOLATED
    return out, states
