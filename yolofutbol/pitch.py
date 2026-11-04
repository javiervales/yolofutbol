"""Homografía imagen -> plano del campo (metros) a partir de los puntos clave del modelo de campo."""

from typing import Optional, Tuple

import cv2
import numpy as np
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()
VERTICES_M = np.array(CONFIG.vertices, dtype=np.float32) / 100.0
PITCH_LENGTH_M = CONFIG.length / 100.0
PITCH_WIDTH_M = CONFIG.width / 100.0

KEYPOINT_CONF_MIN = 0.5
MIN_POINTS = 4
MIN_HULL_AREA_M2 = 20.0
RANSAC_REPROJ_M = 1.0


def valid_keypoints(kp_xy: Optional[np.ndarray], kp_conf: Optional[np.ndarray],
                    conf_min: float = KEYPOINT_CONF_MIN) -> np.ndarray:
    """Máscara de puntos clave utilizables: confianza > conf_min y distintos de (0,0)."""
    if kp_xy is None:
        return np.zeros(len(VERTICES_M), dtype=bool)
    return (kp_conf > conf_min) & (kp_xy[:, 0] > 1) & (kp_xy[:, 1] > 1)


def estimate_homography(kp_xy: Optional[np.ndarray], kp_conf: Optional[np.ndarray],
                        conf_min: float = KEYPOINT_CONF_MIN) -> Optional[np.ndarray]:
    """Estima H (imagen -> metros) con RANSAC; None si no hay puntos suficientes o son degenerados."""
    mask = valid_keypoints(kp_xy, kp_conf, conf_min)
    if mask.sum() < MIN_POINTS:
        return None
    src = kp_xy[mask].astype(np.float32)
    dst = VERTICES_M[mask]
    hull_area = cv2.contourArea(cv2.convexHull(dst.reshape(-1, 1, 2)))
    if hull_area < MIN_HULL_AREA_M2:
        return None
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_M)
    if H is None or inliers is None or int(inliers.sum()) < MIN_POINTS:
        return None
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


class HomographyTracker:
    """Mantiene la última H válida y, opcionalmente, la suaviza con media exponencial."""

    def __init__(self, ema_alpha: float = 1.0, kp_conf_min: float = KEYPOINT_CONF_MIN):
        self.ema_alpha = ema_alpha
        self.kp_conf_min = kp_conf_min
        self.H: Optional[np.ndarray] = None

    def update(self, kp_xy, kp_conf) -> Tuple[Optional[np.ndarray], bool, bool]:
        """Devuelve (H a usar, h_valid, h_reused) para este fotograma."""
        H_new = estimate_homography(kp_xy, kp_conf, self.kp_conf_min)
        if H_new is not None:
            if self.H is not None and self.ema_alpha < 1.0:
                H_new = self.ema_alpha * H_new + (1.0 - self.ema_alpha) * self.H
                H_new = H_new / H_new[2, 2]
            self.H = H_new
            return self.H, True, False
        if self.H is not None:
            return self.H, False, True
        return None, False, False


def project(H: Optional[np.ndarray], xy: np.ndarray) -> np.ndarray:
    """Proyecta puntos de imagen (N,2) al campo en metros; NaN si no hay H."""
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if H is None or len(xy) == 0:
        return np.full((len(xy), 2), np.nan)
    finite = np.isfinite(xy).all(axis=1)
    out = np.full((len(xy), 2), np.nan)
    if finite.any():
        pts = cv2.perspectiveTransform(xy[finite].reshape(-1, 1, 2), H.astype(np.float64))
        out[finite] = pts.reshape(-1, 2)
    return out
