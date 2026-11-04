"""Clasificación de equipos por color: baseline del TFG (5 roles + umbral) y versión corregida (torso, K=2, voto por track)."""

import logging
from collections import Counter, deque
from typing import Dict, List, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

from yolofutbol.detect import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID

log = logging.getLogger(__name__)

from yolofutbol.pressure import REFEREE, UNKNOWN  # noqa: E402  (códigos de equipo compartidos)

# --- Baseline (TFG, secciones 4.3 y 5.3.2) ---------------------------------

BASELINE_K = 5
CALIB_SQUARE_FRAC = 0.15   # lado del cuadrado en calibración (Listing 5.4)
INFER_SQUARE_FRAC = 0.40   # lado del cuadrado en inferencia (Listing 5.7)

# --- Versión corregida -------------------------------------------------------

TORSO_Y = (0.15, 0.45)
TORSO_X = (0.25, 0.75)
GRASS_H = (35, 85)         # tono OpenCV (0-179)
GRASS_S_MIN = 40
GRASS_V_MIN = 40
MIN_NON_GRASS_PIXELS = 5


def center_square_hsv_mean(frame: np.ndarray, xyxy: np.ndarray, side_frac: float) -> Optional[np.ndarray]:
    """Media H, S, V de un cuadrado centrado en la caja, de lado side_frac x ancho (como el TFG)."""
    x1, y1, x2, y2 = xyxy
    side = side_frac * (x2 - x1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = frame.shape[:2]
    xa, ya = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    xb, yb = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
    roi = frame[ya:yb, xa:xb]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    return hsv.mean(axis=0)


class BaselineColorClassifier:
    """K-means con 5 roles sobre el color medio del centro de la caja y umbral de color (TFG)."""

    ROLE_NAMES = {0: "EQUIPO_0", 1: "EQUIPO_1", REFEREE: "ARBITRO"}

    def __init__(self, threshold_factor: float = 0.3):
        self.threshold_factor = threshold_factor
        self.centers: Optional[np.ndarray] = None
        self.cluster_team: Dict[int, int] = {}
        self.cluster_role: Dict[int, str] = {}
        self.threshold = np.nan

    def fit(self, feats: np.ndarray, det_class: np.ndarray, frame_id: np.ndarray,
            foot_xy: np.ndarray) -> None:
        """Ajusta k-means (K=5) y asigna cada cluster a un rol de forma automática."""
        km = KMeans(n_clusters=BASELINE_K, n_init=10, random_state=0).fit(feats)
        self.centers = km.cluster_centers_
        labels = km.labels_
        players_per_cluster = np.array([np.sum((labels == c) & (det_class == PLAYER_CLASS_ID))
                                        for c in range(BASELINE_K)])
        team_clusters = list(np.argsort(-players_per_cluster)[:2])
        rest = [c for c in range(BASELINE_K) if c not in team_clusters]
        referee_cluster = max(rest, key=lambda c: np.sum((labels == c) & (det_class == REFEREE_CLASS_ID)))
        self.cluster_team = {int(team_clusters[0]): 0, int(team_clusters[1]): 1,
                             int(referee_cluster): REFEREE}
        self.cluster_role = {int(team_clusters[0]): "EQUIPO_0", int(team_clusters[1]): "EQUIPO_1",
                             int(referee_cluster): "ARBITRO"}
        for c in rest:
            if c == referee_cluster:
                continue
            team = self._goalkeeper_cluster_team(c, labels, frame_id, foot_xy, team_clusters)
            self.cluster_team[int(c)] = team
            self.cluster_role[int(c)] = f"PORTERO_EQUIPO_{team}"
        dists = [np.linalg.norm(self.centers[i] - self.centers[j])
                 for i in range(BASELINE_K) for j in range(i + 1, BASELINE_K)]
        self.threshold = self.threshold_factor * max(dists)
        log.info("Baseline: roles por cluster %s, UMBRAL_COLOR=%.2f", self.cluster_role, self.threshold)

    def _goalkeeper_cluster_team(self, c, labels, frame_id, foot_xy, team_clusters) -> int:
        """Equipo de un cluster de portero: voto por cercanía al centroide de cada equipo en su fotograma."""
        votes = [0, 0]
        for i in np.flatnonzero(labels == c):
            same = frame_id == frame_id[i]
            cents = [foot_xy[same & (labels == tc)] for tc in team_clusters]
            if any(len(p) == 0 for p in cents):
                continue
            d = [np.linalg.norm(foot_xy[i] - p.mean(axis=0)) for p in cents]
            votes[int(np.argmin(d))] += 1
        if sum(votes) == 0:
            d = [np.linalg.norm(self.centers[c] - self.centers[tc]) for tc in team_clusters]
            return int(np.argmin(d))
        return int(np.argmax(votes))

    def predict(self, feat: Optional[np.ndarray]) -> int:
        """Equipo (0/1), REFEREE o UNKNOWN si el color está lejos de todos los prototipos."""
        if feat is None:
            return UNKNOWN
        d = np.linalg.norm(self.centers - feat, axis=1)
        c = int(np.argmin(d))
        return self.cluster_team[c] if d[c] <= self.threshold else UNKNOWN

    def describe(self) -> dict:
        """Parámetros ajustados, para summary.json."""
        return {"k": BASELINE_K, "centers_hsv": self.centers, "cluster_role": self.cluster_role,
                "color_threshold": self.threshold, "threshold_factor": self.threshold_factor,
                "calib_square_frac": CALIB_SQUARE_FRAC, "infer_square_frac": INFER_SQUARE_FRAC}


def torso_feature(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Color del torso sin césped, como media de (S·cos H, S·sin H, V) para respetar que H es circular."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    xa = int(np.clip(x1 + TORSO_X[0] * bw, 0, w_img - 1))
    xb = int(np.clip(x1 + TORSO_X[1] * bw, xa + 1, w_img))
    ya = int(np.clip(y1 + TORSO_Y[0] * bh, 0, h_img - 1))
    yb = int(np.clip(y1 + TORSO_Y[1] * bh, ya + 1, h_img))
    hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    grass = (hue >= GRASS_H[0]) & (hue <= GRASS_H[1]) & (sat >= GRASS_S_MIN) & (val >= GRASS_V_MIN)
    keep = ~grass if (~grass).sum() >= MIN_NON_GRASS_PIXELS else np.ones(len(hsv), dtype=bool)
    ang = hue[keep] * (2 * np.pi / 180.0)
    s, v = sat[keep] / 255.0, val[keep] / 255.0
    return np.array([np.mean(s * np.cos(ang)), np.mean(s * np.sin(ang)), np.mean(v)])


class TorsoTeamClassifier:
    """K-means con K=2 sobre el color del torso; asigna siempre al prototipo más cercano."""

    def __init__(self):
        self.centers: Optional[np.ndarray] = None

    def fit(self, feats: np.ndarray) -> None:
        """Ajusta los dos prototipos de equipo."""
        if len(feats) < 2:
            raise ValueError("Muy pocos recortes de jugadores para ajustar el clasificador de equipos")
        self.centers = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats).cluster_centers_
        log.info("Full: prototipos de equipo (S·cosH, S·sinH, V) = %s", np.round(self.centers, 3).tolist())

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """Equipo 0/1 del prototipo más cercano para cada vector."""
        if len(feats) == 0:
            return np.zeros(0, dtype=int)
        d = np.linalg.norm(feats[:, None, :] - self.centers[None, :, :], axis=2)
        return d.argmin(axis=1)

    def describe(self) -> dict:
        """Parámetros ajustados, para summary.json."""
        return {"k": 2, "centers_scosh_ssinh_v": self.centers, "torso_y": TORSO_Y, "torso_x": TORSO_X,
                "grass_h": GRASS_H, "grass_s_min": GRASS_S_MIN, "grass_v_min": GRASS_V_MIN}


class TeamVoter:
    """Moda de las últimas asignaciones por track id para que un jugador no cambie de equipo."""

    def __init__(self, window: int = 30):
        self.window = window
        self.history: Dict[int, deque] = {}

    def vote(self, track_id: int, team: int) -> int:
        """Añade la asignación del fotograma y devuelve la moda de la ventana."""
        hist = self.history.setdefault(int(track_id), deque(maxlen=self.window))
        hist.append(int(team))
        return Counter(hist).most_common(1)[0][0]


def goalkeeper_teams(gk_xy: np.ndarray, player_xy: np.ndarray, player_team: np.ndarray) -> np.ndarray:
    """Equipo de cada portero: el del centroide de jugadores más cercano (resolve_goalkeepers_team_id)."""
    out = np.full(len(gk_xy), UNKNOWN, dtype=int)
    ok = np.isfinite(player_xy).all(axis=1)
    cents: List[Optional[np.ndarray]] = []
    for t in (0, 1):
        pts = player_xy[ok & (player_team == t)]
        cents.append(pts.mean(axis=0) if len(pts) else None)
    for i, xy in enumerate(gk_xy):
        if not np.isfinite(xy).all():
            continue
        d = [np.linalg.norm(xy - c) if c is not None else np.inf for c in cents]
        if np.isfinite(min(d)):
            out[i] = int(np.argmin(d))
    return out


def assign_teams_full(frame: np.ndarray, xyxy: np.ndarray, class_id: np.ndarray, tracker_id: np.ndarray,
                      ref_xy: np.ndarray, classifier: TorsoTeamClassifier, voter: TeamVoter) -> np.ndarray:
    """Equipo por detección en modo full: jugadores por color, porteros por cercanía, árbitro aparte."""
    teams = np.full(len(xyxy), REFEREE, dtype=int)
    is_player = class_id == PLAYER_CLASS_ID
    is_gk = class_id == GOALKEEPER_CLASS_ID
    if is_player.any():
        feats = np.array([torso_feature(frame, b) for b in xyxy[is_player]])
        teams[is_player] = classifier.predict(feats)
        for i in np.flatnonzero(is_player):
            teams[i] = voter.vote(tracker_id[i], teams[i])
    if is_gk.any():
        gk_team = goalkeeper_teams(ref_xy[is_gk], ref_xy[is_player], teams[is_player])
        missing = gk_team == UNKNOWN
        if missing.any():
            feats = np.array([torso_feature(frame, b) for b in xyxy[is_gk][missing]])
            gk_team[missing] = classifier.predict(feats)
        teams[is_gk] = gk_team
        for i in np.flatnonzero(is_gk):
            teams[i] = voter.vote(tracker_id[i], teams[i])
    return teams
