"""Carga de los tres modelos YOLO y detección por fotograma (personas, balón, puntos del campo)."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO

log = logging.getLogger(__name__)

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

PLAYER_WEIGHTS = "football-player-detection.pt"
BALL_WEIGHTS = "football-ball-detection.pt"
PITCH_WEIGHTS = "football-pitch-detection.pt"

PLAYER_IMGSZ = 1280
BALL_SLICE_WH = (640, 640)
BALL_NMS_IOU = 0.1


def _load(weights_dir: Path, name: str) -> YOLO:
    """Carga un modelo YOLO comprobando antes que el fichero existe."""
    path = Path(weights_dir).expanduser() / name
    if not path.is_file():
        raise FileNotFoundError(f"No encuentro los pesos {path}")
    log.info("Cargando %s", path)
    return YOLO(str(path))


class Detectors:
    """Agrupa los modelos de jugadores, balón y campo con sus parámetros de inferencia."""

    def __init__(self, weights_dir: Path, device: Optional[str], conf: float,
                 ball_conf: float, use_ball_model: bool, use_pitch_model: bool,
                 player_weights: Optional[Path] = None):
        self.device = device
        self.conf = conf
        self.ball_conf = ball_conf
        if player_weights is None:
            player_weights = Path(weights_dir).expanduser() / PLAYER_WEIGHTS
        self.player_weights = Path(player_weights).expanduser()
        self.player_model = _load(self.player_weights.parent, self.player_weights.name)
        self.ball_model = _load(weights_dir, BALL_WEIGHTS) if use_ball_model else None
        self.pitch_model = _load(weights_dir, PITCH_WEIGHTS) if use_pitch_model else None
        self.slicer = None
        if self.ball_model is not None:
            self.slicer = sv.InferenceSlicer(callback=self._ball_slice_callback,
                                             slice_wh=BALL_SLICE_WH,
                                             overlap_filter=sv.OverlapFilter.NONE)

    def detect_all(self, frame: np.ndarray) -> sv.Detections:
        """Detecciones del modelo de jugadores (clases balón, portero, jugador, árbitro)."""
        result = self.player_model.predict(frame, imgsz=PLAYER_IMGSZ, conf=self.conf,
                                           device=self.device, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def _ball_slice_callback(self, image_slice: np.ndarray) -> sv.Detections:
        """Inferencia del detector de balón sobre una rodaja de 640x640."""
        result = self.ball_model.predict(image_slice, imgsz=BALL_SLICE_WH[0],
                                         conf=self.ball_conf, device=self.device,
                                         verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        """Candidatos a balón del detector específico, por rodajas y con NMS."""
        return self.slicer(frame).with_nms(threshold=BALL_NMS_IOU)

    def detect_pitch(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Puntos clave del campo (32x2) y su confianza (32), o (None, None)."""
        result = self.pitch_model.predict(frame, device=self.device, verbose=False)[0]
        kp = result.keypoints
        if kp is None or kp.xy is None or kp.xy.shape[0] == 0:
            return None, None
        xy = kp.xy[0].cpu().numpy().astype(np.float32)
        if kp.conf is None:
            conf = np.ones(len(xy), dtype=np.float32)
        else:
            conf = kp.conf[0].cpu().numpy().astype(np.float32)
        return xy, conf


def split_people_and_ball(detections: sv.Detections) -> Tuple[sv.Detections, sv.Detections]:
    """Separa personas (portero, jugador, árbitro) de balones."""
    is_ball = detections.class_id == BALL_CLASS_ID
    return detections[~is_ball], detections[is_ball]
