"""Lectura del vídeo por tramos y escritura de resultados (CSV, JSON)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import supervision as sv

log = logging.getLogger(__name__)


@dataclass
class ClipRange:
    """Tramo del vídeo a procesar, en índices de fotograma [start, end)."""

    video_path: str
    fps: float
    width: int
    height: int
    start: int
    end: int

    @property
    def n_frames(self) -> int:
        """Número de fotogramas del tramo."""
        return self.end - self.start

    def video_info(self) -> sv.VideoInfo:
        """VideoInfo del tramo, para sv.VideoSink."""
        return sv.VideoInfo(width=self.width, height=self.height, fps=self.fps,
                            total_frames=self.n_frames)


def clip_range(video_path: str, start_s: float, duration_s: Optional[float]) -> ClipRange:
    """Convierte inicio y duración en segundos a un tramo de fotogramas."""
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"No existe el vídeo: {video_path}")
    info = sv.VideoInfo.from_video_path(video_path)
    fps = float(info.fps)
    total = int(info.total_frames or 0)
    start = int(round(start_s * fps))
    end = total if duration_s is None else start + int(round(duration_s * fps))
    if total > 0:
        end = min(end, total)
    if end <= start:
        raise ValueError(f"Tramo vacío: inicio {start_s} s, duración {duration_s} s, "
                         f"el vídeo tiene {total} fotogramas a {fps:.2f} fps")
    return ClipRange(video_path, fps, int(info.width), int(info.height), start, end)


def iter_frames(rng: ClipRange, stride: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
    """Itera (índice absoluto de fotograma, imagen BGR) dentro del tramo."""
    gen = sv.get_video_frames_generator(source_path=rng.video_path, stride=stride,
                                        start=rng.start, end=rng.end)
    for i, frame in enumerate(gen):
        yield rng.start + i * stride, frame


def sample_frame_indices(rng: ClipRange, n: int) -> np.ndarray:
    """Índices de n fotogramas equiespaciados dentro del tramo."""
    n = max(1, min(n, rng.n_frames))
    return np.unique(np.linspace(rng.start, rng.end - 1, n).round().astype(int))


def read_frames_at(rng: ClipRange, indices: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    """Lee secuencialmente el tramo y devuelve solo los fotogramas pedidos."""
    wanted = set(int(i) for i in indices)
    for idx, frame in iter_frames(rng):
        if idx in wanted:
            yield idx, frame


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Escribe un DataFrame en CSV."""
    df.to_csv(path, index=False, float_format="%.4f")
    log.info("Escrito %s (%d filas)", path, len(df))


def _json_default(obj):
    """Convierte tipos de numpy/pathlib a tipos serializables."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"No serializable: {type(obj)}")


def _nan_to_none(obj):
    """Sustituye NaN por None para que el JSON sea estándar."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _nan_to_none(obj.tolist())
    if isinstance(obj, (float, np.floating)) and np.isnan(obj):
        return None
    return obj


def write_json(data: dict, path: Path) -> None:
    """Escribe un diccionario en JSON legible."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(data), f, indent=2, ensure_ascii=False, default=_json_default)
    log.info("Escrito %s", path)


DETECTIONS_CSV = "detections.csv"
BALL_CSV = "ball.csv"
SUMMARY_JSON = "summary.json"


def write_cache(det: pd.DataFrame, ball: pd.DataFrame, out: Path) -> None:
    """Escribe detections.csv y ball.csv con precisión completa (se releen sin pérdida)."""
    det.to_csv(out / DETECTIONS_CSV, index=False)
    ball.to_csv(out / BALL_CSV, index=False)
    log.info("Escritos %s (%d filas) y %s (%d filas)", out / DETECTIONS_CSV, len(det), out / BALL_CSV, len(ball))


def read_cache(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lee detections.csv y ball.csv de una ejecución, con los mismos tipos que al escribirlos."""
    run_dir = Path(run_dir).expanduser()
    for name in (DETECTIONS_CSV, BALL_CSV):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Falta {run_dir / name}: vuelve a lanzar `run` con esta versión del paquete")
    det = pd.read_csv(run_dir / DETECTIONS_CSV, float_precision="round_trip",
                      dtype={"track_id": "Int64", "team": "Int64", "class_id": int, "h_valid": int})
    ball = pd.read_csv(run_dir / BALL_CSV, float_precision="round_trip",
                       dtype={"ball_state": str, "h_valid": int, "h_reused": int})
    return det, ball


def read_json(path: Path) -> dict:
    """Lee un JSON."""
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        return json.load(f)
