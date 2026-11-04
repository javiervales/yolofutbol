"""Posesión (por fotograma o por equipo), índice de presión, secuencias de posesión, picos y agregados."""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from yolofutbol.track_ball import DETECTED, INTERPOLATED, LOST

CARRIED = "carried"   # balón no visto, asumido en los pies de quien lo llevaba (solo en frames.csv)

UNKNOWN = -1          # equipo sin asignar (solo baseline)
REFEREE = -2          # árbitro: fuera del cálculo
TEAM_CLASS_IDS = (1, 2)   # portero y jugador (detect.py)


@dataclass
class FramePressure:
    """Resultado de posesión y presión en un fotograma."""

    possessor_idx: Optional[int] = None
    possessor_team: Optional[int] = None
    team_possession: Optional[int] = None
    possession_state: str = "none"
    n_pressers: int = 0
    n_pressers_active: Optional[int] = None
    nearest_rival_dist: float = np.nan
    presser_idx: List[int] = field(default_factory=list)


def _pressers(pressure_xy: np.ndarray, teams: np.ndarray, ball_xy: np.ndarray, possessing_team: int,
              dist_pressure: float) -> Tuple[List[int], float]:
    """Rivales del equipo poseedor a ≤ umbral del balón y distancia del rival más cercano."""
    rivals = np.flatnonzero(teams == 1 - possessing_team)
    d_riv = np.linalg.norm(pressure_xy[rivals] - ball_xy, axis=1)
    finite = np.isfinite(d_riv)
    rivals, d_riv = rivals[finite], d_riv[finite]
    if len(rivals) == 0:
        return [], np.nan
    return [int(i) for i in rivals[d_riv <= dist_pressure]], float(d_riv.min())


def frame_pressure(possession_xy: np.ndarray, pressure_xy: np.ndarray, teams: np.ndarray,
                   ball_xy: np.ndarray, dist_possession: float, dist_pressure: float) -> FramePressure:
    """Poseedor = jugador con equipo más cercano al balón (≤ umbral); presión = rivales a ≤ umbral del balón."""
    res = FramePressure()
    if not np.isfinite(ball_xy).all() or len(teams) == 0:
        return res
    has_team = (teams == 0) | (teams == 1)
    d_pos = np.linalg.norm(possession_xy - ball_xy, axis=1)
    d_pos[~has_team | ~np.isfinite(d_pos)] = np.inf
    best = int(np.argmin(d_pos))
    if not d_pos[best] <= dist_possession:
        return res
    res.possessor_idx = best
    res.possessor_team = int(teams[best])
    res.team_possession = res.possessor_team
    res.possession_state = "controlled"
    res.presser_idx, res.nearest_rival_dist = _pressers(pressure_xy, teams, ball_xy, res.possessor_team,
                                                        dist_pressure)
    res.n_pressers = len(res.presser_idx)
    return res


# --- Posesión por equipo (ventanas entre toques controlados) -----------------

CONTROL_LOST_TOLERANCE = 2   # fotogramas de balón perdido que no rompen la cuenta de control


@dataclass
class PossessionWindow:
    """Ventana de posesión de un equipo, en índices de fotograma del tramo (ambos incluidos)."""

    team: int
    start: int
    end: int = -1
    n_touches: int = 0
    end_reason: str = "clip_end"


def team_possession_windows(feet: List[np.ndarray], teams: List[np.ndarray], track_ids: List[np.ndarray],
                            ball: np.ndarray, dist_possession: float, control_frames: int,
                            gap_max: int) -> Tuple[List[PossessionWindow], int]:
    """Ventanas de posesión (abre con un toque controlado, cierra con el del rival o al perder el balón) y nº de cortas descartadas."""
    windows: List[PossessionWindow] = []
    cur: Optional[PossessionWindow] = None
    count, start_of = {}, {}
    lost_run, last_known, n_short = 0, -1, 0
    for i in range(len(ball)):
        if not np.isfinite(ball[i]).all():
            lost_run += 1
            if lost_run > CONTROL_LOST_TOLERANCE:
                count.clear()
                start_of.clear()
            if cur is not None and lost_run > gap_max:
                n_short += _close_window(windows, cur, last_known, "ball_lost", control_frames)
                cur = None
            continue
        lost_run, last_known = 0, i
        near = _near_players(feet[i], teams[i], track_ids[i], ball[i], dist_possession)
        for tid in [t for t in count if t not in near]:
            del count[tid], start_of[tid]
        onsets = []
        for tid, (team, d) in near.items():
            if tid not in count:
                count[tid], start_of[tid] = 0, i
            count[tid] += 1
            if count[tid] == control_frames:
                onsets.append((d, team, start_of[tid]))
        if not onsets:
            continue
        _, team, start = min(onsets)
        if windows and windows[-1].end >= start:
            start = windows[-1].end + 1
        if cur is None:
            cur = PossessionWindow(team=team, start=start)
        elif team != cur.team:
            cur, short = _switch_team(windows, cur, team, start, control_frames)
            n_short += short
        cur.n_touches += sum(1 for o in onsets if o[1] == cur.team)
    if cur is not None:
        n_short += _close_window(windows, cur, last_known, "clip_end", control_frames)
    return windows, n_short


def _close_window(windows: List[PossessionWindow], cur: PossessionWindow, end: int, reason: str,
                  control_frames: int) -> int:
    """Cierra la ventana; si dura menos que un toque controlado la descarta y devuelve 1."""
    cur.end, cur.end_reason = end, reason
    if end - cur.start + 1 < control_frames:
        return 1
    windows.append(cur)
    return 0


def _switch_team(windows: List[PossessionWindow], cur: PossessionWindow, team: int, start: int,
                 control_frames: int) -> Tuple[PossessionWindow, int]:
    """Cambio de equipo; una ventana saliente más corta que control_s se descarta y su tramo lo absorbe la vecina."""
    if start - cur.start >= control_frames:
        cur.end, cur.end_reason = start - 1, "lost_to_opponent"
        windows.append(cur)
        return PossessionWindow(team=team, start=start), 0
    prev = windows[-1] if windows else None
    if prev is not None and prev.team == team and prev.end == cur.start - 1:
        windows.pop()
        prev.end, prev.end_reason = -1, "clip_end"
        return prev, 1
    return PossessionWindow(team=team, start=cur.start), 1


def _near_players(feet: np.ndarray, teams: np.ndarray, track_ids: np.ndarray, ball_xy: np.ndarray,
                  dist_possession: float) -> dict:
    """{track_id: (equipo, distancia)} del jugador con equipo más cercano al balón, si está a ≤ umbral de posesión.

    Solo el más cercano suma para el toque controlado; a los demás se les reinicia la cuenta.
    """
    if len(teams) == 0:
        return {}
    d = np.linalg.norm(feet - ball_xy, axis=1)
    d[~((teams == 0) | (teams == 1)) | ~np.isfinite(d)] = np.inf
    k = int(np.argmin(d))
    if not d[k] <= dist_possession:
        return {}
    return {int(track_ids[k]): (int(teams[k]), float(d[k]))}


def control_frames_from_seconds(control_s: float, fps: float) -> int:
    """Fotogramas consecutivos necesarios para un toque controlado."""
    return max(1, math.ceil(control_s * fps - 1e-9))


def window_frame_pressure(feet: np.ndarray, teams: np.ndarray, ball_xy: np.ndarray, team: Optional[int],
                          dist_possession: float, dist_pressure: float) -> FramePressure:
    """Presión dentro de una ventana: rivales del equipo poseedor cerca del balón, haya o no poseedor individual."""
    res = FramePressure()
    if team is None:
        return res
    res.team_possession = team
    res.possession_state = "in_flight"
    if not np.isfinite(ball_xy).all():
        return res
    d = np.linalg.norm(feet - ball_xy, axis=1) if len(teams) else np.zeros(0)
    mates = np.flatnonzero((teams == team) & np.isfinite(d) & (d <= dist_possession))
    if len(mates):
        res.possessor_idx = int(mates[np.argmin(d[mates])])
        res.possessor_team = team
        res.possession_state = "controlled"
    res.presser_idx, res.nearest_rival_dist = _pressers(feet, teams, ball_xy, team, dist_pressure)
    res.n_pressers = len(res.presser_idx)
    return res


def window_sequences(windows: List[PossessionWindow], df: pd.DataFrame, fps: float,
                     dist_col: str) -> pd.DataFrame:
    """Una fila por ventana de posesión de equipo, con presión, toques y cómo acabó."""
    rows = []
    for k, w in enumerate(windows):
        sub = df.iloc[w.start:w.end + 1]
        pressers = sub["n_pressers"].to_numpy()
        first = sub["frame"].to_numpy()[pressers >= 1]
        start_frame = int(sub["frame"].iloc[0])
        rows.append({
            "seq_id": k, "team": w.team,
            "start_frame": start_frame, "end_frame": int(sub["frame"].iloc[-1]),
            "start_t_s": float(sub["t_s"].iloc[0]),
            "duration_s": len(sub) / fps,
            "n_frames": len(sub),
            "n_controlled_frames": int((sub["possession_state"] == "controlled").sum()),
            "n_touches": w.n_touches,
            "mean_pressers": float(pressers.mean()),
            "max_pressers": int(pressers.max()),
            "time_to_first_presser_s": (first[0] - start_frame) / fps if len(first) else np.nan,
            f"mean_{dist_col}": float(sub[dist_col].mean()) if sub[dist_col].notna().any() else np.nan,
            "end_reason": w.end_reason,
        })
    return pd.DataFrame(rows)


def possession_sequences(df: pd.DataFrame, fps: float, gap_max_frames: int, dist_col: str) -> pd.DataFrame:
    """Agrupa fotogramas en secuencias del mismo equipo poseedor (tolera huecos sin poseedor ≤ gap)."""
    rows = []
    cur = None
    with_pos = df[df["possessor_team"].notna()]
    for r in with_pos.itertuples(index=False):
        team, frame = int(r.possessor_team), int(r.frame)
        if cur is not None and (team != cur["team"] or frame - cur["last"] - 1 > gap_max_frames):
            rows.append(_close_sequence(cur, fps))
            cur = None
        if cur is None:
            cur = {"team": team, "start": frame, "last": frame, "pressers": [], "dists": [], "frames": []}
        cur["last"] = frame
        cur["frames"].append(frame)
        cur["pressers"].append(int(r.n_pressers))
        cur["dists"].append(getattr(r, dist_col))
    if cur is not None:
        rows.append(_close_sequence(cur, fps))
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "seq_id", np.arange(len(out)))
        out = out.rename(columns={"mean_nearest_presser_dist": f"mean_{dist_col}"})
    return out


def _close_sequence(cur: dict, fps: float) -> dict:
    """Métricas de una secuencia de posesión cerrada."""
    pressers = np.array(cur["pressers"])
    frames = np.array(cur["frames"])
    dists = np.array(cur["dists"], dtype=float)
    first = frames[pressers >= 1]
    return {
        "team": cur["team"],
        "start_frame": cur["start"],
        "end_frame": cur["last"],
        "start_t_s": cur["start"] / fps,
        "duration_s": (cur["last"] - cur["start"] + 1) / fps,
        "n_possessor_frames": len(frames),
        "mean_pressers": float(pressers.mean()),
        "max_pressers": int(pressers.max()),
        "time_to_first_presser_s": (first[0] - cur["start"]) / fps if len(first) else np.nan,
        "mean_nearest_presser_dist": float(np.nanmean(dists)) if np.isfinite(dists).any() else np.nan,
    }


def possession_column(df: pd.DataFrame) -> str:
    """Columna que define los fotogramas con posesión (team_possession si existe)."""
    return "team_possession" if "team_possession" in df.columns else "possessor_team"


def pressure_series(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Serie del índice en fotogramas con posesión: crudo, media móvil y media acumulada."""
    col = possession_column(df)
    sub = df[df[col].notna()][["frame", "t_s", "n_pressers", col]].copy()
    sub = sub.rename(columns={col: "possession_team"})
    sub["rolling"] = sub["n_pressers"].rolling(window, min_periods=1, center=True).mean()
    sub["cumulative"] = sub["n_pressers"].expanding().mean()
    return sub


def pct(x: float, total: float) -> float:
    """Porcentaje seguro ante total cero."""
    return float(100.0 * x / total) if total else float("nan")


def pressure_stats(df: pd.DataFrame, seq: pd.DataFrame) -> dict:
    """Agregados del índice de presión y de las secuencias para summary.json."""
    n = len(df)
    col = possession_column(df)
    pos = df[df[col].notna()].rename(columns={col: "possession_team"})
    stats = {
        "frames_with_possessor_pct": pct(df["possessor_team"].notna().sum(), n),
        "frames_with_team_possession_pct": pct(len(pos), n),
        "pressure_index_mean": float(pos["n_pressers"].mean()) if len(pos) else float("nan"),
        "pressure_index_max": int(pos["n_pressers"].max()) if len(pos) else 0,
        "possession_share_pct": {f"team{t}": pct((pos["possession_team"] == t).sum(), len(pos)) for t in (0, 1)},
        "pressure_index_mean_by_possessor_team": {
            f"team{t}": float(pos.loc[pos["possession_team"] == t, "n_pressers"].mean())
            if (pos["possession_team"] == t).any() else float("nan") for t in (0, 1)},
        "n_sequences": int(len(seq)),
    }
    if len(seq) and "time_to_first_presser_s" in seq:
        stats["sequence_duration_s_mean"] = float(seq["duration_s"].mean())
        stats["time_to_first_presser_s_mean"] = float(seq["time_to_first_presser_s"].mean())
        stats["sequences_with_presser_pct"] = pct(seq["time_to_first_presser_s"].notna().sum(), len(seq))
    return stats


# --- Picos de presión (validación del TFG, cuadro 6.4) -------------------------

PEAK_RAW_WINDOW_S = 0.5
PEAK_COLUMNS = ["peak_id", "t_s", "frame", "rolling", "raw_max_pm05s", "possessor_team",
                "prominence", "width_s", "edge"]


def _find_peaks_simple(x: np.ndarray, prominence: float):
    """Máximos locales con prominencia y anchura a media prominencia (respaldo sin scipy)."""
    idx, prom, left, right = [], [], [], []
    n = len(x)
    for i in range(1, n - 1):
        if not (x[i] > x[i - 1] and x[i] >= x[i + 1]):
            continue
        lo = i
        while lo > 0 and x[lo - 1] <= x[i]:
            lo -= 1
        hi = i
        while hi < n - 1 and x[hi + 1] <= x[i]:
            hi += 1
        p = x[i] - max(x[lo:i + 1].min(), x[i:hi + 1].min())
        if p < prominence:
            continue
        level = x[i] - 0.5 * p
        a = i
        while a > 0 and x[a - 1] > level:
            a -= 1
        b = i
        while b < n - 1 and x[b + 1] > level:
            b += 1
        la = a - (x[a] - level) / (x[a] - x[a - 1]) if a > 0 and x[a] != x[a - 1] else float(a)
        rb = b + (x[b] - level) / (x[b] - x[b + 1]) if b < n - 1 and x[b] != x[b + 1] else float(b)
        idx.append(i), prom.append(p), left.append(la), right.append(rb)
    return np.array(idx, dtype=int), np.array(prom), np.array(left), np.array(right)


def _detect_peaks(x: np.ndarray, prominence: float):
    """Índices, prominencias y bordes (fraccionarios) a media prominencia; scipy si está disponible."""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return _find_peaks_simple(x, prominence)
    peaks, props = find_peaks(x, prominence=prominence, width=0, rel_height=0.5)
    return peaks, props["prominences"], props["left_ips"], props["right_ips"]


def _enforce_separation(t: np.ndarray, height: np.ndarray, min_sep_s: float) -> np.ndarray:
    """Máscara de picos que respetan la separación mínima en tiempo, priorizando los más altos."""
    keep = np.zeros(len(t), dtype=bool)
    for i in np.argsort(-height, kind="stable"):
        if not np.any(keep & (np.abs(t - t[i]) < min_sep_s)):
            keep[i] = True
    return keep


def find_pressure_peaks(df: pd.DataFrame, window: int, prominence: float, min_sep_s: float) -> pd.DataFrame:
    """Picos de la media móvil del índice (solo fotogramas con poseedor) con prominencia y separación mínimas."""
    s = pressure_series(df, window).reset_index(drop=True)
    if len(s) < 2:
        return pd.DataFrame(columns=PEAK_COLUMNS)
    x = s["rolling"].to_numpy(dtype=float)
    t = s["t_s"].to_numpy(dtype=float)
    idx, prom, left, right = _detect_peaks(x, prominence) if len(s) >= 3 else (np.zeros(0, int),) + (np.zeros(0),) * 3
    edge = np.zeros(len(idx), dtype=int)
    e_idx, e_prom = _edge_peaks(s, x, t, prominence, min_sep_s)
    new = ~np.isin(e_idx, idx)
    idx = np.concatenate([idx, e_idx[new]]).astype(int)
    prom = np.concatenate([prom, e_prom[new]])
    left = np.concatenate([left, np.full(new.sum(), np.nan)])
    right = np.concatenate([right, np.full(new.sum(), np.nan)])
    edge = np.concatenate([edge, np.ones(new.sum(), dtype=int)])
    if len(idx) == 0:
        return pd.DataFrame(columns=PEAK_COLUMNS)
    keep = _enforce_separation(t[idx], x[idx], min_sep_s)
    idx, prom, left, right, edge = idx[keep], prom[keep], left[keep], right[keep], edge[keep]
    order = np.argsort(idx)
    idx, prom, left, right, edge = idx[order], prom[order], left[order], right[order], edge[order]
    samples = np.arange(len(t))
    rows = []
    for k, (i, p, a, b, e) in enumerate(zip(idx, prom, left, right, edge)):
        near = np.abs(t - t[i]) <= PEAK_RAW_WINDOW_S
        rows.append({
            "peak_id": k, "t_s": t[i], "frame": int(s.at[i, "frame"]), "rolling": x[i],
            "raw_max_pm05s": int(s.loc[near, "n_pressers"].max()),
            "possessor_team": int(s.at[i, "possession_team"]),
            "prominence": float(p),
            "width_s": float(np.interp(b, samples, t) - np.interp(a, samples, t)) if not e else np.nan,
            "edge": int(e),
        })
    return pd.DataFrame(rows, columns=PEAK_COLUMNS)


def _edge_peaks(s: pd.DataFrame, x: np.ndarray, t: np.ndarray, prominence: float,
                min_sep_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Picos en el primer o último fotograma de cada tramo continuo de posesión (find_peaks no mira los bordes).

    Un tramo acaba donde se salta un fotograma o cambia el equipo poseedor. El borde inicial es pico si supera
    en ≥ prominence al mínimo de los min_sep_s segundos posteriores del tramo; el final, al de los anteriores.
    """
    frames = s["frame"].to_numpy()
    team = s["possession_team"].to_numpy()
    cut = np.flatnonzero((np.diff(frames) != 1) | (team[1:] != team[:-1])) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut - 1, [len(s) - 1]])
    idx, prom = [], []
    for a, b in zip(starts, ends):
        if b <= a:
            continue
        seg = np.arange(a, b + 1)
        for i, ref in ((a, seg[(t[seg] > t[a]) & (t[seg] <= t[a] + min_sep_s)]),
                       (b, seg[(t[seg] < t[b]) & (t[seg] >= t[b] - min_sep_s)])):
            if len(ref) and x[i] - x[ref].min() >= prominence:
                idx.append(i)
                prom.append(float(x[i] - x[ref].min()))
    return np.array(idx, dtype=int), np.array(prom, dtype=float)


# --- Análisis completo desde las tablas de caché (run, replay, tune) -----------

DEFAULT_PARAMS = {
    "possession": None, "dist_posesion_px": 50.0, "dist_presion_px": 150.0,
    "dist_posesion_m": 2.0, "dist_presion_m": 5.0, "control_s": 0.4, "ball_gap_max": 15,
    "seq_gap_s": 1.0, "smooth_window": 25, "peak_prominence": 0.5, "peak_min_sep_s": 1.0,
    "carry_max_s": 3.0, "closing_min_mps": 0.0, "closing_window_s": 0.5, "dist_presion_core_m": 2.5,
}
TUNABLE_PARAMS = ("control_s", "dist_posesion_m", "dist_presion_m", "dist_posesion_px", "dist_presion_px",
                  "smooth_window", "peak_prominence", "peak_min_sep_s", "carry_max_s", "closing_min_mps",
                  "closing_window_s", "dist_presion_core_m")


@dataclass
class PressureParams:
    """Parámetros de la etapa de posesión y presión, ya resueltos para un modo."""

    mode: str
    possession: str
    dist_possession: float
    dist_pressure: float
    control_s: float
    ball_gap_max: int
    seq_gap_s: float
    smooth_window: int
    peak_prominence: float
    peak_min_sep_s: float
    carry_max_s: float = 0.0
    closing_min_mps: float = 0.0
    closing_window_s: float = 0.5
    dist_pressure_core: float = 2.5

    @property
    def dist_col(self) -> str:
        """Nombre de la columna de distancia según la unidad del modo."""
        return "nearest_presser_dist_px" if self.mode == "baseline" else "nearest_presser_dist_m"


def pressure_params(mode: str, params: dict) -> PressureParams:
    """Resuelve los parámetros (CLI, summary.json o rejilla) para un modo; los None toman el valor por defecto."""
    p = {**DEFAULT_PARAMS, **{k: v for k, v in params.items() if k in DEFAULT_PARAMS and v is not None}}
    possession = p["possession"] or ("team" if mode == "full" else "frame")
    if possession == "team" and mode != "full":
        raise ValueError("La posesión por equipo (--possession team) solo está disponible en modo full")
    unit = "px" if mode == "baseline" else "m"
    return PressureParams(mode=mode, possession=possession,
                          dist_possession=float(p[f"dist_posesion_{unit}"]),
                          dist_pressure=float(p[f"dist_presion_{unit}"]),
                          control_s=float(p["control_s"]), ball_gap_max=int(p["ball_gap_max"]),
                          seq_gap_s=float(p["seq_gap_s"]), smooth_window=int(p["smooth_window"]),
                          peak_prominence=float(p["peak_prominence"]), peak_min_sep_s=float(p["peak_min_sep_s"]),
                          carry_max_s=float(p["carry_max_s"]), closing_min_mps=float(p["closing_min_mps"]),
                          closing_window_s=float(p["closing_window_s"]),
                          dist_pressure_core=float(p["dist_presion_core_m"]))


@dataclass
class FrameData:
    """Datos de un fotograma tal como salen de detections.csv y ball.csv."""

    frame: int
    track_id: np.ndarray
    team: np.ndarray
    possession_xy: np.ndarray
    pressure_xy: np.ndarray
    ball_xy: np.ndarray


def frames_from_tables(det: pd.DataFrame, ball: pd.DataFrame, mode: str) -> List[FrameData]:
    """Agrupa las detecciones por fotograma (en el orden del CSV) y elige coordenadas según el modo."""
    groups = {int(f): g for f, g in det.groupby("frame", sort=False)}
    empty = det.iloc[0:0]
    out = []
    for row in ball.itertuples(index=False):
        g = groups.get(int(row.frame), empty)
        feet_px = g[["x_px", "y_px"]].to_numpy(dtype=float)
        if mode == "baseline":
            centers = np.column_stack([(g["x1_px"] + g["x2_px"]) / 2, (g["y1_px"] + g["y2_px"]) / 2]).astype(float)
            pos_xy, press_xy = feet_px, centers.reshape(-1, 2)
            ball_xy = np.array([row.ball_x, row.ball_y], dtype=float)
        else:
            pos_xy = press_xy = g[["x_pitch_m", "y_pitch_m"]].to_numpy(dtype=float)
            ball_xy = np.array([row.ball_pitch_x_m, row.ball_pitch_y_m], dtype=float)
        out.append(FrameData(frame=int(row.frame),
                             track_id=g["track_id"].fillna(-1).to_numpy(dtype=int),
                             team=g["team"].fillna(REFEREE).to_numpy(dtype=int),
                             possession_xy=pos_xy, pressure_xy=press_xy, ball_xy=ball_xy))
    return out


@dataclass
class Analysis:
    """Resultado de la etapa de posesión y presión."""

    frames: pd.DataFrame
    sequences: pd.DataFrame
    peaks: pd.DataFrame
    press: List[FramePressure]
    windows: List[PossessionWindow]
    stats: dict


def analyze(frames: List[FrameData], det: pd.DataFrame, ball: pd.DataFrame, fps: float,
            params: PressureParams) -> Analysis:
    """Posesión, presión, secuencias, picos y agregados a partir de las tablas de caché."""
    windows: List[PossessionWindow] = []
    info = {"possession_model": params.possession,
            "control_s": params.control_s if params.possession == "team" else None,
            "carry_max_s": params.carry_max_s, "closing_min_mps": params.closing_min_mps,
            "closing_window_s": params.closing_window_s, "dist_presion_core_m": params.dist_pressure_core}
    if params.mode == "full" and params.carry_max_s > 0:
        frames, ball = apply_carry(frames, ball, fps, params.dist_possession, params.carry_max_s)
    if params.possession == "team":
        control_frames = control_frames_from_seconds(params.control_s, fps)
        windows, n_short = team_possession_windows(
            [f.possession_xy for f in frames], [f.team for f in frames], [f.track_id for f in frames],
            np.array([f.ball_xy for f in frames]).reshape(-1, 2), params.dist_possession, control_frames,
            params.ball_gap_max)
        team_of: List[Optional[int]] = [None] * len(frames)
        for w in windows:
            for i in range(w.start, w.end + 1):
                team_of[i] = w.team
        press = [window_frame_pressure(f.possession_xy, f.team, f.ball_xy, t, params.dist_possession,
                                       params.dist_pressure) for f, t in zip(frames, team_of)]
        info.update({"control_frames": control_frames, "n_possession_windows": len(windows),
                     "possession_changes": sum(1 for w in windows if w.end_reason == "lost_to_opponent"),
                     "short_windows_discarded": n_short})
    else:
        press = [frame_pressure(f.possession_xy, f.pressure_xy, f.team, f.ball_xy, params.dist_possession,
                                params.dist_pressure) for f in frames]
    if params.mode == "full" and params.closing_min_mps > 0:
        apply_active_pressure(frames, press, fps, params)
    df = frames_table(frames, ball, press, params.dist_col)
    if params.possession == "team":
        seq = window_sequences(windows, df, fps, params.dist_col)
    else:
        seq = possession_sequences(df, fps, int(round(params.seq_gap_s * fps)), params.dist_col)
    peaks = find_pressure_peaks(df, params.smooth_window, params.peak_prominence, params.peak_min_sep_s)
    stats = {
        **info, **ball_stats(ball),
        "h_valid_pct": pct(ball["h_valid"].sum(), len(ball)),
        "h_reused_pct": pct(ball["h_reused"].sum(), len(ball)),
        **team_coverage(det, params.mode),
        **pressure_stats(df, seq),
        "n_peaks": int(len(peaks)),
        "n_edge_peaks": int(peaks["edge"].sum()) if len(peaks) else 0,
        "peak_times_s": [round(float(t), 3) for t in peaks["t_s"]],
    }
    return Analysis(df, seq, peaks, press, windows, stats)


def frames_table(frames: List[FrameData], ball: pd.DataFrame, press: List[FramePressure],
                 dist_col: str) -> pd.DataFrame:
    """Tabla frames.csv (una fila por fotograma)."""
    df = ball[["frame", "t_s", "ball_x", "ball_y", "ball_state", "ball_pitch_x_m", "ball_pitch_y_m",
               "h_valid", "h_reused"]].reset_index(drop=True).copy()
    pid = []
    for f, p in zip(frames, press):
        tid = int(f.track_id[p.possessor_idx]) if p.possessor_idx is not None else -1
        pid.append(tid if tid >= 0 else None)
    df["possessor_id"] = pd.array(pid, dtype="Int64")
    df["possessor_team"] = pd.array([p.possessor_team for p in press], dtype="Int64")
    df["team_possession"] = pd.array([p.team_possession for p in press], dtype="Int64")
    df["possession_state"] = [p.possession_state for p in press]
    df["n_pressers"] = [p.n_pressers for p in press]
    df["n_pressers_active"] = pd.array([p.n_pressers_active for p in press], dtype="Int64")
    df[dist_col] = [p.nearest_rival_dist for p in press]
    df["n_players_team0"] = [int(np.sum(f.team == 0)) for f in frames]
    df["n_players_team1"] = [int(np.sum(f.team == 1)) for f in frames]
    df["n_unknown"] = [int(np.sum(f.team == UNKNOWN)) for f in frames]
    return df


def ball_stats(df: pd.DataFrame) -> dict:
    """Porcentaje de fotogramas con balón detectado, interpolado y perdido."""
    n = len(df)
    return {f"ball_{s}_pct": pct((df["ball_state"] == s).sum(), n) for s in (DETECTED, INTERPOLATED, CARRIED, LOST)}


def team_coverage(det: pd.DataFrame, mode: str) -> dict:
    """Detecciones sin equipo: en baseline sobre todas las personas (como el TFG); en full sobre jugadores y porteros."""
    total = len(det) if mode == "baseline" else int(det["class_id"].isin(TEAM_CLASS_IDS).sum())
    unknown = int((det["team"] == UNKNOWN).sum())
    return {"detections_total": total, "detections_without_team": unknown,
            "detections_without_team_pct": pct(unknown, total)}


# --- Balón portado y presión activa (solo modo full) -------------------------

def apply_carry(frames: List[FrameData], ball: pd.DataFrame, fps: float, dist_possession: float,
                carry_max_s: float) -> Tuple[List[FrameData], pd.DataFrame]:
    """Balón perdido junto a un jugador: hasta carry_max_s se asume en sus pies (estado carried); no modifica la caché."""
    max_frames = int(round(carry_max_s * fps))
    out, ball = list(frames), ball.copy().reset_index(drop=True)
    carrier, since = None, 0
    for i, f in enumerate(frames):
        if np.isfinite(f.ball_xy).all():
            carrier, since = _nearest_with_team(f, dist_possession), 0
            continue
        since += 1
        if carrier is None or since > max_frames:
            carrier = None
            continue
        k = np.flatnonzero(f.track_id == carrier)
        if len(k) == 0 or not np.isfinite(f.possession_xy[k[0]]).all():
            carrier = None
            continue
        xy = f.possession_xy[k[0]].astype(float)
        out[i] = dataclasses.replace(f, ball_xy=xy)
        ball.loc[i, ["ball_pitch_x_m", "ball_pitch_y_m"]] = xy
        ball.loc[i, "ball_state"] = CARRIED
    return out, ball


def _nearest_with_team(f: FrameData, dist_possession: float) -> Optional[int]:
    """track_id del jugador con equipo más cercano al balón si está a ≤ umbral, o None."""
    if len(f.team) == 0:
        return None
    d = np.linalg.norm(f.possession_xy - f.ball_xy, axis=1)
    d[~((f.team == 0) | (f.team == 1)) | ~np.isfinite(d) | (f.track_id < 0)] = np.inf
    k = int(np.argmin(d))
    return int(f.track_id[k]) if d[k] <= dist_possession else None


def apply_active_pressure(frames: List[FrameData], press: List[FramePressure], fps: float,
                          params: PressureParams) -> None:
    """Deja como presionantes solo a los rivales que cierran (≥ closing_min_mps) o están a ≤ dist_presion_core_m."""
    window = max(1, int(round(params.closing_window_s * fps)))
    dist_hist = []
    for f in frames:
        d = np.linalg.norm(f.pressure_xy - f.ball_xy, axis=1) if len(f.team) else np.zeros(0)
        dist_hist.append({int(t): float(x) for t, x in zip(f.track_id, d) if t >= 0 and np.isfinite(x)})
    for i, (f, p) in enumerate(zip(frames, press)):
        if p.team_possession is None and p.possessor_team is None:
            continue
        active = []
        for k in p.presser_idx:
            tid = int(f.track_id[k])
            d_now = dist_hist[i].get(tid, np.nan)
            if d_now <= params.dist_pressure_core:
                active.append(k)
                continue
            past = [(j, dist_hist[j][tid]) for j in range(max(0, i - window), i) if tid in dist_hist[j]]
            if past:
                j, d_then = past[0]
                if (d_then - d_now) / ((i - j) / fps) >= params.closing_min_mps:
                    active.append(k)
        p.presser_idx = active
        p.n_pressers = p.n_pressers_active = len(active)
