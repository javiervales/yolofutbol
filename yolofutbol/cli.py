"""CLI: run (inferencia + análisis), replay (solo análisis), compare, tune, evaluate y sheets."""

import argparse
import logging
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from yolofutbol import __version__
from yolofutbol import io as yio
from yolofutbol import tune
from yolofutbol.detect import PLAYER_CLASS_ID, Detectors, split_people_and_ball
from yolofutbol.pitch import HomographyTracker, project
from yolofutbol.pressure import (DEFAULT_PARAMS, Analysis, analyze, ball_stats, find_pressure_peaks,
                                 frames_from_tables, pct, possession_column, pressure_params)
from yolofutbol.teams import (CALIB_SQUARE_FRAC, INFER_SQUARE_FRAC, REFEREE,
                              BaselineColorClassifier, TeamVoter, TorsoTeamClassifier,
                              assign_teams_full, center_square_hsv_mean, torso_feature)
from yolofutbol.track_ball import DETECTED, LOST, BallFilter, interpolate_ball
from yolofutbol.viz import (annotate_frame, contact_sheets, default_story_times, export_frames, overlay_radar,
                           plot_storyboard, plot_compare, plot_pressure,
                           render_radar)

log = logging.getLogger("yolofutbol")

# Avisos internos de las librerías (p. ej. sv.ByteTrack obsoleto desde supervision 0.28): no aportan nada
# a quien usa el programa; la versión de supervision ya está acotada en requirements.txt.
for _category in (FutureWarning, DeprecationWarning, UserWarning):
    warnings.filterwarnings("ignore", category=_category)

LOG_EVERY = 100
TRACK_MIN_CONSECUTIVE = 3
REFERENCE_HEIGHT = 1080


@dataclass
class FrameRecord:
    """Todo lo medido en un fotograma, antes de calcular posesión y presión."""

    frame: int
    t_s: float
    xyxy: np.ndarray
    class_id: np.ndarray
    tracker_id: Optional[np.ndarray]
    teams: np.ndarray
    feet_px: np.ndarray
    centers_px: np.ndarray
    feet_m: np.ndarray
    ball_px: np.ndarray
    ball_state: str = LOST
    ball_m: np.ndarray = None
    H: Optional[np.ndarray] = None
    h_valid: bool = False
    h_reused: bool = False


# --- Calibración del clasificador de equipos --------------------------------

def fit_baseline_classifier(rng: yio.ClipRange, det: Detectors, n_calib: int,
                            factor: float) -> BaselineColorClassifier:
    """Calibración del TFG: n fotogramas, cuadrado central del 15 %, k-means K=5."""
    feats, classes, frames, feet = [], [], [], []
    for idx, frame in yio.read_frames_at(rng, yio.sample_frame_indices(rng, n_calib)):
        people, _ = split_people_and_ball(det.detect_all(frame))
        foot = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for box, cls, f in zip(people.xyxy, people.class_id, foot):
            c = center_square_hsv_mean(frame, box, CALIB_SQUARE_FRAC)
            if c is not None:
                feats.append(c)
                classes.append(cls)
                frames.append(idx)
                feet.append(f)
    if len(feats) < 5:
        raise RuntimeError("La calibración de color no ha encontrado suficientes personas (mínimo 5)")
    log.info("Calibración baseline: %d recortes en %d fotogramas", len(feats), len(set(frames)))
    clf = BaselineColorClassifier(threshold_factor=factor)
    clf.fit(np.array(feats), np.array(classes), np.array(frames), np.array(feet))
    return clf


def fit_full_classifier(rng: yio.ClipRange, det: Detectors, stride: int) -> TorsoTeamClassifier:
    """Muestra de torsos de jugadores cada `stride` fotogramas de todo el tramo y k-means K=2."""
    feats = []
    for _, frame in yio.iter_frames(rng, stride=stride):
        people, _ = split_people_and_ball(det.detect_all(frame))
        players = people[people.class_id == PLAYER_CLASS_ID]
        feats += [torso_feature(frame, b) for b in players.xyxy]
    log.info("Calibración full: %d recortes de jugadores (stride %d)", len(feats), stride)
    if len(feats) < 10:
        log.warning("Pocos recortes para ajustar los equipos (%d); el resultado puede ser inestable", len(feats))
    clf = TorsoTeamClassifier()
    clf.fit(np.array(feats))
    return clf


# --- Pasada principal --------------------------------------------------------

def _record(idx, fps, people: sv.Detections, teams, tracker_id, ball_px, H=None, valid=False, reused=False):
    """Construye el FrameRecord de un fotograma."""
    feet_px = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(float)
    return FrameRecord(frame=idx, t_s=idx / fps, xyxy=people.xyxy.astype(float), class_id=people.class_id,
                       tracker_id=tracker_id, teams=np.asarray(teams, dtype=int), feet_px=feet_px,
                       centers_px=people.get_anchors_coordinates(sv.Position.CENTER).astype(float),
                       feet_m=project(H, feet_px), ball_px=np.asarray(ball_px, dtype=float),
                       H=H, h_valid=valid, h_reused=reused)


def process_baseline(rng: yio.ClipRange, det: Detectors, clf: BaselineColorClassifier) -> List[FrameRecord]:
    """TFG: detección sin tracker, balón del modelo de jugadores, equipo por color del centro de la caja."""
    records = []
    for i, (idx, frame) in enumerate(yio.iter_frames(rng)):
        people, balls = split_people_and_ball(det.detect_all(frame))
        # Como en el Listing 5.9, si hay varios balones se queda el último de la lista.
        ball_px = balls.get_anchors_coordinates(sv.Position.CENTER)[-1] if len(balls) else np.full(2, np.nan)
        teams = [clf.predict(center_square_hsv_mean(frame, b, INFER_SQUARE_FRAC)) for b in people.xyxy]
        rec = _record(idx, rng.fps, people, teams, None, ball_px)
        rec.ball_state = DETECTED if np.isfinite(rec.ball_px).all() else LOST
        records.append(rec)
        _log_progress(i, rng.n_frames)
    return records


def process_full(rng: yio.ClipRange, det: Detectors, clf: TorsoTeamClassifier,
                 args) -> Tuple[List[FrameRecord], dict]:
    """Versión mejorada: ByteTrack, detector de balón con filtro, homografía y equipos por torso con voto."""
    tracker = sv.ByteTrack(frame_rate=rng.fps, minimum_consecutive_frames=TRACK_MIN_CONSECUTIVE)
    jump_px = args.ball_jump_max * rng.height / REFERENCE_HEIGHT
    ball_filter = BallFilter(jump_max_px=jump_px, gap_max=args.ball_gap_max)
    htrack = HomographyTracker(ema_alpha=args.h_ema, kp_conf_min=args.kp_conf)
    voter = TeamVoter(window=args.vote_window)
    records = []
    for i, (idx, frame) in enumerate(yio.iter_frames(rng)):
        people, _ = split_people_and_ball(det.detect_all(frame))
        people = tracker.update_with_detections(people)
        ball_px = ball_filter.update(det.detect_ball(frame))
        H, valid, reused = htrack.update(*det.detect_pitch(frame))
        feet_px = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(float)
        ref_xy = project(H, feet_px) if H is not None else feet_px
        teams = assign_teams_full(frame, people.xyxy, people.class_id, people.tracker_id, ref_xy, clf, voter)
        records.append(_record(idx, rng.fps, people, teams, people.tracker_id, ball_px, H, valid, reused))
        _log_progress(i, rng.n_frames)
    xy, states = interpolate_ball(np.array([r.ball_px for r in records]), args.ball_gap_max)
    for rec, p, s in zip(records, xy, states):
        rec.ball_px, rec.ball_state = p, s
        rec.ball_m = project(rec.H, p[None])[0]
    return records, {"ball_jump_max_px_scaled": jump_px, "ball_jumps_rejected": ball_filter.n_rejected,
                     "kp_conf": args.kp_conf}


def _log_progress(i: int, n: int) -> None:
    """Registro de progreso cada LOG_EVERY fotogramas."""
    if i % LOG_EVERY == 0:
        log.info("Fotograma %d/%d", i, n)


# --- Caché de la etapa cara --------------------------------------------------

def cache_tables(records: List[FrameRecord]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """detections.csv (una fila por detección) y ball.csv (una fila por fotograma) a partir de los registros."""
    det_rows, ball_rows = [], []
    for r in records:
        for i in range(len(r.xyxy)):
            team = int(r.teams[i])
            det_rows.append({
                "frame": r.frame, "t_s": r.t_s,
                "track_id": None if r.tracker_id is None else int(r.tracker_id[i]),
                "class_id": int(r.class_id[i]), "team": None if team == REFEREE else team,
                "x_px": r.feet_px[i, 0], "y_px": r.feet_px[i, 1],
                "x1_px": r.xyxy[i, 0], "y1_px": r.xyxy[i, 1], "x2_px": r.xyxy[i, 2], "y2_px": r.xyxy[i, 3],
                "x_pitch_m": r.feet_m[i, 0], "y_pitch_m": r.feet_m[i, 1], "h_valid": int(r.h_valid),
            })
        ball_m = r.ball_m if r.ball_m is not None else np.full(2, np.nan)
        ball_rows.append({"frame": r.frame, "t_s": r.t_s, "ball_state": r.ball_state,
                          "ball_x": r.ball_px[0], "ball_y": r.ball_px[1],
                          "ball_pitch_x_m": ball_m[0], "ball_pitch_y_m": ball_m[1],
                          "h_valid": int(r.h_valid), "h_reused": int(r.h_reused)})
    det_cols = ["frame", "t_s", "track_id", "class_id", "team", "x_px", "y_px", "x1_px", "y1_px", "x2_px",
                "y2_px", "x_pitch_m", "y_pitch_m", "h_valid"]
    det = pd.DataFrame(det_rows, columns=det_cols)
    det["track_id"] = det["track_id"].astype("Int64")
    det["team"] = det["team"].astype("Int64")
    return det, pd.DataFrame(ball_rows)


def run_analysis(run_dir: Path, mode: str, fps: float, params: dict) -> Analysis:
    """Relee la caché de un directorio y ejecuta la etapa de posesión y presión (común a run y replay)."""
    det, ball = yio.read_cache(run_dir)
    pp = pressure_params(mode, params)
    return analyze(frames_from_tables(det, ball, mode), det, ball, fps, pp)


META_KEYS = ("video", "mode", "fps", "resolution", "frame_start", "frame_end", "n_frames", "distance_unit",
             "ball_jump_max_px_scaled", "ball_jumps_rejected", "kp_conf", "player_weights", "player_weights_path",
             "team_classifier", "versions")


def write_analysis(out: Path, analysis: Analysis, summary: dict, title: str, color_slot: int,
                   smooth_window: int) -> None:
    """Escribe frames.csv, sequences.csv, peaks.csv, summary.json y pressure.png."""
    yio.write_csv(analysis.frames, out / "frames.csv")
    yio.write_csv(analysis.sequences, out / "sequences.csv")
    yio.write_csv(analysis.peaks, out / "peaks.csv")
    yio.write_json(summary, out / yio.SUMMARY_JSON)
    plot_pressure(analysis.frames, out / "pressure.png", title, smooth_window, color_slot=color_slot,
                  peaks=analysis.peaks)
    st = analysis.stats
    if st["possession_model"] == "team":
        log.info("Posesión por equipo: %d ventanas, %d cambios de posesión de equipo, %d ventanas cortas descartadas",
                 st["n_possession_windows"], st["possession_changes"], st["short_windows_discarded"])
    log.info("Balón detectado %.1f %%, sin equipo %.1f %%, presión media %s, %d picos",
             st["ball_detected_pct"], st["detections_without_team_pct"], st["pressure_index_mean"], st["n_peaks"])


def _params_dict(args) -> dict:
    """Parámetros de la CLI serializables."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}


def _versions() -> dict:
    """Versiones de las librerías relevantes."""
    import ultralytics
    out = {"python": platform.python_version(), "supervision": sv.__version__,
           "ultralytics": ultralytics.__version__, "yolofutbol": __version__}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    return out


def write_video(rng: yio.ClipRange, records, press, args, path: Path) -> None:
    """Segunda pasada sobre el vídeo para escribir annotated.mp4."""
    ring = (args.dist_posesion_px, args.dist_presion_px) if args.mode == "baseline" else None
    by_frame = {r.frame: (r, p) for r, p in zip(records, press)}
    with sv.VideoSink(str(path), rng.video_info()) as sink:
        for i, (idx, frame) in enumerate(yio.iter_frames(rng)):
            if idx not in by_frame:
                continue
            rec, fp = by_frame[idx]
            img = annotate_frame(frame, rec, fp, ring)
            if args.mode == "full":
                img = overlay_radar(img, render_radar(rec, fp, args.dist_presion_m), args.radar_pos)
            sink.write_frame(img)
            if i % LOG_EVERY == 0:
                log.info("Vídeo: fotograma %d/%d", i, rng.n_frames)
    log.info("Escrito %s", path)


def cmd_run(args) -> None:
    """Inferencia sobre el clip, caché (detections.csv, ball.csv) y análisis a partir de esa caché."""
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    full = args.mode == "full"
    if full and not hasattr(sv, "ByteTrack"):
        raise SystemExit(f"Esta versión de supervision ({sv.__version__}) ya no incluye sv.ByteTrack. "
                         "Instala supervision<0.31 (ver requirements.txt).")
    try:
        pressure_params(args.mode, vars(args))
    except ValueError as e:
        raise SystemExit(str(e))
    rng = yio.clip_range(args.video, args.start_s, args.duration_s)
    log.info("Clip %s: fotogramas %d-%d (%d) a %.2f fps, %dx%d, modo %s", args.video, rng.start, rng.end - 1,
             rng.n_frames, rng.fps, rng.width, rng.height, args.mode)
    det = Detectors(Path(args.weights_dir), args.device, args.conf, args.ball_conf,
                    use_ball_model=full, use_pitch_model=full,
                    player_weights=Path(args.player_weights) if args.player_weights else None)
    extra = {}
    if full:
        clf = fit_full_classifier(rng, det, args.stride)
        records, extra = process_full(rng, det, clf, args)
    else:
        clf = fit_baseline_classifier(rng, det, args.calib_frames, args.color_threshold_factor)
        records = process_baseline(rng, det, clf)
    yio.write_cache(*cache_tables(records), out)
    analysis = run_analysis(out, args.mode, rng.fps, vars(args))
    args.possession = analysis.stats["possession_model"]
    summary = {
        "video": args.video, "mode": args.mode, "fps": rng.fps, "resolution": [rng.width, rng.height],
        "frame_start": rng.start, "frame_end": rng.end, "n_frames": len(analysis.frames),
        "distance_unit": "m" if full else "px",
        **analysis.stats, **extra,
        "player_weights": det.player_weights.name,
        "player_weights_path": str(det.player_weights),
        "team_classifier": clf.describe(),
        "params": _params_dict(args),
        "versions": _versions(),
    }
    title = args.title or f"Índice de presión · {Path(args.video).stem} · {args.mode}"
    write_analysis(out, analysis, summary, title, 1 if full else 0, args.smooth_window)
    if not args.no_video:
        write_video(rng, records, analysis.press, args, out / "annotated.mp4")


REPLAY_OVERRIDES = ("possession", "dist_posesion_m", "dist_presion_m", "dist_posesion_px", "dist_presion_px",
                    "control_s", "seq_gap_s", "smooth_window", "peak_prominence", "peak_min_sep_s",
                    "carry_max_s", "closing_min_mps", "closing_window_s", "dist_presion_core_m")


def cmd_replay(args) -> None:
    """Recalcula posesión, presión, secuencias y picos desde la caché de otra ejecución, sin modelos ni vídeo."""
    src = Path(args.from_dir).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    source = yio.read_json(src / yio.SUMMARY_JSON)
    mode = source["mode"]
    overrides = {k: getattr(args, k) for k in REPLAY_OVERRIDES if getattr(args, k) is not None}
    params = {**source.get("params", {}), **overrides}
    try:
        analysis = run_analysis(src, mode, float(source["fps"]), params)
    except ValueError as e:
        raise SystemExit(str(e))
    params["possession"] = analysis.stats["possession_model"]
    summary = {**{k: source[k] for k in META_KEYS if k in source}, **analysis.stats,
               "params": params, "replay_of": str(src), "replay_overrides": overrides}
    title = args.title or f"Índice de presión · {Path(str(source.get('video', src.name))).stem} · {mode} · replay"
    write_analysis(out, analysis, summary, title, 1 if mode == "full" else 0, int(params.get(
        "smooth_window", DEFAULT_PARAMS["smooth_window"])))


def cmd_sheets(args) -> None:
    """Hojas de contacto para marcar a ojo los picos de presión."""
    video, offset_s = args.video, 0.0
    if args.annotated:
        video = args.annotated
        sibling = Path(args.annotated).expanduser().parent / yio.SUMMARY_JSON
        if sibling.is_file():
            s = yio.read_json(sibling)
            offset_s = s["frame_start"] / s["fps"]
            log.info("Vídeo anotado: el tiempo mostrado empieza en %.2f s (frame_start de %s)", offset_s, sibling)
    if args.at_s:
        times = [float(x) for x in args.at_s.split(",") if x.strip()]
        n = export_frames(video, Path(args.out).expanduser(), times, offset_s)
        log.info("Escritos %d fotogramas en %s", n, args.out)
        return
    n = contact_sheets(video, Path(args.out).expanduser(), args.every_s, args.cols, args.rows, args.width, offset_s)
    log.info("Escritas %d hojas en %s", n, args.out)


def _compare_stats(df: pd.DataFrame) -> dict:
    """Agregados básicos de un frames.csv para compare.json."""
    pos = df[df[possession_column(df)].notna()]
    return {"n_frames": len(df), **ball_stats(df),
            "h_valid_pct": pct(df["h_valid"].sum(), len(df)),
            "frames_with_possessor_pct": pct(df["possessor_team"].notna().sum(), len(df)),
            "frames_with_team_possession_pct": pct(len(pos), len(df)),
            "pressure_index_mean": float(pos["n_pressers"].mean()) if len(pos) else float("nan"),
            "n_unknown_mean": float(df["n_unknown"].mean())}


def cmd_compare(args) -> None:
    """Gráfica conjunta y resumen de dos ejecuciones sobre el mismo clip."""
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dtypes = {"possessor_team": "Int64", "possessor_id": "Int64", "team_possession": "Int64"}
    df_a = pd.read_csv(args.a, dtype=dtypes)
    df_b = pd.read_csv(args.b, dtype=dtypes)
    if len(df_a) != len(df_b) or (df_a["frame"].to_numpy()[[0, -1]] != df_b["frame"].to_numpy()[[0, -1]]).any():
        log.warning("Los dos CSV no cubren los mismos fotogramas; se comparan tal cual")
    label_a = args.label_a or Path(args.a).parent.name
    label_b = args.label_b or Path(args.b).parent.name
    peaks_a = find_pressure_peaks(df_a, args.smooth_window, args.peak_prominence, args.peak_min_sep_s)
    peaks_b = find_pressure_peaks(df_b, args.smooth_window, args.peak_prominence, args.peak_min_sep_s)
    yio.write_csv(peaks_a, out / "peaks_a.csv")
    yio.write_csv(peaks_b, out / "peaks_b.csv")
    plot_compare(df_a, df_b, label_a, label_b, out / "pressure_compare.png", args.smooth_window,
                 peaks_a, peaks_b)
    yio.write_json({label_a: {**_compare_stats(df_a), "n_peaks": len(peaks_a),
                              "peak_times_s": [round(float(t), 3) for t in peaks_a["t_s"]]},
                    label_b: {**_compare_stats(df_b), "n_peaks": len(peaks_b),
                              "peak_times_s": [round(float(t), 3) for t in peaks_b["t_s"]]},
                    "a": str(args.a), "b": str(args.b), "smooth_window": args.smooth_window},
                   out / "compare.json")


def cmd_storyboard(args) -> None:
    """Figura del artículo: curva de presión del run y fotogramas del vídeo anotado en los instantes elegidos."""
    run = Path(args.run).expanduser()
    summary = yio.read_json(run / yio.SUMMARY_JSON)
    window = int(summary.get("params", {}).get("smooth_window", DEFAULT_PARAMS["smooth_window"]))
    frames = pd.read_csv(run / "frames.csv", dtype={"possessor_team": "Int64", "possessor_id": "Int64",
                                                    "team_possession": "Int64"})
    peaks = pd.read_csv(run / "peaks.csv") if (run / "peaks.csv").is_file() else pd.DataFrame(columns=["t_s"])
    if args.times:
        times = [float(x) for x in args.times.split(",") if x.strip()]
    else:
        times = default_story_times(frames, peaks, window)
    if not times:
        raise SystemExit("No hay instantes: pasa --times o usa un run con picos")
    labels = [x.strip() for x in args.labels.split(",")] if args.labels else \
        [chr(ord("a") + k) for k in range(len(times))]
    if len(labels) != len(times):
        raise SystemExit(f"--labels tiene {len(labels)} etiquetas y --times {len(times)} instantes")
    rng = yio.clip_range(args.video, 0.0, None)
    offset_s = summary.get("frame_start", 0) / float(summary["fps"])
    idx = [int(round((t - offset_s) * rng.fps)) for t in times]
    bad = [t for t, i in zip(times, idx) if not rng.start <= i < rng.end]
    if bad:
        raise SystemExit(f"Instantes fuera del vídeo {args.video}: {bad}")
    got = dict(yio.read_frames_at(rng, np.array(sorted(set(idx)), dtype=int)))
    images = []
    for i in idx:
        img = got[i][args.crop_top:]
        h, w = img.shape[:2]
        img = cv2.resize(img, (args.thumb_w, int(round(h * args.thumb_w / w))), interpolation=cv2.INTER_AREA)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_storyboard(frames, peaks, window, images, times, labels, out, args.title, args.ncols)
    log.info("Escrito %s (%d fotogramas, instantes %s)", out, len(times), times)


def _add_peak_args(p: argparse.ArgumentParser, defaults: bool = True) -> None:
    """Parámetros de la extracción de picos."""
    d = DEFAULT_PARAMS if defaults else {}
    p.add_argument("--peak-prominence", type=float, default=d.get("peak_prominence"),
                   help="prominencia mínima de un pico de la media móvil (presionantes)")
    p.add_argument("--peak-min-sep-s", type=float, default=d.get("peak_min_sep_s"),
                   help="separación mínima entre picos (s)")


def _add_pressure_args(p: argparse.ArgumentParser, defaults: bool = True) -> None:
    """Parámetros de posesión y presión, comunes a run (con valores por defecto) y replay (sin ellos)."""
    d = DEFAULT_PARAMS if defaults else {}
    p.add_argument("--dist-posesion-px", type=float, default=d.get("dist_posesion_px"),
                   help="baseline: umbral de posesión (px)")
    p.add_argument("--dist-presion-px", type=float, default=d.get("dist_presion_px"),
                   help="baseline: umbral de presión (px)")
    p.add_argument("--dist-posesion-m", type=float, default=d.get("dist_posesion_m"), help="full: umbral de posesión (m)")
    p.add_argument("--dist-presion-m", type=float, default=d.get("dist_presion_m"), help="full: umbral de presión (m)")
    p.add_argument("--possession", choices=["team", "frame"], default=None,
                   help="modelo de posesión: team (por defecto en full) o frame (único en baseline)")
    p.add_argument("--control-s", type=float, default=d.get("control_s"),
                   help="posesión team: segundos continuos a ≤ dist-posesion para un toque controlado")
    p.add_argument("--seq-gap-s", type=float, default=d.get("seq_gap_s"),
                   help="posesión frame: hueco máximo sin poseedor dentro de una secuencia (s)")
    p.add_argument("--carry-max-s", type=float, default=d.get("carry_max_s"),
                   help="full: segundos que se asume el balón en los pies de quien lo llevaba al perderlo (0 = no)")
    p.add_argument("--closing-min-mps", type=float, default=d.get("closing_min_mps"),
                   help="full: velocidad mínima de cierre (m/s) para contar como presionante (0 = desactivado)")
    p.add_argument("--closing-window-s", type=float, default=d.get("closing_window_s"),
                   help="full: ventana para medir el cierre (s)")
    p.add_argument("--dist-presion-core-m", type=float, default=d.get("dist_presion_core_m"),
                   help="full: a esta distancia o menos el rival cuenta siempre como presionante (m)")
    p.add_argument("--smooth-window", type=int, default=d.get("smooth_window"),
                   help="ventana de la media móvil (fotogramas con posesión)")
    _add_peak_args(p, defaults)


def build_parser() -> argparse.ArgumentParser:
    """Argumentos de la CLI."""
    p = argparse.ArgumentParser(prog="yolofutbol", description="Índice de presión en fútbol desde vídeo.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="procesa un clip (inferencia + análisis)")
    r.add_argument("--video", required=True, help="vídeo de entrada")
    r.add_argument("--out", required=True, help="directorio de salida del clip")
    r.add_argument("--mode", choices=["baseline", "full"], required=True)
    r.add_argument("--weights-dir", default="~/YOLOFUTBOL/data", help="carpeta con los tres .pt")
    r.add_argument("--player-weights", default=None,
                   help="pesos del detector de jugadores que sustituyen a football-player-detection.pt")
    r.add_argument("--start-s", type=float, default=0.0, help="inicio del tramo (s)")
    r.add_argument("--duration-s", type=float, default=None, help="duración del tramo (s); por defecto hasta el final")
    r.add_argument("--device", default=None, help="dispositivo de inferencia (0, cpu, cuda:0)")
    r.add_argument("--no-video", action="store_true", help="no escribir annotated.mp4")
    r.add_argument("--radar-pos", choices=["center", "right"], default="right",
                   help="full: posición de la vista cenital en annotated.mp4 (abajo centrada o abajo a la derecha)")
    r.add_argument("--conf", type=float, default=0.4, help="confianza mínima del detector de jugadores (TFG: 0,4)")
    r.add_argument("--ball-conf", type=float, default=0.25, help="confianza mínima del detector de balón (full)")
    r.add_argument("--calib-frames", type=int, default=10, help="baseline: fotogramas de calibración de color")
    r.add_argument("--color-threshold-factor", type=float, default=0.3, help="baseline: factor del UMBRAL_COLOR")
    r.add_argument("--stride", type=int, default=60, help="full: muestreo de recortes para k-means")
    r.add_argument("--vote-window", type=int, default=30, help="full: asignaciones por track para la moda")
    r.add_argument("--ball-gap-max", type=int, default=DEFAULT_PARAMS["ball_gap_max"],
                   help="full: hueco máximo interpolable (fotogramas)")
    r.add_argument("--ball-jump-max", type=float, default=150.0, help="full: salto máximo por fotograma (px a 1080p)")
    r.add_argument("--kp-conf", type=float, default=0.5,
                   help="full: confianza mínima de los puntos clave del campo para la homografía")
    r.add_argument("--h-ema", type=float, default=1.0,
                   help="full: peso de la H nueva en la media exponencial (1 = sin suavizar)")
    _add_pressure_args(r)
    r.add_argument("--title", default=None, help="título de pressure.png (sustituye al automático)")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="recalcula el análisis desde la caché de otra ejecución, sin inferencia")
    rp.add_argument("--from", dest="from_dir", required=True, help="directorio de una ejecución de run")
    rp.add_argument("--out", required=True, help="directorio de salida")
    _add_pressure_args(rp, defaults=False)
    rp.add_argument("--title", default=None, help="título de pressure.png (sustituye al automático)")
    rp.set_defaults(func=cmd_replay)

    c = sub.add_parser("compare", help="compara dos frames.csv")
    c.add_argument("--a", required=True, help="frames.csv de la ejecución A (p. ej. baseline)")
    c.add_argument("--b", required=True, help="frames.csv de la ejecución B (p. ej. full)")
    c.add_argument("--out", required=True, help="directorio de salida")
    c.add_argument("--label-a", default=None)
    c.add_argument("--label-b", default=None)
    c.add_argument("--smooth-window", type=int, default=DEFAULT_PARAMS["smooth_window"])
    _add_peak_args(c)
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser("tune", help="barrido de parámetros contra picos etiquetados a mano")
    t.add_argument("--runs", nargs="+", required=True, help="directorios de ejecuciones de run")
    t.add_argument("--labels", required=True, help="CSV con columnas clip,t_s")
    t.add_argument("--grid", required=True, help="JSON parámetro -> lista de valores")
    t.add_argument("--out", required=True, help="directorio de salida")
    t.add_argument("--tol-s", type=float, default=1.5, help="tolerancia de emparejamiento (s)")
    t.add_argument("--chosen", default=None, help="JSON con formato best.json para marcar el valor elegido")
    t.add_argument("--ncols", type=int, default=3, help="columnas de la rejilla de sensitivity.png")
    t.set_defaults(func=tune.cmd_tune)

    se = sub.add_parser("sensitivity", help="sensitivity.png desde tune_results.csv existentes")
    se.add_argument("--results", nargs="+", required=True, help="uno o varios tune_results.csv")
    se.add_argument("--out", required=True, help="directorio de salida")
    se.add_argument("--chosen", default=None, help="JSON con formato best.json para marcar el valor elegido")
    se.add_argument("--ncols", type=int, default=3, help="columnas de la rejilla de sensitivity.png")
    se.set_defaults(func=tune.cmd_sensitivity)

    e = sub.add_parser("evaluate", help="evalúa una combinación de parámetros contra los picos etiquetados")
    e.add_argument("--runs", nargs="+", required=True, help="directorios de ejecuciones de run")
    e.add_argument("--labels", required=True, help="CSV con columnas clip,t_s")
    e.add_argument("--params", default=None, help="best.json de tune; si falta, los parámetros de cada ejecución")
    e.add_argument("--out", required=True, help="directorio de salida")
    e.add_argument("--tol-s", type=float, default=1.5, help="tolerancia de emparejamiento (s)")
    e.set_defaults(func=tune.cmd_evaluate)

    s = sub.add_parser("sheets", help="hojas de contacto PNG para etiquetar picos a ojo")
    s.add_argument("--video", required=True, help="vídeo original")
    s.add_argument("--out", required=True, help="directorio de salida")
    s.add_argument("--every-s", type=float, default=0.5, help="segundos entre miniaturas")
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--rows", type=int, default=3)
    s.add_argument("--width", type=int, default=480, help="ancho de cada miniatura (px)")
    s.add_argument("--annotated", default=None, help="usar este vídeo anotado en lugar del original")
    s.add_argument("--at-s", default=None,
                   help="instantes (s) separados por comas: exporta frame_<t>.png a resolución completa")
    s.set_defaults(func=cmd_sheets)

    sb = sub.add_parser("storyboard", help="figura con la curva de presión y fotogramas anotados en instantes clave")
    sb.add_argument("--run", required=True, help="directorio de un run o replay (frames.csv, peaks.csv)")
    sb.add_argument("--video", required=True, help="vídeo anotado (annotated.mp4) de ese run")
    sb.add_argument("--out", required=True, help="PNG de salida")
    sb.add_argument("--times", default=None, help="instantes (s) separados por comas; por defecto picos y mínimos")
    sb.add_argument("--labels", default=None, help="letras separadas por comas; por defecto a, b, c…")
    sb.add_argument("--title", default=None, help="título pequeño opcional")
    sb.add_argument("--crop-top", type=int, default=0, help="píxeles a recortar arriba (cabecera técnica)")
    sb.add_argument("--thumb-w", type=int, default=620, help="ancho de cada miniatura (px)")
    sb.add_argument("--ncols", type=int, default=3, help="miniaturas por fila")
    sb.set_defaults(func=cmd_storyboard)
    return p


def main(argv=None) -> None:
    """Punto de entrada de la CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)
