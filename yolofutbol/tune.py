"""Ajuste y evaluación de parámetros contra etiquetas manuales: picos (TP/FP/FN) e intensidad media por clip (ρ, r)."""

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from yolofutbol import io as yio
from yolofutbol.pressure import (TUNABLE_PARAMS, Analysis, FrameData, analyze, frames_from_tables,
                                 possession_column, pressure_params)
from yolofutbol.viz import plot_evaluation, plot_intensity, plot_sensitivity

log = logging.getLogger(__name__)

RUN_SUFFIXES = ("full", "baseline", "replay")


@dataclass
class RunData:
    """Caché de una ejecución lista para analizar con distintos parámetros."""

    name: str
    mode: str
    fps: float
    params: dict
    det: pd.DataFrame
    ball: pd.DataFrame
    frames: List[FrameData]


def clip_name(run_dir: Path) -> str:
    """Nombre del clip: directorio de la ejecución sin el sufijo _full, _baseline o _replay."""
    name = Path(run_dir).name
    head, _, tail = name.rpartition("_")
    return head if head and tail in RUN_SUFFIXES else name


def load_run(run_dir: Path) -> RunData:
    """Lee caché y summary.json de una ejecución."""
    run_dir = Path(run_dir).expanduser()
    summary = yio.read_json(run_dir / yio.SUMMARY_JSON)
    det, ball = yio.read_cache(run_dir)
    mode = summary["mode"]
    return RunData(clip_name(run_dir), mode, float(summary["fps"]), summary.get("params", {}), det, ball,
                   frames_from_tables(det, ball, mode))


def load_labels(path: Path) -> Dict[str, np.ndarray]:
    """Picos reales por clip (columnas clip, t_s)."""
    df = pd.read_csv(Path(path).expanduser(), dtype={"clip": str})
    return {c: np.sort(g["t_s"].to_numpy(dtype=float)) for c, g in df.groupby("clip")}


def load_label_weights(path: Path) -> Dict[str, float]:
    """Suma de n_pressers de las marcas de cada clip (1 por marca si la columna falta o está vacía)."""
    df = pd.read_csv(Path(path).expanduser(), dtype={"clip": str})
    n = df["n_pressers"].fillna(1) if "n_pressers" in df.columns else pd.Series(1, index=df.index)
    return {c: float(g.sum()) for c, g in n.groupby(df["clip"])}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """ρ de Spearman: Pearson sobre rangos (empates con rango medio)."""
    return pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """r de Pearson; NaN con menos de 3 puntos o con una serie constante."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def intensity(run: RunData, analysis: Analysis, label_weight: float, n_marks: int) -> dict:
    """Intensidad del sistema (media de n_pressers con posesión) y de referencia (presionantes marcados por minuto)."""
    df = analysis.frames
    pos = df[df[possession_column(df)].notna()]
    duration_s = len(run.ball) / run.fps
    return {"intensity_system": float(pos["n_pressers"].mean()) if len(pos) else float("nan"),
            "intensity_reference": label_weight / (duration_s / 60.0) if duration_s else float("nan"),
            "n_marks": n_marks, "duration_s": duration_s}


def expand_grid(grid: dict) -> List[dict]:
    """Todas las combinaciones de la rejilla parámetro -> lista de valores."""
    bad = [k for k in grid if k not in TUNABLE_PARAMS]
    if bad:
        raise SystemExit(f"Parámetros no ajustables en la rejilla: {bad}. Válidos: {list(TUNABLE_PARAMS)}")
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def match_peaks(detected: np.ndarray, labels: np.ndarray, tol_s: float) -> List[Tuple[int, int]]:
    """Emparejamiento uno a uno por cercanía: pares (detectado, etiqueta) con |Δt| ≤ tol, del más cercano al más lejano."""
    pairs = sorted((abs(d - lab), i, j) for i, d in enumerate(detected) for j, lab in enumerate(labels)
                   if abs(d - lab) <= tol_s)
    used_d, used_l, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_d and j not in used_l:
            used_d.add(i)
            used_l.add(j)
            out.append((i, j))
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precisión, exhaustividad y F1."""
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_runs(runs: List[RunData], labels: Dict[str, np.ndarray], weights: Dict[str, float], overrides: dict,
                  tol_s: float) -> Tuple[Dict[str, dict], dict, Dict[str, Tuple[Analysis, np.ndarray, list]]]:
    """Analiza cada clip: TP, FP y FN de picos, intensidades, y en total además ρ y r entre intensidades."""
    per_clip, details = {}, {}
    for run in runs:
        params = pressure_params(run.mode, {**run.params, **overrides})
        analysis = analyze(run.frames, run.det, run.ball, run.fps, params)
        detected = analysis.peaks["t_s"].to_numpy(dtype=float)
        lab = labels.get(run.name, np.zeros(0))
        pairs = match_peaks(detected, lab, tol_s)
        per_clip[run.name] = {"n_labels": len(lab), "n_detected": len(detected),
                              **prf(len(pairs), len(detected) - len(pairs), len(lab) - len(pairs)),
                              **intensity(run, analysis, weights.get(run.name, 0.0), len(lab))}
        details[run.name] = (analysis, lab, pairs)
    tp, fp, fn = (sum(c[k] for c in per_clip.values()) for k in ("tp", "fp", "fn"))
    sys_i = np.array([c["intensity_system"] for c in per_clip.values()])
    ref_i = np.array([c["intensity_reference"] for c in per_clip.values()])
    total = {"n_labels": sum(c["n_labels"] for c in per_clip.values()),
             "n_detected": sum(c["n_detected"] for c in per_clip.values()), **prf(tp, fp, fn),
             "spearman": spearman(sys_i, ref_i), "pearson": pearson(sys_i, ref_i), "n_clips": len(per_clip)}
    return per_clip, total, details


def _load_inputs(args) -> Tuple[List[RunData], Dict[str, np.ndarray], Dict[str, float]]:
    """Carga ejecuciones y etiquetas, avisando de clips sin etiquetas o etiquetas sin clip."""
    runs = [load_run(Path(d)) for d in args.runs]
    labels = load_labels(Path(args.labels))
    names = {r.name for r in runs}
    for r in runs:
        if r.name not in labels:
            log.warning("El clip %s no tiene etiquetas: todos sus picos contarán como falsos positivos", r.name)
    for c in labels:
        if c not in names:
            log.warning("Hay etiquetas del clip %s pero no se ha pasado su ejecución", c)
    return runs, labels, load_label_weights(Path(args.labels))


OBJECTIVE_TIEBREAK = {"f1": ["f1", "precision"], "spearman": ["spearman", "pearson", "f1"]}


def cmd_tune(args) -> None:
    """Barrido de la rejilla; escribe tune_results.csv (ordenado por el objetivo) y best.json."""
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    runs, labels, weights = _load_inputs(args)
    with open(Path(args.grid).expanduser(), encoding="utf-8") as f:
        combos = expand_grid(json.load(f))
    log.info("Tune: %d combinaciones x %d clips", len(combos), len(runs))
    rows = []
    for k, combo in enumerate(combos):
        per_clip, total, _ = evaluate_runs(runs, labels, weights, combo, args.tol_s)
        row = {**combo, **total}
        for clip, m in per_clip.items():
            row.update({f"{clip}_{key}": m[key] for key in ("tp", "fp", "fn", "f1", "intensity_system")})
        rows.append(row)
        if k % 50 == 0:
            log.info("Combinación %d/%d", k, len(combos))
    res = pd.DataFrame(rows).sort_values(OBJECTIVE_TIEBREAK[args.objective], ascending=False, na_position="last")
    yio.write_csv(res, out / "tune_results.csv")
    best = rows[res.index[0]]
    yio.write_json({"params": {k: best[k] for k in combos[0]},
                    "metrics": {k: best[k] for k in ("n_labels", "n_detected", "tp", "fp", "fn",
                                                     "precision", "recall", "f1", "spearman", "pearson",
                                                     "n_clips")},
                    "objective": args.objective,
                    "runs": [str(d) for d in args.runs], "labels": str(args.labels), "tol_s": args.tol_s},
                   out / "best.json")
    log.info("Mejor combinación (%s): %s (ρ %.3f, r %.3f, F1 %.3f)", args.objective,
             {k: best[k] for k in combos[0]}, best["spearman"], best["pearson"], best["f1"])
    plot_sensitivity(res, list(combos[0]), out / "sensitivity.png", _chosen(args.chosen), args.ncols,
                     metric=args.objective)


def _chosen(path) -> dict:
    """Parámetros elegidos de un JSON con el formato de best.json, o {}."""
    return yio.read_json(Path(path))["params"] if path else {}


def cmd_sensitivity(args) -> None:
    """sensitivity.png a partir de uno o varios tune_results.csv existentes."""
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    res = pd.concat([pd.read_csv(Path(p).expanduser()) for p in args.results], ignore_index=True)
    params = [p for p in TUNABLE_PARAMS if p in res.columns and res[p].notna().any()]
    if args.metric not in res.columns:
        raise SystemExit(f"Los CSV no tienen la columna {args.metric}; vuelve a lanzar tune con esta versión")
    plot_sensitivity(res, params, out / "sensitivity.png", _chosen(args.chosen), args.ncols, metric=args.metric)
    log.info("Escrito %s (%d parámetros, %d combinaciones)", out / "sensitivity.png", len(params), len(res))


def cmd_evaluate(args) -> None:
    """Evalúa una combinación; escribe evaluation.csv y evaluation.png."""
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    runs, labels, weights = _load_inputs(args)
    overrides = yio.read_json(Path(args.params))["params"] if args.params else {}
    per_clip, total, details = evaluate_runs(runs, labels, weights, overrides, args.tol_s)
    rows = [{"clip": c, **m} for c, m in per_clip.items()] + [{"clip": "TOTAL", **total}]
    yio.write_csv(pd.DataFrame(rows), out / "evaluation.csv")
    panels = []
    for run in runs:
        analysis, lab, pairs = details[run.name]
        tp_mask = np.zeros(len(analysis.peaks), dtype=bool)
        tp_mask[np.array([i for i, _ in pairs], dtype=int)] = True
        window = pressure_params(run.mode, {**run.params, **overrides}).smooth_window
        panels.append((run.name, analysis.frames, analysis.peaks, lab, tp_mask, window))
    plot_evaluation(panels, out / "evaluation.png")
    inten = pd.DataFrame([{"clip": c, "intensidad_sistema": m["intensity_system"],
                           "intensidad_referencia": m["intensity_reference"], "n_marcas": m["n_marks"],
                           "duracion_s": m["duration_s"]} for c, m in per_clip.items()])
    yio.write_csv(inten, out / "intensity.csv")
    names = yio.read_json(Path(args.names)) if args.names else None
    plot_intensity(inten, total["spearman"], total["pearson"], out / "intensity.png", names)
    log.info("Evaluación: TP %d, FP %d, FN %d, F1 %s; intensidad ρ %s, r %s (%d clips)", total["tp"], total["fp"],
             total["fn"], total["f1"], total["spearman"], total["pearson"], total["n_clips"])
