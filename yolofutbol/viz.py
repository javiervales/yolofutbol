"""Gráficas del índice de presión y anotación del vídeo (cajas, poseedor, presionantes, radar)."""

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import pandas as pd  # noqa: E402
import supervision as sv  # noqa: E402
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch  # noqa: E402

from yolofutbol import io as yio  # noqa: E402
from yolofutbol.pitch import CONFIG  # noqa: E402
from yolofutbol.pressure import FramePressure, pressure_series  # noqa: E402
from yolofutbol.teams import REFEREE, UNKNOWN  # noqa: E402
from yolofutbol.track_ball import DETECTED, INTERPOLATED  # noqa: E402

# Paleta (hex): equipo 0 azul, equipo 1 naranja, desconocido gris, árbitro amarillo, presión rojo.
HEX = {0: "#2a78d6", 1: "#eb6834", UNKNOWN: "#9a9994", REFEREE: "#eda100"}
PRESSER_HEX = "#d81b9a"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
RAW_GREY = "#b4b2a9"
GAP_BREAK_S = 1.0

RADAR_SCALE = 0.1      # px por cm, como draw_pitch por defecto
RADAR_PADDING = 50
RADAR_WIDTH_FRAC = 0.4
RADAR_OPACITY = 0.6


def _num(v: float) -> str:
    """Número con coma decimal (formato español)."""
    return f"{v:g}".replace(".", ",")


_COMMA = FuncFormatter(lambda v, _pos: _num(v))


def _comma_axes(ax) -> None:
    """Coma decimal en las marcas de los ejes x e y."""
    ax.xaxis.set_major_formatter(_COMMA)
    ax.yaxis.set_major_formatter(_COMMA)


def _bgr(hex_color: str):
    """Color hex a tupla BGR para OpenCV."""
    return sv.Color.from_hex(hex_color).as_bgr()


# --- Gráficas ---------------------------------------------------------------

def _break_gaps(t: np.ndarray, y: np.ndarray):
    """Inserta NaN donde hay huecos largos sin poseedor para no unir tramos con una línea."""
    if len(t) < 2:
        return t, y
    cut = np.flatnonzero(np.diff(t) > GAP_BREAK_S) + 1
    return np.insert(t.astype(float), cut, np.nan), np.insert(y.astype(float), cut, np.nan)


def _draw_peaks(ax, peaks: Optional[pd.DataFrame], color: str) -> None:
    """Marca cada pico con un triángulo sobre la media móvil y la etiqueta de su tiempo."""
    if peaks is None or len(peaks) == 0:
        return
    ax.scatter(peaks["t_s"], peaks["rolling"] + 0.15, marker="v", s=60, color=color,
               edgecolors=SURFACE, linewidths=1.5, zorder=4, label=f"momentos de presión máxima ({len(peaks)})")
    for t, y in zip(peaks["t_s"], peaks["rolling"]):
        ax.annotate(f"{t:.1f} s".replace(".", ","), (t, y + 0.3), ha="center", va="bottom", fontsize=8, color=TEXT_2)


def _draw_pressure(ax, df: pd.DataFrame, window: int, color: str, label: str,
                   peaks: Optional[pd.DataFrame] = None) -> None:
    """Dibuja crudo (gris), media móvil (color), media acumulada (discontinua) y picos en un eje."""
    s = pressure_series(df, window)
    ax.set_facecolor(SURFACE)
    ax.scatter(s["t_s"], s["n_pressers"], s=8, color=RAW_GREY, label="por fotograma", zorder=1)
    t, y = _break_gaps(s["t_s"].to_numpy(), s["rolling"].to_numpy())
    ax.plot(t, y, color=color, linewidth=2, label=f"media móvil ({window} fotogramas)", zorder=3)
    t, y = _break_gaps(s["t_s"].to_numpy(), s["cumulative"].to_numpy())
    ax.plot(t, y, color=TEXT_2, linewidth=1.2, linestyle="--", label="media acumulada", zorder=2)
    _draw_peaks(ax, peaks, color)
    ax.set_title(label, color=TEXT, fontsize=11, loc="left")
    ax.set_ylabel("Rivales presionando", color=TEXT_2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_2)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    _comma_axes(ax)


def _y_max(*dfs: pd.DataFrame) -> float:
    """Límite superior común del eje y."""
    m = max([d["n_pressers"].max() for d in dfs if len(d)] + [3])
    return float(m) + 1.0


def plot_pressure(df: pd.DataFrame, path: Path, title: str, window: int, color_slot: int = 0,
                  peaks: Optional[pd.DataFrame] = None) -> None:
    """Gráfica del índice de presión frente al tiempo, con los picos marcados."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    _draw_pressure(ax, df, window, HEX[color_slot], title, peaks)
    ax.set_ylim(-0.2, _y_max(df))
    ax.set_xlabel("Tiempo (s)", color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
                 path: Path, window: int, peaks_a: Optional[pd.DataFrame] = None,
                 peaks_b: Optional[pd.DataFrame] = None) -> None:
    """Dos paneles con mismos ejes: A arriba (azul), B abajo (naranja), con sus picos."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, sharey=True, facecolor=SURFACE)
    _draw_pressure(axes[0], df_a, window, HEX[0], label_a, peaks_a)
    _draw_pressure(axes[1], df_b, window, HEX[1], label_b, peaks_b)
    axes[0].set_ylim(-0.2, _y_max(df_a, df_b))
    axes[1].set_xlabel("Tiempo (s)", color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --- Vídeo -------------------------------------------------------------------

def _label(team: int, track_id: Optional[int]) -> str:
    """Texto de la etiqueta de una caja."""
    name = {UNKNOWN: "?", REFEREE: "arb"}.get(team, f"E{team}")
    return name if track_id is None else f"#{track_id} {name}"


def annotate_frame(frame: np.ndarray, rec, fp: FramePressure, ring_px: Optional[tuple]) -> np.ndarray:
    """Cajas por equipo, poseedor resaltado, presionantes marcados, balón y texto de estado."""
    img = frame.copy()
    presser = set(fp.presser_idx)
    for i, box in enumerate(rec.xyxy.astype(int)):
        team = int(rec.teams[i])
        color = _bgr(HEX.get(team, HEX[UNKNOWN]))
        thick = 4 if i == fp.possessor_idx else 2
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color, thick)
        tid = None if rec.tracker_id is None else int(rec.tracker_id[i])
        cv2.putText(img, _label(team, tid), (box[0], box[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if i in presser:
            cx = (box[0] + box[2]) // 2
            tri = np.array([[cx, box[1] - 22], [cx - 9, box[1] - 38], [cx + 9, box[1] - 38]])
            cv2.fillPoly(img, [tri], _bgr(PRESSER_HEX))
        if i == fp.possessor_idx:
            cv2.putText(img, "POSEEDOR", (box[0], box[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if np.isfinite(rec.ball_px).all():
        bx, by = rec.ball_px.astype(int)
        ball_color = (0, 255, 0) if rec.ball_state == DETECTED else (0, 165, 255)
        cv2.circle(img, (bx, by), 9, ball_color, 2)
        if ring_px is not None:
            for r in ring_px:
                cv2.circle(img, (bx, by), int(r), (255, 255, 255), 1)
    return _draw_header(img, header_text(rec, fp))


TEAM_NAME = {0: "azul", 1: "naranja"}
BALL_TXT = {DETECTED: "detectado", INTERPOLATED: "interpolado"}
POSSESSION_TXT = {"controlled": "controlada", "in_flight": "en vuelo"}


def header_text(rec, fp: FramePressure) -> str:
    """Cabecera del vídeo anotado en castellano llano."""
    if fp.team_possession is None:
        pos_txt = "sin asignar"
    else:
        team = TEAM_NAME.get(fp.team_possession, str(fp.team_possession))
        pos_txt = f"equipo {team}, {POSSESSION_TXT.get(fp.possession_state, 'sin poseedor')}"
    proj = "ok" if rec.h_valid else ("la anterior" if rec.h_reused else "no disponible")
    t_txt = f"{rec.t_s:.1f}".replace(".", ",")
    return (f"f {rec.frame} · t {t_txt} s · balón: {BALL_TXT.get(rec.ball_state, 'perdido')} · "
            f"posesión: {pos_txt} · presionando: {fp.n_pressers} · proyección: {proj}")


@lru_cache(maxsize=4)
def _header_font(size: int) -> ImageFont.FreeTypeFont:
    """DejaVu Sans (incluida en matplotlib), que admite tildes y eñes, a diferencia de las fuentes de OpenCV."""
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), size)


def _draw_header(img: np.ndarray, text: str) -> np.ndarray:
    """Franja negra superior con el texto en blanco, escalada a la altura del vídeo."""
    size = max(12, round(24 * img.shape[0] / 1080))
    band = int(size * 1.6)
    strip = Image.new("RGB", (img.shape[1], band), (0, 0, 0))
    ImageDraw.Draw(strip).text((10, (band - size) // 2 - 1), text, font=_header_font(size), fill=(255, 255, 255))
    img[:band] = np.asarray(strip)[:, :, ::-1]
    return img


_PITCH_CACHE = {}


def _base_pitch() -> np.ndarray:
    """Imagen base del campo (se dibuja una sola vez)."""
    if "pitch" not in _PITCH_CACHE:
        _PITCH_CACHE["pitch"] = draw_pitch(config=CONFIG, scale=RADAR_SCALE, padding=RADAR_PADDING)
    return _PITCH_CACHE["pitch"].copy()


def render_radar(rec, fp: FramePressure, dist_pressure_m: float) -> np.ndarray:
    """Vista cenital con jugadores proyectados y círculo de presión alrededor del balón."""
    pitch = _base_pitch()
    feet_cm = rec.feet_m * 100.0
    ok = np.isfinite(feet_cm).all(axis=1)
    for team in (0, 1, REFEREE):
        sel = ok & (rec.teams == team)
        if sel.any():
            pitch = draw_points_on_pitch(config=CONFIG, xy=feet_cm[sel], face_color=sv.Color.from_hex(HEX[team]),
                                         radius=14, scale=RADAR_SCALE, padding=RADAR_PADDING, pitch=pitch)
    if fp.presser_idx:
        sel = np.array(fp.presser_idx)
        sel = sel[ok[sel]]
        pitch = draw_points_on_pitch(config=CONFIG, xy=feet_cm[sel], face_color=sv.Color.from_hex(PRESSER_HEX),
                                     radius=14, scale=RADAR_SCALE, padding=RADAR_PADDING, pitch=pitch)
    if np.isfinite(rec.ball_m).all():
        c = (int(rec.ball_m[0] * 100 * RADAR_SCALE) + RADAR_PADDING,
             int(rec.ball_m[1] * 100 * RADAR_SCALE) + RADAR_PADDING)
        cv2.circle(pitch, c, int(dist_pressure_m * 100 * RADAR_SCALE), (255, 255, 255), 2)
        cv2.circle(pitch, c, 8, (255, 255, 255), -1)
    return pitch


def overlay_radar(img: np.ndarray, radar: np.ndarray, position: str = "right") -> np.ndarray:
    """Superpone el radar semitransparente en la parte inferior central del fotograma."""
    h, w = img.shape[:2]
    rw = int(w * RADAR_WIDTH_FRAC)
    rh = int(rw * radar.shape[0] / radar.shape[1])
    small = cv2.resize(radar, (rw, rh), interpolation=cv2.INTER_AREA)
    x0 = w - rw if position == "right" else (w - rw) // 2
    y0 = h - rh
    roi = img[y0:h, x0:x0 + rw]
    img[y0:h, x0:x0 + rw] = cv2.addWeighted(small, RADAR_OPACITY, roi, 1 - RADAR_OPACITY, 0)
    return img


# --- Evaluación y hojas de contacto -------------------------------------------

GOOD_HEX = "#0ca30c"
BAD_HEX = "#d03b3b"


def plot_evaluation(panels: list, path: Path) -> None:
    """Un panel por clip: media móvil, etiquetas manuales (verticales) y picos detectados (verde acierto, rojo FP)."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.8 * len(panels)), squeeze=False, facecolor=SURFACE)
    for ax, (name, frames, peaks, labels, tp_mask, window) in zip(axes[:, 0], panels):
        s = pressure_series(frames, window)
        ax.set_facecolor(SURFACE)
        t, y = _break_gaps(s["t_s"].to_numpy(), s["rolling"].to_numpy())
        ax.plot(t, y, color=HEX[0], linewidth=2, label=f"media móvil ({window})")
        for k, tl in enumerate(labels):
            ax.axvline(tl, color=TEXT_2, linewidth=1, linestyle="--", label="momento etiquetado a mano" if k == 0 else None)
        if len(peaks):
            ok = np.asarray(tp_mask, dtype=bool)
            for mask, color, lab in ((ok, GOOD_HEX, "momento detectado (acierto)"), (~ok, BAD_HEX, "momento detectado (falso)")):
                if mask.any():
                    ax.scatter(peaks["t_s"][mask], peaks["rolling"][mask] + 0.15, marker="v", s=60, color=color,
                               edgecolors=SURFACE, linewidths=1.5, zorder=4, label=lab)
        ax.set_title(name, color=TEXT, fontsize=11, loc="left")
        ax.set_ylabel("Rivales presionando", color=TEXT_2)
        ax.set_ylim(-0.2, _y_max(frames))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=TEXT_2)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        _comma_axes(ax)
    axes[-1, 0].set_xlabel("Tiempo (s)", color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _thumbnail(frame: np.ndarray, width: int, t_s: float) -> np.ndarray:
    """Miniatura con el instante sobreimpreso."""
    h, w = frame.shape[:2]
    thumb = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    text = f"{t_s:.1f} s"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(thumb, (0, 0), (tw + 12, th + 14), (0, 0, 0), -1)
    cv2.putText(thumb, text, (6, th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return thumb


def contact_sheets(video_path: str, out_dir: Path, every_s: float, cols: int, rows: int, width: int,
                   offset_s: float = 0.0) -> int:
    """Mosaicos cols x rows de miniaturas cada every_s segundos; escribe sheet_NN.png y devuelve cuántas hojas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = yio.clip_range(video_path, 0.0, None)
    stride = max(1, int(round(every_s * rng.fps)))
    per_sheet = cols * rows
    thumbs, n_sheets = [], 0

    def flush():
        nonlocal thumbs, n_sheets
        th, tw = thumbs[0].shape[:2]
        sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
        for k, im in enumerate(thumbs):
            r, c = divmod(k, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im[:th, :tw]
        cv2.imwrite(str(out_dir / f"sheet_{n_sheets:02d}.png"), sheet)
        n_sheets += 1
        thumbs = []

    for idx, frame in yio.iter_frames(rng, stride=stride):
        thumbs.append(_thumbnail(frame, width, idx / rng.fps + offset_s))
        if len(thumbs) == per_sheet:
            flush()
    if thumbs:
        flush()
    return n_sheets


def export_frames(video_path: str, out_dir: Path, times_s: list, offset_s: float = 0.0) -> int:
    """Escribe frame_<t>.png a resolución completa para cada instante (t en tiempo del vídeo original)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = yio.clip_range(video_path, 0.0, None)
    wanted = {}
    for t in times_s:
        idx = int(round((t - offset_s) * rng.fps))
        if rng.start <= idx < rng.end:
            wanted.setdefault(idx, []).append(t)
    n = 0
    for idx, frame in yio.read_frames_at(rng, np.array(sorted(wanted), dtype=int)):
        for t in wanted[idx]:
            cv2.imwrite(str(out_dir / f"frame_{t:.2f}.png"), frame)
            n += 1
    return n


# --- Sensibilidad del F1 a cada parámetro (tune) -------------------------------

PARAM_LABELS = {
    "control_s": "control (s)", "dist_posesion_m": "radio posesión (m)",
    "dist_presion_m": "radio presión (m)", "smooth_window": "suavizado (fotogramas)",
    "peak_prominence": "prominencia", "peak_min_sep_s": "separación (s)",
    "closing_min_mps": "acercamiento (m/s)", "dist_presion_core_m": "núcleo (m)",
    "carry_max_s": "balón portado (s)", "dist_posesion_px": "radio posesión (px)",
    "dist_presion_px": "radio presión (px)", "closing_window_s": "ventana acercamiento (s)",
}
CHOSEN_HEX = "#DE1E25"
BAND_GREY = "#d9d8d3"


def plot_sensitivity(results: pd.DataFrame, params: list, path: Path, chosen: Optional[dict] = None,
                     ncols: int = 3) -> None:
    """Rejilla de paneles, uno por parámetro: F1 total máximo (línea con puntos) y rango p25-p75 (banda gris)."""
    n = max(1, len(params))
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    panel_in = 5 / 2.54
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_in * ncols, panel_in * nrows + 0.4),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    for ax, p in zip(flat, params):
        g = results.dropna(subset=[p, "f1"]).groupby(p)["f1"]
        keys = sorted(g.groups)
        x = np.array(keys, dtype=float)
        best = g.max().loc[keys].to_numpy(dtype=float)
        q25 = g.quantile(0.25).loc[keys].to_numpy(dtype=float)
        q75 = g.quantile(0.75).loc[keys].to_numpy(dtype=float)
        ax.set_facecolor(SURFACE)
        ax.fill_between(x, q25, q75, color=BAND_GREY, linewidth=0, label="p25-p75")
        ax.plot(x, best, color=HEX[0], linewidth=1.5, marker="o", markersize=5, label="máximo")
        if chosen and p in chosen and len(x):
            cx = float(chosen[p])
            ax.scatter([cx], [np.interp(cx, x, best)], marker="v", s=70, color=CHOSEN_HEX, zorder=4,
                       label="elegido")
        ax.set_title(PARAM_LABELS.get(p, p), color=TEXT, fontsize=9, loc="left")
        ax.set_ylim(0.3, 0.9)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=TEXT_2, labelsize=8)
        _comma_axes(ax)
    for r in range(nrows):
        axes[r, 0].set_ylabel("F1 total", color=TEXT_2, fontsize=8)
    handles = {}
    for ax in flat[:n]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --- Storyboard: curva de presión con fotogramas repartidos en el tiempo -------

STORY_WIDTH_CM = 17.0
STORY_CURVE_CM = 4.5
STORY_GAP_CM = 0.8


def default_story_times(frames: pd.DataFrame, peaks: pd.DataFrame, window: int, n_max: int = 6) -> list:
    """Instantes por defecto: picos del run y mínimos de la media móvil entre picos consecutivos (hasta n_max)."""
    s = pressure_series(frames, window)
    peak_t = sorted(float(t) for t in peaks["t_s"]) if len(peaks) else []
    minima = []
    for a, b in zip(peak_t[:-1], peak_t[1:]):
        seg = s[(s["t_s"] > a) & (s["t_s"] < b)]
        if len(seg):
            minima.append(float(seg.loc[seg["rolling"].idxmin(), "t_s"]))
    return sorted((peak_t + minima)[:n_max])


def plot_storyboard(frames: pd.DataFrame, peaks: pd.DataFrame, window: int, images: list, times: list,
                    labels: list, path: Path, title: Optional[str] = None, ncols: int = 3) -> None:
    """Figura de ancho 17 cm: curva de presión con marcas (a), (b)… y debajo la rejilla de fotogramas anotados."""
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    img_h, img_w = images[0].shape[:2]
    cm = 1 / 2.54
    thumb_h_cm = STORY_WIDTH_CM / ncols * img_h / img_w
    height_cm = STORY_CURVE_CM + STORY_GAP_CM + nrows * thumb_h_cm + (0.6 if title else 0.2)
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7}):
        fig = plt.figure(figsize=(STORY_WIDTH_CM * cm, height_cm * cm), facecolor="white")
        gs = GridSpec(2 + nrows, ncols, figure=fig,
                      height_ratios=[STORY_CURVE_CM, STORY_GAP_CM] + [thumb_h_cm] * nrows,
                      hspace=0.08, wspace=0.03, left=0.07, right=0.99, bottom=0.005,
                      top=1 - (0.6 if title else 0.2) / height_cm)
        ax = fig.add_subplot(gs[0, :])
        s = pressure_series(frames, window)
        y_top = min(4.0, float(s["rolling"].max()) + 1.2) if len(s) else 4.0
        ax.set_facecolor("white")
        raw = s[s["n_pressers"] <= y_top]
        ax.scatter(raw["t_s"], raw["n_pressers"], s=3, color=RAW_GREY, label="por fotograma", zorder=1)
        t, y = _break_gaps(s["t_s"].to_numpy(), s["rolling"].to_numpy())
        ax.plot(t, y, color=HEX[0], linewidth=1.5, label=f"media móvil ({window} fotogramas)", zorder=3)
        if len(peaks):
            ax.scatter(peaks["t_s"], peaks["rolling"] + 0.15, marker="v", s=22, color=HEX[0],
                       edgecolors="white", linewidths=0.8, zorder=4, label="momentos de presión máxima")
        ax.set_ylim(0, y_top)
        if len(s):
            ax.set_xlim(s["t_s"].min() - 0.3, s["t_s"].max() + 0.3)
        for tm, lab in zip(times, labels):
            ax.axvline(tm, color="#8c8b86", linewidth=0.6, linestyle=(0, (3, 2)), zorder=2)
            ax.text(tm, 0.97, f"({lab})", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color=TEXT, fontsize=7.5, zorder=5,
                    bbox={"boxstyle": "square,pad=0.2", "facecolor": "white", "edgecolor": "#8c8b86",
                          "linewidth": 0.5})
        ax.set_xlabel("Tiempo (s)", color=TEXT_2, labelpad=1, loc="left")
        ax.set_ylabel("rivales presionando", color=TEXT_2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=TEXT_2, width=0.6, length=2.5)
        ax.legend(loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1.0, -0.13), borderaxespad=0,
                  handletextpad=0.4, columnspacing=1.0)
        _comma_axes(ax)
        if title:
            ax.set_title(title, color=TEXT, loc="right", fontsize=9, pad=10)
        for k, (img, tm, lab) in enumerate(zip(images, times, labels)):
            r, c = divmod(k, ncols)
            tax = fig.add_subplot(gs[2 + r, c])
            tax.imshow(img)
            tax.set_xticks([])
            tax.set_yticks([])
            for sp in tax.spines.values():
                sp.set_linewidth(72 / 300)
                sp.set_color("#8c8b86")
            tax.text(0.015, 0.97, f"({lab}) {tm:.1f} s".replace(".", ","), transform=tax.transAxes, ha="left",
                     va="top", fontsize=7.5, color=TEXT,
                     bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "none"})
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
