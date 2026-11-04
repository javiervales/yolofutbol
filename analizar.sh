#!/usr/bin/env bash
# Analiza un clip de fútbol con los parámetros del artículo.
# Uso: bash analizar.sh CLIP.mp4 [SALIDA] [opciones extra de "python -m yolofutbol run"]
#   SALIDA  carpeta de resultados; por defecto salidas/<nombre_del_clip>/
#   Las opciones extra se añaden al final y sustituyen a las de serie (p. ej. --duration-s 5 --no-video).
# Variables de entorno opcionales:
#   YOLOFUTBOL_PYTHON  intérprete de Python (o carpeta de un entorno virtual) que se usará
#   YOLOFUTBOL_DEVICE  dispositivo de inferencia (0 = primera GPU, cpu); por defecto se detecta solo
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,9p' "$0"
  exit 1
fi
CLIP="$1"; shift
if [ ! -f "$CLIP" ]; then
  echo "ERROR: no encuentro el vídeo: $CLIP" >&2
  echo "Indica la ruta del vídeo, por ejemplo: bash analizar.sh ~/Videos/mi_partido.mp4" >&2
  exit 1
fi
CLIP="$(cd "$(dirname "$CLIP")" && pwd)/$(basename "$CLIP")"
NOMBRE="$(basename "${CLIP%.*}")"

SALIDA="$REPO/salidas/$NOMBRE"
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
  SALIDA="$1"; shift
fi

# Intérprete: YOLOFUTBOL_PYTHON, si no el .venv de instalar.sh, si no python3 (p. ej. en Colab).
PY="${YOLOFUTBOL_PYTHON:-}"
if [ -n "$PY" ] && [ -d "$PY" ]; then PY="$PY/bin/python"; fi
if [ -z "$PY" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"; else PY=python3; fi
fi

# Comprobaciones antes de crear nada: entorno con las librerías y modelos descargados.
if ! "$PY" -c "import torch, ultralytics, supervision, sports, scipy" >/dev/null 2>&1; then
  echo "ERROR: falta el entorno de yolofutbol o está incompleto ($PY no tiene las librerías necesarias)." >&2
  echo "Ejecuta primero:  bash instalar.sh        (con tarjeta NVIDIA: bash instalar.sh --gpu)" >&2
  exit 1
fi
FALTAN=0
for m in football-player-detection.pt football-ball-detection.pt football-pitch-detection.pt; do
  if [ ! -s "$REPO/data/$m" ]; then echo "ERROR: falta el modelo data/$m" >&2; FALTAN=1; fi
done
if [ "$FALTAN" = 1 ]; then
  echo "Ejecuta primero:  bash descargar_modelos.sh" >&2
  exit 1
fi

DEVICE="${YOLOFUTBOL_DEVICE:-$("$PY" -c 'import torch; print(0 if torch.cuda.is_available() else "cpu")')}"
if [ "$DEVICE" = "cpu" ]; then
  echo "Sin tarjeta gráfica: el análisis irá por el procesador y tardará al menos 10 veces más."
fi

mkdir -p "$SALIDA"
SALIDA="$(cd "$SALIDA" && pwd)"
echo "Analizando $CLIP -> $SALIDA"

cd "$REPO"
"$PY" -m yolofutbol run \
  --video "$CLIP" --out "$SALIDA" \
  --mode full --possession team --weights-dir "$REPO/data" --device "$DEVICE" \
  --h-ema 0.3 --closing-min-mps 0.5 --dist-presion-core-m 2.5 --carry-max-s 0 \
  --radar-pos right \
  "$@"

# Resumen legible a partir de sequences.csv, peaks.csv y summary.json.
"$PY" - "$SALIDA" <<'PYEOF'
import json
import math
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
summary = json.loads((out / "summary.json").read_text())
seq = pd.read_csv(out / "sequences.csv") if (out / "sequences.csv").stat().st_size > 1 else pd.DataFrame()
peaks = pd.read_csv(out / "peaks.csv") if (out / "peaks.csv").stat().st_size > 1 else pd.DataFrame()

def num(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}".replace(".", ",")

EQUIPO = {0: "equipo 0 (azul)", 1: "equipo 1 (naranja)"}


def color_camiseta(proto):
    """Nombre aproximado del color de camiseta a partir del prototipo (S·cos H, S·sin H, V) del k-means."""
    a, b, v = proto
    sat = math.hypot(a, b)
    if sat < 0.2:  # poco color: lo decide el brillo
        return "blanca o clara" if v >= 0.65 else ("negra u oscura" if v < 0.35 else "gris")
    hue = math.degrees(math.atan2(b, a)) % 360  # H de OpenCV (0-180) pasado a grados
    for limite, nombre in ((20, "roja"), (45, "naranja"), (70, "amarilla"), (160, "verde"), (200, "celeste"),
                           (260, "azul"), (300, "morada"), (340, "rosa o granate"), (360, "roja")):
        if hue < limite:
            return nombre + (" oscura" if v < 0.4 else "")
FINAL = {
    "lost_to_opponent": "la recupera el rival",
    "ball_lost": "se pierde el balón de vista (pase largo, balón fuera de plano o no detectado)",
    "clip_end": "sigue en juego al terminar el clip",
}

dur_clip = summary["n_frames"] / summary["fps"]
print()
print("=" * 72)
print(f"RESUMEN · {Path(summary['video']).name} · {num(dur_clip)} s")
print("=" * 72)
print(f"Balón detectado en el {num(summary.get('ball_detected_pct'))} % de los fotogramas.")
protos = (summary.get("team_classifier") or {}).get("centers_scosh_ssinh_v")
colores = [color_camiseta(p) for p in protos] if protos and len(protos) == 2 else []
if len(colores) == 2 and colores[0] != colores[1]:
    print(f"Equipos (pista por el color medio del torso): azul = camiseta {colores[0]}, "
          f"naranja = camiseta {colores[1]}. Compruébalo en annotated.mp4.")
else:
    print("Equipos: azul y naranja son los dos equipos detectados, en el orden en que aparecen; "
          "mira annotated.mp4 para saber cuál es cada uno.")
share = summary.get("possession_share_pct", {})
if share:
    print(f"Reparto de la posesión asignada: azul {num(share.get('team0'))} %, naranja {num(share.get('team1'))} %.")
print(f"Rivales presionando, media con posesión: {num(summary.get('pressure_index_mean'), 2)}"
      f" (máximo {summary.get('pressure_index_max', '—')}).")
print("  (referencia: en los 24 clips del artículo la media va de 0,03 a 0,65 rivales, con mediana 0,29)")
edge = peaks["edge"].fillna(0).astype(int).astype(bool) if len(peaks) and "edge" in peaks else pd.Series(False, index=peaks.index)
dentro, borde = peaks[~edge], peaks[edge]
if len(dentro):
    print("Momentos de presión máxima, para ir a verlos (s): " + "; ".join(num(t) for t in dentro["t_s"]))
else:
    print("No se han detectado momentos de presión máxima dentro del clip.")
if len(borde):
    print("  En el borde de una posesión (primer o último instante: al recibir o justo antes de perder el balón;\n"
          "  pueden deberse al cambio de posesión o al corte del vídeo, conviene verlos con más cautela): "
          + "; ".join(num(t) for t in borde["t_s"]) + " s")
print()
print(f"Posesiones: {len(seq)}")
for i, r in enumerate(seq.itertuples(index=False), start=1):
    fin = r.start_t_s + r.duration_s
    first = getattr(r, "time_to_first_presser_s", float("nan"))
    if math.isnan(first):
        primero = "ningún rival llega a presionar"
    elif first <= 0.1:
        primero = "ya con un rival encima al recibir"
    else:
        primero = f"primer presionante a los {num(first)} s"
    print(f"  {i:2d}. {EQUIPO.get(r.team, r.team)}, de {num(r.start_t_s)} a {num(fin)} s ({num(r.duration_s)} s); "
          f"{primero}; presionantes medios {num(r.mean_pressers, 2)} (máx. {r.max_pressers}); "
          f"termina: {FINAL.get(getattr(r, 'end_reason', ''), getattr(r, 'end_reason', '—'))}.")
print()
print(f"Resultados en {out}:")
for f in ("pressure.png", "annotated.mp4", "sequences.csv", "frames.csv", "peaks.csv", "summary.json"):
    print(f"  {f}{'' if (out / f).exists() else '  (no generado)'}")
PYEOF
