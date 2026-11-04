#!/usr/bin/env bash
# Descarga a data/ los tres modelos públicos de roboflow/sports (jugadores, balón, campo).
# Uso: bash descargar_modelos.sh [--clip-ejemplo]
#   --clip-ejemplo  descarga también el clip de prueba 0bfacc_0.mp4 (Bundesliga, roboflow/sports)
# Usa el Python del entorno: YOLOFUTBOL_PYTHON si está definido; si no, .venv/ (creado por instalar.sh).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$REPO/data"

CLIP=0
for a in "$@"; do
  case "$a" in
    --clip-ejemplo) CLIP=1 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "Opción desconocida: $a" >&2; exit 2 ;;
  esac
done

# Intérprete del entorno; sin entorno, solo se acepta un python3 que ya tenga gdown (p. ej. Colab).
PY="${YOLOFUTBOL_PYTHON:-}"
if [ -n "$PY" ] && [ -d "$PY" ]; then PY="$PY/bin/python"; fi
if [ -z "$PY" ] && [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"; fi
if [ -z "$PY" ]; then
  if python3 -c "import gdown" >/dev/null 2>&1; then
    PY=python3
  else
    cat >&2 <<'MSG'
ERROR: no hay entorno de yolofutbol en esta carpeta (.venv/).
Ejecuta primero:  bash instalar.sh        (con tarjeta NVIDIA: bash instalar.sh --gpu)
que crea el entorno, instala lo necesario y descarga los modelos.
MSG
    exit 1
  fi
fi
if ! "$PY" -c "import gdown" >/dev/null 2>&1; then
  echo "Instalando gdown en el entorno ($PY) ..."
  "$PY" -m pip install --quiet gdown
fi

mkdir -p "$DATA"
descargar() {  # descargar NOMBRE ID_DE_GOOGLE_DRIVE
  local destino="$DATA/$1"
  if [ -s "$destino" ]; then
    echo "Ya existe data/$1, no se descarga de nuevo."
    return
  fi
  echo "Descargando data/$1 ..."
  if ! "$PY" -m gdown -q "https://drive.google.com/uc?id=$2" -O "$destino.part"; then
    rm -f "$destino.part"
    echo "ERROR: no se pudo descargar $1 (¿sin conexión o Google Drive limita las descargas? Prueba más tarde)." >&2
    exit 1
  fi
  mv "$destino.part" "$destino"
}

descargar football-ball-detection.pt   1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V
descargar football-player-detection.pt 17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q
descargar football-pitch-detection.pt  1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf
if [ "$CLIP" = 1 ]; then
  descargar 0bfacc_0.mp4 12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF
fi
echo "Modelos listos en data/"
