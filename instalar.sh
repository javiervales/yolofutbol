#!/usr/bin/env bash
# Instala yolofutbol en un entorno virtual .venv dentro de esta carpeta y descarga los modelos.
# Uso: bash instalar.sh [--gpu] [--clip-ejemplo]
#   --gpu           instala PyTorch con CUDA 12.4 (tarjeta gráfica NVIDIA); sin ella, versión para procesador
#   --clip-ejemplo  descarga también el clip de prueba 0bfacc_0.mp4
# Para elegir el intérprete: PYTHON=/ruta/a/python3.11 bash instalar.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPU=0
DESCARGA=()
for a in "$@"; do
  case "$a" in
    --gpu) GPU=1 ;;
    --clip-ejemplo) DESCARGA+=(--clip-ejemplo) ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "Opción desconocida: $a" >&2; exit 2 ;;
  esac
done

# Python 3.10, 3.11 o 3.12 (las versiones con ruedas de PyTorch y ultralytics probadas).
version_ok() { "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; }
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && version_ok "$c"; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ] || ! version_ok "$PY"; then
  echo "Hace falta Python 3.10, 3.11 o 3.12 (https://www.python.org/downloads/)." >&2
  exit 1
fi
echo "Usando $("$PY" -V) ($PY)"

if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
VPY="$REPO/.venv/bin/python"
"$VPY" -m pip install --upgrade pip

if [ "$GPU" = 1 ]; then
  "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
elif [ "$(uname -s)" = "Darwin" ]; then
  "$VPY" -m pip install torch torchvision
else
  "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# requirements.txt incluye supervision>=0.26,<0.31 y el paquete sports (git+https://github.com/roboflow/sports.git).
"$VPY" -m pip install -r requirements.txt gdown

bash "$REPO/descargar_modelos.sh" "${DESCARGA[@]+"${DESCARGA[@]}"}"

echo
echo "Instalación terminada. Para analizar un clip:"
echo "  bash analizar.sh ruta/al/clip.mp4"
