#!/usr/bin/env bash
# Descarga a data/ los tres modelos públicos de roboflow/sports (jugadores, balón, campo).
# Uso: bash descargar_modelos.sh [--clip-ejemplo]
#   --clip-ejemplo  descarga también el clip de prueba 0bfacc_0.mp4 (Bundesliga, roboflow/sports)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$REPO/data"
mkdir -p "$DATA"

CLIP=0
for a in "$@"; do
  case "$a" in
    --clip-ejemplo) CLIP=1 ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) echo "Opción desconocida: $a" >&2; exit 2 ;;
  esac
done

# gdown: el del entorno .venv si existe; si no, el del sistema; si no, como módulo de Python.
if [ -x "$REPO/.venv/bin/gdown" ]; then
  GDOWN=("$REPO/.venv/bin/gdown")
elif command -v gdown >/dev/null 2>&1; then
  GDOWN=(gdown)
elif python3 -c "import gdown" >/dev/null 2>&1; then
  GDOWN=(python3 -m gdown)
else
  echo "Falta gdown. Instálalo con: pip install gdown (o ejecuta instalar.sh)" >&2
  exit 1
fi

descargar() {  # descargar NOMBRE ID_DE_GOOGLE_DRIVE
  local destino="$DATA/$1"
  if [ -s "$destino" ]; then
    echo "Ya existe $1, no se descarga de nuevo."
    return
  fi
  echo "Descargando $1 ..."
  "${GDOWN[@]}" "https://drive.google.com/uc?id=$2" -O "$destino"
}

descargar football-ball-detection.pt   1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V
descargar football-player-detection.pt 17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q
descargar football-pitch-detection.pt  1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf
if [ "$CLIP" = 1 ]; then
  descargar 0bfacc_0.mp4 12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF
fi
echo "Modelos listos en $DATA"
