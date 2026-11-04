#!/usr/bin/env bash
# Instala yolofutbol en un entorno virtual .venv dentro de esta carpeta y descarga los modelos.
# Uso: bash instalar.sh [--gpu] [--clip-ejemplo]
#   --gpu           instala PyTorch con CUDA 12.4 (tarjeta gráfica NVIDIA); sin ella, versión para procesador
#   --clip-ejemplo  descarga también el clip de prueba 0bfacc_0.mp4
# Para elegir el intérprete: PYTHON=/ruta/a/python3.11 bash instalar.sh
# Se puede volver a ejecutar sin problema: si el entorno quedó a medias, se borra y se crea de nuevo.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
VENV="$REPO/.venv"
VPY="$VENV/bin/python"

GPU=0
DESCARGA=()
for a in "$@"; do
  case "$a" in
    --gpu) GPU=1 ;;
    --clip-ejemplo) DESCARGA+=(--clip-ejemplo) ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "Opción desconocida: $a" >&2; exit 2 ;;
  esac
done

# --- 1. Python 3.10, 3.11 o 3.12 --------------------------------------------------------------
version_ok() { "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; }
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && version_ok "$c"; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ] || ! version_ok "$PY"; then
  cat >&2 <<'MSG'
ERROR: hace falta Python 3.10, 3.11 o 3.12.
  - Ubuntu/Debian: sudo apt install python3 python3-venv
  - macOS o Windows: https://www.python.org/downloads/
  - Sin permisos de administrador: pipx install uv   (uv descarga su propio Python)
  - O usa el cuaderno de Google Colab, que no necesita instalar nada (ver README).
MSG
  exit 1
fi
PYV="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
echo "Usando Python $PYV ($PY)"

# --- 2. Entorno virtual .venv -----------------------------------------------------------------
fallo_entorno() {
  rm -rf "$VENV"
  cat >&2 <<MSG

ERROR: no se pudo crear el entorno de Python en .venv/ (se ha borrado lo que quedó a medias).
Soluciones, de más a menos sencilla:
  - Ubuntu/Debian, con permisos de administrador:  sudo apt install python3-venv python${PYV}-venv
  - Sin permisos de administrador:                 pipx install uv     (o: pip install --user uv)
  - Comprueba la conexión a internet (se descarga pip si el sistema no lo trae).
Después vuelve a ejecutar: bash instalar.sh $*
O usa el cuaderno de Google Colab (ver README), que no necesita instalar nada.
MSG
  exit 1
}

descargar() {  # descargar URL FICHERO
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then wget -q "$1" -O "$2"
  else "$PY" -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
  fi
}

crear_entorno() {
  if "$PY" -m ensurepip --version >/dev/null 2>&1; then
    echo "Creando el entorno con venv ..."
    "$PY" -m venv "$VENV"
  elif command -v uv >/dev/null 2>&1; then
    echo "El sistema no trae ensurepip; creando el entorno con uv ..."
    uv venv --seed --python "$PY" "$VENV"
  elif command -v virtualenv >/dev/null 2>&1; then
    echo "El sistema no trae ensurepip; creando el entorno con virtualenv ..."
    virtualenv -p "$PY" "$VENV"
  else
    echo "El sistema no trae ensurepip (paquete python3-venv); creando el entorno sin pip y descargando get-pip.py ..."
    "$PY" -m venv --without-pip "$VENV"
    descargar https://bootstrap.pypa.io/get-pip.py "$VENV/get-pip.py"
    "$VPY" "$VENV/get-pip.py" --quiet
    rm -f "$VENV/get-pip.py"
  fi
}

if [ -x "$VPY" ] && "$VPY" -m pip --version >/dev/null 2>&1; then
  echo "Entorno .venv ya creado; se reutiliza."
else
  [ -e "$VENV" ] && { echo "Entorno .venv incompleto de un intento anterior; se borra y se crea de nuevo."; rm -rf "$VENV"; }
  crear_entorno || fallo_entorno "$@"
  "$VPY" -m pip --version >/dev/null 2>&1 || fallo_entorno "$@"
fi

# --- 3. Librerías -----------------------------------------------------------------------------
fallo_pip() {
  cat >&2 <<'MSG'

ERROR: falló la instalación de librerías (mira el mensaje de pip justo encima).
Causas habituales: sin conexión a internet, poco espacio en disco o una versión de Python no soportada.
Puedes volver a ejecutar bash instalar.sh: lo ya instalado se reutiliza.
MSG
  exit 1
}

echo "Instalando librerías (puede tardar varios minutos) ..."
"$VPY" -m pip install --quiet --upgrade pip || fallo_pip
if [ "$GPU" = 1 ]; then
  "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || fallo_pip
elif [ "$(uname -s)" = "Darwin" ]; then
  "$VPY" -m pip install torch torchvision || fallo_pip
else
  "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || fallo_pip
fi
# requirements.txt incluye supervision>=0.26,<0.31 y el paquete sports (git+https://github.com/roboflow/sports.git).
"$VPY" -m pip install -r requirements.txt gdown || fallo_pip

if [ "$GPU" = 1 ] && ! "$VPY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "AVISO: PyTorch no ve la tarjeta gráfica (¿controlador NVIDIA instalado y reciente?, ver nvidia-smi)."
  echo "       El análisis funcionará igualmente, pero por el procesador y mucho más despacio."
fi

# --- 4. Modelos -------------------------------------------------------------------------------
YOLOFUTBOL_PYTHON="$VPY" bash "$REPO/descargar_modelos.sh" "${DESCARGA[@]+"${DESCARGA[@]}"}"

echo
echo "Instalación terminada. Para analizar un clip:"
echo "  bash analizar.sh ruta/al/clip.mp4"
if [ ${#DESCARGA[@]} -gt 0 ]; then
  echo "Prueba rápida con el clip de ejemplo (5 s):"
  echo "  bash analizar.sh data/0bfacc_0.mp4 --duration-s 5"
fi
