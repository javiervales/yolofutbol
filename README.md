# YOLOFUTBOL · índice de presión en fútbol a partir de vídeo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/javiervales/yolofutbol/blob/main/notebooks/YOLOFUTBOL_Colab.ipynb)

YOLOFUTBOL analiza un clip de fútbol grabado con una sola cámara en plano general, cuenta en cada instante cuántos rivales están presionando al equipo que tiene el balón y **señala al entrenador dónde mirar**: los momentos de presión máxima de la jugada, para ir directamente a ellos en el vídeo. No sustituye el visionado; lo dirige.

## Qué necesitas

- **Un vídeo de plano general** (cámara de retransmisión o de grada que vea buena parte del campo), en mp4 y a ser posible a 1080p. Clips de unos segundos a pocos minutos.
- **Opcional, una tarjeta gráfica NVIDIA** con un controlador reciente (compatible con CUDA 12.4; `nvidia-smi` debe funcionar). Sin ella funciona, pero tarda al menos 10 veces más; para clips largos conviene Colab con GPU. Con una tarjeta de gama media, un clip de 30 s a 1080p se analiza en unos 4 minutos, vídeo anotado incluido.
- **Según dónde lo uses:**
  - En Google Colab (opción 1): nada.
  - En tu ordenador (opción 2): Linux o macOS (en Windows, mediante WSL), conexión a internet y Python 3.10, 3.11 o 3.12. En Ubuntu/Debian conviene tener además el paquete `python3-venv` (`sudo apt install python3-venv`); si no lo tienes ni puedes instalarlo, `instalar.sh` se las arregla descargando `pip` por su cuenta.

## Tres formas de usarlo

### 1. En Google Colab, sin instalar nada (la más fácil)

Pulsa el botón **Open in Colab** de arriba. Se abre un cuaderno en el navegador con cinco pasos; ejecútalos en orden con el botón ▶. Te pedirá subir tu vídeo (o usará uno de ejemplo), lo analizará y te dejará descargar todos los resultados en un zip. Antes de empezar, elige en *Entorno de ejecución → Cambiar tipo de entorno de ejecución* la opción **GPU**.

### 2. En tu ordenador, con dos órdenes

Descarga el repositorio (`git clone https://github.com/javiervales/yolofutbol.git`, o el botón *Code → Download ZIP* de GitHub) y, en una terminal dentro de su carpeta:

```bash
bash instalar.sh                      # con tarjeta NVIDIA: bash instalar.sh --gpu
bash analizar.sh ~/Videos/mi_partido.mp4
```

`instalar.sh` crea un entorno de Python en `.venv/` dentro de la carpeta (no toca el resto del sistema), instala lo necesario y descarga los tres modelos de detección a `data/`. Solo hay que ejecutarlo una vez, y si algo falla se puede repetir sin más.

**El vídeo puede estar donde quieras**: a `analizar.sh` se le da su ruta, absoluta (`~/Videos/mi_partido.mp4`) o relativa a la carpeta del repositorio (`../mi_partido.mp4`). No hace falta copiarlo dentro.

**Para comprobar que todo funciona** antes de usar un vídeo propio, instala con el clip de ejemplo y analiza sus 5 primeros segundos (menos de un minuto con GPU):

```bash
bash instalar.sh --gpu --clip-ejemplo
bash analizar.sh data/0bfacc_0.mp4 --duration-s 5
```

`analizar.sh` analiza el clip con los parámetros del artículo, deja los resultados en `salidas/<nombre_del_vídeo>/` y escribe al final un resumen de las posesiones. Este es el resumen real de un clip de 30 s de la Bundesliga (uno de los de Kaggle usados en el artículo):

```
RESUMEN · test16.mp4 · 30,0 s
Balón detectado en el 70,0 % de los fotogramas.
Reparto de la posesión asignada: azul 100,0 %, naranja 0,0 %.
Rivales presionando, media con posesión: 0,26 (máximo 2).
  (referencia: en los 24 clips del artículo la media va de 0,03 a 0,65 rivales, con mediana 0,29)
Momentos de presión máxima, para ir a verlos (s): 4,2; 12,7; 15,8

Posesiones: 2
   1. equipo 0 (azul), de 1,3 a 8,4 s (7,2 s); primer presionante a los 2,7 s; presionantes medios 0,07 (máx. 2); termina: se pierde el balón de vista (pase largo, balón fuera de plano o no detectado).
   2. equipo 0 (azul), de 10,6 a 20,2 s (9,6 s); primer presionante a los 1,1 s; presionantes medios 0,40 (máx. 2); termina: se pierde el balón de vista (pase largo, balón fuera de plano o no detectado).
```

Los equipos se numeran 0 (azul) y 1 (naranja) según el color con que aparecen en el vídeo anotado; cuál es cada uno se ve en `annotated.mp4`. Para leer los «presionantes medios»: en los 24 clips del artículo, la media de un clip completo va de 0,03 a 0,65 rivales, con mediana 0,29. Un pico en el último instante del clip se lista aparte, porque puede deberse solo al corte del vídeo.

Se puede indicar otra carpeta de salida como segundo argumento (`bash analizar.sh mi_partido.mp4 resultados/partido1`) y añadir al final cualquier opción de la orden avanzada; por ejemplo, `--duration-s 10` analiza solo los 10 primeros segundos y `--no-video` no genera el vídeo anotado.

### Si algo falla

| Mensaje | Qué pasa y qué hacer |
|---|---|
| `ensurepip is not available` o `apt install python3.10-venv` | Falta el paquete `python3-venv` de Ubuntu/Debian. `instalar.sh` ya lo resuelve solo descargando `pip`; si aun así falla, instálalo (`sudo apt install python3-venv`) o, sin permisos de administrador, instala `uv` (`pipx install uv`) y repite `bash instalar.sh`. |
| `No module named pip` o un `.venv/` que quedó a medias | Viene de un intento de instalación anterior. Basta con repetir `bash instalar.sh`: detecta el entorno incompleto, lo borra y lo crea de nuevo. |
| `Falta gdown` o `falta el modelo data/...` | No se han descargado los modelos. Ejecuta `bash instalar.sh` (si no lo has hecho) o `bash descargar_modelos.sh`, que usa el entorno `.venv/`. |
| `falta el entorno de yolofutbol` | `analizar.sh` no encuentra las librerías: ejecuta antes `bash instalar.sh`. |
| `PyTorch no ve la tarjeta gráfica` | El controlador NVIDIA no está instalado o es antiguo. Funciona igual, pero por el procesador y mucho más despacio. |

Si no hay forma de instalarlo en tu equipo, usa el cuaderno de Colab (opción 1).

### 3. Uso avanzado

Todo pasa por una única orden con varias subórdenes:

```bash
source .venv/bin/activate
python -m yolofutbol --help          # subórdenes: run, replay, compare, tune, sensitivity, evaluate, sheets, storyboard
python -m yolofutbol run --help      # todas las opciones del análisis
```

`analizar.sh` equivale a:

```bash
python -m yolofutbol run --video mi_partido.mp4 --out salidas/mi_partido \
    --mode full --possession team --weights-dir data --device 0 \
    --h-ema 0.3 --closing-min-mps 0.5 --dist-presion-core-m 2.5 --carry-max-s 0 --radar-pos right
```

(`--device cpu` si no hay tarjeta gráfica). Solo se escriben las opciones cuyo valor en el artículo difiere del que trae el programa por defecto; el resto (radio de presión de 5 m, posesión a 2 m, etc.) son los valores de serie, listados en la tabla siguiente. El modo `baseline` reproduce el sistema original del trabajo fin de grado, en píxeles y sin seguimiento.

## Qué obtienes y cómo leerlo

En `salidas/<nombre_del_clip>/`:

| Fichero | Qué es |
|---|---|
| `pressure.png` | **La curva de presión.** Eje horizontal, tiempo; vertical, rivales presionando. Puntos grises: el recuento en cada fotograma. Línea de color: su media móvil (1 s a 25 fotogramas/s), que es la que se lee. Triángulos con el instante: momentos de presión máxima, los que conviene ir a ver. Donde no hay línea, el sistema no ha podido asignar la posesión a ningún equipo (balón no visto, disputa), y no significa que no haya presión. |
| `annotated.mp4` | **El vídeo anotado.** Recuadros azules y naranjas para los dos equipos, recuadro grueso para el poseedor, triángulo rojo sobre cada rival que presiona y círculo verde en el balón detectado. Abajo a la derecha, una vista cenital del campo con los jugadores y un círculo de 5 m alrededor del balón. Sirve para comprobar si lo que dice la curva es cierto. |
| `sequences.csv` | **La tabla de posesiones**, una fila por posesión de un equipo (columnas abajo). Se abre con Excel o LibreOffice. |
| `peaks.csv` | Los momentos de presión máxima: instante (`t_s`) y valor de la media móvil en ese instante. |
| `frames.csv` | Una fila por fotograma: tiempo, posición y estado del balón, equipo con la posesión, poseedor, rivales presionando (`n_pressers`) y distancia del rival más cercano. Para quien quiera hacer sus propios cálculos. |
| `summary.json` | Resumen técnico: porcentaje de fotogramas con balón detectado, reparto de la posesión, presión media, parámetros usados y versiones del software. |
| `detections.csv`, `ball.csv` | Caché de las detecciones; permiten recalcular con otros parámetros sin volver a procesar el vídeo (`replay`). |

Columnas de `sequences.csv`, en lenguaje de banquillo:

| Columna | Significado |
|---|---|
| `seq_id` | Número de la posesión dentro del clip (desde 0). |
| `team` | Equipo con el balón: 0 = azul, 1 = naranja en el vídeo anotado. Los colores los asigna el sistema según las camisetas; mira el vídeo para saber cuál es cada uno. |
| `start_t_s`, `duration_s` | Cuándo empieza la posesión y cuánto dura, en segundos. |
| `start_frame`, `end_frame`, `n_frames` | Lo mismo en fotogramas. |
| `n_controlled_frames` | Fotogramas en que un jugador concreto tiene el balón controlado (el resto, el balón viaja entre compañeros). |
| `n_touches` | Toques controlados del equipo durante la posesión. |
| `time_to_first_presser_s` | Segundos desde que el equipo gana el balón hasta que llega el primer rival a presionar. Vacío si nadie presiona. Mide la reacción del rival. |
| `mean_pressers` | Rivales presionando, de media, durante la posesión. Es la intensidad de la presión sufrida. |
| `max_pressers` | El mayor número de rivales presionando a la vez. |
| `mean_nearest_presser_dist_m` | Distancia media, en metros, del rival más cercano al balón. |
| `end_reason` | Cómo termina: `lost_to_opponent` (la recupera el rival), `ball_lost` (el sistema deja de ver el balón: pase largo, fuera de plano o no detectado) o `clip_end` (sigue al terminar el clip). |

Un rival cuenta como **presionante** si está a 5 m o menos del balón y, además, está a 2,5 m o menos o se acerca a 0,5 m/s o más. Así no cuenta quien solo acompaña de lejos.

## Parámetros del artículo y cómo cambiarlos

Parámetros que cambian la lectura táctica:

| Qué | Valor | Opción | Efecto práctico |
|---|---|---|---|
| Distancia para tener el balón | 2 m | `--dist-posesion-m` | Un jugador cuenta como poseedor si está a esta distancia del balón o menos. |
| Tiempo para ganar la posesión | 0,4 s | `--control-s` | El equipo gana la posesión cuando uno de sus jugadores se queda este tiempo junto al balón; evita contar balones que solo pasan cerca. |
| Radio de presión | 5 m | `--dist-presion-m` | Solo los rivales a esta distancia del balón o menos pueden contar como presionantes. Subirlo hace la curva más alta y menos selectiva. |
| Cerca: presiona siempre | 2,5 m | `--dist-presion-core-m` | Un rival a esta distancia o menos cuenta como presionante aunque esté quieto. |
| Velocidad mínima de acercamiento | 0,5 m/s | `--closing-min-mps` | Entre 2,5 y 5 m, el rival cuenta solo si se acerca al balón al menos a esta velocidad; así no cuenta quien solo acompaña. 0 lo desactiva. |
| Media móvil de la curva | 25 fotogramas | `--smooth-window` | Cuántos fotogramas se promedian para dibujar la línea de la curva (1 s a 25 fps). Más, curva más suave y con menos picos. |
| Altura mínima de un pico | 0,5 presionantes | `--peak-prominence` | Cuánto tiene que sobresalir la curva respecto a su entorno para marcar un momento de presión máxima. Más, menos picos. |
| Separación mínima entre picos | 1 s | `--peak-min-sep-s` | Dos picos más cercanos que esto se cuentan como uno. |

Parámetros técnicos, que normalmente no hace falta tocar:

| Qué | Valor | Opción | Efecto práctico |
|---|---|---|---|
| Suavizado de la proyección al campo | 0,3 | `--h-ema` | Suaviza el temblor de la vista cenital y de las distancias en metros cuando la cámara se mueve: 1 = sin suavizar; valores menores, posiciones más estables pero que tardan más en seguir un movimiento brusco de cámara. |
| Huecos del balón que se rellenan | 15 fotogramas | `--ball-gap-max` | Cuántos fotogramas seguidos sin balón detectado se rellenan en línea recta (0,6 s a 25 fps); si el balón falta más tiempo, se da por perdido y se cierra la posesión. |
| Balón portado | desactivado (0 s) | `--carry-max-s` | Cuando el detector pierde el balón, cuántos segundos se supone que sigue en los pies de quien lo llevaba. Desactivado en el artículo porque en el ajuste no mejoró la detección de picos. |
| Ventana del acercamiento | 0,5 s | `--closing-window-s` | Intervalo sobre el que se mide si un rival se acerca al balón. |

Para cambiar uno, añádelo al final: `bash analizar.sh mi_partido.mp4 salidas/radio6 --dist-presion-m 6`. Los parámetros de posesión y presión también se cambian sin volver a procesar el vídeo, en segundos, con `replay` (ver «Reproducir el ajuste del artículo»).

## Límites conocidos

- **Balón pequeño o lejano.** Si el balón ocupa pocos píxeles (plano muy abierto, baja resolución) el detector lo pierde a menudo; sin balón no hay posesión y sin posesión no hay curva.
- **Banda lejana.** Con una sola cámara la profundidad se comprime y el error al pasar a metros crece con la distancia: la presión en la banda opuesta a la cámara se infravalora.
- **Portero con el balón en las manos.** El detector no ve el balón en las manos, así que no se abre la posesión y se pierde la presión sobre el saque.
- **Falsos balones.** Un punto blanco cerca de un pie puede tomarse por el balón y crear un pico que no existe. Por eso la curva señala dónde mirar: el vídeo anotado permite descartarlo en segundos.
- El sistema no conoce la altura del balón ni la orientación de los jugadores.

## Reproducir el ajuste del artículo

Las etiquetas de referencia están en `etiquetas/` y las rejillas de parámetros y el script de descarga de los clips de Kaggle en `codigo_articulo/` (cada carpeta tiene su `LEEME.md`). Con el entorno activado (`source .venv/bin/activate`):

```bash
# 1. Procesar cada clip una vez; el nombre de la carpeta debe coincidir con la columna clip de las etiquetas
bash analizar.sh data/0bfacc_0.mp4 salidas/dfl_0bfacc_0 --no-video

# replay: recalcular posesión y presión con otros parámetros desde la caché, sin modelos ni vídeo
python -m yolofutbol replay --from salidas/dfl_0bfacc_0 --out salidas/radio6/dfl_0bfacc_0 --dist-presion-m 6

# tune: barrido de parámetros contra los picos etiquetados (escribe tune_results.csv, best.json y sensitivity.png)
python -m yolofutbol tune --runs salidas/dfl_* --labels etiquetas/labels_dfl.csv \
    --grid codigo_articulo/tune_grid_final.json --out salidas/tune_1

# evaluate: precisión, exhaustividad y F1 de una combinación frente a una referencia
python -m yolofutbol evaluate --runs salidas/dfl_* --labels etiquetas/labels_consenso.csv \
    --params salidas/tune_1/best.json --out salidas/eval_consenso

# sheets: hojas de miniaturas cada 0,5 s para marcar picos a ojo
python -m yolofutbol sheets --video data/0bfacc_0.mp4 --out salidas/hojas_0bfacc_0 --every-s 0.5 --cols 4 --rows 3

# storyboard: figura con la curva y fotogramas anotados en los picos y valles
python -m yolofutbol storyboard --run salidas/0bfacc_0 --video salidas/0bfacc_0/annotated.mp4 \
    --out salidas/storyboard_0bfacc_0.png
```

Los cinco clips de roboflow/sports se descargan de su repositorio (`data/0bfacc_0.mp4` con `descargar_modelos.sh --clip-ejemplo`); los quince de Kaggle, con `codigo_articulo/kaggle_download.py`. `tune` admite rejillas de cualquiera de los parámetros de posesión y presión de la tabla anterior.

## Estructura del repositorio

```
yolofutbol/               paquete de Python
  cli.py                  órdenes run, replay, compare, tune, evaluate, sheets, storyboard
  detect.py               detectores YOLO (personas, balón, puntos del campo)
  track_ball.py           filtro de saltos e interpolación del balón
  pitch.py                homografía: de píxeles a metros sobre el campo
  teams.py                equipos por color de la camiseta (k-means)
  pressure.py             posesión, presionantes, curva y picos
  viz.py                  curva, vídeo anotado, vista cenital, figuras
  tune.py                 barrido y evaluación frente a etiquetas
  labels.py               consenso y unión de las etiquetas de dos analistas
  io.py                   lectura de vídeo y CSV
notebooks/YOLOFUTBOL_Colab.ipynb   cuaderno para Google Colab
instalar.sh               instalación en un entorno .venv
descargar_modelos.sh      descarga de los modelos (y del clip de ejemplo)
analizar.sh               análisis de un clip con los parámetros del artículo
etiquetas/                picos de presión marcados en los clips de ajuste
codigo_articulo/          rejillas de parámetros y descarga de clips de Kaggle
requirements.txt          dependencias (torch se instala aparte)
```

`data/` (modelos y vídeos) y `salidas/` (resultados) no se suben al repositorio.

## Cómo citarlo

Artículo (en revisión):

> Vales-Alonso, J. y Van Eijk, A. (2026). Prototipado rápido de herramientas de análisis táctico con visión por computador para deportes colectivos: un índice de presión en fútbol. *Journal of Universal Movement and Performance (JUMP)*, en revisión.

Trabajo fin de grado de origen:

> Van Eijk, A. (2026). *Estudio mediante YOLO de la detección automática de situaciones de presión en fútbol*. Trabajo Fin de Grado, Grado en Ingeniería Telemática, Universidad Politécnica de Cartagena.

## Licencia y agradecimientos

Código bajo licencia MIT (ver `LICENSE`). Los modelos de detección y los clips de ejemplo son los publicados por [roboflow/sports](https://github.com/roboflow/sports), a cuyo equipo agradecemos que los pusiera a disposición de todos; el seguimiento y las anotaciones usan [supervision](https://github.com/roboflow/supervision) y los detectores [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). Los clips de Kaggle proceden del conjunto «DFL Bundesliga 460 MP4 Videos in 30Sec» (CC BY-SA 4.0).

---

## In English

YOLOFUTBOL turns a single-camera, wide-angle football clip into a **pressing index**: the number of opponents pressing the team in possession at each instant, with the peak moments flagged so the coach knows where to look. Outputs: `pressure.png` (pressing curve), `annotated.mp4` (teams, ball carrier, pressers and a top-down radar view) and `sequences.csv` (one row per team possession: time to first presser, mean and maximum pressers, how it ended).

- **No install:** click *Open in Colab* above and run the cells in order (select a GPU runtime).
- **Local:** `bash instalar.sh` (`--gpu` for NVIDIA/CUDA 12.4), then `bash analizar.sh path/to/my_clip.mp4`; results go to `salidas/my_clip/`. On Ubuntu/Debian without `python3-venv`, the installer bootstraps pip itself; re-running it repairs a half-created `.venv/`. Runs on CPU too, at least 10 times slower.
- **Advanced / reproducing the paper:** `python -m yolofutbol --help`; subcommands `run`, `replay`, `tune`, `evaluate`, `sheets`, `storyboard`. Reference labels are in `etiquetas/`, parameter grids in `codigo_articulo/`.

Requires Python 3.10–3.12, `supervision>=0.26,<0.31` and the public roboflow/sports models (downloaded by `descargar_modelos.sh`). MIT licence. Please cite the paper above (under review) and the original bachelor's thesis.
