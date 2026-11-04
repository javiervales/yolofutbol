# Material para reproducir el ajuste del artículo

- `kaggle_download.py`: descarga clips del conjunto público de Kaggle «DFL Bundesliga 460 MP4 Videos in 30Sec» (CC BY-SA 4.0). Necesita un token de Kaggle en `~/.kaggle/access_token` y el listado de ficheros del conjunto en JSON (respuesta de `/api/v1/datasets/list/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv`). Uso: `python codigo_articulo/kaggle_download.py LISTA.json data/kaggle [MAX]`.
- Rejillas de parámetros para `python -m yolofutbol tune --grid ...` (diccionario parámetro → lista de valores):
  - `tune_grid.json`: primer barrido, 576 combinaciones de tiempo de control, radios de posesión y presión, suavizado, prominencia y separación entre picos (cinco clips de roboflow/sports, frente al analista A).
  - `tune_grid_fix.json`: segundo barrido, 64 combinaciones de arrastre del balón, velocidad de cierre, radio de presión segura y radio de presión (mismos clips y referencia).
  - `tune_grid_final.json`: tercer barrido, 108 combinaciones sobre los 20 clips de ajuste, frente al analista A y al consenso.
- Las etiquetas de referencia están en `etiquetas/`.
