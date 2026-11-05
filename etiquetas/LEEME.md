# Etiquetas de picos de presión (clips de ajuste)

## `fable/`: marcas usadas en el artículo

Marcas del analista automático del artículo (Claude Fable 5.1), que siguiendo un protocolo escrito marcó los picos de presión de los 20 clips de ajuste sobre hojas de miniaturas cada 0,5 s. Pico = uno o más rivales cerrando activamente al poseedor a 5 m o menos. `fable/<clip>.csv` tiene las columnas `clip`, `t_s`, `n_pressers`, `confidence`, `note` (las filas sin `t_s` resumen las fases de posesión); `fable/labels_A.csv` reúne las 84 marcas y es el fichero que usan `tune` y `evaluate` (`--labels etiquetas/fable/labels_A.csv`). La intensidad de referencia de cada clip es la suma de `n_pressers` de sus marcas por minuto; `codigo_articulo/nombres_clips.json` da los rótulos K01–K15 de la figura (`--names`).

## Raíz: marcas previas de dos analistas

Marcas anteriores de dos analistas automáticos (analista A: Claude Opus; analista B: Claude Sonnet) con el mismo protocolo, conservadas como material de contraste; no son las del artículo.

- `<clip>.csv` (analista A) y `<clip>_b.csv` (analista B): `clip`, `t_s` (instante en segundos), `n_pressers` (rivales que presionan), `confidence` (baja/media/alta), `note` (descripción de la jugada).
- Clips: `dfl_<id>` son los cinco de roboflow/sports (`<id>.mp4`); `kg_testN` son del conjunto de Kaggle «DFL Bundesliga 460 MP4 Videos in 30Sec» (CC BY-SA 4.0), ver `codigo_articulo/kaggle_download.py`.
- `labels_A.csv`: todas las marcas del analista A (`clip`, `t_s`, `confidence`); `labels_dfl.csv` y `labels_dfl_ma.csv`: las del analista A en los clips de roboflow/sports, todas y solo las de confianza media o alta.
- `labels_consenso.csv`: picos marcados por ambos (emparejados a ≤ 1,5 s; instante medio). `labels_union.csv`: picos de cualquiera, con `origen` = ambos, solo_A o solo_B.
- `acuerdo.csv`: por clip, número de marcas de A y de B, coincidentes e índice de Dice.
- Consenso, unión y acuerdo se regeneran con `python -m yolofutbol.labels etiquetas` (los clips duplicados de Kaggle quedan excluidos); con una carpeta de un solo analista, como `etiquetas/fable`, escribe solo `labels_A.csv` y la unión.
- La columna `clip` debe coincidir con el nombre de la carpeta de cada ejecución para usarlas con `tune` y `evaluate`.
