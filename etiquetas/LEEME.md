# Etiquetas de picos de presión (clips de ajuste)

Marcas de los dos analistas automáticos del artículo (analista A: Claude Opus; analista B: Claude Sonnet), que siguiendo el mismo protocolo marcaron por separado los picos de presión de los 20 clips de ajuste sobre hojas de miniaturas cada 0,5 s. Pico = uno o más rivales cerrando activamente al poseedor a 5 m o menos.

- `<clip>.csv` (analista A) y `<clip>_b.csv` (analista B): `clip`, `t_s` (instante en segundos), `n_pressers` (rivales que presionan), `confidence` (baja/media/alta), `note` (descripción de la jugada).
- Clips: `dfl_<id>` son los cinco de roboflow/sports (`<id>.mp4`); `kg_testN` son del conjunto de Kaggle «DFL Bundesliga 460 MP4 Videos in 30Sec» (CC BY-SA 4.0), ver `codigo_articulo/kaggle_download.py`.
- `labels_A.csv`: todas las marcas del analista A (`clip`, `t_s`, `confidence`); `labels_dfl.csv` y `labels_dfl_ma.csv`: las del analista A en los clips de roboflow/sports, todas y solo las de confianza media o alta.
- `labels_consenso.csv`: picos marcados por ambos (emparejados a ≤ 1,5 s; instante medio). `labels_union.csv`: picos de cualquiera, con `origen` = ambos, solo_A o solo_B.
- `acuerdo.csv`: por clip, número de marcas de A y de B, coincidentes e índice de Dice.
- Consenso, unión y acuerdo se regeneran con `python -m yolofutbol.labels etiquetas` (los clips duplicados de Kaggle quedan excluidos).
- La columna `clip` debe coincidir con el nombre de la carpeta de cada ejecución para usarlas con `tune` y `evaluate`.
