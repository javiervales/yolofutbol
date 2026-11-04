"""Descarga clips del dataset público de Kaggle «DFL Bundesliga 460 MP4 Videos in 30Sec» (CC BY-SA 4.0).

Uso: python kaggle_download.py LISTA_JSON DIR_SALIDA [MAX]
LISTA_JSON: respuesta de /api/v1/datasets/list/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv
Token: ~/.kaggle/access_token (cabecera Authorization: Bearer). Salta los ficheros ya descargados.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile

DATASET = "saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv"


def main(list_json, out_dir, max_files=None):
    token = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
    data = json.load(open(list_json))
    if isinstance(data, dict):
        data = data.get("datasetFiles") or data.get("files") or []
    files = [f["name"] for f in data if f["name"].endswith(".mp4")]
    if max_files:
        files = files[: int(max_files)]
    os.makedirs(out_dir, exist_ok=True)
    for name in files:
        out = os.path.join(out_dir, os.path.basename(name).replace(" ", "").replace("(", "").replace(")", ""))
        if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
            continue
        enc = urllib.parse.quote(name, safe="")
        url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/{enc}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=600) as r, open(out + ".dl", "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        # Kaggle sirve cada fichero comprimido en zip: se extrae el mp4.
        with zipfile.ZipFile(out + ".dl") as z:
            inner = [n for n in z.namelist() if n.endswith(".mp4")][0]
            with z.open(inner) as src, open(out, "wb") as dst:
                dst.write(src.read())
        os.remove(out + ".dl")
        print(out, os.path.getsize(out), flush=True)
        time.sleep(2)
    print("FIN")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
