"""Fusión de etiquetas manuales de dos analistas (A y B) en etiquetas de consenso.

Entrada: results/labels/dfl_<clip>.csv (analista A) y dfl_<clip>_b.csv (analista B),
columnas clip,t_s,n_pressers,confidence,note (filas con t_s vacío son resúmenes).
Salida: labels_consenso.csv (picos marcados por ambos a <= tol s, instante medio, n_pressers medio redondeado),
labels_union.csv (todos los picos de ambos, fusionando los coincidentes), labels_A.csv (solo el analista A)
y acuerdo.csv (acuerdo entre analistas por clip). Todas las de picos llevan columna n_pressers.
Uso: python -m yolofutbol.labels DIR_LABELS [--tol-s 1.5]
"""
import csv
import glob
import math
import os
import sys


def _read(path):
    """Lee los picos (filas con t_s) de un CSV de etiquetas: (t_s, confianza, n_pressers o None)."""
    rows = []
    for r in csv.DictReader(open(path, newline="")):
        if r["t_s"].strip():
            n = (r.get("n_pressers") or "").strip()
            rows.append((float(r["t_s"]), r["confidence"], int(float(n)) if n else None))
    return sorted(rows, key=lambda x: x[0])


def _mean_n(na, nb):
    """Media redondeada (mitades hacia arriba) de dos n_pressers; si falta uno, el otro."""
    if na is None or nb is None:
        return nb if na is None else na
    return int(math.floor((na + nb) / 2 + 0.5))


def _match(a, b, tol):
    """Empareja uno a uno los instantes de a y b por cercanía dentro de tol."""
    pairs, used_b = [], set()
    cands = sorted((abs(a[i][0] - b[j][0]), i, j) for i in range(len(a)) for j in range(len(b))
                   if abs(a[i][0] - b[j][0]) <= tol)
    used_a = set()
    for _, i, j in cands:
        if i in used_a or j in used_b:
            continue
        pairs.append((i, j)); used_a.add(i); used_b.add(j)
    return pairs, used_a, used_b


def main(labels_dir, tol=1.5):
    """Escribe consenso, unión y acuerdo en labels_dir."""
    cons, union, acuerdo, only_a = [], [], [], []
    excluded = {"kg_test13", "kg_test22", "kg_test23", "kg_test24", "kg_test27"}  # duplicados (md5) de clips ya usados
    for pa in sorted(glob.glob(os.path.join(labels_dir, "dfl_*.csv")) + glob.glob(os.path.join(labels_dir, "kg_*.csv"))):
        if pa.endswith("_b.csv") or os.path.basename(pa).startswith("labels"):
            continue
        if os.path.basename(pa)[:-4] in excluded:
            continue
        clip = os.path.basename(pa)[:-4]
        pb = pa[:-4] + "_b.csv"
        a = _read(pa)
        if not os.path.exists(pb):
            # Solo analista A: entra en labels_A y en la unión; sin consenso ni acuerdo
            only_a.extend((clip, t, c, n) for t, c, n in a)
            union.extend((clip, t, c, "solo_A", n) for t, c, n in a)
            continue
        b = _read(pb)
        only_a.extend((clip, t, c, n) for t, c, n in a)
        pairs, ua, ub = _match(a, b, tol)
        for i, j in pairs:
            t = round((a[i][0] + b[j][0]) / 2, 2)
            conf = "alta" if "baja" not in (a[i][1], b[j][1]) else "media"
            n = _mean_n(a[i][2], b[j][2])
            cons.append((clip, t, conf, n)); union.append((clip, t, conf, "ambos", n))
        for i, (t, c, n) in enumerate(a):
            if i not in ua:
                union.append((clip, t, c, "solo_A", n))
        for j, (t, c, n) in enumerate(b):
            if j not in ub:
                union.append((clip, t, c, "solo_B", n))
        acuerdo.append((clip, len(a), len(b), len(pairs), round(2 * len(pairs) / (len(a) + len(b)), 3)))
    with open(os.path.join(labels_dir, "labels_consenso.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "t_s", "confidence", "n_pressers"]); w.writerows(cons)
    with open(os.path.join(labels_dir, "labels_union.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "t_s", "confidence", "origen", "n_pressers"])
        w.writerows(sorted(union, key=lambda r: (r[0], r[1])))
    with open(os.path.join(labels_dir, "labels_A.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "t_s", "confidence", "n_pressers"]); w.writerows(only_a)
    with open(os.path.join(labels_dir, "acuerdo.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "n_A", "n_B", "coincidentes", "acuerdo_dice"]); w.writerows(acuerdo)
    tot_a = sum(r[1] for r in acuerdo); tot_b = sum(r[2] for r in acuerdo); tot_p = sum(r[3] for r in acuerdo)
    print("acuerdo por clip:", acuerdo)
    dice = f"{2*tot_p/(tot_a+tot_b):.3f}" if tot_a + tot_b else "n/a"
    print(f"TOTAL A={len(only_a)+tot_a} B={tot_b} coincidentes={tot_p} Dice={dice}; consenso={len(cons)} union={len(union)}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[3]) if len(sys.argv) > 3 else 1.5)
