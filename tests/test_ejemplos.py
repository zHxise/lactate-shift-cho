"""Los scripts de examples/ deben correr con una instalacion minima.

Existe porque el script del CSV fallaba si solo se instalaban las
dependencias de desarrollo: usaba matplotlib, que no es dependencia del
paquete. La prueba de usuario del autor no lo detecto porque ya tenia todo
instalado.
"""

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_detectar_en_csv_sin_figuras(tmp_path):
    csv = tmp_path / "cultivos.csv"
    csv.write_text((RAIZ / "examples" / "ejemplo_cultivos.csv").read_text())
    r = subprocess.run([sys.executable, str(RAIZ / "examples" / "detectar_en_mi_csv.py"),
                        str(csv), "--sin-figuras"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "6 de 7 cultivos con shift" in r.stdout


def test_detectar_en_csv_con_columnas_equivocadas(tmp_path):
    csv = tmp_path / "malo.csv"
    csv.write_text("cultivo,dia,lactato\nA,1,0.1\n")
    r = subprocess.run([sys.executable, str(RAIZ / "examples" / "detectar_en_mi_csv.py"),
                        str(csv)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "faltan columnas" in r.stdout


# --- Casos que encontro la prueba con datos inventados (24 sep 2026). Antes
# de corregirlos: el CSV vacio tronaba con una traza de pandas, el de Excel en
# espanol decia que faltaban columnas que si estaban, el texto en una celda
# daba un error en ingles y la tabla ponia C10 antes que C2.

def _correr(tmp_path, contenido, *extra):
    csv = tmp_path / "datos.csv"
    csv.write_text(contenido, encoding="utf-8")
    return subprocess.run([sys.executable, str(RAIZ / "examples" / "detectar_en_mi_csv.py"),
                           str(csv), "--sin-figuras", *extra], capture_output=True, text=True)


def test_csv_vacio_da_mensaje_claro(tmp_path):
    r = _correr(tmp_path, "culture,day,lactate\n")
    assert r.returncode == 1
    assert "no tiene datos" in r.stdout
    assert "Traceback" not in r.stderr


def test_texto_en_una_celda_dice_la_fila(tmp_path):
    r = _correr(tmp_path, "culture,day,lactate\nA,1,0.5\nA,2,tres\n")
    assert r.returncode == 1
    assert "fila 3" in r.stdout and "'tres' no es un numero" in r.stdout


def test_csv_de_excel_en_espanol(tmp_path):
    filas = ["culture;day;lactate"] + [f"A;{d};{v}" for d, v in
                                        enumerate(["0,5", "1,5", "3,0", "4,0", "2,0", "1,0", "0,8"], 1)]
    r = _correr(tmp_path, "\n".join(filas) + "\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 de 1 cultivos con shift" in r.stdout


def test_orden_natural_de_cultivos(tmp_path):
    subida = [0.5, 1.5, 3.0, 4.0, 4.2, 4.4, 4.5]
    filas = ["culture,day,lactate"] + [f"{c},{d},{v}" for c in ("C10", "C2")
                                       for d, v in enumerate(subida, 1)]
    r = _correr(tmp_path, "\n".join(filas) + "\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.index("C2 ") < r.stdout.index("C10")
