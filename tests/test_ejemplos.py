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
