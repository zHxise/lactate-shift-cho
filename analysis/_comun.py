"""Carga del dataset del caso de estudio.

Aislar esto en un modulo evita repetir en cada script los dos detalles del
archivo que se olvidan facil: los nombres de columna con espacios sobrantes
y el identificador que mezcla cultivo y dia en un solo string.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
XLSX = RAIZ / "data" / "raw" / "1-s2_0-S0098135421000041-mmc3.xlsx"
TABLAS = RAIZ / "outputs" / "tablas"
FIGURAS = RAIZ / "outputs" / "figuras"
TABLAS.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

# Ventana de observacion: los dias que el modelo puede ver.
VENTANA = 4


def cargar(hoja: str = "Raw Data") -> pd.DataFrame:
    """Lee una hoja de series de tiempo en formato largo.

    'Culture ID' viene como 'C12_d7': cultivo y dia pegados. Se separan porque
    el dia es el eje temporal y el cultivo es la unidad de analisis. Ademas, en
    la hoja Gap-Filled la columna de lactato se llama '[Lactate] ', con espacio
    final; sin .strip() el mismo nombre no casa entre hojas.
    """
    if not XLSX.exists():
        raise FileNotFoundError(
            f"No esta el dataset en {XLSX}.\n"
            "Es material suplementario bajo copyright de Elsevier y no se "
            "distribuye con el repositorio. Ver data/raw/README.md."
        )
    df = pd.read_excel(XLSX, sheet_name=hoja)
    df.columns = [c.strip() for c in df.columns]
    cid = df["Culture ID"].astype(str)
    df["cult"] = cid.str.extract(r"^(C\d+)_")[0]
    df["day"] = cid.str.extract(r"_d(\d+)$")[0].astype(float)
    return df.sort_values(["cult", "day"]).reset_index(drop=True)
