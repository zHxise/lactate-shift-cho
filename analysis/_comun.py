"""Carga del dataset del caso de estudio.

Aislar esto en un modulo evita repetir en cada script los dos detalles del
archivo que se olvidan facil: los nombres de columna con espacios sobrantes
y el identificador que mezcla cultivo y dia en un solo string.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# Variables predictoras: UNA sola definicion para todo el analisis.
# Estaba duplicada en 03 y 06, y la duplicacion ya produjo un bug (06 corria
# sin vcd_crec_rel). Cualquier script que necesite la matriz de variables la
# construye con esta funcion.
# ---------------------------------------------------------------------------
NUCLEO = ["[Lactate]", "[Glucose]", "VCD", "[NH3]", "pH", "[Glutamate]"]
EXTENDIDAS = ["[Glutamine]", "Osmolality"]
COCIENTES = [("[Lactate]", "[Glucose]"), ("[Lactate]", "VCD")]


def _limpiar(c: str) -> str:
    return (c.replace("[", "").replace("]", "").replace("+", "")
             .replace("-", "").replace(" ", "_").lower())


def construir_variables(raw: pd.DataFrame, extendidas: bool = True) -> pd.DataFrame:
    """Matriz de variables de los dias 1..VENTANA, una fila por cultivo."""
    from lactateshift import early_window_features

    cols = NUCLEO + (EXTENDIDAS if extendidas else [])
    X = early_window_features(raw, id_col="cult", day_col="day",
                              value_cols=cols, window=VENTANA,
                              ratios=COCIENTES, static_cols=["Culture Volume"])
    X = X.rename(columns=_limpiar).rename(columns={"culture_volume": "escala"})

    # Crecimiento relativo de la biomasa en la ventana, con valores medidos
    # (primer y ultimo dato no nulo). Un cultivo que crece mas rapido agota
    # sustrato antes: es la hipotesis mas obvia y hay que vencerla antes de
    # contar historias metabolicas mas finas.
    v = raw[raw["day"] <= VENTANA].sort_values("day").groupby("cult")["VCD"]
    X["vcd_crec_rel"] = v.last() / v.first().replace(0, np.nan)
    return X
