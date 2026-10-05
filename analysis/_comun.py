"""Funciones comunes para cargar el dataset del caso de estudio.

Desarrollado por Arturo Rodriguez.
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

    'Culture ID' viene como 'C12_d7' (cultivo y dia juntos), se separa en dos
    columnas. En Gap-Filled la columna de lactato trae un espacio al final
    ('[Lactate] '), por eso el .strip().
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


# Variables predictoras. Se definen solo aqui para que todos los scripts usen
# la misma matriz.
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

    # Crecimiento relativo de la biomasa en la ventana (primer y ultimo dato
    # medido). Un cultivo que crece mas rapido agota antes el sustrato.
    v = raw[raw["day"] <= VENTANA].sort_values("day").groupby("cult")["VCD"]
    X["vcd_crec_rel"] = v.last() / v.first().replace(0, np.nan)
    return X
