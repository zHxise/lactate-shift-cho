"""Variables predictoras de una ventana temprana.

Ninguna variable depende de datos posteriores al ultimo dia de la ventana ni
de otras series. Los faltantes se dejan como NaN; la imputacion se hace en el
pipeline de modelado, dentro de cada fold.

Desarrollado por Arturo Rodriguez.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["slope", "early_window_features"]


def slope(days: Sequence[float], values: Sequence[float]) -> float:
    """Pendiente por minimos cuadrados. NaN si hay menos de dos puntos."""
    d = np.asarray(days, dtype=float)
    v = np.asarray(values, dtype=float)
    m = np.isfinite(d) & np.isfinite(v)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(d[m], v[m], 1)[0])


def _causal_window(
    g: pd.DataFrame, day_col: str, window: int, value_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dias 1..window en rejilla completa, rellenando solo con el dia anterior
    (ffill). Devuelve la tabla rellenada y la mascara de lo medido antes de
    rellenar."""
    w = g[g[day_col] <= window]
    if w[day_col].duplicated().any():
        raise ValueError("hay dias repetidos dentro de una misma serie")
    w = w.set_index(day_col).reindex(np.arange(1, window + 1))
    medido = w[value_cols].notna()
    w[value_cols] = w[value_cols].ffill()
    return w, medido


def early_window_features(
    data: pd.DataFrame,
    *,
    id_col: str,
    day_col: str,
    value_cols: Sequence[str],
    window: int = 4,
    ratios: Sequence[tuple[str, str]] = (),
    static_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Construye una fila de variables por serie a partir de sus primeros dias.

    Para cada columna en ``value_cols`` genera cuatro variables:

    ``<col>_last``
        Ultimo valor dentro de la ventana (si falta el ultimo dia, el del dia
        medido mas reciente).
    ``<col>_slope``
        Pendiente por minimos cuadrados, solo con los dias medidos.
    ``<col>_mean``
        Promedio de los dias medidos.
    ``<col>_n``
        Numero de dias medidos (sin contar los rellenados).

    ``ratios``: pares ``(numerador, denominador)``; agrega el cociente al
    cierre de la ventana.

    ``static_cols``: columnas constantes dentro de cada serie y conocidas
    desde el inicio (por ejemplo la escala). Se toma su primer valor.
    """
    value_cols = list(value_cols)
    filas = []
    for key, g in data.groupby(id_col, sort=True):
        g = g.sort_values(day_col)
        w, medido = _causal_window(g, day_col, window, value_cols)
        days = w.index.to_numpy(dtype=float)
        f: dict[str, float] = {}

        for col in value_cols:
            x = w[col].to_numpy(dtype=float)
            m = medido[col].to_numpy()
            x_med = np.where(m, x, np.nan)
            f[f"{col}_last"] = x[-1]
            f[f"{col}_slope"] = slope(days, x_med)
            f[f"{col}_mean"] = float(np.nanmean(x_med)) if m.any() else np.nan
            f[f"{col}_n"] = int(m.sum())

        for num, den in ratios:
            a, b = w[num].to_numpy(dtype=float)[-1], w[den].to_numpy(dtype=float)[-1]
            f[f"{num}_over_{den}"] = a / b if np.isfinite(b) and abs(b) > 1e-12 else np.nan

        for col in static_cols:
            f[col] = g[col].iloc[0]

        f[id_col] = key
        filas.append(f)

    return pd.DataFrame(filas).set_index(id_col).sort_index()
