"""Variables predictoras de una ventana temprana, sin fuga de informacion.

La regla que gobierna este modulo: ninguna cantidad calculada aqui puede
depender de un dato posterior al ultimo dia de la ventana, ni de otras series
del conjunto.

Lo primero evita mirar el futuro. Lo segundo evita que el conjunto de prueba
se filtre al de entrenamiento: por eso los valores que faltan por completo se
dejan como ``NaN`` en lugar de imputarse aqui. La imputacion pertenece al
pipeline de modelado, donde se ajusta solo con el fold de entrenamiento.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["slope", "early_window_features"]


def slope(days: Sequence[float], values: Sequence[float]) -> float:
    """Pendiente de una recta ajustada por minimos cuadrados.

    Se prefiere sobre la diferencia entre extremos (``ultimo - primero``)
    porque con tres o cuatro puntos ruidosos esa diferencia depende por
    completo de dos mediciones, y una sola lectura mala la arruina. La recta
    usa todos los puntos disponibles. Con menos de dos puntos validos no hay
    pendiente definida y devuelve ``NaN``.
    """
    d = np.asarray(days, dtype=float)
    v = np.asarray(values, dtype=float)
    m = np.isfinite(d) & np.isfinite(v)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(d[m], v[m], 1)[0])


def _causal_window(g: pd.DataFrame, day_col: str, window: int, value_cols: list[str]) -> pd.DataFrame:
    """Dias 1..window en rejilla completa, con relleno solo hacia atras.

    ``ffill`` hace que un hueco herede del dia anterior, nunca del posterior.
    Es la unica forma de relleno admisible dentro de la ventana: el dia 3 puede
    heredar del dia 2, jamas del dia 5.
    """
    w = g[g[day_col] <= window].set_index(day_col)
    w = w.reindex(np.arange(1, window + 1))
    w[value_cols] = w[value_cols].ffill()
    return w


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
        Valor al cerrar la ventana: el nivel alcanzado.
    ``<col>_slope``
        Pendiente por minimos cuadrados: la velocidad de cambio.
    ``<col>_mean``
        Promedio en la ventana.
    ``<col>_n``
        Cuantos dias se midieron de verdad. Importa: una pendiente calculada
        sobre dos puntos no merece la misma confianza que una sobre cuatro, y
        dejar esa cuenta como variable permite que el modelo lo tenga en cuenta
        y que tu lo audites despues.

    ``ratios`` acepta pares ``(numerador, denominador)`` y agrega el cociente
    al cierre de la ventana. Los cocientes suelen ser mas comparables entre
    escalas y lotes que los niveles absolutos.

    ``static_cols`` son columnas constantes dentro de cada serie y conocidas
    desde el inicio (escala del reactor, consigna de temperatura). Se toma su
    primer valor. No incluyas aqui nada que solo se sepa al terminar el
    cultivo, como su duracion.
    """
    value_cols = list(value_cols)
    filas = []
    for key, g in data.groupby(id_col, sort=True):
        g = g.sort_values(day_col)
        w = _causal_window(g, day_col, window, value_cols)
        days = w.index.to_numpy(dtype=float)
        f: dict[str, float] = {}

        for col in value_cols:
            x = w[col].to_numpy(dtype=float)
            f[f"{col}_last"] = x[-1]
            f[f"{col}_slope"] = slope(days, x)
            f[f"{col}_mean"] = np.nanmean(x) if np.any(np.isfinite(x)) else np.nan
            f[f"{col}_n"] = int(np.isfinite(x).sum())

        for num, den in ratios:
            a, b = w[num].to_numpy(dtype=float)[-1], w[den].to_numpy(dtype=float)[-1]
            f[f"{num}_over_{den}"] = a / b if np.isfinite(b) and abs(b) > 1e-12 else np.nan

        for col in static_cols:
            f[col] = g[col].iloc[0]

        f[id_col] = key
        filas.append(f)

    return pd.DataFrame(filas).set_index(id_col).sort_index()
