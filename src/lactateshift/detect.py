"""Deteccion del lactate shift en series de tiempo de lactato.

El lactate shift es el momento en que un cultivo de celulas de mamifero pasa
de producir lactato netamente a consumirlo. Este modulo lo detecta a partir de
una serie de concentracion de lactato, sin suponer nada sobre el organismo, la
escala ni las unidades.

Uso minimo:

    >>> from lactateshift import detect_shift
    >>> resultado = detect_shift(days=[1,2,3,4,5,6,7,8], values=[1,2,4,7,9,6,3,2])
    >>> resultado.occurred, resultado.day
    (True, 5.0)

Diseno de la regla
------------------
Hay shift en el primer dia ``t`` tal que:

1. la derivada de la serie suavizada es negativa durante ``n_consecutive``
   dias a partir de ``t``, y
2. el valor cae desde el maximo local en ``t`` al menos ``drop_threshold``
   veces ese maximo, medido **antes** de que la serie vuelva a superarlo.

La condicion 2, con esa restriccion, es lo que separa un cambio de regimen de
un bache transitorio: si la serie se recupera enseguida, la caida nunca
alcanza el umbral.

Por que el *primer* maximo local y no el maximo global: una parte de los
cultivos hace el shift, consume lactato varios dias y despues vuelve a
producirlo al final, superando el maximo inicial. Anclar en el maximo global
marca esos cultivos como "sin shift", lo cual es falso: el shift ocurrio, solo
fue reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["ShiftResult", "detect_shift", "detect_shift_batch", "regularize"]


@dataclass(frozen=True)
class ShiftResult:
    """Resultado de la deteccion sobre una serie.

    Attributes
    ----------
    occurred:
        True si se detecto el shift.
    day:
        Dia del shift (el maximo local a partir del cual empieza el descenso),
        o None si no ocurrio.
    drop_fraction:
        Caida relativa alcanzada respecto al valor del maximo, o None.
    peak_value:
        Valor suavizado en el dia del shift, o None.
    censoring_day:
        Ultimo dia observado. Es el tiempo de censura para analisis de
        supervivencia cuando ``occurred`` es False.
    reason:
        Texto corto explicando por que no se detecto, util para depurar.
    """

    occurred: bool
    day: float | None
    drop_fraction: float | None
    peak_value: float | None
    censoring_day: float
    reason: str = ""

    @property
    def time(self) -> float:
        """Tiempo para analisis de supervivencia: dia del evento o de censura."""
        return self.day if self.occurred else self.censoring_day

    def as_dict(self) -> dict:
        d = asdict(self)
        d["time"] = self.time
        return d


def regularize(
    days: Sequence[float],
    values: Sequence[float],
    *,
    interpolate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Lleva la serie a una rejilla diaria completa 1..max(day).

    Series reales traen dias enteros ausentes. Sin regularizar, una derivada
    calculada por diferencias sucesivas trataria un salto de tres dias como si
    fuera de uno. ``interpolate=True`` rellena los huecos linealmente; es
    apropiado para construir la etiqueta (que se lee despues del experimento),
    no para construir variables predictoras.
    """
    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)
    if days.size == 0:
        raise ValueError("la serie esta vacia")
    if days.size != values.size:
        raise ValueError("days y values deben tener la misma longitud")

    grid = np.arange(1, int(np.nanmax(days)) + 1)
    s = pd.Series(values, index=days.astype(int)).reindex(grid)
    if interpolate:
        s = s.interpolate(method="linear", limit_direction="both")
    return grid.astype(float), s.to_numpy(dtype=float)


def smooth(values: Sequence[float], window: int = 3) -> np.ndarray:
    """Media movil centrada. ``window<=1`` devuelve la serie sin cambios.

    Con muestreo diario y series de 9 a 18 puntos, una ventana de 3 quita el
    ruido analitico de un solo dia sin borrar un cambio de tendencia, que dura
    varios dias.
    """
    v = np.asarray(values, dtype=float)
    if window <= 1:
        return v
    return (
        pd.Series(v)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def detect_shift(
    days: Sequence[float],
    values: Sequence[float],
    *,
    smooth_window: int = 3,
    n_consecutive: int = 2,
    drop_threshold: float = 0.30,
    min_day: float | None = None,
    regularize_grid: bool = True,
    refine_peak: bool = True,
) -> ShiftResult:
    """Detecta el lactate shift en una serie.

    Parameters
    ----------
    days, values:
        Dias de muestreo y concentracion de lactato. Pueden traer NaN y dias
        faltantes.
    smooth_window:
        Ventana de la media movil centrada, en dias. 1 desactiva el suavizado.
        Es el parametro al que la deteccion es mas sensible: conviene
        reportarlo y hacerle analisis de sensibilidad.
    n_consecutive:
        Dias consecutivos con derivada negativa exigidos tras el maximo.
    drop_threshold:
        Caida minima, como fraccion del valor del maximo local.
    min_day:
        Si se da, se ignoran los shifts anteriores a ese dia. Sirve cuando la
        serie se usa para prediccion y hay una ventana de observacion previa:
        un shift dentro de esa ventana no es un evento anticipable.
    regularize_grid:
        Si True, la serie se lleva a rejilla diaria e interpola huecos antes de
        derivar.
    refine_peak:
        Corrige el sesgo del suavizado. Una media movil centrada desplaza el
        maximo hacia atras cuando la caida es mas rapida que la subida, que es
        el caso tipico de un cultivo: sin esta correccion el dia detectado sale
        sistematicamente un dia antes del real. Con ``refine_peak=True`` el dia
        se reajusta al maximo de la serie sin suavizar dentro de
        ``+-(smooth_window // 2)`` dias. Detectado con los cultivos sinteticos
        de :mod:`lactateshift.datasets`, donde el dia real se conoce.

    Returns
    -------
    ShiftResult
    """
    if not 0 < drop_threshold < 1:
        raise ValueError("drop_threshold debe estar entre 0 y 1")
    if n_consecutive < 1:
        raise ValueError("n_consecutive debe ser >= 1")

    if regularize_grid:
        d, v = regularize(days, values)
    else:
        d = np.asarray(days, dtype=float)
        v = np.asarray(values, dtype=float)

    last_day = float(d[-1])
    if np.all(np.isnan(v)):
        return ShiftResult(False, None, None, None, last_day, "serie sin datos")

    s = smooth(v, smooth_window)
    if len(s) < n_consecutive + 1:
        return ShiftResult(False, None, None, None, last_day, "serie demasiado corta")

    deriv = np.diff(s)
    for i in range(len(s) - n_consecutive):
        if not np.isfinite(s[i]) or s[i] <= 0:
            continue
        if not np.all(deriv[i : i + n_consecutive] < 0):
            continue
        # minimo alcanzado antes de que la serie vuelva a superar este maximo
        low = s[i]
        for j in range(i + 1, len(s)):
            if s[j] > s[i]:
                break
            low = min(low, s[j])
        drop = (s[i] - low) / s[i]
        if drop < drop_threshold:
            continue
        idx = i
        if refine_peak and smooth_window > 1:
            w = smooth_window // 2
            lo, hi = max(0, i - w), min(len(v), i + w + 1)
            vecindad = v[lo:hi]
            if np.any(np.isfinite(vecindad)):
                idx = lo + int(np.nanargmax(vecindad))

        if min_day is not None and d[idx] < min_day:
            return ShiftResult(
                False, None, float(drop), float(s[i]), last_day,
                f"shift en dia {d[idx]:.0f}, anterior a min_day={min_day:.0f}",
            )
        return ShiftResult(True, float(d[idx]), float(drop), float(s[i]), last_day, "shift")

    return ShiftResult(False, None, None, None, last_day, "sin descenso sostenido suficiente")


def detect_shift_batch(
    data: pd.DataFrame,
    *,
    id_col: str,
    day_col: str,
    value_col: str,
    **kwargs,
) -> pd.DataFrame:
    """Aplica :func:`detect_shift` a cada serie de un DataFrame largo.

    Devuelve una fila por serie, con las columnas de :class:`ShiftResult` mas
    ``time`` (dia del evento o de censura), lista para pasar a un modelo de
    supervivencia.
    """
    filas = []
    for key, g in data.groupby(id_col, sort=True):
        g = g.sort_values(day_col)
        r = detect_shift(g[day_col].to_numpy(), g[value_col].to_numpy(), **kwargs)
        filas.append({id_col: key, **r.as_dict()})
    return pd.DataFrame(filas).set_index(id_col)
