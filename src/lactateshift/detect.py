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
2. el valor cae desde ``t`` al menos ``drop_threshold`` veces el valor en
   ``t``, medido **antes** de que la serie vuelva a superarlo.

Que detecta y que no
--------------------
El punto ``t`` que elige la regla es siempre un maximo local de la serie
suavizada (en una meseta, su ultimo punto). No hace falta imponerlo: se sigue
de la regla. Si ``t`` estuviera en plena bajada (valor anterior mayor), el
punto anterior tambien tendria derivada negativa los dias siguientes y una
caida relativa todavia mayor, asi que habria cumplido primero. Esta propiedad
tiene un test que la verifica sobre miles de series aleatorias.

(Una auditoria externa afirmo que el codigo no garantizaba el maximo local, y
una verificacion mal hecha lo "confirmo": comparaba el dia ya refinado por
``refine_peak``, no el punto que detecta la regla. Se corrigio.)

Una caida profunda seguida de recuperacion **si** cuenta como shift. Es
deliberado: en cultivos reales el cambio a consumo de lactato es a menudo
reversible, y el lactato vuelve a subir en fase tardia. Lo que la regla
excluye son las caidas que no alcanzan ``drop_threshold`` antes de recuperar
el nivel previo, no las que se revierten mas tarde.

Por eso ``min_peak`` existe: sin un umbral de amplitud absoluta, una serie que
oscile cerca de cero puede producir una caida relativa del 30% que es solo
ruido analitico. Por omision no se aplica, porque el valor depende de las
unidades de cada dataset.

Por que el *primer* punto que cumple y no el maximo global: una parte de los
cultivos hace el shift, consume lactato varios dias y despues vuelve a
producirlo al final, superando el maximo inicial. Anclar en el maximo global
marca esos cultivos como "sin shift", lo cual es falso: el shift ocurrio, solo
fue reversible.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["ShiftResult", "detect_shift", "detect_shift_batch", "regularize", "smooth"]


@dataclass(frozen=True)
class ShiftResult:
    """Resultado de la deteccion sobre una serie.

    Attributes
    ----------
    occurred:
        True si se detecto el shift.
    day:
        Dia del shift, o None si no ocurrio. Con ``refine_peak`` puede
        diferir en un dia del punto donde la regla se cumplio sobre la serie
        suavizada (ver ``detect_shift``).
    drop_fraction:
        Caida relativa alcanzada, medida sobre la serie suavizada desde el
        punto de deteccion, o None.
    peak_value:
        Valor de la serie suavizada en el punto de deteccion, o None.
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
    # Sin estas comprobaciones, un dia 2.5 se truncaba a 2 en silencio, un dia
    # repetido hacia fallar reindex con un error opaco, y un dia 0 desaparecia
    # de la rejilla sin aviso.
    if np.any(~np.isfinite(days)):
        raise ValueError("days contiene NaN o infinitos")
    if np.any(days != np.round(days)):
        raise ValueError("days debe contener dias enteros (muestreo diario)")
    if len(np.unique(days)) != len(days):
        raise ValueError("days contiene dias repetidos")
    if np.min(days) < 1:
        raise ValueError("los dias empiezan en 1")

    grid = np.arange(1, int(np.max(days)) + 1)
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
    min_peak: float | None = None,
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
        Si el primer shift ocurre antes de este dia, devuelve
        ``occurred=False`` con la razon en ``reason``. No busca un shift
        posterior: el primer cambio ya ocurrio. Cuidado al usarlo en analisis
        de supervivencia: ese resultado NO es una censura, y tratarlo como tal
        seria un error; esos casos se excluyen, no se censuran.
    regularize_grid:
        Si True, la serie se lleva a rejilla diaria e interpola huecos antes de
        derivar.
    refine_peak:
        Corrige el desplazamiento que introduce el suavizado. Una media movil
        centrada corre el maximo hacia el lado donde la curva es mas suave: si
        la caida es mas brusca que la subida lo corre hacia atras, y si la
        caida es mas lenta lo corre hacia adelante. Con ``refine_peak=True``
        el dia se reajusta al maximo de la serie sin suavizar dentro de
        ``+-(smooth_window // 2)`` dias, lo que corrige en ambos sentidos.

        En los cultivos sinteticos de :mod:`lactateshift.datasets` (subida que
        se aplana, caida brusca) el desplazamiento es hacia atras. En los 101
        eventos del caso de estudio el refinamiento movio el dia en 40: 31
        hacia atras y 9 hacia adelante, es decir, ahi la caida suele ser mas
        lenta que la subida. Los sinteticos no reproducen bien esa forma: son
        utiles para verificar el mecanismo, no para describir cultivos reales.
    min_peak:
        Amplitud minima del maximo, en las unidades de la serie. Sin este
        filtro, una serie que oscile cerca de cero puede dar una caida
        relativa del 30% que es solo ruido analitico. Por omision None,
        porque el valor apropiado depende de las unidades de cada dataset.

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
        if min_peak is not None and s[i] < min_peak:
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
        try:
            r = detect_shift(g[day_col].to_numpy(), g[value_col].to_numpy(), **kwargs)
        except ValueError as e:
            # con muchas series, un error sin el identificador obliga a buscar
            # a mano cual de ellas tiene el problema
            raise ValueError(f"serie {key!r}: {e}") from e
        filas.append({id_col: key, **r.as_dict()})
    return pd.DataFrame(filas).set_index(id_col)
