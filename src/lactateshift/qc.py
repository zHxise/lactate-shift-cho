"""Control de calidad de una serie de lactato.

Un solo punto muy bajo cerca del final de la serie (por ejemplo 3.2, 0.7,
4.3) puede crear un shift falso, porque la media movil no lo compensa en el
borde. Esta funcion solo senala esos puntos para revisarlos; el detector no
se modifica.

Desarrollado por Arturo Rodriguez.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["isolated_low_points"]


def isolated_low_points(
    days: Sequence[float],
    values: Sequence[float],
    *,
    factor: float = 0.5,
    min_rel: float = 0.2,
) -> list[float]:
    """Dias con una medicion sospechosamente baja.

    Un punto se senala si cumple las tres condiciones:

    1. esta por debajo de sus dos vecinos medidos;
    2. vale menos de ``factor`` veces el promedio de esos vecinos;
    3. el promedio de los vecinos supera ``min_rel`` veces el maximo de la
       serie (para ignorar el ruido cerca de cero).

    Los NaN se ignoran: los vecinos son las mediciones anterior y siguiente.
    """
    d = np.asarray(days, dtype=float)
    v = np.asarray(values, dtype=float)
    orden = np.argsort(d)
    d, v = d[orden], v[orden]
    medido = np.isfinite(v)
    d, v = d[medido], v[medido]
    if len(v) < 3:
        return []
    vmax = float(np.max(v))
    fuera = []
    for i in range(1, len(v) - 1):
        a, b, x = v[i - 1], v[i + 1], v[i]
        promedio = (a + b) / 2
        if x < min(a, b) and x < factor * promedio and promedio > min_rel * vmax:
            fuera.append(float(d[i]))
    return fuera
