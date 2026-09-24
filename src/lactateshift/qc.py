"""Control de calidad de una serie antes de interpretar la deteccion.

Por que existe: en una prueba con datos inventados, un cultivo en meseta con
UNA medicion muy baja cerca del final (3.2, 0.7, 4.3) se marco como shift.
La media movil de 3 dias no basta para anular un error de medicion de un dia
cuando ese dia esta cerca del borde de la serie: en el ultimo dia el
promedio solo tiene dos puntos y uno es el erroneo.

No se cambia el detector (su version esta congelada por la validacion
externa). Esta funcion solo SENALA los puntos que conviene revisar; decidir
si son errores le toca a quien conoce el experimento.
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

    1. esta por debajo de sus DOS vecinos medidos (un minimo aislado, no un
       tramo de bajada);
    2. vale menos de ``factor`` veces el promedio de esos vecinos;
    3. los vecinos no estan cerca de cero: su promedio supera ``min_rel``
       veces el maximo de la serie. Despues del consumo el lactato queda
       cerca de cero y ahi las diferencias relativas son solo ruido.

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
