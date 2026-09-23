"""Datos sinteticos de cultivos con shift conocido.

El dataset real usado en el caso de estudio (material suplementario de
Gangadharan et al., 2021) esta bajo copyright de Elsevier y no puede
redistribuirse aqui. Este modulo genera cultivos sinteticos con el dia del
shift conocido de antemano, para que cualquiera pueda probar el paquete,
correr los tests y ver un ejemplo completo sin conseguir ese archivo.

Los datos sinteticos NO reproducen la biologia de un cultivo real. Sirven
para verificar que el detector hace lo que dice, no para sacar conclusiones.

Tampoco reproducen su forma: aqui la subida se aplana antes del maximo y la
caida es brusca, mientras que en los 106 cultivos del caso de estudio la caida
suele ser mas lenta que la subida. Se comprobo al ver que la correccion del
suavizado movia el dia en sentidos opuestos en unos y otros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_culture", "make_synthetic_cultures"]


def make_culture(
    shift_day: float | None,
    duration: int = 15,
    *,
    peak: float = 1.0,
    rise_rate: float = 0.25,
    fall_rate: float = 0.30,
    rebound_day: float | None = None,
    noise: float = 0.02,
    missing_frac: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Genera un cultivo sintetico.

    Parameters
    ----------
    shift_day:
        Dia en que la serie cambia de subir a bajar. ``None`` produce una serie
        monotona creciente (un cultivo que nunca hace el shift).
    rebound_day:
        Si se da, la serie vuelve a subir a partir de ese dia. Reproduce el
        rebote tardio observado en cultivos reales, que es el caso que rompe
        cualquier regla anclada en el maximo global.
    missing_frac:
        Fraccion de dias que se eliminan, para simular muestreo incompleto.
    """
    rng = rng or np.random.default_rng(0)
    days = np.arange(1, duration + 1, dtype=float)
    v = np.empty_like(days)

    for k, t in enumerate(days):
        if shift_day is None:
            v[k] = peak * (1 - np.exp(-rise_rate * t))
        elif t <= shift_day:
            v[k] = peak * (1 - np.exp(-rise_rate * t))
        else:
            base = peak * (1 - np.exp(-rise_rate * shift_day))
            v[k] = base * np.exp(-fall_rate * (t - shift_day))
        if rebound_day is not None and t > rebound_day:
            v[k] += peak * 1.3 * (1 - np.exp(-0.4 * (t - rebound_day)))

    v = v + rng.normal(0, noise * peak, size=v.shape)
    v = np.clip(v, 0, None)

    df = pd.DataFrame({"day": days, "lactate": v})
    if missing_frac > 0:
        keep = rng.random(len(df)) >= missing_frac
        keep[0] = keep[-1] = True  # conservar extremos
        df = df[keep]
    return df.reset_index(drop=True)


def make_synthetic_cultures(
    n: int = 40,
    *,
    seed: int = 0,
    rebound_frac: float = 0.25,
    never_shift_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera ``n`` cultivos sinteticos.

    Returns
    -------
    (series, verdad)
        ``series`` en formato largo con columnas ``culture``, ``day``,
        ``lactate``. ``verdad`` con el dia del shift real por cultivo
        (``NaN`` si no lo hay), para comparar contra lo que detecte el modelo.
    """
    rng = np.random.default_rng(seed)
    partes, verdad = [], []
    for i in range(n):
        cid = f"S{i+1:03d}"
        duration = int(rng.integers(12, 19))
        if rng.random() < never_shift_frac:
            sd = None
        else:
            sd = float(rng.integers(5, min(11, duration - 3)))
        reb = None
        if sd is not None and rng.random() < rebound_frac:
            reb = float(min(sd + rng.integers(4, 7), duration - 2))
        df = make_culture(
            sd, duration, rebound_day=reb,
            peak=float(rng.uniform(0.5, 1.5)),
            rise_rate=float(rng.uniform(0.18, 0.35)),
            fall_rate=float(rng.uniform(0.20, 0.45)),
            noise=float(rng.uniform(0.01, 0.04)),
            missing_frac=float(rng.uniform(0, 0.15)),
            rng=rng,
        )
        df.insert(0, "culture", cid)
        partes.append(df)
        verdad.append({"culture": cid, "shift_day_real": np.nan if sd is None else sd,
                       "rebound_day": reb, "duration": duration})
    return pd.concat(partes, ignore_index=True), pd.DataFrame(verdad).set_index("culture")
