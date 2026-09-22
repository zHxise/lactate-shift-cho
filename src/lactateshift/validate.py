"""Controles contra la fuga de informacion y el sobreajuste.

Con muestras pequenas, un resultado que parece bueno casi siempre lo parece
por una razon equivocada. Estas funciones existen para que refutarlo sea tan
facil como producirlo.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

__all__ = ["permutation_test", "leave_one_group_out"]


def permutation_test(
    score_fn: Callable[[pd.Series], float],
    y: pd.Series,
    *,
    n_permutations: int = 200,
    seed: int = 0,
    lower_is_better: bool = True,
) -> dict:
    """Compara un puntaje real contra su distribucion con la etiqueta barajada.

    Es la pregunta mas basica que hay que hacerle a un modelo: si la etiqueta
    no tuviera ninguna relacion con las variables, .cuantas veces obtendria
    por azar un resultado como este? ``score_fn`` debe recibir una etiqueta y
    devolver el puntaje ya validado de forma cruzada.

    Returns
    -------
    dict con ``observed``, ``null_mean``, ``null_sd`` y ``p_value`` empirico.
    """
    rng = np.random.default_rng(seed)
    observed = float(score_fn(y))
    nulls = np.array([
        float(score_fn(pd.Series(rng.permutation(y.to_numpy()), index=y.index)))
        for _ in range(n_permutations)
    ])
    p = (nulls <= observed).mean() if lower_is_better else (nulls >= observed).mean()
    return {"observed": observed, "null_mean": float(nulls.mean()),
            "null_sd": float(nulls.std()), "p_value": float(p),
            "n_permutations": n_permutations}


def leave_one_group_out(
    fit_predict: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metric: Callable[[Sequence[float], Sequence[float]], float],
    *,
    min_group_size: int = 1,
) -> pd.DataFrame:
    """Entrena excluyendo un grupo completo y predice sobre el.

    Es la prueba dura cuando los datos vienen de contextos distintos (escalas
    de reactor, sitios, lineas celulares): mide si el modelo transfiere a un
    contexto que nunca vio, no solo si interpola dentro de los que conoce.
    Un modelo puede tener buen desempeno con validacion cruzada aleatoria y
    fallar por completo aqui; cuando eso pasa, lo que hay que reportar es
    esto.
    """
    tam = groups.value_counts()
    usables = tam[tam >= min_group_size].index
    filas = []
    for g in usables:
        te = groups == g
        tr = ~te
        if tr.sum() == 0:
            continue
        pred = fit_predict(X[tr], y[tr], X[te])
        filas.append({"group": g, "n_test": int(te.sum()),
                      "n_train": int(tr.sum()),
                      "score": float(metric(y[te], pred))})
    d = pd.DataFrame(filas)
    if not d.empty:
        d.attrs["weighted_mean"] = float(np.average(d["score"], weights=d["n_test"]))
    return d
