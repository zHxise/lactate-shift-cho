"""Controles contra la fuga de informacion y el sobreajuste.

Con muestras pequenas, un resultado que parece bueno casi siempre lo parece
por una razon equivocada. Estas funciones existen para que refutarlo sea tan
facil como producirlo.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

__all__ = ["permutation_test", "leave_one_group_out", "out_of_fold_shap"]


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

    El p-valor usa la correccion estandar (k + 1) / (n + 1), donde k es el
    numero de barajadas al menos tan buenas como el resultado real. Sin el +1
    el p-valor puede salir exactamente 0, lo que no es un p-valor valido: con
    n barajadas, lo minimo que se puede afirmar es 1 / (n + 1).

    Returns
    -------
    dict con ``observed``, ``null_mean``, ``null_sd``, ``p_value`` y
    ``p_min`` (el menor p-valor alcanzable con ese numero de barajadas).
    """
    rng = np.random.default_rng(seed)
    observed = float(score_fn(y))
    nulls = np.array([
        float(score_fn(pd.Series(rng.permutation(y.to_numpy()), index=y.index)))
        for _ in range(n_permutations)
    ])
    k = int((nulls <= observed).sum() if lower_is_better else (nulls >= observed).sum())
    return {"observed": observed, "null_mean": float(nulls.mean()),
            "null_sd": float(nulls.std()),
            "p_value": (k + 1) / (n_permutations + 1),
            "p_min": 1 / (n_permutations + 1),
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


def out_of_fold_shap(
    make_model: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 0,
    impute: str = "median",
):
    """Valores SHAP calculados siempre fuera del fold de entrenamiento.

    Por que no basta entrenar con todo y explicar eso: SHAP explica al modelo,
    no al fenomeno. Un modelo sobreajustado produce explicaciones nitidas de su
    propio sobreajuste. Calculando los valores solo sobre observaciones que el
    modelo no vio, lo que queda es la parte de la explicacion que sobrevive a
    datos nuevos.

    ``make_model`` debe devolver un modelo de arboles nuevo en cada llamada
    (``TreeExplainer`` solo admite modelos basados en arboles). La imputacion
    se ajusta dentro de cada fold.

    Returns
    -------
    (shap_df, importancia, folds)
        ``shap_df`` con una fila por observacion y los valores SHAP promediados
        sobre las repeticiones, para leer la DIRECCION del efecto.
        ``importancia`` es la media de ``|SHAP|`` calculada dentro de cada fold
        y promediada despues: es la forma correcta de medir magnitud.
        ``folds`` es la lista de Series por fold, para ver si el orden de
        importancia es estable o cambia con la particion.

    Notes
    -----
    Importa el orden de las operaciones. Tomar el valor absoluto DESPUES de
    promediar los valores con signo entre folds cancela las contribuciones de
    las variables cuyo efecto cambia de signo segun la particion, y deja
    intactas las de efecto monotono consistente. El resultado es un ranking que
    exagera a unas variables y borra a otras. El valor absoluto va primero,
    dentro de cada fold, y el promedio despues.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import RepeatedKFold
    import shap as _shap

    acum = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    veces = pd.Series(0, index=X.index)
    por_fold = []

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for tr, te in cv.split(X):
        imp = SimpleImputer(strategy=impute).fit(X.iloc[tr])
        Xtr = pd.DataFrame(imp.transform(X.iloc[tr]), columns=X.columns, index=X.index[tr])
        Xte = pd.DataFrame(imp.transform(X.iloc[te]), columns=X.columns, index=X.index[te])
        m = make_model()
        m.fit(Xtr, y.iloc[tr])
        vals = _shap.TreeExplainer(m).shap_values(Xte, check_additivity=False)
        vals = np.asarray(vals)
        if vals.ndim == 3:  # algunos modelos devuelven (n, p, salidas)
            vals = vals[..., 0]
        acum.loc[Xte.index] += vals
        veces.loc[Xte.index] += 1
        por_fold.append(pd.Series(np.abs(vals).mean(axis=0), index=X.columns))

    shap_df = acum.div(veces.replace(0, np.nan), axis=0)
    importancia = pd.concat(por_fold, axis=1).mean(axis=1).sort_values(ascending=False)
    return shap_df, importancia, por_fold
