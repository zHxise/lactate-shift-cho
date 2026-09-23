"""Tests de los controles de validacion."""

import numpy as np
import pandas as pd

from lactateshift.validate import leave_one_group_out, permutation_test


def test_p_valor_nunca_es_cero():
    """Con n barajadas, el menor p-valor posible es 1/(n+1), no 0."""
    y = pd.Series(np.arange(20.0))
    res = permutation_test(lambda yy: 0.0 if yy.equals(y) else 1.0, y,
                           n_permutations=49, seed=0)
    assert np.isclose(res["p_value"], 1 / 50)
    assert np.isclose(res["p_min"], 1 / 50)


def test_p_valor_sin_senal_es_alto():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=30))
    res = permutation_test(lambda yy: float(rng.normal()), y, n_permutations=99)
    assert res["p_value"] > 0.05


def test_leave_one_group_out_excluye_el_grupo_completo():
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = pd.Series(np.arange(10.0))
    g = pd.Series(["a"] * 5 + ["b"] * 5)
    vistos = []

    def fp(Xtr, ytr, Xte):
        vistos.append(set(Xtr.index) & set(Xte.index))
        return np.zeros(len(Xte))

    d = leave_one_group_out(fp, X, y, g, lambda a, b: 0.0)
    assert len(d) == 2 and all(len(v) == 0 for v in vistos)
