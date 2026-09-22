"""Tests de la construccion de variables. El punto central: que nada de lo
que se calcula dependa de informacion posterior a la ventana."""

import numpy as np
import pandas as pd
import pytest

from lactateshift import early_window_features, slope


def _serie(valores_por_dia, cultivo="A", extra=None):
    df = pd.DataFrame({"culture": cultivo,
                       "day": list(range(1, len(valores_por_dia) + 1)),
                       "lactate": valores_por_dia})
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


class TestSlope:
    def test_pendiente_de_una_recta_exacta(self):
        assert np.isclose(slope([1, 2, 3, 4], [2, 4, 6, 8]), 2.0)

    def test_menos_de_dos_puntos_es_nan(self):
        assert np.isnan(slope([1], [5]))
        assert np.isnan(slope([1, 2], [np.nan, 5]))

    def test_ignora_nan(self):
        assert np.isclose(slope([1, 2, 3, 4], [2, np.nan, 6, 8]), 2.0)


class TestSinFuga:
    def test_los_dias_posteriores_a_la_ventana_no_afectan_nada(self):
        base = [1.0, 2.0, 3.0, 4.0]
        corta = _serie(base)
        larga = _serie(base + [99.0, -99.0, 50.0])
        fc = early_window_features(corta, id_col="culture", day_col="day",
                                   value_cols=["lactate"], window=4)
        fl = early_window_features(larga, id_col="culture", day_col="day",
                                   value_cols=["lactate"], window=4)
        pd.testing.assert_frame_equal(fc, fl)

    def test_hueco_hereda_del_dia_anterior_no_del_posterior(self):
        df = pd.DataFrame({"culture": "A", "day": [1, 2, 4],
                           "lactate": [1.0, 2.0, 10.0]})
        f = early_window_features(df, id_col="culture", day_col="day",
                                  value_cols=["lactate"], window=4)
        # el dia 3 falta: debe heredar el 2.0 del dia 2, nunca el 10.0 del dia 4
        assert np.isclose(f.loc["A", "lactate_mean"], (1 + 2 + 2 + 10) / 4)

    def test_no_se_imputa_lo_que_falta_por_completo(self):
        df = pd.DataFrame({"culture": "A", "day": [1, 2, 3, 4],
                           "lactate": [1.0, 2.0, 3.0, 4.0],
                           "glucose": [np.nan] * 4})
        f = early_window_features(df, id_col="culture", day_col="day",
                                  value_cols=["lactate", "glucose"], window=4)
        assert np.isnan(f.loc["A", "glucose_last"])
        assert f.loc["A", "glucose_n"] == 0


class TestVariables:
    def test_cuenta_de_dias_medidos(self):
        df = pd.DataFrame({"culture": "A", "day": [1, 3], "lactate": [1.0, 3.0]})
        f = early_window_features(df, id_col="culture", day_col="day",
                                  value_cols=["lactate"], window=4)
        assert f.loc["A", "lactate_n"] == 4  # tras ffill hay 4 valores
        df2 = pd.DataFrame({"culture": "A", "day": [3, 4], "lactate": [1.0, 3.0]})
        f2 = early_window_features(df2, id_col="culture", day_col="day",
                                   value_cols=["lactate"], window=4)
        assert f2.loc["A", "lactate_n"] == 2  # los dias 1 y 2 no tienen de donde heredar

    def test_cociente(self):
        df = pd.DataFrame({"culture": "A", "day": [1, 2, 3, 4],
                           "lactate": [1.0, 2, 3, 8.0], "vcd": [1.0, 1, 1, 2.0]})
        f = early_window_features(df, id_col="culture", day_col="day",
                                  value_cols=["lactate", "vcd"], window=4,
                                  ratios=[("lactate", "vcd")])
        assert np.isclose(f.loc["A", "lactate_over_vcd"], 4.0)

    def test_division_entre_cero_es_nan(self):
        df = pd.DataFrame({"culture": "A", "day": [1, 2, 3, 4],
                           "lactate": [1.0, 2, 3, 8.0], "vcd": [1.0, 1, 1, 0.0]})
        f = early_window_features(df, id_col="culture", day_col="day",
                                  value_cols=["lactate", "vcd"], window=4,
                                  ratios=[("lactate", "vcd")])
        assert np.isnan(f.loc["A", "lactate_over_vcd"])

    def test_columnas_estaticas(self):
        df = _serie([1.0, 2, 3, 4], extra={"scale": 0.5})
        f = early_window_features(df, id_col="culture", day_col="day",
                                  value_cols=["lactate"], window=4,
                                  static_cols=["scale"])
        assert f.loc["A", "scale"] == 0.5

    def test_una_fila_por_serie(self):
        df = pd.concat([_serie([1.0, 2, 3, 4], "A"), _serie([2.0, 3, 4, 5], "B")])
        f = early_window_features(df, id_col="culture", day_col="day",
                                  value_cols=["lactate"], window=4)
        assert list(f.index) == ["A", "B"]
