"""Tests del detector con series de respuesta conocida (no usan el dataset).

Desarrollado por Arturo Rodriguez.
"""

import numpy as np
import pandas as pd
import pytest

from lactateshift import detect_shift, detect_shift_batch, make_culture, make_synthetic_cultures
from lactateshift.detect import regularize, smooth


class TestCasosBasicos:
    def test_pico_limpio(self):
        r = detect_shift([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4, 7, 9, 6, 3, 2])
        assert r.occurred
        assert r.day == 5.0
        assert r.drop_fraction > 0.5

    def test_monotona_creciente_no_hay_shift(self):
        r = detect_shift(range(1, 11), np.linspace(0.1, 1.0, 10))
        assert not r.occurred
        assert r.day is None
        assert r.time == 10.0  # censurado al ultimo dia

    def test_serie_plana(self):
        r = detect_shift(range(1, 11), [0.5] * 10)
        assert not r.occurred

    def test_bache_superficial_no_cuenta(self):
        # baja dos dias sin llegar al umbral y se recupera: no es shift
        v = [1, 2, 3, 4, 3.8, 3.6, 5, 6, 7, 8]
        r = detect_shift(range(1, 11), v, drop_threshold=0.30)
        assert not r.occurred

    def test_caida_profunda_con_rebote_SI_cuenta(self):
        """Una caida que alcanza el umbral y luego se revierte cuenta como
        shift (la regla no distingue cambio permanente de transitorio)."""
        v = [1, 2, 3, 4, 5, 6, 7, 10, 7, 5, 4, 3, 12, 13]
        assert detect_shift(range(1, 15), v).occurred

    def test_caida_por_debajo_del_umbral(self):
        v = [1, 2, 3, 4, 3.9, 3.8, 3.7, 3.6]
        assert not detect_shift(range(1, 9), v, drop_threshold=0.50).occurred
        assert detect_shift(range(1, 9), v, drop_threshold=0.05).occurred


class TestRebote:
    """Shift seguido de rebote tardio por encima del pico inicial."""

    def test_shift_con_rebote_tardio_se_detecta(self):
        df = make_culture(shift_day=6, duration=16, rebound_day=11, noise=0.0)
        r = detect_shift(df["day"], df["lactate"])
        assert r.occurred, "un shift seguido de rebote sigue siendo un shift"
        assert abs(r.day - 6) <= 1

    def test_el_maximo_global_queda_despues_del_shift(self):
        df = make_culture(shift_day=6, duration=16, rebound_day=11, noise=0.0)
        dia_max_global = df.loc[df["lactate"].idxmax(), "day"]
        r = detect_shift(df["day"], df["lactate"])
        assert dia_max_global > r.day, "el rebote supera el pico inicial"


class TestDatosImperfectos:
    def test_muestreo_completo_da_el_dia_exacto(self):
        df = make_culture(shift_day=7, duration=15, missing_frac=0.0, noise=0.01)
        r = detect_shift(df["day"], df["lactate"])
        assert r.occurred and abs(r.day - 7) <= 1

    def test_el_muestreo_esparso_degrada_la_precision_del_dia(self):
        """Con 30% de dias faltantes el error en el dia puede llegar a 2."""
        errores = {}
        for frac in (0.0, 0.3):
            errs = []
            for seed in range(30):
                rng = np.random.default_rng(seed)
                df = make_culture(shift_day=7, duration=15, missing_frac=frac,
                                  noise=0.01, rng=rng)
                r = detect_shift(df["day"], df["lactate"])
                if r.occurred:
                    errs.append(abs(r.day - 7))
            errores[frac] = np.mean(errs)
        assert errores[0.0] <= 1.0
        assert errores[0.3] <= 2.0
        assert errores[0.3] >= errores[0.0]

    def test_nan_internos(self):
        v = np.array([1, 2, np.nan, 7, 9, 6, 3, 2], dtype=float)
        r = detect_shift([1, 2, 3, 4, 5, 6, 7, 8], v)
        assert r.occurred

    def test_serie_demasiado_corta(self):
        r = detect_shift([1, 2], [1, 2])
        assert not r.occurred
        assert "corta" in r.reason

    def test_serie_vacia_falla(self):
        with pytest.raises(ValueError):
            detect_shift([], [])

    def test_longitudes_distintas_fallan(self):
        with pytest.raises(ValueError):
            detect_shift([1, 2, 3], [1, 2])


class TestFiltrosOpcionales:

    def test_min_peak_descarta_oscilaciones_cerca_de_cero(self):
        # caida relativa del 87% sobre una serie de amplitud despreciable
        v = [0.001, 0.002, 0.003, 0.002, 0.001, 0.0005, 0.0004, 0.0004, 0.0004]
        assert detect_shift(range(1, 10), v).occurred
        assert not detect_shift(range(1, 10), v, min_peak=0.01).occurred


class TestPropiedadMaximoLocal:
    """El punto detectado es un maximo local de la serie suavizada."""

    def test_en_series_aleatorias(self):
        rng = np.random.default_rng(0)
        eventos = 0
        for _ in range(300):
            n = int(rng.integers(5, 18))
            v = np.clip(np.cumsum(rng.normal(0, 1, n)) + 10, 0.01, None)
            for w in (1, 3, 5):
                for k in (1, 2, 3):
                    r = detect_shift(range(1, n + 1), v, smooth_window=w,
                                     n_consecutive=k, refine_peak=False)
                    if r.occurred:
                        eventos += 1
                        s = smooth(v, w)
                        i = int(r.day) - 1
                        assert i == 0 or s[i] >= s[i - 1]
        assert eventos > 100, "la prueba debe cubrir suficientes eventos"

    def test_meseta_se_detecta_en_su_ultimo_punto(self):
        v = [1.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.5, 0.4]
        r = detect_shift(range(1, 9), v, smooth_window=1)
        assert r.occurred and r.day == 4.0


class TestValidacionDeEntrada:
    def test_dias_no_enteros(self):
        with pytest.raises(ValueError, match="enteros"):
            detect_shift([1, 2.5, 3], [1, 2, 3])

    def test_dias_repetidos(self):
        with pytest.raises(ValueError, match="repetidos"):
            detect_shift([1, 2, 2, 3], [1, 2, 3, 4])

    def test_dia_cero(self):
        with pytest.raises(ValueError, match="empiezan en 1"):
            detect_shift([0, 1, 2], [1, 2, 3])

    def test_batch_dice_que_serie_tiene_el_problema(self):
        df = pd.DataFrame({"culture": ["A", "A", "B", "B", "B"],
                           "day": [1, 2, 1, 2, 2], "lactate": [1, 2, 1, 2, 3]})
        with pytest.raises(ValueError, match="serie 'B'"):
            detect_shift_batch(df, id_col="culture", day_col="day", value_col="lactate")

    def test_dias_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            detect_shift([1, np.nan, 3], [1, 2, 3])


class TestParametros:
    def test_umbral_invalido(self):
        with pytest.raises(ValueError):
            detect_shift([1, 2, 3], [1, 2, 3], drop_threshold=1.5)

    def test_n_consecutive_invalido(self):
        with pytest.raises(ValueError):
            detect_shift([1, 2, 3], [1, 2, 3], n_consecutive=0)

    def test_min_day_descarta_shift_temprano(self):
        v = [5, 9, 6, 3, 2, 1.5, 1.2, 1.0]
        assert detect_shift(range(1, 9), v).occurred
        r = detect_shift(range(1, 9), v, min_day=5)
        assert not r.occurred and "min_day" in r.reason

    def test_mas_dias_consecutivos_es_mas_estricto(self):
        v = [1, 2, 3, 4, 2.0, 3.5, 4.5, 5.5, 6.5, 7.5]
        laxo = detect_shift(range(1, 11), v, n_consecutive=1, smooth_window=1)
        estricto = detect_shift(range(1, 11), v, n_consecutive=3, smooth_window=1)
        assert laxo.occurred and not estricto.occurred


class TestRefinamientoDelPico:
    """refine_peak corrige el desplazamiento del dia causado por el suavizado."""

    def test_refinamiento_reduce_el_sesgo(self):
        series, verdad = make_synthetic_cultures(40, seed=2)
        errores = {}
        for refine in (False, True):
            r = detect_shift_batch(series, id_col="culture", day_col="day",
                                   value_col="lactate", refine_peak=refine)
            c = r.join(verdad)
            det = c[c["occurred"] & c["shift_day_real"].notna()]
            errores[refine] = (det["day"] - det["shift_day_real"]).mean()
        assert errores[False] < -0.5, "sin refinamiento el sesgo deberia ser notorio"
        assert abs(errores[True]) < 0.3, "con refinamiento el sesgo casi desaparece"

    def test_refinamiento_corrige_tambien_hacia_adelante(self):
        """Pico brusco y caida lenta: sin refinar da dia 8, con refinar dia 7."""
        v = [1, 2, 3, 4, 5, 6, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6, 5]
        sin = detect_shift(range(1, 16), v, refine_peak=False)
        con = detect_shift(range(1, 16), v, refine_peak=True)
        assert sin.day == 8.0
        assert con.day == 7.0

    def test_dia_detectado_cae_dentro_de_un_dia_del_real(self):
        series, verdad = make_synthetic_cultures(40, seed=3)
        r = detect_shift_batch(series, id_col="culture", day_col="day", value_col="lactate")
        c = r.join(verdad)
        det = c[c["occurred"] & c["shift_day_real"].notna()]
        dentro = (det["day"] - det["shift_day_real"]).abs() <= 1
        assert dentro.mean() >= 0.90


class TestAuxiliares:
    def test_regularize_rellena_dias_faltantes(self):
        d, v = regularize([1, 2, 5], [1.0, 2.0, 5.0])
        assert list(d) == [1, 2, 3, 4, 5]
        assert np.isclose(v[2], 3.0) and np.isclose(v[3], 4.0)

    def test_smooth_ventana_1_no_cambia_nada(self):
        v = [1.0, 5.0, 2.0]
        assert np.allclose(smooth(v, 1), v)

    def test_time_es_dia_de_censura_si_no_hay_evento(self):
        r = detect_shift(range(1, 13), np.linspace(0.1, 1, 12))
        assert r.time == 12.0

    def test_batch_devuelve_una_fila_por_serie(self):
        series, _ = make_synthetic_cultures(12, seed=0)
        r = detect_shift_batch(series, id_col="culture", day_col="day", value_col="lactate")
        assert len(r) == 12
        assert {"occurred", "day", "time", "censoring_day"} <= set(r.columns)
