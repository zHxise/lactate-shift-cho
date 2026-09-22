"""Tests del detector sobre series con respuesta conocida.

Estos tests no dependen del dataset del caso de estudio, que esta bajo
copyright y no se distribuye con el repositorio. Todo lo que se verifica aqui
se construye en el momento, asi que cualquiera puede correrlos.
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
        """Documenta el limite real de la regla, que la auditoria externa
        senalo como no cubierto por los tests.

        Una caida que alcanza el umbral y despues se revierte cuenta como
        shift. Es deliberado: en cultivos reales el cambio a consumo suele ser
        reversible. Pero significa que la regla NO distingue un cambio de
        regimen permanente de una caida profunda transitoria, y eso hay que
        decirlo en vez de suponer lo contrario.
        """
        v = [1, 2, 3, 4, 5, 6, 7, 10, 7, 5, 4, 3, 12, 13]
        assert detect_shift(range(1, 15), v).occurred

    def test_caida_por_debajo_del_umbral(self):
        v = [1, 2, 3, 4, 3.9, 3.8, 3.7, 3.6]
        assert not detect_shift(range(1, 9), v, drop_threshold=0.50).occurred
        assert detect_shift(range(1, 9), v, drop_threshold=0.05).occurred


class TestRebote:
    """El caso que rompe cualquier regla anclada en el maximo global."""

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
        """Caracteriza la degradacion en lugar de fingir que no existe.

        Si faltan dias justo alrededor del pico, la interpolacion mueve el
        maximo y el dia detectado se corre. Con 30% de dias ausentes el error
        puede llegar a 2 dias. Quien use el detector sobre datos de muestreo
        esparso necesita saberlo: el evento se sigue detectando, el dia no es
        igual de confiable.
        """
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
    """Parametros anadidos tras una auditoria externa."""

    def test_min_peak_descarta_oscilaciones_cerca_de_cero(self):
        # caida relativa del 87% sobre una serie de amplitud despreciable
        v = [0.001, 0.002, 0.003, 0.002, 0.001, 0.0005, 0.0004, 0.0004, 0.0004]
        assert detect_shift(range(1, 10), v).occurred
        assert not detect_shift(range(1, 10), v, min_peak=0.01).occurred

    def test_require_local_max_es_mas_estricto(self):
        # meseta: el descenso sostenido empieza en el ultimo punto del plano,
        # que no es un maximo local estricto
        v = [1.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.5, 0.4]
        laxo = detect_shift(range(1, 9), v, smooth_window=1)
        estricto = detect_shift(range(1, 9), v, smooth_window=1, require_local_max=True)
        assert laxo.occurred
        assert (not estricto.occurred) or estricto.day != laxo.day


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
    """El suavizado centrado sesga el dia detectado hacia atras cuando la
    caida es mas rapida que la subida. Este es el test que documenta el bug
    y su correccion."""

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
