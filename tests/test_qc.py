"""Tests del control de calidad (lactateshift.qc)."""

from lactateshift import detect_shift, isolated_low_points

DIAS = list(range(1, 15))


def test_senala_punto_bajo_aislado():
    # meseta con una medicion erronea el dia 13 que produce un shift falso
    v = [0.45, 0.7, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 3.9, 3.75, 3.2, 0.7, 4.3]
    assert detect_shift(DIAS, v).occurred
    assert isolated_low_points(DIAS, v) == [13.0]


def test_no_senala_una_bajada_real():
    v = [0.5, 1.2, 2.5, 3.8, 4.6, 3.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4]
    assert isolated_low_points(DIAS, v) == []


def test_no_senala_el_fondo_de_una_caida_reversible():
    # 1.2 esta por debajo de sus vecinos, pero no de la mitad de su promedio
    v = [0.5, 1.5, 2.8, 3.9, 4.5, 2.5, 1.5, 1.2, 1.8, 2.5, 3.5, 4.2, 4.8, 4.9]
    assert isolated_low_points(DIAS, v) == []


def test_ignora_el_ruido_cerca_de_cero():
    # 0.02 entre 0.10 y 0.12 es muy pequeno frente al maximo de 4
    v = [0.5, 1.5, 3.0, 4.0, 2.0, 1.0, 0.5, 0.10, 0.02, 0.12, 0.1, 0.1, 0.1, 0.1]
    assert isolated_low_points(DIAS, v) == []


def test_usa_las_mediciones_vecinas_si_hay_nan():
    v = [1.0, 2.0, 3.0, float("nan"), 0.5, 3.2, 3.3]
    assert isolated_low_points([1, 2, 3, 4, 5, 6, 7], v) == [5.0]
