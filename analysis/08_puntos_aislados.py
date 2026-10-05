"""
08 - Revisa si alguna etiqueta depende de un punto bajo aislado.

Un solo punto muy bajo cerca del final puede crear un shift falso (ver
lactateshift/qc.py). Se buscan esos puntos con isolated_low_points y se
repite la deteccion sin ellos (el hueco se interpola).

Ejecutar:  python analysis/08_puntos_aislados.py

Desarrollado por Arturo Rodriguez.
"""

import numpy as np
import pandas as pd

from lactateshift import detect_shift, isolated_low_points

from _comun import TABLAS, cargar

# mismos parametros que 02_definicion_evento.py
PARAMETROS = dict(smooth_window=3, n_consecutive=2, drop_threshold=0.30)


def main() -> None:
    raw = cargar("Raw Data")
    filas = []
    for cult, g in raw.groupby("cult"):
        d = g["day"].to_numpy(float)
        v = g["[Lactate]"].to_numpy(float)
        sospechosos = isolated_low_points(d, v)
        if not sospechosos:
            continue
        antes = detect_shift(d, v, **PARAMETROS)
        v2 = np.where(np.isin(d, sospechosos), np.nan, v)
        despues = detect_shift(d, v2, **PARAMETROS)
        filas.append({
            "cult": cult,
            "dias_sospechosos": " ".join(f"{x:.0f}" for x in sospechosos),
            "shift_antes": antes.occurred, "dia_antes": antes.day,
            "shift_sin_ellos": despues.occurred, "dia_sin_ellos": despues.day,
            "cambia": (antes.occurred != despues.occurred) or (antes.day != despues.day),
        })
    t = pd.DataFrame(filas)
    t.to_csv(TABLAS / "puntos_aislados.csv", index=False)
    print(t.to_string(index=False))
    print(f"\n{len(t)} de {raw['cult'].nunique()} cultivos tienen al menos un punto bajo aislado.")
    print(f"Quitarlos cambia el resultado en {int(t['cambia'].sum())}.")


if __name__ == "__main__":
    main()
