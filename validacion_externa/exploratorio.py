"""Analisis exploratorios de la validacion externa (NO confirmatorios).

Lo confirmatorio es solo lo que calcula comparar.py con las reglas de
PROTOCOLO.md. Esto se escribio despues de ver el resultado y sirve para
entenderlo, no para cambiarlo:

1. Cuanto cae el lactato en cada curva (crudo y suavizado), para ver por que
   fallaron las que fallaron. Descriptivo; no se usa para ajustar el umbral.
2. El resultado sin la curva yin2025/vvm_alto. Este analisis se anuncio en
   DIGITALIZACION.md antes de correr el detector (commit d530162), porque su
   respuesta es un intervalo de 8 dias y no un dia.

    python validacion_externa/exploratorio.py

Desarrollado por Arturo Rodriguez.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "src"))
sys.path.insert(0, str(AQUI))

from comparar import clopper_pearson, preparar_dias  # noqa: E402
from lactateshift.detect import regularize, smooth  # noqa: E402


def caida_maxima(dias, valores):
    """Mayor caida relativa desde un punto hasta el minimo posterior."""
    mejor = (0.0, np.nan, np.nan)
    for i, v in enumerate(valores):
        if v <= 0:
            continue
        j = i + int(np.argmin(valores[i:]))
        c = 1 - valores[j] / v
        if c > mejor[0]:
            mejor = (float(c), float(dias[i]), float(dias[j]))
    return mejor


def main() -> None:
    curvas = pd.read_csv(AQUI / "curvas.csv")
    res = pd.read_csv(AQUI / "resultado.csv")
    filas = []
    for (art, cur), g in curvas.groupby(["articulo", "curva"], sort=False):
        d, v, desp = preparar_dias(g)
        d, v = regularize(d, v)
        s = np.asarray(smooth(v, 3))
        cr, s_ini, s_fin = caida_maxima(d - desp, v)
        cs, _, _ = caida_maxima(d - desp, s)
        filas.append({"articulo": art, "curva": cur, "caida_cruda": round(cr, 3),
                      "caida_suavizada": round(cs, 3), "desde_dia": s_ini,
                      "hasta_dia": s_fin, "ultimo_dia": float(d[-1] - desp)})
    t = pd.DataFrame(filas).merge(res[["articulo", "curva", "definicion", "acierto"]],
                                  on=["articulo", "curva"])
    t.to_csv(AQUI / "exploratorio.csv", index=False)
    print("1. Caida maxima del lactato (el umbral del detector es 30% sobre la suavizada)\n")
    print(t.to_string(index=False))

    sin = res[~((res["articulo"] == "yin2025") & (res["curva"] == "vvm_alto"))]
    k, n = int(sin["acierto"].sum()), len(sin)
    lo, hi = clopper_pearson(k, n)
    print(f"\n2. Sin yin2025/vvm_alto: {k}/{n} = {k / n:.0%}  IC95% [{lo:.0%}, {hi:.0%}]")
    print("   (anunciado antes de detectar; aun asi, no cambia el veredicto confirmatorio)")


if __name__ == "__main__":
    main()
