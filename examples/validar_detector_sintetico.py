"""Valida el detector contra cultivos sinteticos con dia del shift conocido.

No necesita el dataset. Reproduce las cifras del README sobre refine_peak.
Los cultivos sinteticos no tienen la forma de un cultivo real.

Ejecutar:  python examples/validar_detector_sintetico.py

Desarrollado por Arturo Rodriguez.
"""

import pandas as pd

from lactateshift import detect_shift_batch, make_synthetic_cultures

filas = []
for seed in range(10):
    series, verdad = make_synthetic_cultures(40, seed=seed)
    for refinar in (False, True):
        r = detect_shift_batch(series, id_col="culture", day_col="day",
                               value_col="lactate", refine_peak=refinar)
        c = r.join(verdad)
        con_shift = c[c["shift_day_real"].notna()]
        det = con_shift[con_shift["occurred"]]
        err = det["day"] - det["shift_day_real"]
        falsos = int((c["shift_day_real"].isna() & c["occurred"]).sum())
        filas.append({"seed": seed, "refine_peak": refinar,
                      "con_shift_real": len(con_shift), "detectados": len(det),
                      "exactos": int((err == 0).sum()),
                      "a_1_dia": int((err.abs() <= 1).sum()),
                      "sesgo_medio": err.mean(), "falsos_positivos": falsos})

d = pd.DataFrame(filas)
print("Semilla 1 (la que cita el README):")
print(d[d.seed == 1].round(2).to_string(index=False))

print("\nPromedio sobre 10 semillas (400 cultivos sinteticos):")
agg = d.groupby("refine_peak").agg(
    sesgo_medio=("sesgo_medio", "mean"),
    exactos=("exactos", "sum"), detectados=("detectados", "sum"),
    a_1_dia=("a_1_dia", "sum"), con_shift_real=("con_shift_real", "sum"),
    falsos_positivos=("falsos_positivos", "sum"))
agg["pct_exactos"] = (agg["exactos"] / agg["detectados"] * 100).round(1)
agg["pct_detectados"] = (agg["detectados"] / agg["con_shift_real"] * 100).round(1)
print(agg.round(3).to_string())

from pathlib import Path
salida = Path(__file__).resolve().parents[1] / "outputs" / "tablas"
salida.mkdir(parents=True, exist_ok=True)
d.to_csv(salida / "detector_sintetico.csv", index=False)
