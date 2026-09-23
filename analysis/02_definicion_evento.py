"""
02 — Definicion del lactate shift y analisis de sensibilidad.

Produce la etiqueta central: el dia en que cada cultivo pasa de producir a
consumir lactato, con los cultivos que nunca lo hacen como censurados.

La regla vive en el paquete (``lactateshift.detect``); aqui solo se aplica al
dataset del caso de estudio y se documentan las decisiones que son propias de
ESTE dataset.

DECISION — la etiqueta se construye sobre 'Raw Data', no sobre 'Gap-Filled'.
El paper describe el gap-filling de variables dependientes del tiempo como
interpolacion de Stineman seguida de SVR entrenado sobre series completas.
Stineman interpola usando el punto anterior Y el posterior: es retrospectiva.
Para la etiqueta no seria fatal (se lee despues del experimento y puede mirar
todo el cultivo), pero introduce curvatura sintetica justo en la derivada que
estamos midiendo. El lactato solo tiene 3% de celdas faltantes en Raw, asi que
no hace falta pagar ese precio. Gap-Filled se compara al final.

DECISION — un shift ocurrido dentro de la ventana de observacion (dias 1-4) se
excluye del conjunto de modelado en vez de marcarse como "sin shift". No es un
problema de anticipacion: el desenlace ya es visible en las propias variables.

Ejecutar:  python analysis/02_definicion_evento.py
"""

from itertools import product

import pandas as pd

from lactateshift import detect_shift_batch
from _comun import TABLAS, VENTANA, cargar

# Parametros congelados de la definicion.
SUAVIZADO = 3
N_CONSEC = 2
UMBRAL = 0.30


def etiquetar(df: pd.DataFrame, **kw) -> pd.DataFrame:
    r = detect_shift_batch(df, id_col="cult", day_col="day",
                           value_col="[Lactate]", **kw)
    r["excluir"] = r["occurred"] & (r["day"] <= VENTANA)
    return r


def sensibilidad(df: pd.DataFrame) -> pd.DataFrame:
    """Si la proporcion de eventos se mueve mucho con los parametros, la
    definicion es arbitraria y hay que declararlo."""
    filas = []
    for v, n, u in product([1, 3, 5], [2, 3], [0.20, 0.30, 0.40, 0.50]):
        e = etiquetar(df, smooth_window=v, n_consecutive=n, drop_threshold=u)
        m = e[~e["excluir"]]
        dias = m.loc[m["occurred"], "day"]
        filas.append({"suavizado": v, "dias_consec": n, "umbral": u,
                      "excluidos": int(e["excluir"].sum()),
                      "n_modelado": len(m), "n_evento": int(m["occurred"].sum()),
                      "censurados": int((~m["occurred"]).sum()),
                      "dia_mediana": dias.median()})
    return pd.DataFrame(filas)


def main() -> None:
    raw = cargar("Raw Data")
    et = etiquetar(raw, smooth_window=SUAVIZADO, n_consecutive=N_CONSEC,
                   drop_threshold=UMBRAL)
    et.to_csv(TABLAS / "etiquetas_evento.csv")

    exc, mod = et[et["excluir"]], et[~et["excluir"]]
    print(f"=== Definicion congelada (suavizado={SUAVIZADO}, "
          f"consecutivos={N_CONSEC}, umbral={UMBRAL}) ===")
    print(f"cultivos totales: {len(et)}")
    print(f"excluidos por shift dentro de dias 1-{VENTANA}: {len(exc)} -> {list(exc.index)}")
    n = int(mod["occurred"].sum())
    print(f"conjunto de modelado: {len(mod)} | con evento {n} | "
          f"censurados {len(mod)-n} -> {list(mod[~mod['occurred']].index)}")

    d = mod.loc[mod["occurred"], "day"]
    print(f"\ndia del evento: mediana {d.median():.0f} (p25 {d.quantile(.25):.0f}, "
          f"p75 {d.quantile(.75):.0f}, rango {d.min():.0f}-{d.max():.0f})")
    print("distribucion:", d.value_counts().sort_index().to_dict())
    print(f"anticipacion sobre la ventana: mediana {d.median()-VENTANA:.0f} dias; "
          f"{int((d - VENTANA <= 1).sum())}/{len(d)} eventos a un solo dia del cierre")

    print("\n=== Sensibilidad a los parametros ===")
    sens = sensibilidad(raw)
    sens.to_csv(TABLAS / "sensibilidad_definicion.csv", index=False)
    print(sens.pivot_table(index=["suavizado", "dias_consec"], columns="umbral",
                           values="n_evento").to_string())

    # Cuanto y hacia donde mueve el dia la correccion del suavizado en los
    # datos reales. La explicacion original (el suavizado corre el maximo
    # hacia atras) venia de los datos sinteticos y resulto no describir bien
    # a los cultivos reales: aqui se mide en vez de suponerse.
    sin_ref = etiquetar(raw, smooth_window=SUAVIZADO, n_consecutive=N_CONSEC,
                        drop_threshold=UMBRAL, refine_peak=False)
    ambos = et["occurred"] & sin_ref["occurred"]
    mov = (et.loc[ambos, "day"] - sin_ref.loc[ambos, "day"]).value_counts().sort_index()
    print("\n=== Efecto de refine_peak en los datos reales ===")
    print("desplazamiento del dia (con - sin refinamiento):", mov.to_dict())
    mov.rename_axis("desplazamiento").rename("cultivos").to_csv(TABLAS / "refine_real.csv")

    gap = cargar("Gap-Filled Data")
    et_gap = etiquetar(gap, smooth_window=SUAVIZADO, n_consecutive=N_CONSEC,
                       drop_threshold=UMBRAL)
    comp = et[["occurred", "day"]].join(et_gap[["occurred", "day"]],
                                        lsuffix="_raw", rsuffix="_gap")
    comp.to_csv(TABLAS / "comparacion_raw_gap.csv")
    dif = (comp["day_raw"] - comp["day_gap"]).abs()
    print("\n=== Raw vs Gap-Filled (misma regla) ===")
    print(f"con evento: raw={int(et['occurred'].sum())} gap={int(et_gap['occurred'].sum())} | "
          f"cambian de clase: {int((comp['occurred_raw'] != comp['occurred_gap']).sum())}")
    print(f"diferencia en el dia: mediana {dif.median():.1f}, max {dif.max():.0f}, "
          f"con diferencia >=2 dias: {int((dif >= 2).sum())}")
    print(f"\nTablas en {TABLAS}")


if __name__ == "__main__":
    main()
