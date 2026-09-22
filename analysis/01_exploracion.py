"""
01_exploracion.py — Exploracion estructural del suplemento mmc3.xlsx
(Gangadharan et al. 2021, Computers & Chemical Engineering).

Objetivo: entender que hay en el archivo ANTES de decidir nada de modelado.
No define el evento ni construye features todavia: solo describe.

Ejecutar:  python analysis/01_exploracion.py
Salidas: outputs/tablas/*.csv  y un resumen por consola.

Por que este script y no un notebook: la exploracion debe ser reproducible
y versionable. Los notebooks guardan estado oculto y ordenan mal el historial.
"""

import numpy as np
import pandas as pd

from _comun import TABLAS as OUT, XLSX, cargar

# Variables que importan para la hipotesis del lactate shift.
# Lactato y glucosa son el eje; VCD normaliza por biomasa; glutamina y amonio
# entran por la hipotesis de que el switch acompana el cambio de metabolismo
# de glutamina; pH y osmolalidad son condiciones de proceso que lo modulan.
NUCLEO = ["[Lactate]", "[Glucose]", "VCD", "[Glutamine]", "[NH3]",
          "pH", "Osmolality", "[Glutamate]"]


def describir_series(nombre: str, df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de estructura: cuantos cultivos, duraciones y huecos."""
    dur = df.groupby("cult")["day"].agg(dia_min="min", dia_max="max", n_filas="count")
    # Un cultivo es "continuo" si tiene una fila por dia desde d1 hasta su
    # ultimo dia. Si no, faltan dias completos (no solo celdas sueltas), lo
    # que afecta directamente a features de ventana fija dias 1-4.
    dur["continuo"] = dur["n_filas"] == dur["dia_max"]
    print(f"\n=== {nombre} ===")
    print(f"filas={len(df)}  cultivos={df['cult'].nunique()}  dias {df['day'].min():.0f}-{df['day'].max():.0f}")
    print("distribucion de duracion (ultimo dia):")
    print(dur["dia_max"].value_counts().sort_index().to_string())
    print(f"cultivos con dias completos: {int(dur['continuo'].sum())}/{len(dur)}")
    return dur


def faltantes(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c not in ("Culture ID", "cult", "day")]
    return df[cols].isna().mean().mul(100).round(2).sort_values(ascending=False)


def cobertura_ventana(raw: pd.DataFrame, dias=(1, 4)) -> pd.DataFrame:
    """Cuantos dias REALMENTE medidos hay por cultivo en la ventana 1-4.

    Esto decide que features son honestas. Si glutamina solo tiene 2.5 dias
    medidos de 4 en promedio, una pendiente de glutamina en esa ventana esta
    construida mayormente sobre valores imputados por los autores.
    """
    v = raw[(raw["day"] >= dias[0]) & (raw["day"] <= dias[1])]
    filas = []
    for c in NUCLEO:
        n = v.groupby("cult")[c].count().reindex(raw["cult"].unique()).fillna(0)
        filas.append({"variable": c, "dias_medidos_media": round(n.mean(), 2),
                      "cultivos_>=3dias": int((n >= 3).sum()),
                      "cultivos_>=2dias": int((n >= 2).sum()),
                      "cultivos_0dias": int((n == 0).sum())})
    return pd.DataFrame(filas)


def fraccion_imputada(raw: pd.DataFrame, gap: pd.DataFrame, dias=(1, 4)) -> pd.DataFrame:
    """Que proporcion de la ventana 1-4 viene del gap-filling de los autores.

    RIESGO METODOLOGICO: el gap-filling lo hicieron ellos con el cultivo
    completo disponible. Si para rellenar el dia 3 usaron informacion de los
    dias 8-15, entonces un feature "del dia 3" contiene informacion del
    futuro, y el modelo puede parecer predictivo sin serlo. Cuantificar esto
    es el primer paso; decidir que hacer al respecto viene despues.
    """
    m = raw.merge(gap, on=["cult", "day"], suffixes=("_r", "_g"))
    v = m[(m["day"] >= dias[0]) & (m["day"] <= dias[1])]
    filas = []
    for c in NUCLEO:
        filas.append({"variable": c,
                      "pct_imputado_d1_4": round(v[c + "_r"].isna().mean() * 100, 1),
                      "pct_imputado_cultivo_completo": round(m[c + "_r"].isna().mean() * 100, 1)})
    return pd.DataFrame(filas)


def perfil_lactato(gap: pd.DataFrame) -> pd.DataFrame:
    """Descripcion del pico de lactato por cultivo. NO es la definicion del
    evento: es solo para saber si la ventana 1-4 es anticipacion real."""
    filas = []
    for c, d in gap.groupby("cult"):
        s = d["[Lactate]"].values
        dias = d["day"].values
        i = int(np.argmax(s))
        post_min = s[i + 1:].min() if i < len(s) - 1 else np.nan
        filas.append({"cult": c, "duracion": dias[-1], "dia_pico": dias[i],
                      "frac_duracion": dias[i] / dias[-1],
                      "lactato_pico": s[i], "lactato_min_post": post_min,
                      "caida_rel": np.nan if np.isnan(post_min) else (s[i] - post_min) / s[i] if s[i] > 0 else np.nan})
    return pd.DataFrame(filas)


def cruce_hojas() -> None:
    """Intento de ligar Midpoint/Endpoint (45 filas, con linea celular y lote)
    con las series de tiempo (106 cultivos). No hay Culture ID en esas hojas."""
    mid = pd.read_excel(XLSX, sheet_name="Midpoint Set")
    end = pd.read_excel(XLSX, sheet_name="Endpoint Set")
    mid.columns = [c.strip() for c in mid.columns]
    end.columns = [c.strip() for c in end.columns]
    print("\n=== Midpoint / Endpoint ===")
    print(f"filas: mid={len(mid)} end={len(end)}")
    print("Cell Line:", mid["Cell Line"].value_counts().to_dict())
    print("Batch:", mid["Batch"].value_counts().to_dict())
    print("Dias anonimizados -> Midpoint Day:", sorted(mid["Midpoint Day"].unique()),
          "| Harvest Day:", sorted(end["Harvest Day"].unique()))
    print("Culture Volume unicos (mid):", sorted(mid["Culture Volume"].unique()))
    print("[mAb] identico entre mid y end:", bool(np.allclose(mid["[mAb]"], end["[mAb]"])))


def main() -> None:
    raw = cargar("Raw Data")
    gap = cargar("Gap-Filled Data")

    describir_series("Raw Data", raw).to_csv(OUT / "duracion_raw.csv")
    describir_series("Gap-Filled Data", gap)

    print("\n--- % de celdas faltantes, Raw Data ---")
    nan_raw = faltantes(raw)
    print(nan_raw.to_string())
    nan_raw.to_csv(OUT / "faltantes_raw.csv", header=["pct_nan"])
    print("\nGap-Filled: faltantes totales =", int(faltantes(gap).sum()))

    cob = cobertura_ventana(raw)
    print("\n--- Cobertura real medida en dias 1-4 (Raw) ---")
    print(cob.to_string(index=False))
    cob.to_csv(OUT / "cobertura_dias1_4.csv", index=False)

    imp = fraccion_imputada(raw, gap)
    print("\n--- Fraccion imputada por el gap-filling ---")
    print(imp.to_string(index=False))
    imp.to_csv(OUT / "fraccion_imputada.csv", index=False)

    perf = perfil_lactato(gap)
    perf.to_csv(OUT / "perfil_lactato.csv", index=False)
    print("\n--- Pico de lactato (descriptivo) ---")
    print(f"dia del pico: mediana {perf['dia_pico'].median():.0f} "
          f"(p25 {perf['dia_pico'].quantile(.25):.0f}, p75 {perf['dia_pico'].quantile(.75):.0f})")
    print(f"fraccion de la duracion: mediana {perf['frac_duracion'].median():.3f}")
    print(f"cultivos con pico en dia <= 4: {int((perf['dia_pico'] <= 4).sum())}  "
          "(candidatos a excluir: no hay nada que anticipar)")

    # Escalas: Culture Volume es constante dentro de cada cultivo -> sirve como
    # variable de grupo para la validacion 'dejar fuera una escala completa'.
    cv = raw.groupby("cult")["Culture Volume"].nunique()
    print("\nCulture Volume constante por cultivo:", bool((cv == 1).all()))
    escalas = raw.groupby("cult")["Culture Volume"].first().value_counts().sort_index()
    print("cultivos por escala (volumen normalizado):")
    print(escalas.to_string())
    escalas.to_csv(OUT / "cultivos_por_escala.csv", header=["n_cultivos"])

    cruce_hojas()
    print(f"\nTablas escritas en {OUT}")


if __name__ == "__main__":
    main()
