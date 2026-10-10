"""
13 - Las asociaciones del predictor diario (script 12) se sostienen dentro de
cada grupo de volumen?

Por que: los grupos de volumen difieren en receta de temperatura y
probablemente en producto. Una variable puede parecer asociada al shift solo
porque distingue grupos (por ejemplo, un grupo con pCO2 alta y shifts
tardios), sin relacion dentro de ninguno. Dos pruebas:

A. Modelo conjunto con efectos fijos de grupo: se agrega un indicador por
   grupo de volumen. Asi cada coeficiente se estima comparando dias y
   cultivos DENTRO del mismo grupo. Si una asociacion desaparece al agregar
   los indicadores, era diferencia entre grupos.

B. Modelo por separado en cada uno de los tres grupos grandes (21, 22 y 50
   cultivos). Con tan pocos eventos por grupo los intervalos seran anchos;
   lo que se mira es si la DIRECCION se repite.

Intervalos: bootstrap por cultivo (se remuestrean cultivos enteros).

Ejecutar:  python analysis/13_asociaciones_por_grupo.py

Desarrollado por Arturo Rodriguez.
"""

import warnings
from importlib import import_module

import numpy as np
import pandas as pd

from _comun import TABLAS, cargar

warnings.filterwarnings("ignore")
est = import_module("12_estado_celular")
SEED = 0
N_BOOT = 200


def ajustar(pp, cols, n_boot=N_BOOT):
    """Odds ratio por desviacion estandar con IC por bootstrap de cultivos."""
    full = est.modelo().fit(pp[cols], pp["y"])
    coef = full.named_steps["m"].coef_[0]
    rng = np.random.default_rng(SEED)
    cultivos = pp["cult"].unique()
    por_cult = {c: d for c, d in pp.groupby("cult")}
    boot = []
    for _ in range(n_boot):
        b = pd.concat([por_cult[c] for c in rng.choice(cultivos, len(cultivos), replace=True)])
        if b["y"].nunique() < 2:
            continue
        boot.append(est.modelo().fit(b[cols], b["y"]).named_steps["m"].coef_[0])
    boot = np.array(boot)
    return pd.DataFrame({"or": np.exp(coef),
                         "bajo": np.exp(np.percentile(boot, 2.5, axis=0)),
                         "alto": np.exp(np.percentile(boot, 97.5, axis=0))}, index=cols)


def main() -> None:
    raw = cargar("Raw Data")
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")
    pp = est.persona_periodo(raw, et, adelanto=1)
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]
    receta = ["temperatura", "bajo_T_hoy", "baja_T_manana"]
    medic = list(est.MEDICIONES.values()) + ["dvcd"]
    variables = receta + medic

    grandes = [g for g, n in pp.groupby("grupo")["cult"].nunique().items() if n >= 15]
    pp["grupo_s"] = pp["grupo"].map(lambda v: f"{v:.5f}")
    ind_grupo = []
    for g in grandes[1:]:  # el primero es la referencia
        col = f"g_{g:.5f}"
        pp[col] = (pp["grupo"] == g).astype(int)
        ind_grupo.append(col)
    pp["g_otros"] = (~pp["grupo"].isin(grandes)).astype(int)
    ind_grupo.append("g_otros")

    print("=== A. Modelo conjunto: sin y con efectos fijos de grupo ===")
    sin = ajustar(pp, dia + variables).loc[variables]
    con = ajustar(pp, dia + ind_grupo + variables).loc[variables]
    tabla = pd.DataFrame({"OR_sin_grupo": sin["or"], "IC_sin": [f"{a:.2f}-{b:.2f}" for a, b in zip(sin['bajo'], sin['alto'])],
                          "OR_con_grupo": con["or"], "IC_con": [f"{a:.2f}-{b:.2f}" for a, b in zip(con['bajo'], con['alto'])]})
    tabla["se_sostiene"] = ((con["bajo"] > 1) | (con["alto"] < 1))
    print(tabla.round(2).to_string())

    print("\n=== B. Modelo por grupo (direccion del efecto) ===")
    por_grupo = {}
    for g in grandes:
        sub = pp[pp["grupo"] == g]
        cols = [c for c in dia + variables if sub[c].nunique() > 1]  # sin columnas constantes
        r = ajustar(sub, cols)
        por_grupo[f"{g:.5f} ({sub['cult'].nunique()} cult., {int(sub['y'].sum())} ev.)"] = r["or"]
        print(f"\ngrupo {g:.5f}: {sub['cult'].nunique()} cultivos, {int(sub['y'].sum())} eventos")
        sig = r.loc[[v for v in variables if v in r.index]]
        sig = sig[(sig["bajo"] > 1) | (sig["alto"] < 1)]
        print("  IC que excluye 1:", {k: round(v, 2) for k, v in sig["or"].items()} or "ninguna")
    pg = pd.DataFrame(por_grupo).reindex(variables)
    signo = np.sign(np.log(pg))
    pg["misma_direccion_en_los_3"] = signo.nunique(axis=1).eq(1) & signo.notna().all(axis=1)
    print("\nOdds ratio por grupo (NaN = la variable no varia en ese grupo):")
    print(pg.round(2).to_string())

    salida = tabla.join(pg)
    salida.round(3).to_csv(TABLAS / "estado_asociaciones_por_grupo.csv")
    print(f"\nTabla en {TABLAS / 'estado_asociaciones_por_grupo.csv'}")


if __name__ == "__main__":
    main()
