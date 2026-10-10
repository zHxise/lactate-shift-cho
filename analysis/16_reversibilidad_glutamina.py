"""
16 - Pruebas antes del modelo mecanistico: reversibilidad del shift y glutamina.

A. REVERSIBILIDAD (diagnostico de la estructura 3 de la propuesta de modelos)
   La estructura 3 supone un interruptor con "memoria": una vez que el cultivo
   pasa a consumir lactato, se queda ahi (histeresis). Si muchos cultivos
   vuelven a PRODUCIR lactato de forma sostenida despues del shift, esa idea
   pierde sustento. Se usa la misma logica que define el shift, al reves:
   re-produccion = tasa especifica q > 0 durante 2 intervalos seguidos
   despues del shift Y el lactato sube desde su minimo posterior al shift al
   menos 15% de la AMPLITUD del propio cultivo (maximo - minimo de la serie
   suavizada), confirmado en los valores crudos.
   Correccion de la auditoria de la tarea 5: la version anterior pedia una
   subida de 15% RELATIVA al minimo. Con datos min-max ese minimo es casi
   cero (mediana 0.010), asi que 15% de el es ruido (~0.0015) y ademas
   depende del desplazamiento desconocido de la normalizacion. La amplitud
   del cultivo no depende de ese desplazamiento. La regla anterior y la de 0%
   (sin filtro) se conservan como sensibilidad, junto con cuantos dias
   despues del shift ocurre, y si el cultivo seguia creciendo o ya iba en
   declive (la re-produccion tardia en declive es otro fenomeno, no el mismo
   interruptor que regresa).

B. GLUTAMINA COMO EVENTO DE AGOTAMIENTO
   La literatura (Zagari 2013, Hong 2018) dice que lo que coincide con el
   shift es el AGOTAMIENTO de glutamina, no su nivel. Antes de modelar se
   describe como se comporta la glutamina en este dataset (si casi nunca se
   agota, no hay evento que detectar). Despues:
   - evento = la glutamina (rellenada solo hacia adelante) cae al percentil
     q o por debajo; q = 5, 10, 20. El umbral se calcula SOLO con las filas
     de entrenamiento de cada fold (calcularlo con todo el dataset seria
     fuga de informacion);
   - se agrega el evento al modelo de riesgo diario (M3 del script 15, datos
     con QC) y se mide la mejora de AUC fuera de fold con IC por cultivo;
   - coincidencia temporal: cuantos cultivos tienen su primer dia de
     agotamiento a +-1 dia del shift, contra el azar (permutacion), igual que
     se hizo con la temperatura.

Ejecutar:  python analysis/16_reversibilidad_glutamina.py

Desarrollado por Arturo Rodriguez.
"""

import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lactateshift import regularize, smooth  # noqa: E402

from _comun import TABLAS, cargar  # noqa: E402

warnings.filterwarnings("ignore")
est = import_module("12_estado_celular")
c15 = import_module("15_correcciones_auditoria")
tasa = import_module("10_tasa_especifica")
SEED = 0
PERCENTILES = (5, 10, 20)


# ---------------------------------------------------------------- A
def reproduccion(dias, lac, vcd, dia_shift, subida_min, base="amplitud"):
    """Primer dia (despues del shift) de re-produccion sostenida, o NaN.

    base="amplitud": la subida se mide como fraccion de (max - min) del cultivo
                     y debe verse tambien en los valores crudos (regla principal).
    base="minimo":   regla anterior, subida relativa al minimo (sensibilidad).
    """
    d_ini, q = tasa.tasas(dias, lac, vcd, 0.0)
    _, L = regularize(dias, lac)
    Ls = smooth(L, tasa.SUAVIZADO)
    amplitud = np.nanmax(Ls) - np.nanmin(Ls)
    i0 = int(np.searchsorted(d_ini, dia_shift)) + 1  # primer intervalo ya en consumo
    for i in range(i0, len(q) - 1):
        if q[i] > 0 and q[i + 1] > 0:
            minimo = np.nanmin(Ls[i0:i + 1])
            if base == "minimo":
                if minimo <= 0 and subida_min > 0:
                    continue
                if (np.nanmax(Ls[i + 1:i + 3]) - minimo) / max(minimo, 1e-9) >= subida_min:
                    return float(d_ini[i])
            else:
                umbral = subida_min * amplitud
                suave = np.nanmax(Ls[i + 1:i + 3]) - minimo
                # en crudo: el maximo de los 2 dias siguientes contra el minimo crudo desde el shift
                crudo = np.nanmax(L[i + 1:i + 3]) - np.nanmin(L[i0:i + 1])
                if suave >= umbral and crudo >= umbral:
                    return float(d_ini[i])
    return np.nan


def parte_a(raw, et):
    filas = []
    for cult, g in raw.sort_values("day").groupby("cult"):
        if cult not in et.index or et.loc[cult, "excluir"] or not et.loc[cult, "evento"]:
            continue
        g = g.dropna(subset=["[Lactate]", "VCD"])
        dias, lac, vcd = g["day"].to_numpy(), g["[Lactate]"].to_numpy(), g["VCD"].to_numpy()
        ds = et.loc[cult, "dia"]
        fila = {"cult": cult, "dia_shift": ds, "dias_observados_despues": dias.max() - ds}
        fila["reproduce_15"] = reproduccion(dias, lac, vcd, ds, 0.15)               # principal
        fila["reproduce_15_rel_min"] = reproduccion(dias, lac, vcd, ds, 0.15, "minimo")  # regla anterior
        fila["reproduce_0"] = reproduccion(dias, lac, vcd, ds, 0.0, "minimo")       # sin filtro
        dr = fila["reproduce_15"]
        if np.isfinite(dr):
            # crecimiento en ese momento: VCD del dia de re-produccion vs su maximo previo
            v = g.set_index("day")["VCD"]
            fila["vcd_relativa_al_max"] = v.loc[:dr].iloc[-1] / v.loc[:dr].max()
        filas.append(fila)
    return pd.DataFrame(filas).set_index("cult")


# ---------------------------------------------------------------- B
def evento_glutamina(pp, umbral):
    return (pp["glutamina"] <= umbral).astype(int)  # NaN -> 0 (no observado como agotado)


def oof_con_evento(pp, cols_base, pct, n_rep=5, con_evento=True):
    acum = np.zeros(len(pp))
    for rep in range(n_rep):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for tr, te in cv.split(pp, pp["y"], pp["cult"]):
            ptr, pte = pp.iloc[tr].copy(), pp.iloc[te].copy()
            cols = list(cols_base)
            if con_evento:
                umbral = np.nanpercentile(ptr["glutamina"], pct)  # solo entrenamiento
                ptr["gln_agotada"] = evento_glutamina(ptr, umbral)
                pte["gln_agotada"] = evento_glutamina(pte, umbral)
                cols.append("gln_agotada")
            m = c15.modelo_robusto().fit(ptr[cols], ptr["y"])
            acum[te] += m.predict_proba(pte[cols])[:, 1]
    return acum / n_rep


def main() -> None:
    raw = cargar("Raw Data")
    raw_l, _ = c15.qc_gases(raw)
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")

    # ------------------------------------------------------------ A
    print("=== A. Reversibilidad del shift ===")
    a = parte_a(raw, et)
    n = len(a)
    r15, r0 = a["reproduce_15"].notna(), a["reproduce_0"].notna()
    rrel = a["reproduce_15_rel_min"].notna()
    print(f"cultivos con shift: {n}")
    print(f"vuelven a producir lactato (>=2 intervalos, subida >=15% de la amplitud, confirmada en crudo): "
          f"{int(r15.sum())} ({r15.mean():.0%})")
    print(f"  sensibilidad: regla anterior (15% del minimo) {int(rrel.sum())}; sin filtro {int(r0.sum())}")
    largo = a["dias_observados_despues"] >= 8
    print(f"  con >= 8 dias de seguimiento despues del shift: {int(r15[largo].sum())} de {int(largo.sum())} "
          f"({r15[largo].mean():.0%})")
    dif = (a.loc[r15, "reproduce_15"] - a.loc[r15, "dia_shift"])
    print("dias entre el shift y la re-produccion:", dif.value_counts().sort_index().to_dict())
    print("VCD en ese momento / su maximo previo (1 = sigue en el maximo; <1 = ya en declive): "
          f"mediana {a.loc[r15, 'vcd_relativa_al_max'].median():.2f}")
    print("dias observados despues del shift, con / sin re-produccion (mediana):",
          a.loc[r15, "dias_observados_despues"].median(), "/", a.loc[~r15, "dias_observados_despues"].median())
    a.to_csv(TABLAS / "reversibilidad_shift.csv")

    # ------------------------------------------------------------ B
    print("\n=== B. Glutamina ===")
    q = raw["[Glutamine]"]
    print(f"faltante: {q.isna().mean():.0%} de las filas")
    print("mediana por dia (sube con el tiempo => no es el patron de agotamiento tipico):")
    print(raw.groupby("day")["[Glutamine]"].median().round(3).loc[1:12].to_dict())
    pp = est.persona_periodo(raw_l, et, adelanto=1)
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]
    m3 = dia + ["temperatura", "bajo_T_hoy", "baja_T_manana"] + list(est.MEDICIONES.values()) + ["dvcd"]
    m3_sin_gln = [c for c in m3 if c != "glutamina"]

    p_base = oof_con_evento(pp, m3, None, con_evento=False)
    auc_base = roc_auc_score(pp["y"], p_base)
    print(f"\nAUC M3 (nivel de glutamina): {auc_base:.3f}")
    filas = []
    for pct in PERCENTILES:
        for nombre, cols in (("M3 + evento", m3), ("M3 sin nivel + evento", m3_sin_gln)):
            p = oof_con_evento(pp, cols, pct)
            ic_auc, ic_dif = c15.bootstrap_auc(pp, p_base, p)
            auc = roc_auc_score(pp["y"], p)
            filas.append({"percentil": pct, "modelo": nombre, "auc": auc, "dif_vs_M3": auc - auc_base,
                          "dif_ic_bajo": ic_dif[0], "dif_ic_alto": ic_dif[1]})
            print(f"  p{pct:<3d} {nombre:24s} AUC {auc:.3f}  diferencia {auc - auc_base:+.3f} "
                  f"(IC {ic_dif[0]:+.3f} a {ic_dif[1]:+.3f})")
    pd.DataFrame(filas).round(4).to_csv(TABLAS / "glutamina_evento_auc.csv", index=False)

    # coincidencia temporal: primer dia de agotamiento (p10 de todo el dataset; descriptivo)
    umbral = np.nanpercentile(pp["glutamina"], 10)
    ev = et[(~et["excluir"]) & et["evento"]]
    primero = {}
    for cult, g in raw.sort_values("day").groupby("cult"):
        if cult in ev.index:
            s = g.set_index("day")["[Glutamine]"].ffill()
            d = s[s <= umbral]
            if not d.empty:
                primero[cult] = d.index.min()
    pr = pd.Series(primero)
    j = ev.loc[pr.index, "dia"]
    obs = np.mean(np.abs(j - pr) <= 1)
    rng = np.random.default_rng(SEED)
    nulo = np.array([np.mean(np.abs(j.to_numpy() - rng.permutation(pr.to_numpy())) <= 1) for _ in range(5000)])
    print(f"\ncultivos que llegan alguna vez al p10 de glutamina: {len(pr)} de {len(ev)}")
    print(f"primer agotamiento a +-1 dia del shift: {obs:.0%}; azar {nulo.mean():.0%}; "
          f"p = {(np.sum(nulo >= obs) + 1) / 5001:.3f}")
    print("dia del agotamiento menos dia del shift:", (pr - j).value_counts().sort_index().to_dict())
    pd.DataFrame({"primer_agotamiento": pr, "dia_shift": j}).to_csv(TABLAS / "glutamina_agotamiento_vs_shift.csv")


if __name__ == "__main__":
    main()
