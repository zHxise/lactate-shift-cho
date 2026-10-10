"""
15 - Correcciones tras la auditoria independiente (Claude Science, octubre 2026).

La auditoria de los scripts 11-13 encontro cuatro problemas. Este script los
corrige y deja las cifras que se pueden reportar.

A. CONTROL DE CALIDAD DE GASES EN SANGRE
   C75 dia 2 tiene pH = 0 y pCO2 = 0 con valores normales los dias 1 y 3:
   es una medicion fallida, y como los datos son min-max define el "piso" de
   toda la escala. Regla fijada antes de mirar resultados: una lectura de gases
   es invalida si su pH cae a mas de 3 rangos intercuartiles (IQR) de los
   cuartiles del dataset. Si es invalida se enmascaran pH, pCO2 y pO2 de esa
   fila (es una sola medicion del gasometro). Los huecos se rellenan hacia
   adelante como siempre (nunca con datos futuros).
   Ademas, las razones de momios se reportan por IQR (RobustScaler) y no por
   desviacion estandar: un solo valor extremo infla la desviacion estandar
   pero casi no mueve el IQR.

B. METRICAS QUE CORRESPONDEN AL USO
   El AUC agrupado de 0.92 incluye lo que el modelo sabe solo por el dia del
   cultivo (0.83). Lo que importa es la MEJORA sobre el dia, con intervalo
   por bootstrap de cultivos (no la variacion entre particiones). Tambien:
   AUC dentro de cada dia (separa a los que hacen el shift ese dia de los que
   no) y una alarma con umbral fijado de antemano (p >= 0.5).

C. ASOCIACIONES POR GRUPO CON INTERVALOS
   El criterio "misma direccion en los 3 grupos" tiene 25% de probabilidad
   de cumplirse por azar. Se reemplaza por el intervalo de confianza de cada
   variable dentro de cada grupo.

D. TEMPERATURA: COMPARACION CONTRA EL AZAR Y AJUSTE POR CRECIMIENTO
   - Cuantos cultivos con cambio de temperatura hacen el shift a +-1 dia del
     cambio, comparado con lo que daria el azar (permutando los dias de
     cambio entre cultivos, en general y dentro de cada grupo de volumen).
   - Regresion del dia del shift contra la VCD del dia 3 y el dia del cambio
     de temperatura: si el cambio de temperatura aporta algo una vez que se
     sabe que tan rapido crecio el cultivo.

Ejecutar:  python analysis/15_correcciones_auditoria.py

Desarrollado por Arturo Rodriguez.
"""

import warnings
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from _comun import TABLAS, cargar

warnings.filterwarnings("ignore")
est = import_module("12_estado_celular")
SEED = 0
N_BOOT_OR = 300
N_BOOT_AUC = 1000
N_PERM = 5000
GASES = ["pH", "pCO2", "pO2"]


# ---------------------------------------------------------------- A
def qc_gases(raw):
    """Enmascara lecturas de gases invalidas (regla de 3 IQR sobre el pH)."""
    raw = raw.copy()
    q1, q3 = raw["pH"].quantile([0.25, 0.75])
    iqr = q3 - q1
    malas = (raw["pH"] < q1 - 3 * iqr) | (raw["pH"] > q3 + 3 * iqr)
    registro = raw.loc[malas, ["cult", "day"] + GASES].copy()
    raw.loc[malas, GASES] = np.nan
    return raw, registro


def modelo_robusto():
    """Igual que el script 12, pero escalado por IQR en lugar de desviacion estandar."""
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", RobustScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=2000))])


# ---------------------------------------------------------------- B
def predicciones_fuera_de_fold(pp, cols, n_rep=5):
    """Promedio de las predicciones fuera de fold en n_rep particiones por cultivo."""
    acum = np.zeros(len(pp))
    for rep in range(n_rep):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for tr, te in cv.split(pp, pp["y"], pp["cult"]):
            m = modelo_robusto().fit(pp.iloc[tr][cols], pp.iloc[tr]["y"])
            acum[te] += m.predict_proba(pp.iloc[te][cols])[:, 1]
    return acum / n_rep


def bootstrap_auc(pp, p_a, p_b, n=N_BOOT_AUC):
    """IC de AUC(b) y de AUC(b) - AUC(a) remuestreando cultivos completos."""
    rng = np.random.default_rng(SEED)
    cultivos = pp["cult"].unique()
    idx = {c: np.where(pp["cult"].to_numpy() == c)[0] for c in cultivos}
    y = pp["y"].to_numpy()
    aucs, difs = [], []
    for _ in range(n):
        filas = np.concatenate([idx[c] for c in rng.choice(cultivos, len(cultivos), replace=True)])
        if y[filas].min() == y[filas].max():
            continue
        a, b = roc_auc_score(y[filas], p_a[filas]), roc_auc_score(y[filas], p_b[filas])
        aucs.append(b)
        difs.append(b - a)
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(difs, [2.5, 97.5])


def alarma(pp, p, umbral=0.5):
    """Primer dia en que el riesgo predicho supera el umbral, contra el dia real."""
    res = {"exacto": 0, "antes": 0, "sin_alarma_a_tiempo": 0}
    for _, g in pp.assign(p=p).groupby("cult"):
        if g["y"].sum() == 0:
            continue  # censurados: no hay dia real
        dia_real = int(g.loc[g["y"] == 1, "dia"].iloc[0])
        alarmas = g.loc[g["p"] >= umbral, "dia"]
        if alarmas.empty:
            res["sin_alarma_a_tiempo"] += 1
        elif alarmas.min() == dia_real:
            res["exacto"] += 1
        else:
            res["antes"] += 1
    return res


# ---------------------------------------------------------------- C
def razones_momios(pp, cols, variables, n_boot=N_BOOT_OR):
    """OR por IQR con IC por bootstrap de cultivos."""
    full = modelo_robusto().fit(pp[cols], pp["y"])
    coef = pd.Series(full.named_steps["m"].coef_[0], index=cols)
    rng = np.random.default_rng(SEED)
    cultivos = pp["cult"].unique()
    por_cult = {c: d for c, d in pp.groupby("cult")}
    boot = []
    for _ in range(n_boot):
        b = pd.concat([por_cult[c] for c in rng.choice(cultivos, len(cultivos), replace=True)])
        if b["y"].nunique() < 2:
            continue
        boot.append(pd.Series(modelo_robusto().fit(b[cols], b["y"]).named_steps["m"].coef_[0], index=cols))
    boot = pd.DataFrame(boot)
    v = [x for x in variables if x in cols]
    return pd.DataFrame({"or_por_iqr": np.exp(coef[v]),
                         "ic_bajo": np.exp(boot[v].quantile(0.025)),
                         "ic_alto": np.exp(boot[v].quantile(0.975))})


# ---------------------------------------------------------------- D
def temperatura(raw, et):
    rec = pd.read_csv(TABLAS / "receta_temperatura.csv", index_col="cult")
    vol = raw.groupby("cult")["Culture Volume"].first().round(5)
    v3 = raw[raw["day"] == 3].set_index("cult")["VCD"]
    d = et[(~et["excluir"]) & et["evento"]].join(rec).join(vol.rename("grupo")).join(v3.rename("vcd_d3"))
    t = d[d["tshift"] == 1].copy()
    dentro = lambda dia, td: np.mean(np.abs(dia - td) <= 1)
    obs = dentro(t["dia"].to_numpy(), t["tshift_day"].to_numpy())
    rng = np.random.default_rng(SEED)
    nulo = np.array([dentro(t["dia"].to_numpy(), rng.permutation(t["tshift_day"].to_numpy()))
                     for _ in range(N_PERM)])
    nulo_g = []
    for _ in range(N_PERM):
        td = t["tshift_day"].copy()
        for _, idx in t.groupby("grupo").groups.items():
            td.loc[idx] = rng.permutation(td.loc[idx].to_numpy())
        nulo_g.append(dentro(t["dia"].to_numpy(), td.to_numpy()))
    nulo_g = np.array(nulo_g)

    # Regresion: dia del shift ~ VCD dia 3 + dia del cambio de temperatura
    tt = t.dropna(subset=["vcd_d3"])
    y = tt["dia"].to_numpy()
    filas = []
    for cols in (["vcd_d3"], ["tshift_day"], ["vcd_d3", "tshift_day"]):
        X = tt[cols].to_numpy()
        m = LinearRegression().fit(X, y)
        filas.append({"modelo": " + ".join(cols), "r2": m.score(X, y),
                      "coef_tshift_day": m.coef_[-1] if "tshift_day" in cols else np.nan})
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(tt), len(tt))
        boot.append(LinearRegression().fit(tt[["vcd_d3", "tshift_day"]].to_numpy()[i], y[i]).coef_[1])
    reg = pd.DataFrame(filas)
    ic_t = np.percentile(boot, [2.5, 97.5])

    # Grupo 0.00202: contraste con / sin cambio de temperatura, crudo y ajustado por VCD dia 3
    g = d[(d["grupo"] == 0.00202)].dropna(subset=["vcd_d3"])
    crudo = g.loc[g["tshift"] == 1, "dia"].mean() - g.loc[g["tshift"] == 0, "dia"].mean()
    aj = LinearRegression().fit(g[["tshift", "vcd_d3"]], g["dia"]).coef_[0]
    traslape = (g.groupby("tshift")["vcd_d3"].describe()[["min", "50%", "max"]])
    return {"n": len(t), "observado": obs, "nulo_media": nulo.mean(),
            "p": (np.sum(nulo >= obs) + 1) / (N_PERM + 1),
            "nulo_grupo_media": nulo_g.mean(), "p_grupo": (np.sum(nulo_g >= obs) + 1) / (N_PERM + 1),
            "reg": reg, "ic_coef_tshift": ic_t, "g_crudo": crudo, "g_ajustado": aj, "g_vcd": traslape}


def main() -> None:
    raw = cargar("Raw Data")
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")

    print("=== A. Control de calidad de gases ===")
    raw_l, registro = qc_gases(raw)
    print("lecturas enmascaradas (regla de 3 IQR sobre el pH):")
    print(registro.to_string(index=False))
    registro.to_csv(TABLAS / "qc_gases_enmascarados.csv", index=False)

    pp = est.persona_periodo(raw_l, et, adelanto=1)
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]
    receta = ["temperatura", "bajo_T_hoy", "baja_T_manana"]
    medic = list(est.MEDICIONES.values()) + ["dvcd"]
    m3 = dia + receta + medic

    print("\n=== B. Metricas (adelanto 1 dia, datos limpios, escalado por IQR) ===")
    p0 = predicciones_fuera_de_fold(pp, dia)
    p3 = predicciones_fuera_de_fold(pp, m3)
    auc0, auc3 = roc_auc_score(pp["y"], p0), roc_auc_score(pp["y"], p3)
    ic3, ic_dif = bootstrap_auc(pp, p0, p3)
    print(f"AUC solo dia {auc0:.3f} | M3 {auc3:.3f} (IC 95% {ic3[0]:.3f}-{ic3[1]:.3f})")
    print(f"mejora sobre el dia: {auc3 - auc0:+.3f} (IC 95% {ic_dif[0]:+.3f} a {ic_dif[1]:+.3f})")
    por_dia = []
    for d_, g in pp.assign(p0=p0, p3=p3).groupby("dia"):
        if g["y"].sum() >= 5 and (g["y"] == 0).sum() >= 5:
            por_dia.append({"dia": d_, "eventos": int(g["y"].sum()), "en_riesgo": len(g),
                            "auc_M3": roc_auc_score(g["y"], g["p3"])})
    por_dia = pd.DataFrame(por_dia)
    print("\nAUC dentro de cada dia (solo dias con >= 5 eventos y >= 5 sin evento):")
    print(por_dia.round(3).to_string(index=False))
    al0, al3 = alarma(pp, p0), alarma(pp, p3)
    print(f"\nalarma p >= 0.5 (primer dia en que se dispara): solo dia {al0} | M3 {al3}")
    pd.DataFrame([{"modelo": "solo dia", "auc": auc0, **al0},
                  {"modelo": "M3", "auc": auc3, "auc_ic_bajo": ic3[0], "auc_ic_alto": ic3[1],
                   "mejora": auc3 - auc0, "mejora_ic_bajo": ic_dif[0], "mejora_ic_alto": ic_dif[1], **al3}]
                 ).round(3).to_csv(TABLAS / "metricas_corregidas.csv", index=False)
    por_dia.round(3).to_csv(TABLAS / "auc_por_dia.csv", index=False)

    print("\n=== C. Razones de momios por IQR (M3, datos limpios) ===")
    variables = receta + medic
    conjunto = razones_momios(pp, m3, variables)
    conjunto["ic_excluye_1"] = (conjunto["ic_bajo"] > 1) | (conjunto["ic_alto"] < 1)
    print("todos los cultivos:")
    print(conjunto.round(2).to_string())
    filas = []
    for gr, n in pp.groupby("grupo")["cult"].nunique().items():
        if n < 15:
            continue
        sub = pp[pp["grupo"] == gr]
        cols = [c for c in m3 if sub[c].nunique() > 1]
        r = razones_momios(sub, cols, ["vcd", "ph", "pco2", "amonio", "glutamina", "temperatura"], n_boot=200)
        for v, fila in r.iterrows():
            filas.append({"grupo": f"{gr:.5f}", "cultivos": n, "variable": v, **fila.to_dict(),
                          "ic_excluye_1": (fila["ic_bajo"] > 1) or (fila["ic_alto"] < 1)})
    por_grupo = pd.DataFrame(filas)
    print("\npor grupo de volumen (IC por bootstrap de cultivos dentro del grupo):")
    print(por_grupo.pivot(index="variable", columns="grupo", values="or_por_iqr").round(2).to_string())
    print("IC que excluye 1:")
    print(por_grupo.pivot(index="variable", columns="grupo", values="ic_excluye_1").to_string())
    conjunto.round(3).to_csv(TABLAS / "or_por_iqr_conjunto.csv")
    por_grupo.round(3).to_csv(TABLAS / "or_por_iqr_por_grupo.csv", index=False)

    print("\n=== D. Temperatura: contra el azar y ajustada por crecimiento ===")
    t = temperatura(raw, et)
    print(f"{t['n']} cultivos con cambio de temperatura y shift: {t['observado']:.0%} a +-1 dia")
    print(f"  azar (permutando dias de cambio): {t['nulo_media']:.0%}, p = {t['p']:.3f}")
    print(f"  azar dentro de cada grupo de volumen: {t['nulo_grupo_media']:.0%}, p = {t['p_grupo']:.3f}")
    print("\nregresion del dia del shift:")
    print(t["reg"].round(3).to_string(index=False))
    print(f"coeficiente del dia de cambio de temperatura (con VCD dia 3): IC 95% "
          f"{t['ic_coef_tshift'][0]:.2f} a {t['ic_coef_tshift'][1]:.2f}")
    print(f"\ngrupo 0.00202: diferencia de dia del shift con vs sin cambio de T: cruda {t['g_crudo']:.2f}, "
          f"ajustada por VCD dia 3 {t['g_ajustado']:.2f} dias")
    print("VCD dia 3 por grupo de receta (min, mediana, max) -> poco traslape = el ajuste extrapola:")
    print(t["g_vcd"].round(3).to_string())
    pd.DataFrame([{"n": t["n"], "observado": t["observado"], "azar": t["nulo_media"], "p": t["p"],
                   "azar_dentro_grupo": t["nulo_grupo_media"], "p_grupo": t["p_grupo"],
                   "coef_tshift_ajustado_ic_bajo": t["ic_coef_tshift"][0],
                   "coef_tshift_ajustado_ic_alto": t["ic_coef_tshift"][1],
                   "grupo_00202_crudo": t["g_crudo"], "grupo_00202_ajustado": t["g_ajustado"]}]
                 ).round(3).to_csv(TABLAS / "temperatura_corregida.csv", index=False)
    t["reg"].round(3).to_csv(TABLAS / "temperatura_regresion.csv", index=False)


if __name__ == "__main__":
    main()
