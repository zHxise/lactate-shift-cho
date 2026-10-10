"""
21 - Interruptor endogeno: la forma cinetica del grupo de Kontoravdi (Kotidis 2019).

En Kotidis et al. (2019, Biotechnol Bioeng 116:1612, ec. 11), usada tambien
por Yu, del Rio Chanona y Kontoravdi (2026), la tasa especifica de lactato es

    q_Lac = (mu/Y_X,Lac - Y_Lac/Glc*q_Glc) * (Lacmax1 - L)/Lacmax1
            + m_lac * (Lacmax2 - L)/Lacmax2

No hay un interruptor externo: el propio lactato frena la produccion ligada
al crecimiento (primer termino) y, por encima de Lacmax2, el segundo termino
se vuelve consumo. Es decir, el disparador es la concentracion de lactato.

Traducido a nuestro balance (sin glucosa consumida: la glucosa esta
controlada y no hay datos de alimentacion, asi que el termino de glucosa no
se puede usar; queda solo la parte ligada al crecimiento):

    dL = alfa*dX+ * (1 - L/L1) + [modulacion pH, pCO2] + m*(1 - L/L2)*I

Desarrollando, m*(1 - L/L2)*I = m*I - (m/L2)*L*I, que es EXACTAMENTE beta*I -
k*L*I del modelo 1. Lo unico nuevo es la inhibicion de la produccion por el
lactato: -c*dX+*L, con c = alfa/L1 >= 0. Por eso este modelo es el modelo 1
mas un parametro, y la comparacion es anidada.

Con datos min-max el lactato tiene un desplazamiento desconocido
(L_real = a + b*L). (1 - L/L1) sigue siendo lineal en L con otro
intercepto, asi que la forma se conserva; L1 y L2 NO se interpretan como
concentraciones.

Pregunta: un modelo con disparador endogeno (el lactato) fecha el shift y
describe el perfil tan bien como el interruptor externo del modelo 2?

REGLA DE DECISION (fijada antes de correr): el modelo tipo Kotidis
reemplaza al modelo 2 solo si su RMSE del perfil no es peor que el del
modelo 2 (IC pareado que incluya 0 o sea < 0) Y su acierto del dia exacto
no es menor. Si solo mejora al modelo 1, se reporta como "el disparador
endogeno ayuda pero no alcanza".

Misma ventana, misma particion GroupKFold y misma estimacion de alfa por
cultivo (dias 1-4, Bayes empirico) que el script 17.

Ejecutar:  python analysis/21_modelo_kotidis.py

Desarrollado por Arturo Rodriguez.
"""

import warnings
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.model_selection import GroupKFold

from _comun import TABLAS, cargar

warnings.filterwarnings("ignore")
c15 = import_module("15_correcciones_auditoria")
m1 = import_module("17_modelo1_crecimiento")


def matriz(s, mu, sd):
    """Columnas: alfa, beta, k, a_p, a_c (como el modelo 1) y c (inhibicion por lactato)."""
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    dXp = np.maximum(np.diff(X), 0)
    ivcd = (X[:-1] + X[1:]) / 2
    ph = np.nan_to_num((s["ph_prev"].to_numpy()[:-1] - mu[0]) / sd[0])
    co = np.nan_to_num((s["pco2_prev"].to_numpy()[:-1] - mu[1]) / sd[1])
    A = np.column_stack([dXp, ivcd, -L[:-1] * ivcd, dXp * ph, dXp * co, -dXp * L[:-1]])
    return A, np.diff(L)


def ajustar(S, cultivos, mu, sd):
    A = np.vstack([matriz(S[c], mu, sd)[0] for c in cultivos])
    y = np.concatenate([matriz(S[c], mu, sd)[1] for c in cultivos])
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    lo = [0, 0, 0, -np.inf, -np.inf, 0]
    return lsq_linear(A[ok], y[ok], bounds=(lo, [np.inf] * 6)).x


def lam_pooling(S, cultivos, p, mu, sd):
    est_a, se2, res = [], [], []
    for c in cultivos:
        A, y = matriz(S[c], mu, sd)
        base, x = A[:, 1:] @ p[1:], A[:, 0]
        den = x @ x
        if den > 1e-10:
            a_i = x @ (y - base) / den
            r = y - base - a_i * x
            res.extend(r); est_a.append(a_i)
            se2.append(np.var(r) / den if len(r) > 1 else np.nan)
    return np.var(res) / max(np.nanvar(est_a) - np.nanmean(se2), 1e-6)


def alfa(s, p, mu, sd, lam, hasta=4):
    A, y = matriz(s.loc[:hasta], mu, sd)
    if len(y) == 0:
        return p[0]
    x, base = A[:, 0], A[:, 1:] @ p[1:]
    return max((x @ (y - base) + lam * p[0]) / (x @ x + lam), 0.0)


def simular(s, p, a, mu, sd, i0=0, n=None):
    """El lactato simulado (no el observado) alimenta la inhibicion y el consumo."""
    A, _ = matriz(s, mu, sd)
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    fin = len(L) if n is None else min(len(L), i0 + n + 1)
    sim = np.full(len(L), np.nan)
    sim[i0] = L[i0]
    for t in range(i0, fin - 1):
        dXp, ivcd = A[t, 0], A[t, 1]
        prod = a * dXp + p[3] * A[t, 3] + p[4] * A[t, 4] + p[1] * ivcd - p[5] * dXp * sim[t]
        sim[t + 1] = max(sim[t] + prod - p[2] * sim[t] * ivcd, 0.0)
    return sim


def main() -> None:
    raw = cargar("Raw Data")
    raw_l, _ = c15.qc_gases(raw)
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")
    S = m1.series(raw_l, et, m1.CORTE_VCD)

    cult = np.array(sorted(S))
    filas, params = [], []
    for tr_i, te_i in GroupKFold(n_splits=5).split(cult, groups=cult):
        tr, te = cult[tr_i], cult[te_i]
        todo = pd.concat([S[c] for c in tr])
        mu = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
        sd = (todo["ph_prev"].std(), todo["pco2_prev"].std())
        p = ajustar(S, tr, mu, sd)
        params.append(p)
        lam = lam_pooling(S, tr, p, mu, sd)
        for c in te:
            s = S[c]
            obs = s["L"].to_numpy()
            a = alfa(s, p, mu, sd, lam)
            sim = simular(s, p, a, mu, sd)
            e2 = [simular(s, p, a, mu, sd, i0=i, n=2)[i + 2] - obs[i + 2] for i in range(len(obs) - 2)]
            filas.append({"cult": c, "K_rmse": float(np.sqrt(np.nanmean((sim - obs) ** 2))),
                          "K_rmse2": float(np.sqrt(np.nanmean(np.square(e2)))) if e2 else np.nan,
                          "K_shift": m1.dia_shift_sim(s, sim)})
    k = pd.DataFrame(filas).set_index("cult")
    m2t = pd.read_csv(TABLAS / "modelo2_por_cultivo.csv", index_col="cult")
    t = m2t[["rmse_condicionado", "rmse2_modelo", "dia_shift_simulado", "dia_shift_etiqueta",
             "M2_rmse_pred", "M2_rmse2_pred", "M2_shift_pred"]].join(k)
    t.to_csv(TABLAS / "modelo_kotidis_por_cultivo.csv")

    # Diagnostico: el mismo ajuste con todos los cultivos y c SIN restriccion de signo.
    # Si c sale negativo, la produccion ligada al crecimiento SUBE con el lactato en los
    # datos agrupados (lo contrario de una inhibicion), probablemente porque los cultivos
    # que mas producen son tambien los que mas acumulan (diferencias entre cultivos).
    todo = pd.concat([S[c] for c in cult])
    mu = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
    sd = (todo["ph_prev"].std(), todo["pco2_prev"].std())
    A = np.vstack([matriz(S[c], mu, sd)[0] for c in cult])
    y = np.concatenate([matriz(S[c], mu, sd)[1] for c in cult])
    okf = np.isfinite(A).all(axis=1) & np.isfinite(y)
    libre = lsq_linear(A[okf], y[okf], bounds=([0, 0, 0, -np.inf, -np.inf, -np.inf], [np.inf] * 6)).x
    print(f"c sin restriccion de signo (todos los cultivos): {libre[5]:+.3f}")
    pd.DataFrame({"parametro": ["alfa", "beta", "k", "a_p", "a_c", "c"], "libre": libre}).round(4).to_csv(
        TABLAS / "modelo_kotidis_c_libre.csv", index=False)

    p_med = np.mean(params, axis=0)
    print("parametros promedio (alfa, beta, k, a_p, a_c, c):", np.round(p_med, 4))
    print(f"c (inhibicion de la produccion por lactato) > 0 en {sum(pp[5] > 1e-9 for pp in params)} de 5 folds")

    lab = t["dia_shift_etiqueta"]
    ok = lab.notna()
    def tiempo(col):
        e = (t.loc[ok, col] - lab[ok])
        return float((e == 0).sum() / ok.sum()), float((e.abs() <= 1).sum() / ok.sum()), int(e.notna().sum())

    filas = []
    for nombre, col, col2, sh in (("M1 sin interruptor", "rmse_condicionado", "rmse2_modelo", "dia_shift_simulado"),
                                  ("tipo Kotidis (disparador: lactato)", "K_rmse", "K_rmse2", "K_shift"),
                                  ("M2 interruptor externo (riesgo)", "M2_rmse_pred", "M2_rmse2_pred", "M2_shift_pred")):
        ex, pm, det = tiempo(sh)
        filas.append({"modelo": nombre, "rmse_perfil": t[col].mean(), "rmse_2dias": t[col2].mean(),
                      "exacto": ex, "pm1": pm, "detectados": det, "cultivos": int(ok.sum())})
    res = pd.DataFrame(filas)
    comps = []
    for nombre, a_, b_ in (("Kotidis - M1 (perfil)", "K_rmse", "rmse_condicionado"),
                           ("Kotidis - M1 (2 dias)", "K_rmse2", "rmse2_modelo"),
                           ("Kotidis - M2 (perfil)", "K_rmse", "M2_rmse_pred"),
                           ("Kotidis - M2 (2 dias)", "K_rmse2", "M2_rmse2_pred")):
        d, ic = m1.ic_pareado(t[a_], t[b_])
        comps.append({"comparacion": nombre, "dif": d, "ic_bajo": ic[0], "ic_alto": ic[1]})
    comp = pd.DataFrame(comps)
    res.round(4).to_csv(TABLAS / "modelo_kotidis_resumen.csv", index=False)
    comp.round(4).to_csv(TABLAS / "modelo_kotidis_comparaciones.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + res.round(4).to_string(index=False))
    print("\n" + comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
