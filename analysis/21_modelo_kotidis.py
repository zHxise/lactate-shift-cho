"""
21 - Disparador interno: la forma cinetica del grupo de Kontoravdi (Kotidis 2019).

En Kotidis et al. (2019, Biotechnol Bioeng 116:1612, ec. 11), usada tambien
por Yu, del Rio Chanona y Kontoravdi (2026), la tasa especifica de lactato es

    q_Lac = (mu/Y_X,Lac - Y_Lac/Glc*q_Glc) * (Lacmax1 - L)/Lacmax1
            + m_lac * (Lacmax2 - L)/Lacmax2

No hay interruptor externo: el propio lactato frena la produccion ligada al
crecimiento y, por encima de Lacmax2, el segundo termino se vuelve consumo.

Traducido a nuestro balance (sin glucosa consumida: la glucosa esta
controlada y no hay datos de alimentacion; si el consumo de glucosa por
celula fuera constante, ese termino ya cabe en beta*I - k*L*I):

    dL = alfa*dX+ + a_p*dX+*pH' + a_c*dX+*pCO2' + beta*I - k*L*I  - c*dX+*L

m*(1 - L/L2)*I = m*I - (m/L2)*L*I es beta*I - k*L*I del modelo 1, asi que lo
nuevo es solo c: cuanto frena el lactato la produccion ligada al crecimiento.
(Detalle: con desplazamiento desconocido de la normalizacion, beta podria ser
negativo; aqui se mantiene beta >= 0 como en el modelo 1. La inhibicion se
aplica a alfa*dX+ y no a los terminos de pH y pCO2.)

CORRECCION (auditoria de la tarea 7). La primera version ajustaba una sola
alfa para todos los cultivos y obtenia c = 0 (o negativo sin restriccion):
"el lactato no frena la produccion". Era un artefacto ENTRE cultivos: los
cultivos que producen mas lactato por celula tambien acumulan mas, y una
regresion agrupada lo lee como "la produccion sube con el lactato" (efecto
tipo Simpson). Con alfa propia por cultivo (efectos fijos, m1.ajustar_ef),
c se estima con la variacion DENTRO de cada cultivo. Este script reporta las
dos versiones para que se vea el cambio.

IMPORTANTE: este resultado es POST HOC (el estimador y las variantes se
eligieron despues de ver datos, durante una auditoria). Sus intervalos son
optimistas y debe tratarse como hipotesis para confirmar con otros datos.

Controles de que la inhibicion no es solo "tiempo": se repite con el dia del
cultivo y con la IVCD acumulada en lugar del lactato.

Modelos comparados (todos con alfa por cultivo, mismas particiones):
  M1 (script 17), M2 (script 18), tipo Kotidis sin interruptor, variantes de
  control, y M2 + inhibicion (exploratorio).

Ejecutar:  python analysis/21_modelo_kotidis.py

Desarrollado por Arturo Rodriguez.
"""

import warnings
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from _comun import TABLAS, cargar

warnings.filterwarnings("ignore")
est = import_module("12_estado_celular")
c15 = import_module("15_correcciones_auditoria")
m1 = import_module("17_modelo1_crecimiento")
m2 = import_module("18_modelo2_dos_estados")


def inhibidor(s, tipo, L):
    """Variable que frena la produccion: lactato (estado), dia o IVCD acumulada (controles)."""
    if tipo == "lactato":
        return L
    d = s.index.to_numpy().astype(float)
    if tipo == "dia":
        return d / 10.0
    X = s["X"].to_numpy()
    return np.concatenate([[0.0], np.cumsum((X[:-1] + X[1:]) / 2)])  # IVCD acumulada


def matriz(s, mu, sd, tipo="lactato", F=None):
    """Columnas: alfa, beta, k, a_p, a_c, c. Con F (modelo 2 + inhibicion) se aplican los dos estados."""
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    dXp = np.maximum(np.diff(X), 0)
    ivcd = (X[:-1] + X[1:]) / 2
    ph = np.nan_to_num((s["ph_prev"].to_numpy()[:-1] - mu[0]) / sd[0])
    co = np.nan_to_num((s["pco2_prev"].to_numpy()[:-1] - mu[1]) / sd[1])
    z = inhibidor(s, tipo, L)[:-1]
    if F is None:
        A = np.column_stack([dXp, ivcd, -L[:-1] * ivcd, dXp * ph, dXp * co, -dXp * z])
    else:
        f = F[:-1]
        A = np.column_stack([(1 - f) * dXp, (1 - f) * ivcd, -f * L[:-1] * ivcd,
                             (1 - f) * dXp * ph, (1 - f) * dXp * co, -(1 - f) * dXp * z])
    return A, np.diff(L)


LO = [0, 0, -np.inf, -np.inf, 0]  # beta, k, a_p, a_c, c (alfa por cultivo va aparte)


def ajustar(S, cultivos, mu, sd, tipo, et=None, con_F=False):
    bl = [matriz(S[c], mu, sd, tipo, m2.f_obs(S[c], et.loc[c]) if con_F else None) for c in cultivos]
    return m1.ajustar_ef(bl, LO)[0]


def lam_pooling(S, cultivos, p, mu, sd, tipo, et=None, con_F=False):
    est_a, se2, res = [], [], []
    for c in cultivos:
        A, y = matriz(S[c], mu, sd, tipo, m2.f_obs(S[c], et.loc[c]) if con_F else None)
        base, x = A[:, 1:] @ p[1:], A[:, 0]
        den = x @ x
        if den > 1e-10:
            a_i = x @ (y - base) / den
            r = y - base - a_i * x
            res.extend(r); est_a.append(a_i)
            se2.append(np.var(r) / den if len(r) > 1 else np.nan)
    return np.var(res) / max(np.nanvar(est_a) - np.nanmean(se2), 1e-6)


def alfa(s, p, mu, sd, lam, tipo, F=None, hasta=4):
    n = len(s.loc[:hasta])
    A, y = matriz(s.loc[:hasta], mu, sd, tipo, None if F is None else F[:n])
    if len(y) == 0:
        return p[0]
    x, base = A[:, 0], A[:, 1:] @ p[1:]
    return max((x @ (y - base) + lam * p[0]) / (x @ x + lam), 0.0)


def simular(s, p, a, mu, sd, tipo, F=None, i0=0, n=None):
    """El lactato SIMULADO (no el observado) alimenta la inhibicion y el consumo."""
    A, _ = matriz(s, mu, sd, tipo, F)
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    dos_estados = F is not None
    F = np.zeros(len(L)) if F is None else F
    zfijo = inhibidor(s, tipo, L)
    fin = len(L) if n is None else min(len(L), i0 + n + 1)
    sim = np.full(len(L), np.nan)
    sim[i0] = L[i0]
    for t in range(i0, fin - 1):
        dXp = max(X[t + 1] - X[t], 0)
        ivcd = (X[t] + X[t + 1]) / 2
        z = sim[t] if tipo == "lactato" else zfijo[t]
        prod = (1 - F[t]) * (a * dXp + p[1] * ivcd - p[5] * dXp * z) + p[3] * A[t, 3] + p[4] * A[t, 4]
        consumo = (F[t] if dos_estados else 1.0) * p[2] * sim[t] * ivcd
        sim[t + 1] = max(sim[t] + prod - consumo, 0.0)
    return sim


def rm(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.nanmean(e ** 2))) if np.isfinite(e).any() else np.nan


def main() -> None:
    raw = cargar("Raw Data")
    raw_l, _ = c15.qc_gases(raw)
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")
    S = m1.series(raw_l, et, m1.CORTE_VCD)
    cult = np.array(sorted(S))

    # ------------------------------------------------ 0. diagnosticos (todos los cultivos)
    todo = pd.concat([S[c] for c in cult])
    mu0 = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
    sd0 = (todo["ph_prev"].std(), todo["pco2_prev"].std())
    bl = [matriz(S[c], mu0, sd0) for c in cult]
    A = np.vstack([b[0] for b in bl]); y = np.concatenate([b[1] for b in bl])
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    c_entre = lsq_linear(A[ok], y[ok], bounds=([0, 0, 0, -np.inf, -np.inf, -np.inf], [np.inf] * 6)).x[5]
    c_dentro = m1.ajustar_ef(bl, [0, 0, -np.inf, -np.inf, -np.inf])[0][5]
    # sin modelo: lactato producido por unidad de crecimiento antes del shift contra el lactato
    rho = []
    for c in cult:
        if not et.loc[c, "evento"]:
            continue
        s = S[c].loc[:et.loc[c, "dia"]]
        L, X = s["L"].to_numpy(), s["X"].to_numpy()
        dX, dL = np.diff(X), np.diff(L)
        m = dX > 1e-6
        if m.sum() >= 4:
            rho.append(spearmanr(dL[m] / dX[m], L[:-1][m])[0])
    rho = np.array(rho)
    diag = pd.DataFrame([{"c_una_alfa_comun": c_entre, "c_alfa_por_cultivo": c_dentro,
                          "rho_mediana_sin_modelo": np.nanmedian(rho), "frac_rho_negativo": np.mean(rho < 0),
                          "cultivos_rho": len(rho)}])
    diag.round(4).to_csv(TABLAS / "modelo_kotidis_diagnostico.csv", index=False)
    print("c con una alfa comun (entre cultivos):", round(c_entre, 3))
    print("c con alfa por cultivo (dentro de cultivos):", round(c_dentro, 3))
    print(f"sin modelo: produccion por crecimiento vs lactato antes del shift, Spearman mediana "
          f"{np.nanmedian(rho):.2f}, negativa en {np.mean(rho < 0):.0%} de {len(rho)} cultivos")

    # ------------------------------------------------ 1. validacion cruzada
    pp = est.persona_periodo(raw_l, et, adelanto=1)
    ciego = et.copy(); ciego["evento"] = False; ciego["tiempo"] = ciego["ultimo_dia"]
    pp_ciego = est.persona_periodo(raw_l, ciego, adelanto=1)
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]
    m3 = dia + ["temperatura", "bajo_T_hoy", "baja_T_manana"] + list(est.MEDICIONES.values()) + ["dvcd"]

    variantes = {"K": ("lactato", False), "Kdia": ("dia", False), "Kivcd": ("ivcd", False),
                 "M2inh": ("lactato", True)}
    filas, cs = [], []
    for tr_i, te_i in GroupKFold(n_splits=5).split(cult, groups=cult):
        tr, te = cult[tr_i], cult[te_i]
        tt = pd.concat([S[c] for c in tr])
        mu = (tt["ph_prev"].mean(), tt["pco2_prev"].mean())
        sd = (tt["ph_prev"].std(), tt["pco2_prev"].std())
        Fp = m2.riesgo_predicho(pp_ciego, pp[pp["cult"].isin(tr)], m3, te)
        ajustes = {}
        for v, (tipo, con_F) in variantes.items():
            p = ajustar(S, tr, mu, sd, tipo, et, con_F)
            ajustes[v] = (p, lam_pooling(S, tr, p, mu, sd, tipo, et, con_F))
        cs.append({"fold": len(cs) + 1, **{f"c_{v}": ajustes[v][0][5] for v in variantes}})
        for c in te:
            s = S[c]
            obs = s["L"].to_numpy()
            F = Fp[c].reindex(s.index).fillna(0).to_numpy() if c in Fp else np.zeros(len(s))
            res = {"cult": c}
            for v, (tipo, con_F) in variantes.items():
                p, lam = ajustes[v]
                Fv = F if con_F else None
                a = alfa(s, p, mu, sd, lam, tipo, Fv)
                sim = simular(s, p, a, mu, sd, tipo, Fv)
                e2 = [simular(s, p, a, mu, sd, tipo, Fv, i0=i, n=2)[i + 2] - obs[i + 2] for i in range(len(obs) - 2)]
                res[f"{v}_rmse"], res[f"{v}_rmse2"] = rm(sim - obs), rm(e2)
                res[f"{v}_shift"] = m1.dia_shift_sim(s, sim)
            filas.append(res)
    k = pd.DataFrame(filas).set_index("cult")
    m2t = pd.read_csv(TABLAS / "modelo2_por_cultivo.csv", index_col="cult")
    t = m2t[["rmse_condicionado", "rmse2_modelo", "dia_shift_simulado", "dia_shift_etiqueta",
             "M2_rmse_pred", "M2_rmse2_pred", "M2_rmse2_congelado", "M2_shift_pred"]].join(k)
    t.to_csv(TABLAS / "modelo_kotidis_por_cultivo.csv")
    pd.DataFrame(cs).round(4).to_csv(TABLAS / "modelo_kotidis_c_por_fold.csv", index=False)
    print("\nc por fold (entrenamiento):")
    print(pd.DataFrame(cs).round(3).to_string(index=False))

    lab = t["dia_shift_etiqueta"]
    okl = lab.notna()
    def tiempo(col):
        e = (t.loc[okl, col] - lab[okl])
        return float((e == 0).sum() / okl.sum()), float((e.abs() <= 1).sum() / okl.sum())

    filas = []
    for nombre, col, col2, sh in (
            ("M1 sin interruptor", "rmse_condicionado", "rmse2_modelo", "dia_shift_simulado"),
            ("M2 interruptor externo (riesgo)", "M2_rmse_pred", "M2_rmse2_pred", "M2_shift_pred"),
            ("tipo Kotidis, sin interruptor (inhibe el lactato)", "K_rmse", "K_rmse2", "K_shift"),
            ("control: inhibe el dia del cultivo", "Kdia_rmse", "Kdia_rmse2", "Kdia_shift"),
            ("control: inhibe la IVCD acumulada", "Kivcd_rmse", "Kivcd_rmse2", "Kivcd_shift"),
            ("M2 + inhibicion por lactato (exploratorio)", "M2inh_rmse", "M2inh_rmse2", "M2inh_shift")):
        ex, pm = tiempo(sh)
        filas.append({"modelo": nombre, "rmse_perfil": t[col].mean(), "rmse_2dias": t[col2].mean(),
                      "exacto": ex, "pm1": pm})
    res = pd.DataFrame(filas)
    comps = []
    for nombre, a_, b_ in (("Kotidis - M1 (perfil)", "K_rmse", "rmse_condicionado"),
                           ("Kotidis - M2 (perfil)", "K_rmse", "M2_rmse_pred"),
                           ("Kotidis - M2 congelado (2 dias)", "K_rmse2", "M2_rmse2_congelado"),
                           ("M2 + inhibicion - M2 (perfil)", "M2inh_rmse", "M2_rmse_pred")):
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
