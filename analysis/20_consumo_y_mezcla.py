"""
20 - Dos modificaciones del modelo 2 sugeridas por la revision de literatura (tarea 6).

A. CONSUMO SATURABLE POR CELULA
   El modelo 2 consume lactato como k*L*IVCD (primer orden en L). La
   literatura mide tasas especificas de consumo casi constantes mientras hay
   lactato (Lopez-Meza 2016; Okamura 2024; Ben Yahia 2017) o usa una forma
   saturable (Xing 2024; Wang 2026). Aqui:
       consumo = F * q_c * L/(K + L) * IVCD
   K grande  -> vuelve al primer orden (k = q_c/K): la prueba es anidada.
   K pequeno -> casi orden cero: consume a tasa constante hasta agotar.
   K NO se ajusta con los cultivos de prueba: en cada fold se elige de una
   rejilla con el error de ajuste en los cultivos de entrenamiento.
   Diagnostico previo sin modelo (en este script, parte 0): la tasa especifica
   de consumo despues del shift contra el lactato relativo al pico del cultivo.

B. EL SHIFT COMO EVENTO DEL CULTIVO (MEZCLA DE TRAYECTORIAS)
   El modelo 2 mezcla TASAS: un dia con F = 0.5 produce a la mitad y consume
   a la mitad. Eso equivale a leer F como fraccion de celulas que ya cambiaron
   (como Wang 2026). Nuestro F es otra cosa: la probabilidad de que el cultivo
   ya haya cambiado. La forma coherente con esa lectura es promediar
   TRAYECTORIAS: simular un cambio brusco en cada dia posible tau y pesar
   cada trayectoria por P(tau) = F_tau - F_(tau-1); el resto (1 - F_ultimo)
   es "no cambia en la ventana".

Las dos se evaluan con interruptor perfecto (shift real: aisla la cinetica)
y con F predicho (uso real), con la misma particion y validacion agrupada
por cultivo que los scripts 17 y 18. Tambien se repite el control de
estructura (riesgo solo con el dia) con el consumo nuevo, para ver si la
conclusion "la estructura sola no aporta" cambia.

SENSIBILIDAD (sugerida por la revision): con un desplazamiento c > 0 en la
forma saturable (el 0 de los datos min-max no seria lactato cero), la
conclusion podria cambiar. Se prueba forzando c = 0.05 y 0.10, y se mira que
c prefieren los datos.

REGLAS DE DECISION (fijadas antes de correr):
  A se adopta si, con interruptor perfecto Y con F predicho, el RMSE de los
    dias posteriores al shift mejora sobre el primer orden con IC pareado
    (bootstrap por cultivo) completamente < 0, y la K elegida no queda en el
    extremo superior de la rejilla en la mayoria de los folds.
  B se adopta si mejora el RMSE posterior al shift sobre la mezcla de tasas
    con IC < 0 y no empeora el acierto del dia exacto.
Si no se cumplen, se reporta igual: tambien es resultado.

Ejecutar:  python analysis/20_consumo_y_mezcla.py

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

# K en unidades de lactato normalizado (los picos suelen estar entre 0.1 y 0.4).
# np.inf = primer orden (la forma actual).
REJILLA_K = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf]


# Desplazamiento del lactato en la forma saturable. Los datos son min-max: el
# lactato real es a + b*L con "a" desconocido, asi que el L = 0 de los datos no
# es necesariamente lactato cero. C_OFF = 0 en el analisis principal; la
# sensibilidad (al final de main) lo cambia a 0.05 y 0.10 para ver si la
# conclusion depende de ese supuesto.
C_OFF = 0.0


def g(L, K):
    """Dependencia del consumo con el lactato. K = inf -> L (primer orden)."""
    L = np.maximum(L, 0.0)
    return L if np.isinf(K) else (L + C_OFF) / (K + L + C_OFF)


# ---------------------------------------------------------------- modelo 2 con g(L)
def matriz(s, F, mu, sd, K):
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    dXp = np.maximum(np.diff(X), 0)
    ivcd = (X[:-1] + X[1:]) / 2
    ph = np.nan_to_num((s["ph_prev"].to_numpy()[:-1] - mu[0]) / sd[0])
    co = np.nan_to_num((s["pco2_prev"].to_numpy()[:-1] - mu[1]) / sd[1])
    f = F[:-1]
    A = np.column_stack([(1 - f) * dXp, (1 - f) * ivcd, -f * g(L[:-1], K) * ivcd,
                         (1 - f) * dXp * ph, (1 - f) * dXp * co])
    return A, np.diff(L)


def ajustar(S, cultivos, et, mu, sd, K):
    bl = [matriz(S[c], m2.f_obs(S[c], et.loc[c]), mu, sd, K) for c in cultivos]
    A = np.vstack([b[0] for b in bl]); y = np.concatenate([b[1] for b in bl])
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    r = lsq_linear(A[ok], y[ok], bounds=([0, 0, 0, -np.inf, -np.inf], [np.inf] * 5))
    return r.x, float(np.mean((A[ok] @ r.x - y[ok]) ** 2))


def elegir_K(S, tr, et, mu, sd):
    """K con menor error de ajuste en ENTRENAMIENTO (nunca mira los cultivos de prueba)."""
    errores = {K: ajustar(S, tr, et, mu, sd, K)[1] for K in REJILLA_K}
    K = min(errores, key=errores.get)
    return K, ajustar(S, tr, et, mu, sd, K)[0]


def lam_pooling(S, cultivos, et, p, mu, sd, K):
    est_a, se2, res = [], [], []
    for c in cultivos:
        A, y = matriz(S[c], m2.f_obs(S[c], et.loc[c]), mu, sd, K)
        base, x = A[:, 1:] @ p[1:], A[:, 0]
        den = x @ x
        if den > 1e-10:
            a_i = x @ (y - base) / den
            r = y - base - a_i * x
            res.extend(r); est_a.append(a_i)
            se2.append(np.var(r) / den if len(r) > 1 else np.nan)
    return np.var(res) / max(np.nanvar(est_a) - np.nanmean(se2), 1e-6)


def alfa(s, F, p, mu, sd, lam, K, hasta=4):
    A, y = matriz(s.loc[:hasta], F[:len(s.loc[:hasta])], mu, sd, K)
    if len(y) == 0:
        return p[0]
    base = A[:, 1:] @ p[1:]
    x = A[:, 0]
    return max((x @ (y - base) + lam * p[0]) / (x @ x + lam), 0.0)


def simular(s, F, p, a, mu, sd, K):
    A, _ = matriz(s, F, mu, sd, K)
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    sim = np.full(len(L), np.nan)
    sim[0] = L[0]
    for t in range(len(L) - 1):
        ivcd = (X[t] + X[t + 1]) / 2
        prod = (1 - F[t]) * (a * max(X[t + 1] - X[t], 0) + p[1] * ivcd) + p[3] * A[t, 3] + p[4] * A[t, 4]
        sim[t + 1] = max(sim[t] + prod - F[t] * p[2] * g(np.array([sim[t]]), K)[0] * ivcd, 0.0)
    return sim


def mezcla_trayectorias(s, F, p, mu, sd, lam, K):
    """Promedio de trayectorias con cambio brusco en cada dia tau, pesadas por P(tau)."""
    d = s.index.to_numpy()
    Pc = np.concatenate([[0.0], F])
    P = np.diff(Pc)                      # P(tau = d_i)
    P = np.clip(P, 0, None)
    p_nunca = max(1 - F[-1], 0.0)
    total = np.zeros(len(d))
    for i, w in enumerate(P):
        if w < 1e-6:
            continue
        Fh = (d >= d[i]).astype(float)
        total += w * simular(s, Fh, p, alfa(s, Fh, p, mu, sd, lam, K), mu, sd, K)
    if p_nunca > 1e-6:
        F0 = np.zeros(len(d))
        total += p_nunca * simular(s, F0, p, alfa(s, F0, p, mu, sd, lam, K), mu, sd, K)
    tau_modal = d[int(np.argmax(P))] if P.max() >= p_nunca else np.nan
    return total / max(P.sum() + p_nunca, 1e-9), tau_modal


# ---------------------------------------------------------------- evaluacion
def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.nanmean(e ** 2))) if np.isfinite(e).any() else np.nan


def correr(S, pp, pp_ciego, et, cols, etiqueta):
    cult = np.array(sorted(S))
    filas, ks = [], []
    for tr_i, te_i in GroupKFold(n_splits=5).split(cult, groups=cult):
        tr, te = cult[tr_i], cult[te_i]
        todo = pd.concat([S[c] for c in tr])
        mu = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
        sd = (todo["ph_prev"].std(), todo["pco2_prev"].std())
        Fp = m2.riesgo_predicho(pp_ciego, pp[pp["cult"].isin(tr)], cols, te)
        K_sat, p_sat = elegir_K(S, tr, et, mu, sd)
        ks.append(K_sat)
        p_lin, _ = ajustar(S, tr, et, mu, sd, np.inf)
        modelos = {"lineal": (np.inf, p_lin), "saturable": (K_sat, p_sat)}
        lams = {k: lam_pooling(S, tr, et, pk, mu, sd, Kk) for k, (Kk, pk) in modelos.items()}
        for c in te:
            s = S[c]
            obs = s["L"].to_numpy()
            ds = et.loc[c, "dia"] if et.loc[c, "evento"] and et.loc[c, "dia"] < s.index.max() else np.nan
            post = s.index.to_numpy() > ds if np.isfinite(ds) else np.zeros(len(s), bool)
            F_pred = Fp[c].reindex(s.index).fillna(0).to_numpy() if c in Fp else np.zeros(len(s))
            F_or = m2.f_obs(s, et.loc[c])
            res = {"cult": c, "dia_shift": ds}
            for nombre, (K, p) in modelos.items():
                for fn, F in (("techo", F_or), ("pred", F_pred)):
                    sim = simular(s, F, p, alfa(s, F, p, mu, sd, lams[nombre], K), mu, sd, K)
                    res[f"{nombre}_{fn}_rmse"] = rmse(sim - obs)
                    res[f"{nombre}_{fn}_post"] = rmse((sim - obs)[post]) if post.any() else np.nan
                    res[f"{nombre}_{fn}_shift"] = m1.dia_shift_sim(s, sim)
                mez, tau = mezcla_trayectorias(s, F_pred, p, mu, sd, lams[nombre], K)
                res[f"{nombre}_mezcla_rmse"] = rmse(mez - obs)
                res[f"{nombre}_mezcla_post"] = rmse((mez - obs)[post]) if post.any() else np.nan
                res[f"{nombre}_mezcla_tau"] = tau
            filas.append(res)
    df = pd.DataFrame(filas).set_index("cult")
    df.columns = [f"{etiqueta}_{c}" if c != "dia_shift" else c for c in df.columns]
    return df, ks


def diagnostico_sin_modelo(S, et):
    """Tasa especifica de consumo despues del shift contra el lactato relativo al pico."""
    filas = []
    for c, s in S.items():
        if not et.loc[c, "evento"]:
            continue
        X, L, d = s["X"].to_numpy(), s["L"].to_numpy(), s.index.to_numpy()
        for i in range(len(d) - 1):
            ivcd = (X[i] + X[i + 1]) / 2
            if d[i] >= et.loc[c, "dia"] and ivcd > 0 and L.max() > 0:
                filas.append({"cult": c, "L_rel_pico": L[i] / L.max(), "q_consumo": -(L[i + 1] - L[i]) / ivcd})
    f = pd.DataFrame(filas)
    f["tramo"] = pd.cut(f["L_rel_pico"], [0, 0.25, 0.5, 0.75, 1.0001],
                        labels=["0-25%", "25-50%", "50-75%", "75-100%"])
    t = f.groupby("tramo", observed=True)["q_consumo"].agg(["count", "median"]).reset_index()
    rho = [spearmanr(gg["q_consumo"], gg["L_rel_pico"])[0] for _, gg in f.groupby("cult") if len(gg) >= 4]
    return t, float(np.nanmedian(rho)), len(rho), len(f), f["cult"].nunique()


def main() -> None:
    raw = cargar("Raw Data")
    raw_l, _ = c15.qc_gases(raw)
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")
    S = m1.series(raw_l, et, m1.CORTE_VCD)

    print("=== 0. Diagnostico sin modelo: consumo por celula despues del shift ===")
    t0, rho, n_rho, n_pares, n_cult = diagnostico_sin_modelo(S, et)
    print(f"{n_pares} intervalos posteriores al shift en {n_cult} cultivos")
    print(t0.round(4).to_string(index=False))
    print(f"Spearman dentro de cultivo (q vs lactato), mediana de {n_rho} cultivos: {rho:.2f}")
    t0.round(4).to_csv(TABLAS / "consumo_vs_lactato.csv", index=False)

    pp = est.persona_periodo(raw_l, et, adelanto=1)
    ciego = et.copy(); ciego["evento"] = False; ciego["tiempo"] = ciego["ultimo_dia"]
    pp_ciego = est.persona_periodo(raw_l, ciego, adelanto=1)
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]
    m3 = dia + ["temperatura", "bajo_T_hoy", "baja_T_manana"] + list(est.MEDICIONES.values()) + ["dvcd"]

    t, ks = correr(S, pp, pp_ciego, et, m3, "M2")
    tc, _ = correr(S, pp, pp_ciego, et, dia, "control")
    t = t.join(tc.drop(columns="dia_shift"))
    t.to_csv(TABLAS / "modelo2_consumo_mezcla_por_cultivo.csv")
    print("\nK elegida por fold (entrenamiento):", ks)

    lab = t["dia_shift"]
    def exacto(col):
        ok = lab.notna()
        return float((t.loc[ok, col] == lab[ok]).sum() / ok.sum())
    def pm1(col):
        ok = lab.notna()
        return float(((t.loc[ok, col] - lab[ok]).abs() <= 1).sum() / ok.sum())

    filas = []
    def comparar(nombre, a, b, metrica):
        d, ic = m1.ic_pareado(t[a], t[b])
        filas.append({"comparacion": nombre, "metrica": metrica, "modelo": t[a].mean(), "referencia": t[b].mean(),
                      "dif": d, "ic_bajo": ic[0], "ic_alto": ic[1]})
    for fn, txt in (("techo", "interruptor perfecto"), ("pred", "F predicho")):
        comparar(f"A saturable vs lineal ({txt})", f"M2_saturable_{fn}_post", f"M2_lineal_{fn}_post", "RMSE despues del shift")
        comparar(f"A saturable vs lineal ({txt})", f"M2_saturable_{fn}_rmse", f"M2_lineal_{fn}_rmse", "RMSE del perfil")
    for k in ("lineal", "saturable"):
        comparar(f"B mezcla de trayectorias vs de tasas ({k})", f"M2_{k}_mezcla_post", f"M2_{k}_pred_post", "RMSE despues del shift")
        comparar(f"B mezcla de trayectorias vs de tasas ({k})", f"M2_{k}_mezcla_rmse", f"M2_{k}_pred_rmse", "RMSE del perfil")
    comparar("control de estructura con consumo saturable (riesgo solo dia vs M2)", "control_saturable_pred_rmse",
             "M2_saturable_pred_rmse", "RMSE del perfil")
    res = pd.DataFrame(filas)
    res.round(4).to_csv(TABLAS / "modelo2_consumo_mezcla.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + res.round(4).to_string(index=False))

    tiempo = pd.DataFrame([
        {"metodo": "M2 lineal, mezcla de tasas", "exacto": exacto("M2_lineal_pred_shift"), "pm1": pm1("M2_lineal_pred_shift")},
        {"metodo": "M2 saturable, mezcla de tasas", "exacto": exacto("M2_saturable_pred_shift"), "pm1": pm1("M2_saturable_pred_shift")},
        {"metodo": "M2 mezcla de trayectorias (tau mas probable)", "exacto": exacto("M2_saturable_mezcla_tau"),
         "pm1": pm1("M2_saturable_mezcla_tau")},
    ])
    tiempo.round(4).to_csv(TABLAS / "modelo2_consumo_mezcla_tiempo.csv", index=False)
    print("\n" + tiempo.round(3).to_string(index=False))
    print(f"\nK en el extremo superior (primer orden) en {sum(np.isinf(k) for k in ks)} de {len(ks)} folds")

    # ------------------------------------------------ sensibilidad al desplazamiento
    global C_OFF
    sens = []
    for c_off in (0.05, 0.10):
        C_OFF = c_off
        ts, ks_s = correr(S, pp, pp_ciego, et, m3, "M2")
        for fn in ("techo", "pred"):
            d, ic = m1.ic_pareado(ts[f"M2_saturable_{fn}_post"], ts[f"M2_lineal_{fn}_post"])
            sens.append({"desplazamiento": c_off, "interruptor": fn, "dif_post": d, "ic_bajo": ic[0],
                         "ic_alto": ic[1], "K_por_fold": " ".join(str(k) for k in ks_s)})
    # que desplazamiento prefieren los datos: ajuste con todos los cultivos (descriptivo,
    # no es validacion) para cada desplazamiento, con la mejor K, relativo al primer orden
    todo = pd.concat([S[c] for c in S])
    mu = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
    sd = (todo["ph_prev"].std(), todo["pco2_prev"].std())
    C_OFF = 0.0
    mse_lineal = ajustar(S, sorted(S), et, mu, sd, np.inf)[1]
    pref = []
    for c_off in (0.0, 0.01, 0.02, 0.05, 0.10):
        C_OFF = c_off
        mejor = min(((K, ajustar(S, sorted(S), et, mu, sd, K)[1]) for K in REJILLA_K[:-1]), key=lambda r: r[1])
        pref.append({"desplazamiento": c_off, "mejor_K": mejor[0], "mse_relativo_a_primer_orden": mejor[1] / mse_lineal})
    pref = pd.DataFrame(pref)
    pref.round(4).to_csv(TABLAS / "modelo2_consumo_desplazamiento.csv", index=False)
    print("\nAjuste con todos los cultivos segun el desplazamiento (< 1 = mejor que primer orden):")
    print(pref.round(4).to_string(index=False))
    C_OFF = 0.0
    sens = pd.DataFrame(sens)
    sens.round(4).to_csv(TABLAS / "modelo2_consumo_sensibilidad.csv", index=False)
    print("\nSensibilidad al desplazamiento de la normalizacion (saturable - lineal, despues del shift):")
    print(sens.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
