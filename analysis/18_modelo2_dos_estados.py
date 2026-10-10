"""
18 - Estructura 2: dos estados (produccion / consumo) con transicion predicha.

Version simple de la estructura recomendada en la propuesta de modelos (tarea
3 de Claude Science). Une los dos analisis que ya funcionaban:
- el modelo de riesgo diario (scripts 12 y 15) da, cada dia, la probabilidad
  de que el cultivo ya haya pasado a consumir lactato;
- el balance de lactato del script 17 da cuanto se produce y se consume.

    dL + d*L = (1 - F) * [alfa*dX+*(1 + a_p*pH' + a_c*pCO2') + beta*I]  -  F * k*L*I

    F = fraccion del cultivo en estado de consumo.
        Al AJUSTAR (cultivos de entrenamiento): F observado (0 antes del
        shift de la etiqueta, 1 despues).
        Al EVALUAR (cultivos de prueba): F PREDICHO por el modelo de riesgo
        entrenado sin esos cultivos: F_t = 1 - prod_{s<=t} (1 - h_s),
        la probabilidad acumulada de que el shift ya haya ocurrido.

Diferencia con la estructura 1: ahi el consumo esta SIEMPRE activo y el
shift sale solo cuando el crecimiento se frena; aqui el consumo se "enciende"
segun las condiciones del cultivo (VCD, pH, pCO2, etc. del modelo de riesgo).

Variante "oraculo": F observado tambien en prueba. No es un modelo usable
(usa la etiqueta), es el techo: cuanto ganaria la estructura 2 si el
interruptor se predijera perfecto.

Regla de decision fijada ANTES de correr (de la propuesta de modelos):
la estructura 2 se prefiere a la 1 solo si
  (a) el IC pareado de la diferencia de RMSE a 2 dias (M2 - M1) es < 0, y
  (b) acierta mas dias del shift a +-1 dia.

Modo explicativo, igual que el script 17: se usa la VCD medida y las
variables del modelo de riesgo del dia t; no es un pronostico puro de la
biomasa.

Prueba de "donde actua el pH": se repite quitando el pH de la cinetica
(solo en el interruptor) y quitandolo del interruptor (solo en la cinetica).

Ejecutar:  python analysis/18_modelo2_dos_estados.py

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
est = import_module("12_estado_celular")
c15 = import_module("15_correcciones_auditoria")
m1 = import_module("17_modelo1_crecimiento")
SEED = 0
D = 0.0  # dilucion principal; el script 17 mostro que 0-3% no cambia conclusiones


def matriz2(s, F, mu, sd, ph_en_cinetica=True):
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    dXp = np.maximum(np.diff(X), 0)
    ivcd = (X[:-1] + X[1:]) / 2
    ph = np.nan_to_num((s["ph_prev"].to_numpy()[:-1] - mu[0]) / sd[0])
    co = np.nan_to_num((s["pco2_prev"].to_numpy()[:-1] - mu[1]) / sd[1])
    f = F[:-1]
    A = np.column_stack([(1 - f) * dXp, (1 - f) * ivcd, -f * L[:-1] * ivcd,
                         (1 - f) * dXp * ph * (1 if ph_en_cinetica else 0), (1 - f) * dXp * co])
    return A, np.diff(L) + D * L[:-1]


def f_obs(s, et_c):
    return (s.index.to_numpy() >= et_c["dia"]).astype(float) if et_c["evento"] else np.zeros(len(s))


def ajustar2(S, cultivos, et, mu, sd, ph_cin):
    bloques = [matriz2(S[c], f_obs(S[c], et.loc[c]), mu, sd, ph_cin) for c in cultivos]
    A = np.vstack([b[0] for b in bloques]); y = np.concatenate([b[1] for b in bloques])
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    return lsq_linear(A[ok], y[ok], bounds=([0, 0, 0, -np.inf, -np.inf], [np.inf] * 5)).x


def alfa_cond2(s, F, p, mu, sd, lam, ph_cin, hasta=4):
    A, y = matriz2(s.loc[:hasta], F[:len(s.loc[:hasta])], mu, sd, ph_cin)
    if len(y) == 0:
        return p[0]
    x = A[:, 0]
    base = A[:, 1] * p[1] + A[:, 2] * p[2] + A[:, 3] * p[3] + A[:, 4] * p[4]
    return max((x @ (y - base) + lam * p[0]) / (x @ x + lam), 0.0)


def simular2(s, F, p, alfa, mu, sd, ph_cin, i0=0, n=None):
    A, _ = matriz2(s, F, mu, sd, ph_cin)
    L = s["L"].to_numpy()
    fin = len(L) if n is None else min(len(L), i0 + n + 1)
    sim = np.full(len(L), np.nan)
    sim[i0] = L[i0]
    X = s["X"].to_numpy()
    for t in range(i0, fin - 1):
        ivcd = (X[t] + X[t + 1]) / 2
        prod = (1 - F[t]) * (alfa * max(X[t + 1] - X[t], 0) + p[1] * ivcd) + p[3] * A[t, 3] + p[4] * A[t, 4]
        sim[t + 1] = max(sim[t] + prod - F[t] * p[2] * sim[t] * ivcd - D * sim[t], 0.0)
    return sim


def riesgo_predicho(pp_todos, pp_tr, cols, cultivos_te):
    """F_t para los cultivos de prueba con el modelo de riesgo entrenado sin ellos."""
    m = c15.modelo_robusto().fit(pp_tr[cols], pp_tr["y"])
    te = pp_todos[pp_todos["cult"].isin(cultivos_te)].copy()
    te["h"] = m.predict_proba(te[cols])[:, 1]
    out = {}
    for c, g in te.groupby("cult"):
        g = g.sort_values("dia")
        out[c] = pd.Series(1 - np.cumprod(1 - g["h"].to_numpy()), index=g["dia"].to_numpy())
    return out


def correr(S, pp, pp_ciego, et, cols_riesgo, ph_cin, etiqueta):
    cult = np.array(sorted(S))
    filas = []
    for tr_i, te_i in GroupKFold(n_splits=5).split(cult, groups=cult):
        tr, te = cult[tr_i], cult[te_i]
        todo = pd.concat([S[c] for c in tr])
        mu = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
        sd = (todo["ph_prev"].std(), todo["pco2_prev"].std())
        p = ajustar2(S, tr, et, mu, sd, ph_cin)
        lam = m1.lambda_pooling(S, tr, p, D, mu, sd)
        Fp = riesgo_predicho(pp_ciego, pp[pp["cult"].isin(tr)], cols_riesgo, te)
        for c in te:
            s = S[c]
            obs = s["L"].to_numpy()
            F_pred = Fp[c].reindex(s.index).fillna(0).to_numpy() if c in Fp else np.zeros(len(s))
            F_or = f_obs(s, et.loc[c])
            res = {"cult": c}
            for nombre, F in (("pred", F_pred), ("oraculo", F_or)):
                a = alfa_cond2(s, F, p, mu, sd, lam, ph_cin)
                sim = simular2(s, F, p, a, mu, sd, ph_cin)
                e2 = [simular2(s, F, p, a, mu, sd, ph_cin, i0=i, n=2)[i + 2] - obs[i + 2] for i in range(len(obs) - 2)]
                res[f"rmse_{nombre}"] = float(np.sqrt(np.nanmean((sim - obs) ** 2)))
                res[f"rmse2_{nombre}"] = float(np.sqrt(np.nanmean(np.square(e2)))) if e2 else np.nan
                res[f"shift_{nombre}"] = m1.dia_shift_sim(s, sim)
            filas.append(res)
    df = pd.DataFrame(filas).set_index("cult")
    df.columns = [f"{etiqueta}_{c}" for c in df.columns]
    return df


def main() -> None:
    raw = cargar("Raw Data")
    raw_l, _ = c15.qc_gases(raw)
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")
    S = m1.series(raw_l, et, m1.CORTE_VCD)

    pp = est.persona_periodo(raw_l, et, adelanto=1)
    ciego = et.copy(); ciego["evento"] = False; ciego["tiempo"] = ciego["ultimo_dia"]
    pp_ciego = est.persona_periodo(raw_l, ciego, adelanto=1)  # todas las filas, sin cortar en el evento
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]
    m3 = dia + ["temperatura", "bajo_T_hoy", "baja_T_manana"] + list(est.MEDICIONES.values()) + ["dvcd"]
    m3_sin_ph = [c for c in m3 if c != "ph"]

    m1_ev = pd.read_csv(TABLAS / "modelo1_por_cultivo.csv", index_col="cult")
    base = correr(S, pp, pp_ciego, et, m3, True, "M2")
    v_solo_interruptor = correr(S, pp, pp_ciego, et, m3, False, "pHinterruptor")
    v_solo_cinetica = correr(S, pp, pp_ciego, et, m3_sin_ph, True, "pHcinetica")
    # control negativo: el interruptor se alimenta con un riesgo que solo conoce el dia.
    # Si M2 ganara igual con esto, la ventaja vendria de la forma de dos estados y no
    # de las mediciones del cultivo.
    v_solo_dia = correr(S, pp, pp_ciego, et, dia, True, "solodia")
    t = m1_ev.join(base).join(v_solo_interruptor).join(v_solo_cinetica).join(v_solo_dia)
    t.to_csv(TABLAS / "modelo2_por_cultivo.csv")

    lab = t["dia_shift_etiqueta"]
    def acierto(col):
        ok = lab.notna() & t[col].notna()
        return np.mean(np.abs(t.loc[ok, col] - lab[ok]) <= 1), int(ok.sum()), int(lab.notna().sum())

    def fila(nombre, rmse, rmse2, shift):
        dif, ic = m1.ic_pareado(t[rmse2], t["rmse2_modelo"])
        a, n_ok, n_lab = acierto(shift)
        return {"modelo": nombre, "rmse_perfil": t[rmse].mean(), "rmse_2dias": t[rmse2].mean(),
                "dif_2dias_vs_M1": dif, "ic_bajo": ic[0], "ic_alto": ic[1],
                "shift_pm1": a, "shift_detectado": n_ok, "shift_en_ventana": n_lab}

    a1, n1, nl = acierto("dia_shift_simulado")
    filas = [{"modelo": "M1 sin interruptor (script 17)", "rmse_perfil": t["rmse_condicionado"].mean(),
              "rmse_2dias": t["rmse2_modelo"].mean(), "shift_pm1": a1, "shift_detectado": n1, "shift_en_ventana": nl},
             {"modelo": "perfil promedio", "rmse_perfil": t["rmse_perfil_promedio"].mean(),
              "rmse_2dias": t["rmse2_baseline"].mean()},
             fila("M2 dos estados (F predicho)", "M2_rmse_pred", "M2_rmse2_pred", "M2_shift_pred"),
             fila("M2 oraculo (F observado; techo)", "M2_rmse_oraculo", "M2_rmse2_oraculo", "M2_shift_oraculo"),
             fila("M2 con pH solo en el interruptor", "pHinterruptor_rmse_pred", "pHinterruptor_rmse2_pred",
                  "pHinterruptor_shift_pred"),
             fila("M2 con pH solo en la cinetica", "pHcinetica_rmse_pred", "pHcinetica_rmse2_pred",
                  "pHcinetica_shift_pred"),
             fila("M2 control: riesgo solo con el dia", "solodia_rmse_pred", "solodia_rmse2_pred",
                  "solodia_shift_pred")]
    res = pd.DataFrame(filas)
    res.round(4).to_csv(TABLAS / "modelo2_resumen.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    g = res.iloc[2]
    print(f"\nRegla de decision: (a) IC de la diferencia a 2 dias < 0: {g['ic_alto'] < 0} | "
          f"(b) mas aciertos del shift que M1: {g['shift_pm1'] > a1}")


if __name__ == "__main__":
    main()
