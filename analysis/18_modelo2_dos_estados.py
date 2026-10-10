"""
18 - Estructura 2: dos estados (produccion / consumo) con transicion predicha.

Version simple de la estructura recomendada en la propuesta de modelos (tarea
3 de Claude Science). Une los dos analisis que ya funcionaban:
- el modelo de riesgo diario (scripts 12 y 15) da, cada dia, la probabilidad
  de que el cultivo ya haya pasado a consumir lactato;
- el balance de lactato del script 17 da cuanto se produce y se consume.

    dL + d*L = (1 - F) * [alfa*dX+ + a_p*dX+*pH' + a_c*dX+*pCO2' + beta*I]  -  F * k*L*I

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

Correcciones de la auditoria de la tarea 5 (Claude Science):
- lambda (cuanto se encoge la alfa de cada cultivo) se calculaba con la
  matriz del modelo 1 y los parametros del modelo 2. Ahora usa la matriz del
  modelo 2 con F observado en los cultivos de entrenamiento.
- En el error "a 2 dias", el segundo paso usaba F del dia t+1, calculado con
  mediciones del dia t+1 (futuro respecto al origen t). Se agrega la version
  con esas mediciones congeladas en el dia t (rmse2_congelado). Las dos
  versiones usan la VCD observada: es una RECONSTRUCCION explicativa a 2
  dias, no un pronostico.
- El acierto del dia del shift cuenta como fallo a los cultivos donde el
  modelo no simula shift (antes solo se contaban los detectados), y se
  compara contra dos referencias: el dia mediano de entrenamiento (constante)
  y el riesgo solo (primer dia con F >= 0.5, sin cinetica).
- Se repite todo sobre la serie completa (sin cortar en el declive de la VCD).

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


def riesgo_predicho(pp_todos, pp_tr, cols, cultivos_te, congelar=False):
    """F_t para los cultivos de prueba con el modelo de riesgo entrenado sin ellos.

    congelar=True devuelve ademas Fz: el F del dia t calculado con las
    mediciones del dia t-1 (el dia, la temperatura programada y el riesgo
    acumulado hasta t-1 si se conocen). Sirve para el paso t -> t+1 de un
    pronostico hecho desde t-1 sin usar mediciones futuras.
    """
    m = c15.modelo_robusto().fit(pp_tr[cols], pp_tr["y"])
    te = pp_todos[pp_todos["cult"].isin(cultivos_te)].sort_values(["cult", "dia"]).copy()
    te["h"] = m.predict_proba(te[cols])[:, 1]
    if congelar:
        medidas = [c for c in list(est.MEDICIONES.values()) + ["dvcd"] if c in cols]
        tz = te.copy()
        tz[medidas] = te.groupby("cult")[medidas].shift(1)
        te["hz"] = m.predict_proba(tz[cols])[:, 1]
    out, out_z = {}, {}
    for c, g in te.groupby("cult"):
        F = 1 - np.cumprod(1 - g["h"].to_numpy())
        out[c] = pd.Series(F, index=g["dia"].to_numpy())
        if congelar:
            F_prev = np.concatenate([[0.0], F[:-1]])
            out_z[c] = pd.Series(1 - (1 - F_prev) * (1 - g["hz"].to_numpy()), index=g["dia"].to_numpy())
    return (out, out_z) if congelar else out


def lambda_pooling2(S, cultivos, et, p, mu, sd, ph_cin):
    """Bayes empirico para la alfa de cada cultivo con la matriz del modelo 2 (F observado)."""
    est_a, se2, res = [], [], []
    for c in cultivos:
        A, y = matriz2(S[c], f_obs(S[c], et.loc[c]), mu, sd, ph_cin)
        base = A[:, 1:] @ p[1:]
        x = A[:, 0]
        den = x @ x
        if den > 1e-10:
            a_i = x @ (y - base) / den
            r = y - base - a_i * x
            res.extend(r)
            est_a.append(a_i)
            se2.append(np.var(r) / den if len(r) > 1 else np.nan)
    tau2 = max(np.nanvar(est_a) - np.nanmean(se2), 1e-6)
    return np.var(res) / tau2


def primer_dia(index, F, umbral=0.5):
    i = np.flatnonzero(F >= umbral)
    return float(index[i[0]]) if len(i) else np.nan


def correr(S, pp, pp_ciego, et, cols_riesgo, ph_cin, etiqueta):
    cult = np.array(sorted(S))
    filas = []
    for tr_i, te_i in GroupKFold(n_splits=5).split(cult, groups=cult):
        tr, te = cult[tr_i], cult[te_i]
        todo = pd.concat([S[c] for c in tr])
        mu = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
        sd = (todo["ph_prev"].std(), todo["pco2_prev"].std())
        p = ajustar2(S, tr, et, mu, sd, ph_cin)
        lam = lambda_pooling2(S, tr, et, p, mu, sd, ph_cin)
        Fp, Fz = riesgo_predicho(pp_ciego, pp[pp["cult"].isin(tr)], cols_riesgo, te, congelar=True)
        et_tr = et.loc[[c for c in tr if c in et.index]]
        dia_const = float(et_tr.loc[et_tr["evento"], "dia"].median())  # referencia: dia mediano de entrenamiento
        for c in te:
            s = S[c]
            obs = s["L"].to_numpy()
            F_pred = Fp[c].reindex(s.index).fillna(0).to_numpy() if c in Fp else np.zeros(len(s))
            F_z = Fz[c].reindex(s.index).fillna(0).to_numpy() if c in Fz else np.zeros(len(s))
            F_or = f_obs(s, et.loc[c])
            res = {"cult": c, "dia_const": dia_const, "dia_riesgo": primer_dia(s.index, F_pred)}
            for nombre, F in (("pred", F_pred), ("oraculo", F_or)):
                a = alfa_cond2(s, F, p, mu, sd, lam, ph_cin)
                sim = simular2(s, F, p, a, mu, sd, ph_cin)
                e2 = [simular2(s, F, p, a, mu, sd, ph_cin, i0=i, n=2)[i + 2] - obs[i + 2] for i in range(len(obs) - 2)]
                res[f"rmse_{nombre}"] = float(np.sqrt(np.nanmean((sim - obs) ** 2)))
                res[f"rmse2_{nombre}"] = float(np.sqrt(np.nanmean(np.square(e2)))) if e2 else np.nan
                res[f"shift_{nombre}"] = m1.dia_shift_sim(s, sim)
                if nombre == "pred":
                    # mismo pronostico, pero el segundo paso usa F_{t+1} sin mediciones de t+1
                    e2z = []
                    for i in range(len(obs) - 2):
                        Fi = F.copy()
                        Fi[i + 1] = F_z[i + 1]
                        e2z.append(simular2(s, Fi, p, a, mu, sd, ph_cin, i0=i, n=2)[i + 2] - obs[i + 2])
                    res["rmse2_congelado"] = float(np.sqrt(np.nanmean(np.square(e2z)))) if e2z else np.nan
            filas.append(res)
    df = pd.DataFrame(filas).set_index("cult")
    df.columns = [f"{etiqueta}_{c}" for c in df.columns]
    return df


def tabla_tiempo(lab, dias):
    """Acierto del dia del shift sobre TODOS los cultivos con shift en la ventana.

    lab: dia de la etiqueta (NaN si no hay shift en la ventana).
    dias: {nombre: serie con el dia predicho o simulado (NaN = no lo detecto)}.
    Un cultivo sin shift simulado cuenta como fallo en "exacto" y "a +-1".
    """
    ok = lab.notna()
    filas, por_dia = [], []
    for nombre, d in dias.items():
        d = d.reindex(lab.index)
        err = (d - lab)[ok]
        filas.append({"metodo": nombre, "cultivos": int(ok.sum()), "detectados": int(err.notna().sum()),
                      "exacto": float((err == 0).sum() / ok.sum()), "pm1": float((err.abs() <= 1).sum() / ok.sum()),
                      "mae_detectados": float(err.abs().mean())})
        for dv, g in err.groupby(lab[ok]):
            por_dia.append({"metodo": nombre, "dia_real": dv, "cultivos": len(g),
                            "exacto": float((g == 0).mean()), "pm1": float((g.abs() <= 1).mean())})
    return pd.DataFrame(filas), pd.DataFrame(por_dia)


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

    m3_sin_proxies = [c for c in m3 if c not in ("ph", "pco2", "osmolalidad")]

    m1_ev = pd.read_csv(TABLAS / "modelo1_por_cultivo.csv", index_col="cult")
    base = correr(S, pp, pp_ciego, et, m3, True, "M2")
    v_solo_interruptor = correr(S, pp, pp_ciego, et, m3, False, "pHinterruptor")
    v_solo_cinetica = correr(S, pp, pp_ciego, et, m3_sin_ph, True, "pHcinetica")
    # control negativo: el interruptor se alimenta con un riesgo que solo conoce el dia.
    # Si M2 ganara igual con esto, la ventaja vendria de la forma de dos estados y no
    # de las mediciones del cultivo.
    v_solo_dia = correr(S, pp, pp_ciego, et, dia, True, "solodia")
    # pH, pCO2 y osmolalidad cambian dia a dia junto con el lactato (posible control
    # de pH): si M2 dependiera de ellos, estaria usando el lactato por la puerta de atras
    v_sin_proxies = correr(S, pp, pp_ciego, et, m3_sin_proxies, True, "sinproxies")
    t = (m1_ev.join(base).join(v_solo_interruptor).join(v_solo_cinetica).join(v_solo_dia)
         .join(v_sin_proxies))
    t.to_csv(TABLAS / "modelo2_por_cultivo.csv")

    lab = t["dia_shift_etiqueta"]
    n_lab = int(lab.notna().sum())

    def acierto(col):
        """+-1 dia sobre todos los cultivos con shift en la ventana (no detectado = fallo)."""
        ok = lab.notna()
        return float((np.abs(t.loc[ok, col] - lab[ok]) <= 1).sum() / ok.sum())

    def fila(nombre, rmse, rmse2, shift, rmse2z=None):
        dif, ic = m1.ic_pareado(t[rmse], t["rmse_condicionado"])
        dif2, ic2 = m1.ic_pareado(t[rmse2], t["rmse2_modelo"])
        difz, icz = m1.ic_pareado(t[rmse2z], t["rmse2_modelo"]) if rmse2z else (np.nan, (np.nan, np.nan))
        return {"modelo": nombre, "rmse_perfil": t[rmse].mean(), "rmse_2dias": t[rmse2].mean(),
                "rmse_2dias_congelado": t[rmse2z].mean() if rmse2z else np.nan,
                "dif_perfil_vs_M1": dif, "ic_perfil_bajo": ic[0], "ic_perfil_alto": ic[1],
                "dif_2dias_vs_M1": dif2, "ic_bajo": ic2[0], "ic_alto": ic2[1],
                "dif_congelado_vs_M1": difz, "ic_congelado_bajo": icz[0], "ic_congelado_alto": icz[1],
                "shift_pm1": acierto(shift), "shift_detectado": int(t.loc[lab.notna(), shift].notna().sum()),
                "shift_en_ventana": n_lab}

    dif_pp, ic_pp = m1.ic_pareado(t["rmse_condicionado"], t["rmse_perfil_promedio"])
    filas = [{"modelo": "M1 sin interruptor (script 17)", "rmse_perfil": t["rmse_condicionado"].mean(),
              "rmse_2dias": t["rmse2_modelo"].mean(), "shift_pm1": acierto("dia_shift_simulado"),
              "shift_detectado": int(t.loc[lab.notna(), "dia_shift_simulado"].notna().sum()),
              "shift_en_ventana": n_lab},
             {"modelo": "perfil promedio", "rmse_perfil": t["rmse_perfil_promedio"].mean(),
              "rmse_2dias": t["rmse2_baseline"].mean()},
             fila("M2 dos estados (F predicho)", "M2_rmse_pred", "M2_rmse2_pred", "M2_shift_pred",
                  "M2_rmse2_congelado"),
             fila("M2 oraculo (F observado; techo)", "M2_rmse_oraculo", "M2_rmse2_oraculo", "M2_shift_oraculo"),
             fila("M2 con pH solo en el interruptor", "pHinterruptor_rmse_pred", "pHinterruptor_rmse2_pred",
                  "pHinterruptor_shift_pred", "pHinterruptor_rmse2_congelado"),
             fila("M2 con pH solo en la cinetica", "pHcinetica_rmse_pred", "pHcinetica_rmse2_pred",
                  "pHcinetica_shift_pred", "pHcinetica_rmse2_congelado"),
             fila("M2 control: riesgo solo con el dia", "solodia_rmse_pred", "solodia_rmse2_pred",
                  "solodia_shift_pred", "solodia_rmse2_congelado"),
             fila("M2 sin pH, pCO2 ni osmolalidad en el riesgo", "sinproxies_rmse_pred", "sinproxies_rmse2_pred",
                  "sinproxies_shift_pred", "sinproxies_rmse2_congelado")]
    res = pd.DataFrame(filas)
    res.round(4).to_csv(TABLAS / "modelo2_resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    print(f"\nM1 vs perfil promedio (perfil): {dif_pp:+.4f} [{ic_pp[0]:+.4f}, {ic_pp[1]:+.4f}]")
    g = res.iloc[2]
    print(f"Regla de decision: (a) IC de la diferencia a 2 dias < 0: {g['ic_alto'] < 0} | "
          f"(b) mas aciertos del shift que M1: {g['shift_pm1'] > res.iloc[0]['shift_pm1']}")

    # ---------------- acierto del dia del shift contra referencias simples
    tiempo, por_dia = tabla_tiempo(lab, {
        "dia mediano de entrenamiento (constante)": t["M2_dia_const"],
        "riesgo solo (primer dia con F >= 0.5)": t["M2_dia_riesgo"],
        "M1 sin interruptor": t["dia_shift_simulado"],
        "M2 riesgo solo con el dia (control)": t["solodia_shift_pred"],
        "M2 dos estados": t["M2_shift_pred"],
        "M2 oraculo (techo)": t["M2_shift_oraculo"]})
    tiempo.round(4).to_csv(TABLAS / "modelo2_tiempo.csv", index=False)
    por_dia.round(4).to_csv(TABLAS / "modelo2_tiempo_por_dia.csv", index=False)
    print("\n" + tiempo.round(3).to_string(index=False))
    coincide = (t["M2_shift_pred"] == t["M2_dia_riesgo"])[lab.notna()].mean()
    print(f"M2 simula el shift el mismo dia que marca el riesgo solo en {coincide:.0%} de los cultivos")

    # ---------------- serie completa (sin cortar en el declive de la VCD)
    Sc = m1.series(raw_l, et, None)
    m1c, _ = m1.evaluar(Sc, et, D)
    m2c = correr(Sc, pp, pp_ciego, et, m3, True, "M2")
    dc = m1c.join(m2c)
    filas_c = []
    for nombre, col, col2 in (("perfil promedio", "rmse_perfil_promedio", "rmse2_baseline"),
                              ("M1 sin interruptor", "rmse_condicionado", "rmse2_modelo"),
                              ("M2 dos estados", "M2_rmse_pred", "M2_rmse2_pred")):
        d1, i1 = m1.ic_pareado(dc[col], dc["rmse_perfil_promedio"])
        d2, i2 = m1.ic_pareado(dc[col2], dc["rmse2_baseline"])
        filas_c.append({"modelo": nombre, "rmse_perfil": dc[col].mean(), "rmse_2dias": dc[col2].mean(),
                        "dif_perfil_vs_promedio": d1, "ic_perfil_bajo": i1[0], "ic_perfil_alto": i1[1],
                        "dif_2dias_vs_promedio": d2, "ic_2dias_bajo": i2[0], "ic_2dias_alto": i2[1]})
    comp = pd.DataFrame(filas_c)
    comp.round(4).to_csv(TABLAS / "modelo2_serie_completa.csv", index=False)
    print("\nSerie completa (incluye el declive y la re-produccion tardia):")
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
