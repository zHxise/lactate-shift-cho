"""
17 - Estructura 1: cinetica de lactato acoplada al crecimiento, SIN interruptor.

Es la hipotesis nula de la propuesta de modelos (Claude Science, tarea 3):
si un modelo donde el shift "sale solo" (la produccion baja cuando el
crecimiento se frena y el consumo existe siempre) explica los datos, no hace
falta un interruptor regulado. Cualquier modelo con interruptor (estructura 2)
tiene que ganarle a este.

BALANCE DIARIO (unidades normalizadas)

    dL + d*L = alfa*dX+ + beta*I - k*L*I + (efectos de pH y pCO2 sobre alfa)

    dX+  crecimiento del dia (solo la parte positiva: max(dX, 0))
    I    densidad celular integrada del dia (trapecio)
    d    dilucion por alimentacion: desconocida -> se prueba 0, 1, 2 y 3% por dia
    alfa produccion ligada al crecimiento (>= 0)
    beta produccion por celula no ligada al crecimiento (>= 0; obligarla a ser
         positiva evita que "esconda" la dilucion, como paso en el script 14)
    k    consumo por celula proporcional al lactato (>= 0), SIEMPRE activo
    pH y pCO2 del dia anterior modulan alfa: alfa*(1 + a_p*pH' + a_c*pCO2'),
         con pH' y pCO2' estandarizados con los cultivos de entrenamiento.
         Linealizado asi el modelo sigue siendo lineal en los parametros.

El shift no es un parametro: se define en la SIMULACION igual que en los
datos (primer dia en que la produccion neta simulada es negativa 2 dias
seguidos).

VENTANA: solo la fase antes del declive. El script 16 mostro que mas de la
mitad de los cultivos vuelve a producir lactato tarde, cuando la VCD ya
baja; ese es otro fenomeno. La ventana termina el primer dia en que la VCD
cae por debajo del 85% de su maximo hasta ese dia (regla basada solo en VCD,
para no recortar usando el propio lactato). Sensibilidad: 75% y cultivo
completo.

DIFERENCIAS ENTRE CULTIVOS (parcial): cada cultivo tiene su propia alfa,
"encogida" hacia la alfa global (efectos aleatorios aproximados por
bayes empirico). Para un cultivo NUEVO se estima su alfa solo con los
dias 1-4 (lo que se sabria al cerrar el dia 4): modo "condicionado".
Modo "poblacion": usa la alfa global.

EVALUACION (validacion por cultivo, 5 folds)
- Simulacion del perfil completo dentro de la ventana, desde el lactato del
  dia 1 y con la VCD medida (modo explicativo: no pronostica la biomasa).
- Pronostico a 2 dias: desde el lactato observado en t, simular t+2.
- Dia del shift simulado vs etiqueta (+-1 dia).
Baselines: perfil promedio por dia; a 2 dias, lactato de hoy + incremento
promedio de esos dias en los cultivos de entrenamiento.
Diferencias con IC por bootstrap de cultivos (pareado).

Ejecutar:  python analysis/17_modelo1_crecimiento.py

Desarrollado por Arturo Rodriguez.
"""

import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lactateshift import regularize  # noqa: E402

from _comun import TABLAS, cargar  # noqa: E402

warnings.filterwarnings("ignore")
c15 = import_module("15_correcciones_auditoria")
SEED = 0
DILUCIONES = (0.0, 0.01, 0.02, 0.03)
CORTE_VCD = 0.85
NOMBRES = ["alfa", "beta", "k", "a_p", "a_c"]


def series(raw_l, et, corte):
    """Por cultivo: rejilla diaria con L, X, pH y pCO2 (rezagados 1 dia) y fin de ventana."""
    out = {}
    for cult, g in raw_l.sort_values("day").groupby("cult"):
        if cult not in et.index or et.loc[cult, "excluir"]:
            continue
        gg = g.dropna(subset=["[Lactate]", "VCD"])
        if len(gg) < 5:
            continue
        d, L = regularize(gg["day"].to_numpy(), gg["[Lactate]"].to_numpy())
        _, X = regularize(gg["day"].to_numpy(), gg["VCD"].to_numpy())
        s = pd.DataFrame({"L": L, "X": X}, index=d.astype(int))
        gas = g.set_index("day")[["pH", "pCO2"]].reindex(s.index).ffill().bfill()  # bfill solo cubre el dia 1
        s["ph_prev"] = gas["pH"].shift(1).fillna(gas["pH"])
        s["pco2_prev"] = gas["pCO2"].shift(1).fillna(gas["pCO2"])
        if corte is None:
            fin = int(s.index.max())
        else:
            caida = s.index[s["X"] < corte * s["X"].cummax()]
            fin = int(caida.min()) if len(caida) else int(s.index.max())
        out[cult] = s.loc[:fin]
    return out


def matriz(s, d, mu, sd):
    """Columnas de la regresion para cada intervalo t -> t+1."""
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    dX = np.diff(X)
    dXp = np.maximum(dX, 0)
    ivcd = (X[:-1] + X[1:]) / 2
    ph = (s["ph_prev"].to_numpy()[:-1] - mu[0]) / sd[0]
    co = (s["pco2_prev"].to_numpy()[:-1] - mu[1]) / sd[1]
    ph, co = np.nan_to_num(ph), np.nan_to_num(co)
    A = np.column_stack([dXp, ivcd, -L[:-1] * ivcd, dXp * ph, dXp * co])
    y = np.diff(L) + d * L[:-1]
    return A, y


def ajustar(S, cultivos, d, mu, sd):
    A = np.vstack([matriz(S[c], d, mu, sd)[0] for c in cultivos])
    y = np.concatenate([matriz(S[c], d, mu, sd)[1] for c in cultivos])
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    lo = [0, 0, 0, -np.inf, -np.inf]
    return lsq_linear(A[ok], y[ok], bounds=(lo, [np.inf] * 5)).x, A[ok], y[ok]


def lambda_pooling(S, cultivos, p, d, mu, sd):
    """Bayes empirico para la alfa de cada cultivo: lambda = sigma^2 / tau^2."""
    est, se2, res = [], [], []
    for c in cultivos:
        A, y = matriz(S[c], d, mu, sd)
        base = A[:, 1:] @ p[1:]
        x = A[:, 0] * (1 + 0)  # columna de alfa (sin modulacion para la estimacion simple)
        num = x @ (y - base)
        den = x @ x
        if den > 1e-10:
            a_i = num / den
            r = y - base - a_i * x
            res.extend(r)
            est.append(a_i)
            se2.append(np.var(r) / den if len(r) > 1 else np.nan)
    sigma2 = np.var(res)
    tau2 = max(np.nanvar(est) - np.nanmean(se2), 1e-6)
    return sigma2 / tau2


def alfa_condicionada(s, p, d, mu, sd, lam, hasta_dia=4):
    """Alfa del cultivo estimada solo con los dias 1..hasta_dia, encogida hacia la global."""
    A, y = matriz(s.loc[:hasta_dia], d, mu, sd)
    if len(y) == 0:
        return p[0]
    x = A[:, 0] * (1 + p[3] * A[:, 3] / np.where(A[:, 0] == 0, 1, A[:, 0])
                   + p[4] * A[:, 4] / np.where(A[:, 0] == 0, 1, A[:, 0]))
    base = A[:, 1] * p[1] + A[:, 2] * p[2]
    a = (x @ (y - base) + lam * p[0]) / (x @ x + lam)
    return max(a, 0.0)


def simular(s, p, alfa, d, mu, sd, desde=None):
    """Simula L dentro de la ventana con la VCD medida. desde=(t0, L0) para pronosticos cortos."""
    A, _ = matriz(s, d, mu, sd)
    L = s["L"].to_numpy().copy()
    sim = np.full(len(L), np.nan)
    i0 = 0 if desde is None else desde
    sim[i0] = L[i0]
    for t in range(i0, len(L) - 1):
        dXp, ivcd = A[t, 0], A[t, 1]
        prod = alfa * dXp + p[3] * A[t, 3] + p[4] * A[t, 4] + p[1] * ivcd
        cons = p[2] * sim[t] * ivcd
        sim[t + 1] = max(sim[t] + prod - cons - d * sim[t], 0.0)
    return sim


def dia_shift_sim(s, sim):
    """Primer dia con produccion neta simulada negativa 2 intervalos seguidos."""
    dL = np.diff(sim)
    for t in range(len(dL) - 1):
        if dL[t] < 0 and dL[t + 1] < 0 and np.any(dL[:t] > 0):
            return float(s.index[t])
    return np.nan


def evaluar(S, et, d):
    cult = np.array(sorted(S))
    filas, params = [], []
    for tr_i, te_i in GroupKFold(n_splits=5).split(cult, groups=cult):
        tr, te = cult[tr_i], cult[te_i]
        todo = pd.concat([S[c] for c in tr])
        mu = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
        sd = (todo["ph_prev"].std(), todo["pco2_prev"].std())
        p, _, _ = ajustar(S, tr, d, mu, sd)
        params.append(p)
        lam = lambda_pooling(S, tr, p, d, mu, sd)
        prom = pd.concat([S[c]["L"] for c in tr], axis=1).mean(axis=1)
        inc2 = pd.concat([S[c]["L"].diff(2).shift(-2) for c in tr], axis=1).mean(axis=1)
        for c in te:
            s = S[c]
            obs = s["L"].to_numpy()
            a_c = alfa_condicionada(s, p, d, mu, sd, lam)
            sim_pob = simular(s, p, p[0], d, mu, sd)
            sim_con = simular(s, p, a_c, d, mu, sd)
            base = prom.reindex(s.index).to_numpy()
            # pronostico a 2 dias desde cada t observado
            e2_mod, e2_base = [], []
            for i in range(len(obs) - 2):
                sm = simular(s.iloc[i:i + 3], p, a_c, d, mu, sd)
                e2_mod.append(sm[2] - obs[i + 2])
                b = obs[i] + inc2.reindex([s.index[i]]).to_numpy()[0]
                e2_base.append(b - obs[i + 2])
            rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))) if len(e) else np.nan
            ds_lab = et.loc[c, "dia"] if et.loc[c, "evento"] else np.nan
            ds_sim = dia_shift_sim(s, sim_con)
            filas.append({"cult": c, "dias": len(obs),
                          "rmse_poblacion": rm(sim_pob - obs), "rmse_condicionado": rm(sim_con - obs),
                          "rmse_perfil_promedio": rm(base - obs),
                          "rmse2_modelo": rm(e2_mod), "rmse2_baseline": rm(e2_base),
                          "dia_shift_etiqueta": ds_lab if (np.isfinite(ds_lab) and ds_lab < s.index.max()) else np.nan,
                          "dia_shift_simulado": ds_sim})
    return pd.DataFrame(filas).set_index("cult"), np.mean(params, axis=0)


def ic_pareado(a, b, n=2000):
    rng = np.random.default_rng(SEED)
    d = (a - b).dropna().to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, [2.5, 97.5])


def main() -> None:
    raw = cargar("Raw Data")
    raw_l, _ = c15.qc_gases(raw)
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")

    resumen = []
    for corte in (CORTE_VCD, 0.75, None):
        S = series(raw_l, et, corte)
        for d in DILUCIONES:
            if corte != CORTE_VCD and d not in (0.0, 0.02):
                continue  # sensibilidad de ventana solo en dos diluciones
            ev, p = evaluar(S, et, d)
            dif_sim, ic_sim = ic_pareado(ev["rmse_condicionado"], ev["rmse_perfil_promedio"])
            dif_2, ic_2 = ic_pareado(ev["rmse2_modelo"], ev["rmse2_baseline"])
            both = ev.dropna(subset=["dia_shift_etiqueta", "dia_shift_simulado"])
            acierto = np.mean(np.abs(both["dia_shift_etiqueta"] - both["dia_shift_simulado"]) <= 1) if len(both) else np.nan
            fila = {"ventana": "completa" if corte is None else f"VCD>={corte:.0%} del max",
                    "dilucion": d, "cultivos": len(ev),
                    "rmse_poblacion": ev["rmse_poblacion"].mean(),
                    "rmse_condicionado": ev["rmse_condicionado"].mean(),
                    "rmse_perfil_promedio": ev["rmse_perfil_promedio"].mean(),
                    "dif_vs_promedio": dif_sim, "dif_ic_bajo": ic_sim[0], "dif_ic_alto": ic_sim[1],
                    "rmse2_modelo": ev["rmse2_modelo"].mean(), "rmse2_baseline": ev["rmse2_baseline"].mean(),
                    "dif2": dif_2, "dif2_ic_bajo": ic_2[0], "dif2_ic_alto": ic_2[1],
                    "shift_en_ventana": len(both), "shift_simulado_pm1": acierto,
                    **{f"param_{n}": v for n, v in zip(NOMBRES, p)}}
            resumen.append(fila)
            print(f"[{fila['ventana']}, d={d:.2f}] RMSE perfil: modelo cond. {fila['rmse_condicionado']:.4f} "
                  f"(poblacion {fila['rmse_poblacion']:.4f}) vs promedio {fila['rmse_perfil_promedio']:.4f} "
                  f"| dif {dif_sim:+.4f} [{ic_sim[0]:+.4f}, {ic_sim[1]:+.4f}]")
            print(f"      a 2 dias: modelo {fila['rmse2_modelo']:.4f} vs baseline {fila['rmse2_baseline']:.4f} "
                  f"| dif {dif_2:+.4f} [{ic_2[0]:+.4f}, {ic_2[1]:+.4f}] | shift simulado a +-1 dia: "
                  f"{acierto:.0%} de {len(both)} | params {np.round(p, 3)}")
            if corte == CORTE_VCD and d == 0.0:
                ev.to_csv(TABLAS / "modelo1_por_cultivo.csv")
    pd.DataFrame(resumen).round(4).to_csv(TABLAS / "modelo1_resumen.csv", index=False)
    print("\nNegativo = el modelo erra menos que el baseline. Parametros = promedio de los 5 folds "
          "(alfa, beta, k, a_p, a_c).")


if __name__ == "__main__":
    main()
