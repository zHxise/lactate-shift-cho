"""
14 - Modelo cinetico sencillo del perfil de lactato (paso 3).

La Dra. Kontoravdi prefiere un enfoque mecanistico e interpretable, y le
interesa el perfil de lactato, no solo el dia del shift. Este es el modelo
mas simple que cumple eso, siguiendo la logica de COSMIC-dFBA: el
metabolismo del cultivo es una mezcla de dos estados celulares, cada uno con
su propia cinetica.

BALANCE DE LACTATO POR DIA (sin dilucion, que no se conoce)

    delta L = (1 - f) * [ alfa * delta X  +  beta * IVCD ]  -  f * k * L * IVCD

    L      lactato;  X  VCD;  IVCD  densidad celular integrada del dia
    f      fraccion de celulas en estado de consumo (0 antes del shift, 1 despues)
    alfa   produccion de lactato ligada al crecimiento   } Luedeking-Piret, la
    beta   produccion por celula no ligada al crecimiento } forma clasica
    k      consumo por celula, proporcional al lactato disponible (cinetica de
           primer orden: mientras mas lactato hay, mas rapido se consume)

Tres parametros con significado biologico, ajustados por minimos cuadrados.
Unidades relativas (los datos estan normalizados): los valores sirven para
comparar cultivos entre si, no para reportar mmol/celula/dia.

QUE SE EVALUA
A. Ajuste global con el estado observado (f de la etiqueta del script 11):
   validacion por cultivo (los parametros se ajustan sin el cultivo que se
   simula). Se simula el perfil completo desde el lactato del dia 1, usando la
   VCD medida, y se compara con un baseline: el perfil promedio por dia.
B. Lo mismo pero con f PREDICHA por el modelo de riesgo diario (script 12,
   M3, fuera de fold): f = probabilidad acumulada de haber hecho el shift.
   Es la version que se podria usar sin conocer el desenlace.
C. Parametros por cultivo y su relacion con la receta de temperatura.

Limitaciones que se declaran:
- La VCD futura se toma como medida: el modelo reconstruye el perfil de
  lactato dada la biomasa, no pronostica la biomasa.
- Dos estados no explican el rebote tardio (16 cultivos vuelven a producir);
  por eso la simulacion se evalua hasta el dia 10.
- Sin dilucion: si la alimentacion diluye, k absorbe parte de ese efecto.

Ejecutar:  python analysis/14_modelo_cinetico.py

Desarrollado por Arturo Rodriguez.
"""

import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import mannwhitneyu
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lactateshift import regularize  # noqa: E402

from _comun import TABLAS, cargar  # noqa: E402

warnings.filterwarnings("ignore")
est = import_module("12_estado_celular")
DIA_MAX = 10
SEED = 0


def series(raw, et):
    """Por cultivo: dias, L y X en rejilla diaria (huecos internos interpolados)."""
    out = {}
    for cult, g in raw.sort_values("day").groupby("cult"):
        if et.loc[cult, "excluir"]:
            continue
        g = g.dropna(subset=["[Lactate]", "VCD"])
        if len(g) < 5:
            continue
        d, L = regularize(g["day"].to_numpy(), g["[Lactate]"].to_numpy())
        _, X = regularize(g["day"].to_numpy(), g["VCD"].to_numpy())
        out[cult] = pd.DataFrame({"L": L, "X": X}, index=d.astype(int))
    return out


def intervalos(s, f):
    """Columnas de la regresion para cada dia t -> t+1."""
    L, X = s["L"].to_numpy(), s["X"].to_numpy()
    dX, ivcd = np.diff(X), (X[:-1] + X[1:]) / 2
    f = f[:-1]
    A = np.column_stack([(1 - f) * dX, (1 - f) * ivcd, -f * L[:-1] * ivcd])
    return A, np.diff(L)


def f_observada(s, et_c):
    dias = s.index.to_numpy()
    if not et_c["evento"]:
        return np.zeros(len(dias))
    return (dias >= et_c["dia"]).astype(float)


def ajustar(bloques):
    A = np.vstack([b[0] for b in bloques])
    y = np.concatenate([b[1] for b in bloques])
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    # alfa y beta libres; k >= 0 (consumo)
    r = lsq_linear(A[ok], y[ok], bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]))
    return r.x


def simular(s, f, p):
    L = s["L"].to_numpy()
    X = s["X"].to_numpy()
    sim = np.empty(len(L))
    sim[0] = L[0]
    for t in range(len(L) - 1):
        ivcd = (X[t] + X[t + 1]) / 2
        dL = (1 - f[t]) * (p[0] * (X[t + 1] - X[t]) + p[1] * ivcd) - f[t] * p[2] * sim[t] * ivcd
        sim[t + 1] = max(sim[t] + dL, 0.0)
    return sim


def f_predicha(raw, et, cultivos_tr, cultivos_te):
    """f = P(ya hizo el shift) con el modelo de riesgo M3 entrenado sin los cultivos de prueba."""
    pp = est.persona_periodo(raw, et, adelanto=1)
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]
    cols = dia + ["temperatura", "bajo_T_hoy", "baja_T_manana"] + list(est.MEDICIONES.values()) + ["dvcd"]
    tr = pp[pp["cult"].isin(cultivos_tr)]
    m = est.modelo().fit(tr[cols], tr["y"])
    # filas de TODOS los dias de los cultivos de prueba (sin cortar en el evento)
    et_ciego = et.copy()
    et_ciego["evento"] = False
    et_ciego["tiempo"] = et_ciego["ultimo_dia"]  # hasta el ultimo dia medido
    todos = est.persona_periodo(raw, et_ciego.loc[cultivos_te], adelanto=1)
    todos["h"] = m.predict_proba(todos[cols])[:, 1]
    res = {}
    for c, g in todos.groupby("cult"):
        sobrevive = np.cumprod(1 - g.sort_values("dia")["h"].to_numpy())
        res[c] = pd.Series(1 - sobrevive, index=g.sort_values("dia")["dia"].to_numpy())
    return res


def main() -> None:
    raw = cargar("Raw Data")
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")
    S = series(raw, et)
    cult = np.array(sorted(S))
    print(f"cultivos: {len(cult)}")

    filas, ejemplos = [], {}
    for tr_i, te_i in GroupKFold(n_splits=5).split(cult, groups=cult):
        tr, te = cult[tr_i], cult[te_i]
        p = ajustar([intervalos(S[c], f_observada(S[c], et.loc[c])) for c in tr])
        fp = f_predicha(raw, et, tr, te)
        # baseline: perfil promedio por dia de los cultivos de entrenamiento
        prom = pd.concat([S[c]["L"] for c in tr], axis=1).mean(axis=1)
        for c in te:
            s = S[c]
            dias = s.index.to_numpy()
            f_obs = f_observada(s, et.loc[c])
            f_pr = fp[c].reindex(dias).fillna(0).to_numpy() if c in fp else np.zeros(len(dias))
            sim_obs, sim_pr = simular(s, f_obs, p), simular(s, f_pr, p)
            m = dias <= DIA_MAX
            obs = s["L"].to_numpy()
            base = prom.reindex(dias).to_numpy()
            filas.append({"cult": c,
                          "rmse_f_observada": np.sqrt(np.mean((sim_obs[m] - obs[m]) ** 2)),
                          "rmse_f_predicha": np.sqrt(np.mean((sim_pr[m] - obs[m]) ** 2)),
                          "rmse_perfil_promedio": np.sqrt(np.nanmean((base[m] - obs[m]) ** 2))})
            ejemplos[c] = pd.DataFrame({"observado": obs, "sim_f_observada": sim_obs,
                                        "sim_f_predicha": sim_pr}, index=dias)
    ev = pd.DataFrame(filas).set_index("cult")
    ev.to_csv(TABLAS / "cinetico_rmse.csv")
    pd.concat(ejemplos, names=["cult", "dia"]).to_csv(TABLAS / "cinetico_perfiles.csv")

    print(f"\n=== A/B. Error del perfil de lactato hasta el dia {DIA_MAX} (RMSE, unidades normalizadas) ===")
    print("validacion por cultivo (5 folds); mediana y media entre cultivos:")
    print(ev.agg(["median", "mean"]).round(4).to_string())
    print(f"el modelo con f observada gana al perfil promedio en "
          f"{int((ev['rmse_f_observada'] < ev['rmse_perfil_promedio']).sum())} de {len(ev)} cultivos; "
          f"con f predicha en {int((ev['rmse_f_predicha'] < ev['rmse_perfil_promedio']).sum())}")

    p_all = ajustar([intervalos(S[c], f_observada(S[c], et.loc[c])) for c in cult])
    print(f"\nparametros globales: alfa {p_all[0]:.3f} | beta {p_all[1]:.3f} | k {p_all[2]:.3f}")

    # --- C. parametros por cultivo vs receta ------------------------------
    rec = pd.read_csv(TABLAS / "receta_temperatura.csv", index_col="cult")
    vol = raw.groupby("cult")["Culture Volume"].first().round(5)
    par = []
    for c in cult:
        A, y = intervalos(S[c], f_observada(S[c], et.loc[c]))
        n_crec = int((A[:, 0] != 0).sum() + (A[:, 1] != 0).sum() > 0)
        if (A[:, 2] != 0).sum() < 2 or (A[:, 1] != 0).sum() < 3:
            continue  # pocos dias en algun estado para estimar
        ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
        r = lsq_linear(A[ok], y[ok], bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]))
        par.append({"cult": c, "alfa": r.x[0], "beta": r.x[1], "k": r.x[2]})
    par = pd.DataFrame(par).set_index("cult").join(rec[["T0", "tshift"]]).join(vol.rename("grupo"))
    par.to_csv(TABLAS / "cinetico_parametros.csv")
    print(f"\n=== C. Parametros por cultivo ({len(par)} cultivos con suficientes dias en ambos estados) ===")
    print(par.groupby("tshift")[["alfa", "beta", "k"]].median().round(3).rename(
        index={0: "sin cambio de T", 1: "con cambio de T"}).to_string())
    g = par[par["grupo"] == 0.00202]
    if g["tshift"].nunique() == 2:
        print("\ndentro del grupo 0.00202 (Mann-Whitney, con vs sin cambio de temperatura):")
        for v in ("alfa", "beta", "k"):
            a, b = g.loc[g["tshift"] == 1, v], g.loc[g["tshift"] == 0, v]
            print(f"  {v:5s} mediana {a.median():.3f} vs {b.median():.3f}  p = {mannwhitneyu(a, b).pvalue:.3f}  "
                  f"(n = {len(a)} vs {len(b)})")


if __name__ == "__main__":
    main()
