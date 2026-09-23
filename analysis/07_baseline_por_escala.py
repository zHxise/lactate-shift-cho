"""
07 — ¿Las variables de proceso aportan algo por encima de la escala?

Objecion: predecir el dia del shift podria ser lo mismo que aprender a que
escala pertenece el cultivo. El control de solo-escala ya mostraba que la escala
predice por si sola (MAE 0.795 contra 0.835). Con eso a la vista, "predecir
la mediana global" es un rival debil. Este script usa el rival fuerte y dos
pruebas que atacan la pregunta directamente.

A. Rival fuerte: predecir la mediana de la PROPIA escala (calculada con el
   fold de entrenamiento). Incertidumbre por bootstrap sobre CULTIVOS.

   Por que sobre cultivos y no sobre folds: la primera version de este script
   tomaba el percentil 2.5-97.5 de las diferencias por fold. Eso mide cuanto
   varia UN fold de ~17 cultivos, no la incertidumbre de la diferencia media,
   y con folds tan chicos el intervalo cruzaba cero incluso contra la mediana
   global, donde la permutacion del script 04 ya mostraba que el modelo gana.
   Era un intervalo mal construido que llevaba a una conclusion falsa. La
   unidad independiente es el cultivo: se promedia el error fuera de fold de
   cada cultivo sobre las repeticiones y se remuestrean cultivos.

B. Dentro de una sola escala (la de 43 cultivos). Ahi la escala es constante,
   asi que cualquier mejora sobre la mediana tiene que venir de otras
   variables. Con prueba de permutacion dentro de esa escala.

C. De donde sale la ventaja: error por dia real del evento.

Ejecutar:  python analysis/07_baseline_por_escala.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _comun import TABLAS

warnings.filterwarnings("ignore")
SEED = 0


def pipe(m):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", StandardScaler()), ("m", m)])


def rf(n=300):
    return RandomForestRegressor(n_estimators=n, min_samples_leaf=3,
                                 random_state=SEED, n_jobs=-1)


def cargar():
    t = pd.read_csv(TABLAS / "features_d1_4.csv", index_col="cult")
    t = t[~t["excluir"]]
    meta = ["evento", "dia_evento", "tiempo", "excluir"]
    ext = [c for c in t.columns if c.startswith(("glutamine", "osmolality"))]
    nucleo = [c for c in t.columns if c not in meta + ext]
    return t[t["evento"]], nucleo


def errores_por_cultivo(c, nucleo):
    """Error absoluto fuera de fold de cada cultivo, promediado sobre 5x5."""
    y, esc = c["dia_evento"], c["escala"]
    acum = {k: np.zeros(len(c)) for k in ["global", "escala", "ridge", "rf"]}
    veces = np.zeros(len(c))
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(c):
        ytr, yte = y.iloc[tr], y.iloc[te].to_numpy()
        med_esc = ytr.groupby(esc.iloc[tr]).median()
        preds = {
            "global": np.full(len(te), ytr.median()),
            # si la escala del test no esta en train, cae a la mediana global
            "escala": esc.iloc[te].map(med_esc).fillna(ytr.median()).to_numpy(),
            "ridge": pipe(RidgeCV(alphas=np.logspace(-3, 3, 25)))
                     .fit(c[nucleo].iloc[tr], ytr).predict(c[nucleo].iloc[te]),
            "rf": pipe(rf()).fit(c[nucleo].iloc[tr], ytr).predict(c[nucleo].iloc[te]),
        }
        for k, p in preds.items():
            acum[k][te] += np.abs(yte - p)
        veces[te] += 1
    return pd.DataFrame({k: v / veces for k, v in acum.items()}, index=c.index)


def bootstrap(d, n=5000):
    rng = np.random.default_rng(SEED)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(b, [2.5, 97.5])


def main() -> None:
    c, nucleo = cargar()
    y = c["dia_evento"]

    # ------------------------------------------------------------------ A
    print("=" * 72)
    print("A. Rival fuerte: la mediana de la propia escala")
    print("=" * 72)
    e = errores_por_cultivo(c, nucleo)
    print("MAE fuera de fold (CV 5x5, mismos folds para todos):")
    for k, nombre in [("global", "mediana global"), ("escala", "mediana de la escala"),
                      ("ridge", "ridge"), ("rf", "random forest")]:
        print(f"  {nombre:22s} {e[k].mean():.3f}")
    e.mean().rename("mae").to_csv(TABLAS / "baseline_por_escala_mae.csv")

    print("\nDiferencia de error (baseline - modelo), bootstrap sobre cultivos:")
    print("positivo = el modelo erra menos\n")
    filas = []
    for m, mn in [("ridge", "ridge"), ("rf", "random forest")]:
        for b, bn in [("global", "mediana global"), ("escala", "mediana de la escala")]:
            d = (e[b] - e[m]).to_numpy()
            lo, hi = bootstrap(d)
            # un extremo inferior pegado a cero no es una victoria: es el limite
            if lo >= 0.01:
                veredicto = "gana"
            elif hi < 0:
                veredicto = "pierde"
            elif lo >= 0:
                veredicto = "en el limite (el IC toca cero)"
            else:
                veredicto = "no concluyente"
            print(f"  {mn:14s} vs {bn:22s} {d.mean():+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]  -> {veredicto}")
            filas.append({"modelo": mn, "baseline": bn, "diferencia": round(d.mean(), 3),
                          "ic_bajo": round(lo, 3), "ic_alto": round(hi, 3)})
    pd.DataFrame(filas).to_csv(TABLAS / "baseline_por_escala.csv", index=False)

    # ------------------------------------------------------------------ B
    print("\n" + "=" * 72)
    print("B. Dentro de una sola escala: la escala no puede explicar nada")
    print("=" * 72)
    esc = c["escala"].round(6)
    grande = esc.value_counts().idxmax()
    s = c[esc == grande]
    ys = s["dia_evento"]
    cols = [x for x in nucleo if x != "escala"]
    print(f"escala {grande}: {len(s)} cultivos, dia medio {ys.mean():.2f}, sd {ys.std():.2f}")

    def mae_dentro(yy, modelo, reps):
        out = []
        for tr, te in RepeatedKFold(n_splits=5, n_repeats=reps, random_state=SEED).split(s):
            p = pipe(modelo()).fit(s[cols].iloc[tr], yy.iloc[tr]).predict(s[cols].iloc[te])
            out.append(np.abs(yy.iloc[te].to_numpy() - p).mean())
        return float(np.mean(out))

    med = []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(s):
        med.append(np.abs(ys.iloc[te].to_numpy() - ys.iloc[tr].median()).mean())
    mae_ridge = mae_dentro(ys, lambda: RidgeCV(alphas=np.logspace(-3, 3, 25)), 5)
    print(f"  mediana       {np.mean(med):.3f}")
    print(f"  ridge         {mae_ridge:.3f}")
    obs = mae_dentro(ys, lambda: rf(150), 2)
    print(f"  random forest {obs:.3f}")

    rng = np.random.default_rng(SEED)
    nulos = np.array([mae_dentro(pd.Series(rng.permutation(ys.to_numpy()), index=ys.index),
                                 lambda: rf(150), 1) for _ in range(40)])
    p = ((nulos <= obs).sum() + 1) / (len(nulos) + 1)
    print(f"\n  permutacion dentro de la escala (RF, {len(nulos)} barajadas): nulo {nulos.mean():.3f} "
          f"+- {nulos.std():.3f}, p = {p:.3f}  (minimo posible {1 / (len(nulos) + 1):.3f})")
    pd.DataFrame([{"escala": grande, "n": len(s), "mediana": float(np.mean(med)),
                   "ridge": mae_ridge, "random_forest": obs,
                   "nulo_media": nulos.mean(), "nulo_sd": nulos.std(),
                   "p_valor": p, "barajadas": len(nulos)}]
                 ).to_csv(TABLAS / "dentro_de_escala.csv", index=False)

    # ------------------------------------------------------------------ C
    print("\n" + "=" * 72)
    print("C. De donde sale la ventaja: error medio por dia real del evento")
    print("=" * 72)
    tabla = (e.assign(dia=y.to_numpy())
               .groupby("dia")
               .agg(n=("rf", "size"), mediana_escala=("escala", "mean"),
                    random_forest=("rf", "mean"))
               .round(2))
    print(tabla.to_string())
    tabla.to_csv(TABLAS / "baseline_por_escala_por_dia.csv")
    print(f"\nTablas en {TABLAS}")


if __name__ == "__main__":
    main()
