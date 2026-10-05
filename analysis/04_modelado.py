"""
04 - Prediccion del dia del shift desde los dias 1-4.

Regresion del dia del evento (cultivos con evento) y Cox sobre el conjunto
completo (incluye censurados).

Controles:
1. Imputacion y escalado dentro de un Pipeline (se ajustan solo con el fold
   de entrenamiento).
2. Baseline: mediana del entrenamiento.
3. Control con solo la escala del reactor.
4. Leave-one-scale-out.
5. Prueba de permutacion.

Nota: como hay una fila por cultivo, agrupar por cultivo equivale a KFold.
Lo correcto seria agrupar por linea celular o lote, pero esa informacion no
se puede ligar a las series, asi que el desempeno puede ser optimista.

Ejecutar:  python analysis/04_modelado.py

Desarrollado por Arturo Rodriguez.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lactateshift.validate import leave_one_group_out, permutation_test
from _comun import TABLAS

warnings.filterwarnings("ignore")
SEED = 0
ALPHAS = np.logspace(-3, 3, 25)


def pipe(modelo):
    # clone para que cada fold use un modelo sin entrenar
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("esc", StandardScaler()),
                     ("mod", clone(modelo))])


def mae_cv(modelo, X, y, n_rep=5, seed=SEED) -> tuple[float, float]:
    cv = RepeatedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    m = [mean_absolute_error(y.iloc[te], pipe(modelo).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]))
         for tr, te in cv.split(X)]
    return float(np.mean(m)), float(np.std(m))


def main() -> None:
    t = pd.read_csv(TABLAS / "features_d1_4.csv", index_col="cult")
    t = t[~t["excluir"]]
    meta = ["evento", "dia_evento", "tiempo", "excluir"]
    ext = [c for c in t.columns if c.startswith(("glutamine", "osmolality"))]
    nucleo = [c for c in t.columns if c not in meta + ext]
    todas = [c for c in t.columns if c not in meta]

    con_ev = t[t["evento"]]
    y = con_ev["dia_evento"]
    print("=== Regresion del dia del evento ===")
    print(f"n = {len(con_ev)} | dia: media {y.mean():.2f}, sd {y.std():.2f}, "
          f"mediana {y.median():.0f}")

    modelos = {
        "baseline (mediana)": DummyRegressor(strategy="median"),
        "ridge": RidgeCV(alphas=ALPHAS),
        "random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=SEED, n_jobs=-1),
    }
    filas = []
    for etiqueta, cols in [("nucleo", nucleo), ("nucleo+extendidas", todas),
                           ("control: solo escala", ["escala"])]:
        print(f"\n--- {etiqueta} ({len(cols)} variables) ---")
        for nombre, m in modelos.items():
            mu, sd = mae_cv(m, con_ev[cols], y)
            filas.append({"conjunto": etiqueta, "modelo": nombre,
                          "MAE_dias": round(mu, 3), "sd": round(sd, 3)})
            print(f"  {nombre:20s} MAE {mu:.3f} +- {sd:.3f} dias")
    pd.DataFrame(filas).to_csv(TABLAS / "resultados_regresion.csv", index=False)

    # --- permutacion ------------------------------------------------------
    # En cada permutacion se repite todo el procedimiento, incluida la
    # seleccion de alpha de RidgeCV. Fijar alpha con las etiquetas reales
    # sesga el p-valor. Se corre para Ridge y Random Forest.
    print("\n=== Prueba de permutacion (nucleo) ===")
    Xn = con_ev[nucleo]
    perm_filas = []
    for nombre, hacer in [("ridge", lambda: RidgeCV(alphas=ALPHAS)),
                          ("random forest", lambda: RandomForestRegressor(
                              n_estimators=150, min_samples_leaf=3,
                              random_state=SEED, n_jobs=-1))]:
        # RF es mas lento: menos barajadas (p minimo mas alto)
        n_perm, n_rep = (200, 2) if nombre == "ridge" else (40, 1)
        res = permutation_test(lambda yy: mae_cv(hacer(), Xn, yy, n_rep=n_rep)[0],
                               y, n_permutations=n_perm, seed=SEED, lower_is_better=True)
        print(f"  {nombre:14s} MAE observado {res['observed']:.3f} | "
              f"nulo {res['null_mean']:.3f} +- {res['null_sd']:.3f} | "
              f"p = {res['p_value']:.3f}  ({n_perm} barajadas; minimo posible {res['p_min']:.3f})")
        perm_filas.append({"modelo": nombre, "mae_observado": res["observed"],
                           "nulo_media": res["null_mean"], "nulo_sd": res["null_sd"],
                           "p_valor": res["p_value"], "barajadas": n_perm})
    pd.DataFrame(perm_filas).to_csv(TABLAS / "permutacion.csv", index=False)

    # --- leave-one-scale-out ---------------------------------------------
    print("\n=== Leave-one-scale-out (escalas con >=15 cultivos) ===")
    loso_filas = []
    for nombre, m in modelos.items():
        def fit_predict(Xtr, ytr, Xte, _m=m):
            return pipe(_m).fit(Xtr, ytr).predict(Xte)
        d = leave_one_group_out(fit_predict, con_ev[nucleo], y, con_ev["escala"],
                                mean_absolute_error, min_group_size=15)
        d["group"] = d["group"].map(lambda v: f"{v:.5f}")
        print(f"\n{nombre}:")
        print(d.rename(columns={"score": "MAE"}).round(3).to_string(index=False))
        print(f"  MAE ponderado: {d.attrs['weighted_mean']:.3f}")
        for _, fila in d.iterrows():
            loso_filas.append({"modelo": nombre, "escala": fila["group"],
                               "n_test": fila["n_test"], "mae": fila["score"]})
        loso_filas.append({"modelo": nombre, "escala": "ponderado",
                           "n_test": int(d["n_test"].sum()), "mae": d.attrs["weighted_mean"]})
    pd.DataFrame(loso_filas).to_csv(TABLAS / "loso.csv", index=False)

    # --- Cox --------------------------------------------------------------
    print("\n=== Cox: aprovecha los cultivos censurados ===")
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("lifelines no instalado: pip install 'lactateshift[analysis]'")
        return

    T, E = t["tiempo"].rename("tiempo"), t["evento"].astype(int).rename("evento")

    def cox_cidx(X, T, E, n_rep=2, seed=SEED) -> float:
        """c-index con imputacion y escalado dentro de cada fold (a mano,
        lifelines no usa Pipeline)."""
        out = []
        for tr, te in RepeatedKFold(n_splits=5, n_repeats=n_rep, random_state=seed).split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            med = Xtr.median()
            Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            cols = Xtr.columns[Xtr.std() > 1e-8]
            c = CoxPHFitter(penalizer=0.5)
            c.fit(Xtr[cols].join(T).join(E), duration_col="tiempo", event_col="evento")
            out.append(concordance_index(T.iloc[te],
                                         -c.predict_partial_hazard(Xte[cols]).values,
                                         E.iloc[te]))
        return float(np.mean(out))

    real = cox_cidx(t[nucleo], T, E)
    print(f"c-index: {real:.3f}   (0.5 = azar)")

    # controles: permutacion y solo escala
    rng = np.random.default_rng(SEED)
    nulos = np.array([cox_cidx(t[nucleo],
                               pd.Series(T.to_numpy()[k], index=T.index, name="tiempo"),
                               pd.Series(E.to_numpy()[k], index=E.index, name="evento"))
                      for k in (rng.permutation(len(t)) for _ in range(30))])
    print(f"  permutacion: nulo {nulos.mean():.3f} +- {nulos.std():.3f} "
          f"-> p = {((nulos >= real).sum() + 1) / (len(nulos) + 1):.3f} "
          f"(minimo posible con {len(nulos)} barajadas: {1 / (len(nulos) + 1):.3f})")
    control = cox_cidx(t[['escala']], T, E)
    print(f"  control solo-escala: {control:.3f}")
    pd.DataFrame([{"c_index": real, "nulo_media": nulos.mean(), "nulo_sd": nulos.std(),
                   "p_valor": ((nulos >= real).sum() + 1) / (len(nulos) + 1),
                   "barajadas": len(nulos), "control_solo_escala": control}]
                 ).to_csv(TABLAS / "cox.csv", index=False)
    print(f"  empates en el tiempo: {T.value_counts().sort_index().to_dict()}")
    print(f"\nResultados en {TABLAS}")


if __name__ == "__main__":
    main()
