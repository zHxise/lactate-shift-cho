"""
04 — Prediccion del dia del shift desde los dias 1-4.

Tarea principal: regresion del dia del evento sobre los cultivos con evento
observado. Tarea complementaria: Cox sobre el conjunto completo, que si
aprovecha los censurados.

LO QUE SE HACE PARA NO ENGANARSE
1. Imputacion y estandarizacion dentro de un Pipeline, asi que se ajustan solo
   con el fold de entrenamiento. Imputar antes de partir es el error mas comun
   en trabajos de este tamano y basta para inventar desempeno.
2. Baseline trivial: predecir siempre la mediana del entrenamiento. Si un
   modelo no le gana, no aprendio nada. Con el dia del evento concentrado
   entre 5 y 7, este baseline es fuerte.
3. Control de solo-escala: si predecir con el volumen de reactor da casi lo
   mismo que el modelo completo, lo aprendido es en que equipo se corrio el
   experimento, no biologia.
4. Leave-one-scale-out: entrenar en unas escalas y predecir en otra.
5. Prueba de permutacion: barajar la etiqueta y repetir todo.

CORRECCION AL PLAN ORIGINAL
Agrupar la validacion cruzada por cultivo no hace nada aqui: la matriz tiene
una fila por cultivo, asi que equivale a un KFold normal. La agrupacion que
importaria es por linea celular o lote, y no esta disponible: las hojas
Midpoint/Endpoint traen esa informacion para 45 cultivos pero sin llave a las
series de tiempo. Dos cultivos de la misma linea pueden caer uno en
entrenamiento y otro en prueba, asi que el desempeno reportado es
probablemente optimista.

Ejecutar:  python analysis/04_modelado.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
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
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("esc", StandardScaler()),
                     ("mod", modelo)])


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
    # La primera version fijaba el alpha de Ridge una sola vez, con las
    # etiquetas reales, y lo reutilizaba en todas las barajadas. Eso invalida
    # el p-valor y ademas lo sesga a favor: con etiquetas barajadas, RidgeCV
    # elegiria una regularizacion mucho mas fuerte y predeciria cerca de la
    # media, dando un MAE nulo MENOR. Congelar un alpha pequeno obliga al
    # modelo nulo a sobreajustar ruido y a errar mas de lo que erraria si se
    # le dejara elegir. La prueba correcta repite el procedimiento COMPLETO,
    # seleccion de hiperparametro incluida, dentro de cada permutacion.
    # Error encontrado en auditoria externa.
    # Se corre para los dos modelos. La primera version solo permutaba Ridge,
    # mientras que la cifra destacada en el README era la del Random Forest:
    # el p-valor no respaldaba el numero que se estaba presentando. Error
    # encontrado en auditoria externa.
    print("\n=== Prueba de permutacion (nucleo) ===")
    Xn = con_ev[nucleo]
    for nombre, hacer in [("ridge", lambda: RidgeCV(alphas=ALPHAS)),
                          ("random forest", lambda: RandomForestRegressor(
                              n_estimators=150, min_samples_leaf=3,
                              random_state=SEED, n_jobs=-1))]:
        # El bosque es mucho mas caro, asi que lleva menos barajadas. Eso
        # limita la resolucion del p-valor (minimo 1/n_perm), no su validez.
        n_perm, n_rep = (200, 2) if nombre == "ridge" else (40, 1)
        res = permutation_test(lambda yy: mae_cv(hacer(), Xn, yy, n_rep=n_rep)[0],
                               y, n_permutations=n_perm, seed=SEED, lower_is_better=True)
        print(f"  {nombre:14s} MAE observado {res['observed']:.3f} | "
              f"nulo {res['null_mean']:.3f} +- {res['null_sd']:.3f} | "
              f"p = {res['p_value']:.3f}  ({n_perm} barajadas)")

    # --- leave-one-scale-out ---------------------------------------------
    print("\n=== Leave-one-scale-out (escalas con >=15 cultivos) ===")
    print("La prueba dura: transfiere el modelo a una escala que nunca vio?")
    for nombre, m in modelos.items():
        def fit_predict(Xtr, ytr, Xte, _m=m):
            return pipe(_m).fit(Xtr, ytr).predict(Xte)
        d = leave_one_group_out(fit_predict, con_ev[nucleo], y, con_ev["escala"],
                                mean_absolute_error, min_group_size=15)
        d["group"] = d["group"].map(lambda v: f"{v:.5f}")
        print(f"\n{nombre}:")
        print(d.rename(columns={"score": "MAE"}).round(3).to_string(index=False))
        print(f"  MAE ponderado: {d.attrs['weighted_mean']:.3f}")

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
        """La imputacion y el escalado se ajustan dentro de cada fold.
        lifelines no se integra con Pipeline, asi que se hace a mano; hacerlo
        fuera del fold inflaria el c-index."""
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

    # Un c-index alto con tiempos tan empatados invita a sospechar. Dos controles:
    rng = np.random.default_rng(SEED)
    nulos = np.array([cox_cidx(t[nucleo],
                               pd.Series(T.to_numpy()[k], index=T.index, name="tiempo"),
                               pd.Series(E.to_numpy()[k], index=E.index, name="evento"))
                      for k in (rng.permutation(len(t)) for _ in range(30))])
    print(f"  permutacion: nulo {nulos.mean():.3f} +- {nulos.std():.3f} "
          f"-> p = {(nulos >= real).mean():.3f}")
    print(f"  control solo-escala: {cox_cidx(t[['escala']], T, E):.3f}")
    print(f"  empates en el tiempo: {T.value_counts().sort_index().to_dict()}")
    print(f"\nResultados en {TABLAS}")


if __name__ == "__main__":
    main()
