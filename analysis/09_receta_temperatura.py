"""
09 - El perfil de temperatura (la receta del proceso) explica el dia del shift?

Motivo: la variable Temperature solo toma 4 valores normalizados (0, 0.571,
0.714 y 1). Son setpoints, no mediciones que varien. 37 cultivos tienen un
cambio de temperatura (de 1.0 a 0.0) entre los dias 3 y 8. Ese dia lo fija la
receta antes de inocular, asi que es informacion conocida de antemano y no
depende del estado del cultivo... salvo que en planta el cambio se dispare por
un criterio del cultivo (por ejemplo, alcanzar cierta VCD). Eso hay que
preguntarlo, no suponerlo.

Que hace:
A. Describe la receta de temperatura por cultivo (temperatura inicial, si hay
   cambio y en que dia) y la cruza con el dia del lactate shift.
B. Compara el error de predecir el dia del shift con:
   - baseline (mediana),
   - solo la receta (temperatura inicial, si hay cambio, dia del cambio, volumen),
   - las 28 variables de los dias 1-4 (las del script 04),
   - las 28 variables + la receta.
   Misma validacion que el script 04 (RepeatedKFold 5x5, imputacion y
   escalado dentro del fold).
C. Rebote tardio: cultivos que, despues del shift, vuelven a superar el
   primer pico de lactato. Lo cruza con la receta y con el titulo final.

Ejecutar:  python analysis/09_receta_temperatura.py

Desarrollado por Arturo Rodriguez.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _comun import TABLAS, cargar

warnings.filterwarnings("ignore")
SEED = 0
SIN_CAMBIO = 30  # dia "ficticio" para los cultivos sin cambio de temperatura


def receta_temperatura(raw: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cultivo: temperatura inicial, si cambia y el dia del cambio."""
    filas = []
    for c, g in raw.sort_values("day").groupby("cult"):
        t = g.set_index("day")["Temperature"].round(3).dropna()
        if t.empty:
            filas.append({"cult": c, "T0": np.nan, "tshift": np.nan, "tshift_day": np.nan})
            continue
        cambios = t[t != t.iloc[0]]
        filas.append({"cult": c, "T0": t.iloc[0],
                      "tshift": int(not cambios.empty),
                      "tshift_day": float(cambios.index[0]) if not cambios.empty else np.nan})
    return pd.DataFrame(filas).set_index("cult")


def mae_cv(modelo, X, y) -> float:
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    m = []
    for tr, te in cv.split(X):
        p = Pipeline([("i", SimpleImputer(strategy="median")),
                      ("s", StandardScaler()), ("m", modelo)])
        p.fit(X.iloc[tr], y.iloc[tr])
        m.append(mean_absolute_error(y.iloc[te], p.predict(X.iloc[te])))
    return float(np.mean(m))


def main() -> None:
    raw = cargar("Raw Data")
    rec = receta_temperatura(raw)
    rec.to_csv(TABLAS / "receta_temperatura.csv")

    t = pd.read_csv(TABLAS / "features_d1_4.csv", index_col="cult")
    t = t[~t["excluir"]].join(rec)
    meta = ["evento", "dia_evento", "tiempo", "excluir"]
    ext = [c for c in t.columns if c.startswith(("glutamine", "osmolality"))]
    nucleo = [c for c in t.columns if c not in meta + ext + list(rec.columns)]

    # --- A. receta vs dia del shift ---------------------------------------
    c = t[t["evento"]].copy()
    print("=== A. Receta de temperatura en el conjunto de modelado ===")
    print("cultivos con cambio de temperatura por grupo de volumen:")
    print(pd.crosstab(c["escala"].round(5), c["tshift"]).to_string())
    con = c[c["tshift"] == 1]
    dif = (con["dia_evento"] - con["tshift_day"]).value_counts().sort_index()
    print("\ndia del shift menos dia del cambio de temperatura:", dif.to_dict())
    rho = spearmanr(con["tshift_day"], con["dia_evento"])
    print(f"Spearman dia del cambio vs dia del shift: {rho.statistic:.2f} "
          f"(p = {rho.pvalue:.4f}, n = {len(con)})")
    print("dia del shift, mediana con cambio / sin cambio:",
          con["dia_evento"].median(), "/", c.loc[c["tshift"] == 0, "dia_evento"].median())

    # --- B. cuanto explica la receta --------------------------------------
    c["tshift_day_f"] = c["tshift_day"].fillna(SIN_CAMBIO)
    receta = ["T0", "tshift", "tshift_day_f", "escala"]
    y = c["dia_evento"]
    rf = lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                       random_state=SEED, n_jobs=-1)
    ridge = lambda: RidgeCV(alphas=np.logspace(-3, 3, 25))
    filas = [{"conjunto": "baseline (mediana)", "ridge": mae_cv(DummyRegressor(strategy="median"), c[["escala"]], y),
              "random_forest": np.nan}]
    for nombre, cols in [("solo receta", receta), ("variables d1-4", nucleo),
                         ("variables d1-4 + receta", nucleo + receta)]:
        filas.append({"conjunto": nombre, "ridge": mae_cv(ridge(), c[cols], y),
                      "random_forest": mae_cv(rf(), c[cols], y)})
    res = pd.DataFrame(filas).round(3)
    res.to_csv(TABLAS / "receta_vs_variables.csv", index=False)
    print("\n=== B. MAE en dias (RepeatedKFold 5x5) ===")
    print(res.to_string(index=False))

    # --- C. rebote tardio --------------------------------------------------
    lac = raw.pivot(index="cult", columns="day", values="[Lactate]")
    mab = (raw.sort_values("day").groupby("cult")["[mAb]"]
           .apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan))
    reb = {}
    for cult, fila in c.iterrows():
        s = lac.loc[cult].dropna()
        reb[cult] = bool((s.loc[fila["dia_evento"] + 1:] > s.loc[:fila["dia_evento"]].max()).any())
    c["rebote"] = pd.Series(reb)
    c["mab_final"] = mab
    c[["rebote", "mab_final", "tshift"]].to_csv(TABLAS / "rebote.csv")
    print("\n=== C. Rebote tardio (vuelve a superar el primer pico) ===")
    print(f"{int(c['rebote'].sum())} de {len(c)} cultivos")
    print("rebote vs cambio de temperatura:")
    print(pd.crosstab(c["tshift"], c["rebote"]).to_string())
    print("titulo final normalizado (mediana) sin / con rebote:",
          c.groupby("rebote")["mab_final"].median().round(3).to_dict())

    # Exploratorio: se puede anticipar el rebote desde los dias 1-4?
    # AUC fuera de fold, 10 particiones estratificadas distintas.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    yb = c["rebote"].astype(int)
    aucs = []
    for nombre, cols in [("solo volumen", ["escala"]),
                         ("receta", ["escala", "T0", "tshift"]),
                         ("variables d1-4", nucleo)]:
        a = []
        for s_ in range(10):
            clf = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                            ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                         class_weight="balanced", random_state=s_))])
            pr = cross_val_predict(clf, c[cols], yb, method="predict_proba",
                                   cv=StratifiedKFold(5, shuffle=True, random_state=s_))[:, 1]
            a.append(roc_auc_score(yb, pr))
        aucs.append({"variables": nombre, "auc_media": round(float(np.mean(a)), 3),
                     "auc_sd": round(float(np.std(a)), 3)})
    aucs = pd.DataFrame(aucs)
    aucs.to_csv(TABLAS / "rebote_auc.csv", index=False)
    print("\nAUC exploratoria para anticipar el rebote (10 particiones 5-fold):")
    print(aucs.to_string(index=False))

    # --- D. mismo protocolo en subconjuntos ------------------------------
    # Todo con RF de 300 arboles y 5x5, para que las cifras sean comparables
    # entre si. (El script 07 usa 150 arboles y 2 repeticiones dentro de la
    # escala, por eso su 0.713 difiere un poco del de aqui.)
    print("\n=== D. MAE por subconjunto, mismo protocolo (RF 300, 5x5) ===")
    sub = []
    grupos = {
        "todos (85)": c,
        "sin cambio de temperatura": c[c["tshift"] == 0],
        "grupo de volumen 0.00202": c[c["escala"].round(5) == 0.00202],
    }
    for nombre, g in grupos.items():
        yy = g["dia_evento"]
        rec_cols = [x for x in receta if g[x].nunique() > 1] or ["escala"]
        fila = {"subconjunto": nombre, "n": len(g),
                "mediana": mae_cv(DummyRegressor(strategy="median"), g[["escala"]], yy),
                "solo_receta": mae_cv(rf(), g[rec_cols], yy),
                "variables_d1_4": mae_cv(rf(), g[nucleo], yy),
                "d1_4_mas_receta": mae_cv(rf(), g[nucleo + rec_cols], yy)}
        sub.append(fila)
    sub = pd.DataFrame(sub).round(3)
    sub.to_csv(TABLAS / "receta_por_subconjunto.csv", index=False)
    print(sub.to_string(index=False))

    print("\nRelacion dia del shift vs titulo final por grupo de volumen (Spearman):")
    for v, g in c.groupby(c["escala"].round(5)):
        if g["mab_final"].notna().sum() >= 10:
            r = spearmanr(g["dia_evento"], g["mab_final"], nan_policy="omit")
            print(f"  {v}: rho {r.statistic:+.2f} (p = {r.pvalue:.3f}, n = {len(g)})")


if __name__ == "__main__":
    main()
