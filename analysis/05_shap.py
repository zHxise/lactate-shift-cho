"""
05 - Interpretacion con SHAP.

- SHAP se calcula fuera del fold de entrenamiento.
- Como las variables de una misma medicion estan correlacionadas, tambien se
  reporta la importancia agrupada por variable de origen.
- Se compara con importancia por permutacion.
- Ablacion de la variable principal y revision de si separa escalas.

Sin marcadores redox (NAD+, piruvato) esto son asociaciones, no mecanismo.

Ejecutar:  python analysis/05_shap.py

Desarrollado por Arturo Rodriguez.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from lactateshift.validate import out_of_fold_shap
from _comun import FIGURAS, TABLAS

warnings.filterwarnings("ignore")
SEED = 0
SUFIJOS = ("_last", "_slope", "_mean", "_n")


def variable_base(col: str) -> str:
    """Nombre base de la variable (quita _last, _slope, _mean, _n).
    Los cocientes se dejan separados."""
    for suf in SUFIJOS:
        if col.endswith(suf):
            return col[: -len(suf)]
    return col


def hacer_rf():
    # Sin StandardScaler: al arbol no le afecta y asi SHAP queda en la
    # escala original de cada variable.
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                 random_state=SEED, n_jobs=-1)


def main() -> None:
    t = pd.read_csv(TABLAS / "features_d1_4.csv", index_col="cult")
    t = t[~t["excluir"]]
    meta = ["evento", "dia_evento", "tiempo", "excluir"]
    ext = [c for c in t.columns if c.startswith(("glutamine", "osmolality"))]
    nucleo = [c for c in t.columns if c not in meta + ext]

    con_ev = t[t["evento"]]
    X, y = con_ev[nucleo], con_ev["dia_evento"]
    print(f"=== SHAP fuera de fold (n={len(X)}, {X.shape[1]} variables) ===")

    shap_df, imp, por_fold = out_of_fold_shap(hacer_rf, X, y, n_splits=5,
                                              n_repeats=4, seed=SEED)
    imp.to_csv(TABLAS / "shap_importancia.csv", header=["shap_abs_medio"])
    print("\nimportancia individual (|SHAP| medio, dias), top 12:")
    print(imp.head(12).round(4).to_string())

    # estabilidad del ranking entre folds
    ranks = pd.DataFrame([f.rank(ascending=False) for f in por_fold])
    estab = ranks.std().sort_values()
    top5_por_fold = [set(f.nlargest(5).index) for f in por_fold]
    coincidencias = pd.Series([len(a & b) for i, a in enumerate(top5_por_fold)
                               for b in top5_por_fold[i + 1:]])
    print(f"\nestabilidad del top-5 entre folds: coinciden en promedio "
          f"{coincidencias.mean():.1f} de 5 variables")
    print("variables con ranking mas estable:", list(estab.head(5).index))

    # --- agregado por variable de origen ---------------------------------
    grupos = imp.groupby(variable_base).sum().sort_values(ascending=False)
    grupos.to_csv(TABLAS / "shap_importancia_agrupada.csv", header=["shap_abs_sumado"])
    print("\nimportancia agregada por variable de origen (lo interpretable):")
    print(grupos.round(4).to_string())

    # --- contraste con importancia por permutacion ------------------------
    print("\n=== Contraste: importancia por permutacion sobre datos no vistos ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    perm_acum = pd.Series(0.0, index=X.columns)
    for tr, te in kf.split(X):
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("mod", hacer_rf())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(pipe, X.iloc[te], y.iloc[te], n_repeats=10,
                                   random_state=SEED, scoring="neg_mean_absolute_error")
        perm_acum += pd.Series(r.importances_mean, index=X.columns)
    perm = (perm_acum / kf.get_n_splits()).groupby(variable_base).sum().sort_values(ascending=False)
    comp = pd.DataFrame({"shap": grupos, "permutacion": perm}).dropna()
    comp["rank_shap"] = comp["shap"].rank(ascending=False)
    comp["rank_perm"] = comp["permutacion"].rank(ascending=False)
    comp.to_csv(TABLAS / "shap_vs_permutacion.csv")
    rho = comp["shap"].corr(comp["permutacion"], method="spearman")
    pd.DataFrame([{"rho_spearman": rho}]).to_csv(TABLAS / "shap_rho.csv", index=False)
    print(comp.round(4).to_string())
    print(f"\ncorrelacion de rangos entre los dos metodos: rho = {rho:.2f}")

    # --- direccion del efecto --------------------------------------------
    print("\n=== Direccion del efecto (top 8 individuales) ===")
    print("correlacion entre el valor de la variable y su propio SHAP;")
    print("SHAP positivo = empuja el dia del evento hacia MAS TARDE\n")
    # se usan los valores imputados, que son los que vio el modelo
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                         columns=X.columns, index=X.index)
    filas = []
    for c in imp.head(8).index:
        r = X_imp[c].corr(shap_df[c], method="spearman")
        signo = "mas alto -> shift MAS TARDE" if r > 0 else "mas alto -> shift MAS TEMPRANO"
        filas.append({"variable": c, "rho(valor, shap)": round(r, 3), "lectura": signo})
    print(pd.DataFrame(filas).to_string(index=False))
    pd.DataFrame(filas).to_csv(TABLAS / "shap_direccion.csv", index=False)

    # --- ablacion de la variable principal ---------------------------------
    # SHAP mide cuanto usa el modelo una variable, no si su informacion es
    # unica. Se compara el MAE con y sin ella.
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    def mae_cv(modelo, cols, n_rep=5):
        cv = RepeatedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED)
        pl = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("esc", StandardScaler()), ("mod", clone(modelo))])
        return float(np.mean([
            mean_absolute_error(y.iloc[te], pl().fit(X[cols].iloc[tr], y.iloc[tr])
                                .predict(X[cols].iloc[te]))
            for tr, te in cv.split(X)]))

    principal = grupos.index[0]
    cols_ppal = [c for c in X.columns if variable_base(c) == principal]
    print("\n=== Ablacion: que pasa si quitamos la variable mas importante ===")
    filas = [
        ("baseline (mediana)", mae_cv(DummyRegressor(strategy="median"), list(X.columns))),
        ("todas las variables", mae_cv(hacer_rf(), list(X.columns))),
        (f"solo {principal}", mae_cv(hacer_rf(), cols_ppal)),
        (f"todas SIN {principal}", mae_cv(hacer_rf(),
                                          [c for c in X.columns if c not in cols_ppal])),
    ]
    abl = pd.DataFrame(filas, columns=["conjunto", "MAE_dias"])
    abl.to_csv(TABLAS / "shap_ablacion.csv", index=False)
    print(abl.round(3).to_string(index=False))
    print(f"\nSi quitar {principal} casi no cambia el MAE, su informacion "
          "tambien esta en otras variables.")

    # --- variable principal vs escala ------------------------------------
    # Su grafico de dependencia muestra dos nubes separadas; se revisa si
    # coinciden con la escala del reactor.
    var_ppal = imp.index[0]   # la variable individual mas importante
    print(f"\n=== {var_ppal} por grupos y escala ===")
    v = X_imp[var_ppal]
    corte = float(v.median())
    grupo = v > corte
    print(f"partiendo en la mediana ({corte:.4f}): "
          f"{int(grupo.sum())} arriba, {int((~grupo).sum())} abajo")
    print("\ndia del evento por grupo:")
    print(y.groupby(grupo).agg(["median", "mean", "count"]).round(2).to_string())
    if "escala" in X.columns:
        tabla = pd.crosstab(grupo, X["escala"].round(5))
        print("\nreparto por escala de reactor:")
        print(tabla.to_string())
        concentracion = tabla.max(axis=1) / tabla.sum(axis=1)
        print(f"\nfraccion de cada grupo concentrada en una sola escala: "
              f"{concentracion.round(2).to_dict()}")
        print("\nSi los grupos coinciden con la escala, la variable funciona en "
              "parte como\nmarcador del proceso y no solo como variable metabolica.")

    # --- figuras ----------------------------------------------------------
    top3 = list(imp.head(3).index)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    grupos.head(9)[::-1].plot.barh(ax=axes[0], color="tab:blue")
    axes[0].set_xlabel("|SHAP| sumado (dias)")
    axes[0].set_title("Importancia por variable de origen", fontsize=10)
    for ax, c in zip(axes[1:], top3):
        ax.scatter(X_imp[c], shap_df[c], s=20, alpha=0.75, color="tab:blue",
                   edgecolor="none")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel(c)
        ax.set_ylabel("SHAP (dias)")
        rho_c = X_imp[c].corr(shap_df[c], method="spearman")
        flecha = "mas alto -> mas tarde" if rho_c > 0 else "mas alto -> mas temprano"
        ax.set_title(flecha, fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("SHAP fuera de fold, Random Forest, dia del lactate shift\n"
                 "SHAP positivo empuja el dia del evento hacia mas tarde",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURAS / "shap_resumen.png", dpi=130)
    plt.close(fig)
    print(f"\nFigura en {FIGURAS / 'shap_resumen.png'}")


if __name__ == "__main__":
    main()
