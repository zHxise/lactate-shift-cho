"""
05 — Interpretacion del modelo con SHAP, contrastada con hipotesis mecanisticas.

TRES PRECAUCIONES QUE CAMBIAN COMO SE LEE ESTO

1. Los valores SHAP se calculan FUERA del fold de entrenamiento. SHAP explica
   al modelo, no al fenomeno: un modelo sobreajustado da explicaciones nitidas
   de su propio sobreajuste. Calculandolos solo sobre observaciones que el
   modelo no vio, queda la parte de la explicacion que sobrevive a datos
   nuevos.

2. Las variables estan muy correlacionadas entre si. ``lactate_last``,
   ``lactate_mean`` y ``lactate_slope`` describen la misma medicion. SHAP
   reparte el credito entre variables correlacionadas, asi que el ranking
   individual es inestable por construccion. Por eso se reporta tambien la
   importancia agregada por variable de origen, que es lo interpretable.

3. Se compara con importancia por permutacion sobre datos no vistos. Si las
   dos concuerdan, la lectura es mas solida; si no, la discrepancia es el
   resultado y hay que decirlo.

Y el limite de fondo: el dataset no tiene marcadores redox (NAD+, piruvato).
Todo lo que sigue son asociaciones entre senales de proceso. No es evidencia
de mecanismo.

Ejecutar:  python analysis/05_shap.py
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
    """Agrupa las cuatro variantes de una misma medicion bajo su nombre.

    Los cocientes se dejan intactos: colapsar ``lactate_over_glucose`` y
    ``lactate_over_vcd`` en una sola categoria mezclaba dos senales distintas
    y hacia ininterpretable su importancia agregada. Error encontrado en
    auditoria externa.
    """
    for suf in SUFIJOS:
        if col.endswith(suf):
            return col[: -len(suf)]
    return col


def hacer_rf():
    # Sin StandardScaler, a diferencia del script 04. No cambia el modelo: un
    # arbol ordena y corta cada variable por umbrales, y un reescalado lineal
    # conserva el orden y lleva cada umbral a su equivalente, asi que las
    # particiones y predicciones son las mismas. Si importa para leer SHAP:
    # asi los valores de las variables quedan en sus unidades originales.
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

    # Estabilidad: si el orden cambia con la particion, el ranking individual
    # no significa gran cosa y hay que apoyarse en la version agregada.
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
    # El signo importa mas que la magnitud: dice si un valor alto adelanta o
    # retrasa el shift, y eso es lo que se puede contrastar con la biologia.
    print("\n=== Direccion del efecto (top 8 individuales) ===")
    print("correlacion entre el valor de la variable y su propio SHAP;")
    print("SHAP positivo = empuja el dia del evento hacia MAS TARDE\n")
    # La correlacion se calcula sobre los valores imputados, que son los que
    # el modelo vio al generar esos SHAP. Usar X sin imputar puede invertir el
    # signo aparente si los faltantes no son aleatorios. Error de auditoria.
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                         columns=X.columns, index=X.index)
    filas = []
    for c in imp.head(8).index:
        r = X_imp[c].corr(shap_df[c], method="spearman")
        signo = "mas alto -> shift MAS TARDE" if r > 0 else "mas alto -> shift MAS TEMPRANO"
        filas.append({"variable": c, "rho(valor, shap)": round(r, 3), "lectura": signo})
    print(pd.DataFrame(filas).to_string(index=False))
    pd.DataFrame(filas).to_csv(TABLAS / "shap_direccion.csv", index=False)

    # --- ablacion: importancia no es lo mismo que necesidad ---------------
    # SHAP y la permutacion miden cuanto USA el modelo una variable. No miden
    # cuanta informacion UNICA aporta. Si su informacion tambien esta en otras
    # variables, quitarla no empeora la prediccion: el modelo la recupera de
    # las demas. Esta es la comprobacion que desmiente la lectura ingenua de
    # un grafico de importancias.
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
    print(f"\nSi quitar {principal} apenas cambia el MAE, su informacion es")
    print("redundante con la de otras variables. Es la variable que el modelo")
    print("prefiere usar, no la unica que podria usar.")

    # --- la variable principal, .senal fisiologica o etiqueta del proceso? -
    # El grafico de dependencia de la variable principal mostraba dos nubes
    # separadas en lugar de una relacion continua. Eso no parece una respuesta
    # metabolica graduada; parece que el modelo esta separando dos familias de
    # cultivos. Vale la pena comprobarlo explicitamente, porque cambia por
    # completo lo que se puede afirmar.
    var_ppal = imp.index[0]   # la variable individual mas importante
    print(f"\n=== .{var_ppal} es senal fisiologica o etiqueta del proceso? ===")
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
        # .cuanto del grupo mas grande cae en una sola escala?
        concentracion = tabla.max(axis=1) / tabla.sum(axis=1)
        print(f"\nfraccion de cada grupo concentrada en una sola escala: "
              f"{concentracion.round(2).to_dict()}")
        print("\nSi los grupos coinciden en buena medida con la escala, la "
              "variable esta funcionando\ncomo marcador de a que familia de "
              "procesos pertenece el cultivo, y no solo como\nvariable "
              "metabolica. Leido junto con el leave-one-scale-out del script "
              "04, apunta a\nque parte de la senal es identidad del proceso, "
              "no fisiologia universal del shift.")

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
    fig.suptitle("SHAP fuera de fold — Random Forest, dia del lactate shift\n"
                 "SHAP positivo empuja el dia del evento hacia mas tarde",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURAS / "shap_resumen.png", dpi=130)
    plt.close(fig)
    print(f"\nFigura en {FIGURAS / 'shap_resumen.png'}")


if __name__ == "__main__":
    main()
