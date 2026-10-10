"""
12 - Predictor del shift dia a dia (modelo de riesgo discreto, interpretable).

De donde sale la idea: en COSMIC-dFBA (Gopalakrishnan et al. 2024) el cambio
de estado metabolico se predice con un modelo logistico sencillo a partir de
las condiciones del biorreactor (metabolitos que se agotan, oxigeno y
temperatura). Aqui se adapta esa idea al shift de lactato en fed-batch, con
lo que pidio la Dra. Kontoravdi: momento del shift, que se asocia con el,
interpretabilidad y 1-2 dias de anticipacion.

QUE SE MODELA
Cada cultivo aporta una fila por dia mientras todavia no ha hecho el shift
(formato "persona-periodo"). La pregunta de cada fila es:

    dado que el cultivo sigue produciendo lactato al dia t,
    es el dia t su ultimo dia de produccion?

Eso es la "funcion de riesgo" (hazard) de un modelo de supervivencia en
tiempo discreto, y se estima con una regresion logistica comun. Ventajas:
- la prediccion se actualiza cada dia (ya no hay ventana fija de 4 dias);
- los cultivos sin shift (censurados) entran sin trucos: aportan sus filas
  hasta el ultimo dia, todas con 0;
- cada coeficiente se lee como "cuanto cambia el riesgo de que hoy sea el
  ultimo dia de produccion por cada desviacion estandar de la variable".

Etiqueta: definicion final del script 11 (tasa especifica, filtro de 15%).
Se excluyen C33 y C34 (no acumulan lactato).

ANTICIPACION
- Adelanto de 1 dia: se usan los datos hasta el dia t. La bajada del lactato
  se ve hasta la muestra del dia t+1.
- Adelanto de 2 dias: se usan los datos hasta el dia t-1.

VARIABLES (todas conocidas en el momento de predecir)
- dia del cultivo, como categorias (un indicador por dia): el riesgo basal
  cambia con la edad del cultivo y NO de forma lineal. Con el dia como numero
  el baseline salia artificialmente debil (AUC 0.79 contra 0.83 con
  categorias), y las mediciones parecian aportar mas de lo que aportan:
  la VCD, que sube con el tiempo, absorbia la parte no lineal del dia.
- receta: temperatura del dia, si hoy bajo la temperatura, y si manana baja
  (el cambio esta programado de antemano; lo confirmo la Dra. Kontoravdi).
- mediciones: VCD y su cambio diario, glutamina, glutamato, amonio, pH,
  osmolalidad, glucosa, pO2 y pCO2.
- NO se usa nada de lactato: el lactato define la etiqueta, usarlo seria
  medir el desenlace con el mismo instrumento.
Los huecos se rellenan solo hacia adelante dentro de cada cultivo (nunca con
datos futuros); lo que sigue faltando se imputa con la mediana del fold de
entrenamiento.

VALIDACION
Ahora hay varias filas por cultivo, asi que la particion se hace POR CULTIVO
(StratifiedGroupKFold): las filas de un mismo cultivo nunca quedan repartidas
entre entrenamiento y prueba. Mezclarlas seria fuga de informacion.
Se compara el AUC por fila de cuatro modelos anidados:
  M0 solo dia | M1 dia + receta | M2 dia + mediciones | M3 dia + receta + mediciones
Y M3 dejando fuera un grupo de volumen completo.
Coeficientes de M3 con intervalos por bootstrap de cultivos.

Ejecutar:  python analysis/12_estado_celular.py

Desarrollado por Arturo Rodriguez.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _comun import TABLAS, cargar

warnings.filterwarnings("ignore")
SEED = 0
MEDICIONES = {"VCD": "vcd", "[Glutamine]": "glutamina", "[Glutamate]": "glutamato",
              "[NH3]": "amonio", "pH": "ph", "Osmolality": "osmolalidad",
              "[Glucose]": "glucosa", "pO2": "po2", "pCO2": "pco2"}


def persona_periodo(raw: pd.DataFrame, et: pd.DataFrame, adelanto: int) -> pd.DataFrame:
    """Una fila por cultivo y dia hasta el shift (o la censura)."""
    filas = []
    for cult, g in raw.sort_values("day").groupby("cult"):
        if cult not in et.index or et.loc[cult, "excluir"]:
            continue
        g = g.set_index("day")
        dias = np.arange(1, int(g.index.max()) + 1)
        g = g.reindex(dias)
        x = g[list(MEDICIONES)].ffill().rename(columns=MEDICIONES)  # solo hacia adelante
        x["dvcd"] = x["vcd"].diff()
        temp = g["Temperature"].ffill()
        rec = pd.DataFrame({"temperatura": temp,
                            "bajo_T_hoy": (temp.diff() < 0).astype(int),
                            # programado de antemano: se conoce el dia anterior
                            "baja_T_manana": (temp.shift(-1) < temp).astype(int)}, index=dias)
        X = x.join(rec)
        if adelanto == 2:  # se predice con lo que se sabia el dia anterior...
            med = [c for c in X.columns if c not in ("baja_T_manana",)]
            X[med] = X[med].shift(1)
            # ...y la receta programada sigue conociendose: baja manana o pasado
            X["baja_T_manana"] = ((temp.shift(-1) < temp) | (temp.diff() < 0)).astype(int)
        evento, dia_ev, tiempo = et.loc[cult, ["evento", "dia", "tiempo"]]
        ultimo = int(dia_ev) if evento else int(tiempo)
        for d in range(2, ultimo + 1):
            fila = X.loc[d].to_dict()
            fila.update({"cult": cult, "dia": d,
                         **{f"d{k}": int(d == k) for k in range(3, 12)}, "d12p": int(d >= 12), "y": int(bool(evento) and d == ultimo),
                         "grupo": round(float(g["Culture Volume"].dropna().iloc[0]), 5)})
            filas.append(fila)
    return pd.DataFrame(filas)


def modelo():
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=2000))])


def auc_cv(pp, cols, n_rep=5):
    aucs, briers = [], []
    for rep in range(n_rep):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        pred = np.full(len(pp), np.nan)
        for tr, te in cv.split(pp, pp["y"], pp["cult"]):
            m = modelo().fit(pp.iloc[tr][cols], pp.iloc[tr]["y"])
            pred[te] = m.predict_proba(pp.iloc[te][cols])[:, 1]
        aucs.append(roc_auc_score(pp["y"], pred))
        briers.append(brier_score_loss(pp["y"], pred))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(briers))


def main() -> None:
    raw = cargar("Raw Data")
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")
    medic = list(MEDICIONES.values()) + ["dvcd"]
    receta = ["temperatura", "bajo_T_hoy", "baja_T_manana"]
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]  # dia 2 = referencia
    conjuntos = {"M0 dia": dia, "M1 dia + receta": dia + receta,
                 "M2 dia + mediciones": dia + medic,
                 "M3 dia + receta + mediciones": dia + receta + medic}

    resumen = []
    for adelanto in (1, 2):
        pp = persona_periodo(raw, et, adelanto)
        print(f"\n=== Adelanto de {adelanto} dia(s): {len(pp)} filas, {pp['cult'].nunique()} cultivos, "
              f"{int(pp['y'].sum())} eventos ===")
        for nombre, cols in conjuntos.items():
            auc, sd, br = auc_cv(pp, cols)
            resumen.append({"adelanto_dias": adelanto, "modelo": nombre,
                            "auc": round(auc, 3), "auc_sd": round(sd, 3), "brier": round(br, 4)})
            print(f"  {nombre:30s} AUC {auc:.3f} +- {sd:.3f} | Brier {br:.4f}")
        if adelanto == 1:
            pp1 = pp

    pd.DataFrame(resumen).to_csv(TABLAS / "estado_auc.csv", index=False)

    # --- transferencia: dejar fuera un grupo de volumen ------------------
    print("\n=== M3 dejando fuera un grupo de volumen (adelanto 1 dia) ===")
    cols = conjuntos["M3 dia + receta + mediciones"]
    logo = []
    for gr, n in pp1.groupby("grupo")["cult"].nunique().items():
        if n < 15:
            continue
        te = pp1["grupo"] == gr
        m = modelo().fit(pp1.loc[~te, cols], pp1.loc[~te, "y"])
        p = m.predict_proba(pp1.loc[te, cols])[:, 1]
        m0 = modelo().fit(pp1.loc[~te, dia], pp1.loc[~te, "y"])
        p0 = m0.predict_proba(pp1.loc[te, dia])[:, 1]
        logo.append({"grupo": f"{gr:.5f}", "cultivos": n, "auc_M3": roc_auc_score(pp1.loc[te, "y"], p),
                     "auc_M0_dia": roc_auc_score(pp1.loc[te, "y"], p0)})
    logo = pd.DataFrame(logo).round(3)
    logo.to_csv(TABLAS / "estado_logo.csv", index=False)
    print(logo.to_string(index=False))

    # --- coeficientes con bootstrap por cultivo ---------------------------
    print("\n=== Coeficientes de M3 (adelanto 1 dia): odds ratio por desviacion estandar ===")
    full = modelo().fit(pp1[cols], pp1["y"])
    coef = pd.Series(full.named_steps["m"].coef_[0], index=cols)
    rng = np.random.default_rng(SEED)
    cultivos = pp1["cult"].unique()
    boot = []
    for _ in range(300):
        elegidos = rng.choice(cultivos, size=len(cultivos), replace=True)
        b = pd.concat([pp1[pp1["cult"] == c] for c in elegidos])
        if b["y"].nunique() < 2:
            continue
        boot.append(modelo().fit(b[cols], b["y"]).named_steps["m"].coef_[0])
    boot = np.array(boot)
    tabla = pd.DataFrame({"odds_ratio": np.exp(coef),
                          "ic_bajo": np.exp(np.percentile(boot, 2.5, axis=0)),
                          "ic_alto": np.exp(np.percentile(boot, 97.5, axis=0))}, index=cols)
    tabla = tabla.drop(index=dia)  # los indicadores de dia son el riesgo basal
    tabla["ic_excluye_1"] = (tabla["ic_bajo"] > 1) | (tabla["ic_alto"] < 1)
    tabla = tabla.reindex(tabla["odds_ratio"].apply(lambda v: abs(np.log(v))).sort_values(ascending=False).index)
    tabla.round(3).to_csv(TABLAS / "estado_coeficientes.csv")
    print(tabla.round(2).to_string())
    print("\nOdds ratio > 1: valores altos de la variable se asocian con que HOY sea el ultimo "
          "dia de produccion (shift inminente). < 1: lo retrasan. Asociacion, no causa.")


if __name__ == "__main__":
    main()
