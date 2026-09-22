"""
06 — Experimentos surgidos de una auditoria externa del proyecto.

Controles adicionales sobre objeciones que el analisis original no
respondia.

Las cuatro preguntas:

A. .El modelo solo extrapola la trayectoria de lactato que ya empezo?
   La etiqueta se construye de la curva futura de lactato, y el lactato
   temprano esta entre las variables. Si el modelo pierde casi todo su
   desempeno al quitarle el lactato, lo que hace es continuar una curva, no
   anticipar un cambio metabolico.

B. .El leave-one-scale-out mide transferencia de verdad?
   ``escala`` esta entre las variables, asi que el modelo conoce el volumen de
   la escala nueva aunque no haya visto cultivos de ella. Repetirlo sin esa
   variable mide transferencia mas limpia.

C. .Las conclusiones aguantan otras definiciones del evento?
   La sensibilidad original solo contaba eventos y dias. No decia si el MAE
   cambia cuando cambia la definicion, que es lo que importa.

D. .La correccion del pico (``refine_peak``) sesga los resultados?
   Movio los excluidos de 6 a 16 cultivos. Hay que ver el efecto sobre el
   desempeno, no solo sobre el conteo.

Ejecutar:  python analysis/06_respuesta_auditoria.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lactateshift import detect_shift_batch, early_window_features
from lactateshift.validate import leave_one_group_out
from _comun import TABLAS, VENTANA, cargar

warnings.filterwarnings("ignore")
SEED = 0
ALPHAS = np.logspace(-3, 3, 25)
NUCLEO_COLS = ["[Lactate]", "[Glucose]", "VCD", "[NH3]", "pH", "[Glutamate]"]


def pipe(m):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("esc", StandardScaler()), ("mod", m)])


def mae_cv(m, X, y, n_rep=5, seed=SEED) -> tuple[float, float]:
    cv = RepeatedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    v = [mean_absolute_error(y.iloc[te], pipe(m).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]))
         for tr, te in cv.split(X)]
    return float(np.mean(v)), float(np.std(v))


def rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                 random_state=SEED, n_jobs=-1)


def cargar_tabla():
    t = pd.read_csv(TABLAS / "features_d1_4.csv", index_col="cult")
    t = t[~t["excluir"]]
    meta = ["evento", "dia_evento", "tiempo", "excluir"]
    ext = [c for c in t.columns if c.startswith(("glutamine", "osmolality"))]
    nucleo = [c for c in t.columns if c not in meta + ext]
    return t, nucleo


# ---------------------------------------------------------------- A
def experimento_a(t, nucleo):
    print("=" * 72)
    print("A. .El modelo solo continua la trayectoria de lactato?")
    print("=" * 72)
    con_ev = t[t["evento"]]
    y = con_ev["dia_evento"]

    sin_lac = [c for c in nucleo if not c.startswith(("lactate", "lactate_over"))]
    solo_lac = [c for c in nucleo if c.startswith(("lactate", "lactate_over"))]

    filas = []
    for nombre, cols in [("baseline (mediana)", nucleo), ("todas", nucleo),
                         ("solo variables de lactato", solo_lac),
                         ("SIN ninguna variable de lactato", sin_lac)]:
        m = DummyRegressor(strategy="median") if nombre.startswith("baseline") else rf()
        mu, sd = mae_cv(m, con_ev[cols], y)
        filas.append({"conjunto": nombre, "n_vars": len(cols),
                      "MAE": round(mu, 3), "sd": round(sd, 3)})
    d = pd.DataFrame(filas)
    print(d.to_string(index=False))
    print("\nLectura: si 'SIN lactato' se acerca a 'todas', el desempeno no")
    print("depende de extrapolar la curva de lactato. Si se desploma hacia el")
    print("baseline, la objecion de la auditoria es correcta.")
    d.to_csv(TABLAS / "aud_a_sin_lactato.csv", index=False)


# ---------------------------------------------------------------- B
def experimento_b(t, nucleo):
    print("\n" + "=" * 72)
    print("B. Leave-one-scale-out sin darle al modelo la variable 'escala'")
    print("=" * 72)
    con_ev = t[t["evento"]]
    y = con_ev["dia_evento"]
    sin_escala = [c for c in nucleo if c != "escala"]

    for etiqueta, cols in [("con 'escala' entre las variables", nucleo),
                           ("SIN 'escala'", sin_escala)]:
        print(f"\n-- {etiqueta} --")
        for nombre, m in [("baseline", DummyRegressor(strategy="median")),
                          ("ridge", RidgeCV(alphas=ALPHAS)), ("random forest", rf())]:
            def fp(Xtr, ytr, Xte, _m=m):
                return pipe(_m).fit(Xtr, ytr).predict(Xte)
            d = leave_one_group_out(fp, con_ev[cols], y, con_ev["escala"],
                                    mean_absolute_error, min_group_size=15)
            print(f"  {nombre:14s} MAE ponderado {d.attrs['weighted_mean']:.3f}   "
                  f"por escala: {[round(v,3) for v in d['score']]}")
    print("\nNota: son solo tres escalas (43, 18 y 16 cultivos). Permite decir")
    print("que no transfiere ENTRE ESTAS TRES, no que no transfiera en general.")


# ---------------------------------------------------------------- C y D
def reconstruir(raw, **kw) -> tuple[pd.DataFrame, list[str], int]:
    """Rehace etiquetas y variables con otros parametros de deteccion."""
    et = detect_shift_batch(raw, id_col="cult", day_col="day",
                            value_col="[Lactate]", **kw)
    et["excluir"] = et["occurred"] & (et["day"] <= VENTANA)
    X = early_window_features(raw, id_col="cult", day_col="day",
                              value_cols=NUCLEO_COLS, window=VENTANA,
                              ratios=[("[Lactate]", "[Glucose]"), ("[Lactate]", "VCD")],
                              static_cols=["Culture Volume"])
    X.columns = [c.replace("[", "").replace("]", "").replace("+", "")
                  .replace("-", "").replace(" ", "_").lower() for c in X.columns]
    X = X.rename(columns={"culture_volume": "escala"})
    tabla = X.join(et[["occurred", "day"]].rename(
        columns={"occurred": "evento", "day": "dia_evento"})).join(et[["excluir"]])
    n_excluidos = int(tabla["excluir"].sum())
    return tabla[~tabla["excluir"]], list(X.columns), n_excluidos


def experimento_cd(raw):
    print("\n" + "=" * 72)
    print("C y D. .El desempeno aguanta otras definiciones del evento?")
    print("=" * 72)
    print("La sensibilidad original solo contaba eventos. Aqui se rehace el")
    print("modelo completo con cada definicion y se compara contra su propio")
    print("baseline, que cambia cuando cambia el conjunto.\n")

    filas = []
    configs = [(3, 2, 0.30, True), (3, 2, 0.30, False), (1, 2, 0.30, True),
               (5, 2, 0.30, True), (3, 3, 0.30, True), (3, 2, 0.20, True),
               (3, 2, 0.40, True), (3, 2, 0.50, True)]
    for suav, consec, umbral, refinar in configs:
        t2, cols, n_exc = reconstruir(raw, smooth_window=suav, n_consecutive=consec,
                                      drop_threshold=umbral, refine_peak=refinar)
        con_ev = t2[t2["evento"]]
        if len(con_ev) < 30:
            continue
        y = con_ev["dia_evento"]
        base, _ = mae_cv(DummyRegressor(strategy="median"), con_ev[cols], y, n_rep=3)
        modelo, _ = mae_cv(rf(), con_ev[cols], y, n_rep=3)
        filas.append({"suav": suav, "consec": consec, "umbral": umbral,
                      "refine": refinar, "n": len(con_ev),
                      "excluidos": n_exc,
                      "baseline": round(base, 3), "RF": round(modelo, 3),
                      "mejora": round(base - modelo, 3)})
    d = pd.DataFrame(filas)
    print(d.to_string(index=False))
    d.to_csv(TABLAS / "aud_cd_sensibilidad_desempeno.csv", index=False)
    print(f"\nMejora sobre el baseline en todas las definiciones probadas: "
          f"min {d['mejora'].min():.3f}, max {d['mejora'].max():.3f}, "
          f"mediana {d['mejora'].median():.3f}")
    print("Si la mejora es positiva en todas, la conclusion no depende de los")
    print("parametros elegidos. Si cambia de signo en alguna, si depende.")


def main():
    t, nucleo = cargar_tabla()
    raw = cargar("Raw Data")
    experimento_a(t, nucleo)
    experimento_b(t, nucleo)
    experimento_cd(raw)
    print(f"\nTablas en {TABLAS}")


if __name__ == "__main__":
    main()
