"""
22 - Respuesta a las preguntas de la Dra. Kontoravdi (correo del 13 oct 2026).

El proyecto esta congelado; estos tres analisis son una excepcion porque los
pide ella o porque responden directamente a sus preguntas. Los criterios de
exito se fijaron ANTES de correr el script (ver notas del proyecto):

A. ANTICIPACION DE 2 DIAS
   Mismo modelo de riesgo diario que el script 15 (QC de gases, RobustScaler,
   L2 C = 1, validacion agrupada por cultivo 5 x 5, bootstrap de cultivos),
   pero con las mediciones hasta el dia t-1 (adelanto = 2 en el script 12).
   La receta de temperatura se sigue conociendo porque estaba programada.
   Exito: el IC 95% de la mejora sobre el modelo de solo el dia no incluye 0.
   Se agrega un modelo reducido (dia + receta + VCD y su cambio) para ver si
   casi toda la senal viene del crecimiento (descriptivo, sin criterio).

B. DEJANDO FUERA UN GRUPO DE VOLUMEN
   Se entrena con todos los cultivos menos los de un grupo grande (>= 15
   cultivos) y se evalua en ese grupo. Los grupos chicos siempre entrenan.
   Se compara contra el modelo de solo el dia entrenado igual.
   Exito: el modelo completo le gana al de solo el dia en al menos 2 de 3
   grupos (diferencia de AUC > 0; se reporta el IC por bootstrap de los
   cultivos del grupo evaluado).

C. PCA DE LOS CULTIVOS (descriptivo, sin prueba de hipotesis)
   Una fila por cultivo: trayectorias de los dias 1 a 9 (el ultimo dia que
   tienen todos los cultivos) de VCD, glucosa, glutamina, glutamato, amonio,
   pH, pCO2, pO2, osmolalidad y lactato. Huecos dentro del cultivo por
   interpolacion lineal y, si siguen, mediana de la columna; luego
   estandarizacion. Se colorea por cambio de temperatura, grupo de volumen y
   dia del shift, y se repite dentro del grupo que tiene cultivos con y sin
   cambio de temperatura (para separar algo de temperatura y proceso).
   Se reporta la varianza explicada y que fraccion de la varianza de PC1 y
   PC2 se asocia al grupo y al cambio de temperatura (eta cuadrada).

Ejecutar:  python analysis/22_preguntas_kontoravdi.py

Desarrollado por Arturo Rodriguez.
"""

import warnings
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from _comun import FIGURAS, TABLAS, cargar  # noqa: E402

warnings.filterwarnings("ignore")
est = import_module("12_estado_celular")
c15 = import_module("15_correcciones_auditoria")

DIA = [f"d{k}" for k in range(3, 12)] + ["d12p"]
RECETA = ["temperatura", "bajo_T_hoy", "baja_T_manana"]
MEDIC = list(est.MEDICIONES.values()) + ["dvcd"]
M3 = DIA + RECETA + MEDIC
SOLO_VCD = DIA + RECETA + ["vcd", "dvcd"]


# ---------------------------------------------------------------- A
def anticipacion(raw_l, et):
    filas, por_dia = [], []
    for adelanto in (1, 2):
        pp = est.persona_periodo(raw_l, et, adelanto=adelanto)
        p0 = c15.predicciones_fuera_de_fold(pp, DIA)
        for nombre, cols in (("completo", M3), ("dia + receta + VCD", SOLO_VCD)):
            p = c15.predicciones_fuera_de_fold(pp, cols)
            ic, ic_dif = c15.bootstrap_auc(pp, p0, p)
            auc0, auc = roc_auc_score(pp["y"], p0), roc_auc_score(pp["y"], p)
            filas.append({"adelanto_dias": adelanto, "modelo": nombre, "auc_solo_dia": auc0, "auc": auc,
                          "auc_ic_bajo": ic[0], "auc_ic_alto": ic[1], "mejora": auc - auc0,
                          "mejora_ic_bajo": ic_dif[0], "mejora_ic_alto": ic_dif[1],
                          "filas": len(pp), "eventos": int(pp["y"].sum())})
            if nombre == "completo":
                for d_, g in pp.assign(p=p).groupby("dia"):
                    if g["y"].sum() >= 5 and (g["y"] == 0).sum() >= 5:
                        por_dia.append({"adelanto_dias": adelanto, "dia": d_, "eventos": int(g["y"].sum()),
                                        "en_riesgo": len(g), "auc": roc_auc_score(g["y"], g["p"])})
    return pd.DataFrame(filas), pd.DataFrame(por_dia)


# ---------------------------------------------------------------- B
def grupo_fuera(raw_l, et, n_boot=c15.N_BOOT_AUC):
    pp = est.persona_periodo(raw_l, et, adelanto=1)
    n_por_grupo = pp.groupby("grupo")["cult"].nunique()
    grandes = [g for g, n in n_por_grupo.items() if n >= 15]
    rng = np.random.default_rng(c15.SEED)
    filas = []
    for g in grandes:
        tr, te = pp[pp["grupo"] != g], pp[pp["grupo"] == g].copy()
        # columnas constantes en el grupo evaluado no estorban: el modelo se ajusta con el resto
        p0 = c15.modelo_robusto().fit(tr[DIA], tr["y"]).predict_proba(te[DIA])[:, 1]
        p3 = c15.modelo_robusto().fit(tr[M3], tr["y"]).predict_proba(te[M3])[:, 1]
        y = te["y"].to_numpy()
        auc0, auc3 = roc_auc_score(y, p0), roc_auc_score(y, p3)
        cults = te["cult"].unique()
        idx = {c: np.where(te["cult"].to_numpy() == c)[0] for c in cults}
        difs = []
        for _ in range(n_boot):
            f = np.concatenate([idx[c] for c in rng.choice(cults, len(cults), replace=True)])
            if y[f].min() == y[f].max():
                continue
            difs.append(roc_auc_score(y[f], p3[f]) - roc_auc_score(y[f], p0[f]))
        filas.append({"grupo_fuera": f"{g:.5f}", "cultivos": len(cults), "eventos": int(y.sum()),
                      "auc_solo_dia": auc0, "auc_completo": auc3, "mejora": auc3 - auc0,
                      "mejora_ic_bajo": np.percentile(difs, 2.5), "mejora_ic_alto": np.percentile(difs, 97.5)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- C
VARS_PCA = {"VCD": "VCD", "[Glucose]": "glucose", "[Glutamine]": "glutamine", "[Glutamate]": "glutamate",
            "[NH3]": "ammonium", "pH": "pH", "pCO2": "pCO2", "pO2": "pO2", "Osmolality": "osmolality",
            "[Lactate]": "lactate"}
DIAS_PCA = range(1, 10)


def matriz_pca(raw_l, cultivos):
    filas = {}
    for c in cultivos:
        g = raw_l[raw_l["cult"] == c].set_index("day").sort_index()
        g = g[~g.index.duplicated()].reindex(range(1, int(g.index.max()) + 1))
        g = g[list(VARS_PCA)].interpolate(limit_area="inside").loc[list(DIAS_PCA)]
        filas[c] = {f"{VARS_PCA[v]}_d{d}": g.loc[d, v] for v in VARS_PCA for d in DIAS_PCA}
    X = pd.DataFrame(filas).T
    faltantes = X.isna().mean().mean()
    X = X.fillna(X.median())
    return X, faltantes


def eta2(valores, etiquetas):
    """Fraccion de la varianza de un componente que se explica por una variable categorica."""
    df = pd.DataFrame({"v": valores, "e": etiquetas})
    total = ((df["v"] - df["v"].mean()) ** 2).sum()
    entre = df.groupby("e")["v"].apply(lambda s: len(s) * (s.mean() - df["v"].mean()) ** 2).sum()
    return entre / total


def pca(raw_l, et):
    rec = pd.read_csv(TABLAS / "receta_temperatura.csv", index_col="cult")
    info = et[~et["excluir"]].join(rec[["tshift"]])
    vol = raw_l.groupby("cult")["Culture Volume"].first().round(5)
    info["grupo"] = vol.reindex(info.index)
    X, faltantes = matriz_pca(raw_l, info.index)
    Z = StandardScaler().fit_transform(X)
    modelo = PCA(n_components=5).fit(Z)
    S = pd.DataFrame(modelo.transform(Z)[:, :2], index=X.index, columns=["PC1", "PC2"]).join(info)
    cargas = pd.DataFrame(modelo.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
    # carga promedio por variable (suma sobre dias) para leer que significa cada eje
    cargas_var = cargas.groupby(cargas.index.str.rsplit("_d", n=1).str[0]).mean()
    resumen = {"faltantes_rellenados": faltantes,
               **{f"var_explicada_PC{i + 1}": v for i, v in enumerate(modelo.explained_variance_ratio_)}}
    for pc in ("PC1", "PC2"):
        resumen[f"eta2_{pc}_grupo"] = eta2(S[pc], S["grupo"])
        resumen[f"eta2_{pc}_tshift"] = eta2(S[pc], S["tshift"])
    # dentro del grupo con cultivos con y sin cambio de temperatura (>= 5 de cada uno)
    mixto = [g for g, s in S.groupby("grupo") if (s["tshift"] == 1).sum() >= 5 and (s["tshift"] == 0).sum() >= 5]
    dentro = None
    if mixto:
        g = mixto[0]
        sub = X.loc[S.index[S["grupo"] == g]]
        Zg = StandardScaler().fit_transform(sub)
        mg = PCA(n_components=2).fit(Zg)
        dentro = pd.DataFrame(mg.transform(Zg), index=sub.index, columns=["PC1", "PC2"]).join(info)
        resumen["grupo_mixto"] = f"{g:.5f}"
        resumen["grupo_mixto_n_con_T"] = int((dentro["tshift"] == 1).sum())
        resumen["grupo_mixto_n_sin_T"] = int((dentro["tshift"] == 0).sum())
        resumen["grupo_mixto_var_PC1"] = mg.explained_variance_ratio_[0]
        resumen["grupo_mixto_var_PC2"] = mg.explained_variance_ratio_[1]
        resumen["grupo_mixto_eta2_PC1_tshift"] = eta2(dentro["PC1"], dentro["tshift"])
        resumen["grupo_mixto_eta2_PC2_tshift"] = eta2(dentro["PC2"], dentro["tshift"])
    return S, cargas_var, pd.Series(resumen), dentro


def figura_pca(S, dentro, resumen):
    AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#8a8986"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    v1, v2 = resumen["var_explicada_PC1"], resumen["var_explicada_PC2"]
    xl, yl = f"PC1 ({v1:.0%} of variance)", f"PC2 ({v2:.0%})"
    for val, col, lab in ((0, GRIS, "no temperature shift"), (1, AZUL, "temperature shift")):
        s = S[S["tshift"] == val]
        ax[0].scatter(s["PC1"], s["PC2"], s=18, color=col, label=f"{lab} (n={len(s)})", edgecolor="white", lw=0.5)
    ax[0].set_title("A · by temperature shift", loc="left")
    # grupos ordenados por volumen (de menor a mayor), igual que en la tabla de "grupo fuera"
    conteo = S["grupo"].value_counts()
    grandes = sorted(g for g, n in conteo.items() if n >= 15)
    nombres = ["lowest volume", "intermediate volume", "highest volume"]
    paleta = [AQUA, NARANJA, AZUL]
    for g, col, nom in zip(grandes, paleta, nombres):
        s = S[S["grupo"] == g]
        ax[1].scatter(s["PC1"], s["PC2"], s=18, color=col, label=f"{nom} (n={len(s)})", edgecolor="white", lw=0.5)
    s = S[~S["grupo"].isin(grandes)]
    ax[1].scatter(s["PC1"], s["PC2"], s=18, color=GRIS, edgecolor="white", lw=0.5)
    ax[1].scatter([], [], color=GRIS, s=18, label=f"small groups (n={len(S[~S['grupo'].isin(grandes)])})")
    ax[1].set_title("B · by volume group", loc="left")
    sc = ax[2].scatter(S["PC1"], S["PC2"], s=18, c=S["dia"].where(S["evento"]), cmap="viridis",
                       edgecolor="white", lw=0.5)
    cens = S[~S["evento"]]
    ax[2].scatter(cens["PC1"], cens["PC2"], s=18, color=GRIS, label="no shift")
    fig.colorbar(sc, ax=ax[2], label="shift day")
    ax[2].set_title("C · by shift day", loc="left")
    for a in ax[:3]:
        a.set_xlabel(xl); a.set_ylabel(yl)
    for a in ax[:2]:
        a.legend(frameon=False, fontsize=7)
    ax[2].legend(frameon=False, fontsize=7)
    if dentro is not None:
        for val, col, lab in ((0, GRIS, "no T shift"), (1, AZUL, "T shift")):
            s = dentro[dentro["tshift"] == val]
            ax[3].scatter(s["PC1"], s["PC2"], s=18, color=col, label=f"{lab} (n={len(s)})", edgecolor="white", lw=0.5)
        ax[3].set_title("D · PCA within the mixed volume group", loc="left")
        ax[3].set_xlabel(f"PC1 ({resumen['grupo_mixto_var_PC1']:.0%})")
        ax[3].set_ylabel(f"PC2 ({resumen['grupo_mixto_var_PC2']:.0%})")
        ax[3].legend(frameon=False, fontsize=7)
    fig.text(0.5, -0.04, "One point per culture: days 1-9 of VCD, glucose, glutamine, glutamate, ammonium, pH, "
             "pCO2, pO2, osmolality and lactate (min-max data, standardised). Descriptive only. "
             "Panel A omits culture C27 (no temperature record); 104 cultures in B-C.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig6_pca_cultivos_en.png")
    plt.close(fig)


def main() -> None:
    raw = cargar("Raw Data")
    raw_l, _ = c15.qc_gases(raw)
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")

    print("=== A. Anticipacion de 1 y 2 dias ===")
    a, a_dia = anticipacion(raw_l, et)
    print(a.round(3).to_string(index=False))
    print(a_dia.round(3).to_string(index=False))
    a.round(4).to_csv(TABLAS / "kontoravdi_anticipacion.csv", index=False)
    a_dia.round(3).to_csv(TABLAS / "kontoravdi_anticipacion_por_dia.csv", index=False)

    print("\n=== B. Dejando fuera un grupo de volumen ===")
    b = grupo_fuera(raw_l, et)
    print(b.round(3).to_string(index=False))
    print("criterio (gana en >= 2 de 3):", int((b["mejora"] > 0).sum()), "de", len(b))
    b.round(4).to_csv(TABLAS / "kontoravdi_grupo_fuera.csv", index=False)

    print("\n=== C. PCA ===")
    S, cargas, resumen, dentro = pca(raw_l, et)
    print(resumen.apply(lambda v: round(float(v), 3) if isinstance(v, (int, float, np.floating, np.integer)) else v).to_string())
    print("cargas promedio por variable:")
    print(cargas.round(3).sort_values("PC1").to_string())
    figura_pca(S, dentro, resumen)
    resumen.to_frame("valor").to_csv(TABLAS / "kontoravdi_pca_resumen.csv")
    cargas.round(4).to_csv(TABLAS / "kontoravdi_pca_cargas.csv")
    S.round(4).to_csv(TABLAS / "kontoravdi_pca_puntajes.csv")


if __name__ == "__main__":
    main()
