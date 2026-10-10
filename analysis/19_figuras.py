"""
19 - Figuras finales del analisis actual (scripts 11-18).

Genera en outputs/figuras/:
  fig1_perfiles_modelos.png   lactato observado vs modelo 1 y modelo 2 en 6 cultivos
                              de ejemplo (validacion por cultivo), con el shift real
  fig2_riesgo_diario.png      fraccion F predicha (probabilidad acumulada de que el
                              shift ya ocurrio) en los mismos cultivos
  fig3_asociaciones.png       razones de momios por IQR con IC: todos los cultivos y
                              por grupo de volumen
  fig4_temperatura.png        coincidencia con el cambio de temperatura vs azar, y
                              dia del shift vs VCD del dia 3
  fig5_comparacion_modelos.png error a 2 dias y acierto del shift por modelo

Colores: paleta categorica validada (azul, naranja, aqua) en orden fijo; el gris
se reserva para referencias (modelo 1, azar, todos los cultivos).
Los ejemplos se eligen de forma determinista (primer cultivo, en orden, con
shift en los dias 4, 5, 6, 7 y 8, mas uno sin shift), no por que se vean bien.

Ejecutar:  python analysis/19_figuras.py

Desarrollado por Arturo Rodriguez.
"""

import warnings
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from _comun import FIGURAS, TABLAS, cargar  # noqa: E402

warnings.filterwarnings("ignore")
est = import_module("12_estado_celular")
c15 = import_module("15_correcciones_auditoria")
m1 = import_module("17_modelo1_crecimiento")
m2 = import_module("18_modelo2_dos_estados")

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIS, TINTA, TINTA2 = "#8a8986", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c8c7c2", "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def ejemplos(et, S):
    """Un cultivo por dia del shift (4 a 8) y uno censurado.

    En cada grupo se toma el cultivo con el error mediano del modelo 2 (RMSE del
    perfil, fuera de fold), para que los ejemplos sean tipicos y no los mejores
    ni los peores. Una version anterior tomaba el primero por orden alfabetico y
    salieron seis cultivos que el modelo subestima, que no representan el
    conjunto (sesgo mediano del modelo 2 sobre todos: casi cero).
    """
    err = pd.read_csv(TABLAS / "modelo2_por_cultivo.csv", index_col="cult")["M2_rmse_pred"]
    mod = et[(~et["excluir"])]

    def mediano(cands):
        e = err.reindex([x for x in cands if x in S]).dropna().sort_values()
        return e.index[(len(e) - 1) // 2] if len(e) else None

    elegidos = [mediano(mod.index[mod["evento"] & (mod["dia"] == d)]) for d in (4, 5, 6, 7, 8)]
    elegidos.append(mediano(mod.index[~mod["evento"]]))
    return [c for c in elegidos if c is not None]


def simulaciones(raw_l, et, S, cultivos_ej):
    """Simulaciones fuera de fold (misma particion que los scripts 17-18) para los ejemplos."""
    pp = est.persona_periodo(raw_l, et, adelanto=1)
    ciego = et.copy(); ciego["evento"] = False; ciego["tiempo"] = ciego["ultimo_dia"]
    ppc = est.persona_periodo(raw_l, ciego, adelanto=1)
    dia = [f"d{k}" for k in range(3, 12)] + ["d12p"]
    m3 = dia + ["temperatura", "bajo_T_hoy", "baja_T_manana"] + list(est.MEDICIONES.values()) + ["dvcd"]
    cult = np.array(sorted(S))
    out = {}
    for tr_i, te_i in GroupKFold(n_splits=5).split(cult, groups=cult):
        tr, te = cult[tr_i], cult[te_i]
        te_ej = [c for c in te if c in cultivos_ej]
        if not te_ej:
            continue
        todo = pd.concat([S[c] for c in tr])
        mu = (todo["ph_prev"].mean(), todo["pco2_prev"].mean())
        sd = (todo["ph_prev"].std(), todo["pco2_prev"].std())
        p1, _, _ = m1.ajustar(S, tr, 0.0, mu, sd)
        lam1 = m1.lambda_pooling(S, tr, p1, 0.0, mu, sd)
        p2 = m2.ajustar2(S, tr, et, mu, sd, True)
        Fp = m2.riesgo_predicho(ppc, pp[pp["cult"].isin(tr)], m3, te_ej)
        for c in te_ej:
            s = S[c]
            sim1 = m1.simular(s, p1, m1.alfa_condicionada(s, p1, 0.0, mu, sd, lam1), 0.0, mu, sd)
            F = Fp[c].reindex(s.index).fillna(0).to_numpy()
            sim2 = m2.simular2(s, F, p2, m2.alfa_cond2(s, F, p2, mu, sd, lam1, True), mu, sd, True)
            out[c] = pd.DataFrame({"obs": s["L"], "m1": sim1, "m2": sim2, "F": F}, index=s.index)
    return out


def fig_perfiles(sims, raw, et, orden):
    fig, ejes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=False)
    for ax, c in zip(ejes.flat, orden):
        d = sims[c]
        completo = raw[raw["cult"] == c].set_index("day")["[Lactate]"]
        ax.plot(completo.index, completo.values, "o", ms=3.5, color="#c8c7c2", zorder=1)
        ax.plot(d.index, d["obs"], "o", ms=4, color=TINTA, label="observado", zorder=3)
        ax.plot(d.index, d["m1"], "--", lw=2, color=GRIS, label="modelo 1 (sin interruptor)", zorder=2)
        ax.plot(d.index, d["m2"], "-", lw=2, color=AZUL, label="modelo 2 (dos estados)", zorder=4)
        if et.loc[c, "evento"]:
            # el dia ya va en el titulo; una etiqueta de texto dentro del panel tapaba puntos
            ax.axvline(et.loc[c, "dia"], color=NARANJA, lw=1.5, ls=":", zorder=0, label="shift real (etiqueta)")
            titulo = f"{c} · shift real día {int(et.loc[c, 'dia'])}"
        else:
            titulo = f"{c} · sin shift"
        ax.set_title(titulo, fontsize=9, color=TINTA, loc="left")
        ax.set_xlabel("día")
    for ax in ejes[:, 0]:
        ax.set_ylabel("lactato (normalizado)")
    h, l = ejes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.03, "Puntos grises: días fuera de la ventana modelada (después del declive de la VCD). "
             "Simulaciones fuera de fold; VCD medida.", ha="center", color=TINTA2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig1_perfiles_modelos.png")
    plt.close(fig)


def fig_riesgo(sims, et, orden):
    fig, ejes = plt.subplots(2, 3, figsize=(10, 5.2), sharey=True)
    for ax, c in zip(ejes.flat, orden):
        d = sims[c]
        ax.step(d.index, d["F"], where="post", color=AZUL, lw=2)
        ax.axhline(0.5, color="#c8c7c2", lw=1)
        if et.loc[c, "evento"]:
            ax.axvline(et.loc[c, "dia"], color=NARANJA, lw=1.5, ls=":")
            ax.set_title(f"{c} · shift real día {int(et.loc[c, 'dia'])}", fontsize=9, loc="left", color=TINTA)
        else:
            ax.set_title(f"{c} · sin shift", fontsize=9, loc="left", color=TINTA)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("día")
    for ax in ejes[:, 0]:
        ax.set_ylabel("F predicha\n(prob. de que el shift ya ocurrió)")
    fig.suptitle("Fracción en estado de consumo predicha por el modelo de riesgo diario "
                 "(línea punteada naranja: shift real)", fontsize=10, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig2_riesgo_diario.png")
    plt.close(fig)


def fig_asociaciones():
    conj = pd.read_csv(TABLAS / "or_por_iqr_conjunto.csv", index_col=0)
    grp = pd.read_csv(TABLAS / "or_por_iqr_por_grupo.csv")
    variables = ["vcd", "ph", "pco2", "amonio", "glutamina", "temperatura"]
    nombres = {"vcd": "VCD", "ph": "pH", "pco2": "pCO2", "amonio": "amonio",
               "glutamina": "glutamina", "temperatura": "temperatura"}
    grupos = sorted(grp["grupo"].unique(), key=float)
    colores = dict(zip(grupos, (AZUL, NARANJA, AQUA)))
    marcas = dict(zip(grupos, ("o", "s", "^")))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, v in enumerate(variables):
        y0 = len(variables) - i
        f = conj.loc[v]
        ax.errorbar(f["or_por_iqr"], y0 + 0.3, xerr=[[f["or_por_iqr"] - f["ic_bajo"]], [f["ic_alto"] - f["or_por_iqr"]]],
                    fmt="D", color=TINTA, ms=6, lw=2, capsize=0, label="todos los cultivos" if i == 0 else None)
        for k, g in enumerate(grupos):
            r = grp[(grp["grupo"] == g) & (grp["variable"] == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r["or_por_iqr"], y0 - 0.1 - 0.15 * k,
                        xerr=[[r["or_por_iqr"] - r["ic_bajo"]], [r["ic_alto"] - r["or_por_iqr"]]],
                        fmt=marcas[g], color=colores[g], ms=5, lw=1.5, capsize=0,
                        label=f"grupo {float(g):.5f}" if i == 0 else None)
    ax.axvline(1, color=GRIS, lw=1)
    ax.set_xscale("log")
    ax.set_yticks([len(variables) - i for i in range(len(variables))])
    ax.set_yticklabels([nombres[v] for v in variables])
    ax.set_xlabel("razón de momios por IQR (escala log) · < 1 retrasa el shift, > 1 lo adelanta")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Asociación con el shift en el modelo de riesgo diario (IC 95%, bootstrap por cultivo)",
                 fontsize=10, loc="left", color=TINTA)
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig3_asociaciones.png")
    plt.close(fig)


def fig_temperatura(raw, et):
    rec = pd.read_csv(TABLAS / "receta_temperatura.csv", index_col="cult")
    v3 = raw[raw["day"] == 3].set_index("cult")["VCD"]
    d = et[(~et["excluir"]) & et["evento"]].join(rec).join(v3.rename("vcd_d3"))
    t = d[d["tshift"] == 1]
    obs = np.mean(np.abs(t["dia"] - t["tshift_day"]) <= 1)
    rng = np.random.default_rng(0)
    nulo = [np.mean(np.abs(t["dia"].to_numpy() - rng.permutation(t["tshift_day"].to_numpy())) <= 1) for _ in range(5000)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.hist(nulo, bins=np.arange(0.3, 1.0, 1 / len(t)), color="#c8c7c2", edgecolor="#fcfcfb")
    a.axvline(obs, color=AZUL, lw=2)
    a.text(obs, a.get_ylim()[1] * 0.9, f"  observado {obs:.0%}", color=TINTA, fontsize=8)
    a.text(np.mean(nulo), a.get_ylim()[1] * 0.98, f"azar {np.mean(nulo):.0%}", color=TINTA2, fontsize=8, ha="center")
    a.set_xlabel(f"fracción de los {len(t)} cultivos con shift a ±1 día del cambio de temperatura")
    a.set_ylabel("permutaciones")
    a.set_title("Coincidencia con el cambio de temperatura vs azar", fontsize=10, loc="left", color=TINTA)
    # R² de una recta dia ~ VCD dia 3, por subconjunto. Ojo: el 0.66 que se citaba
    # era solo de los 34 cultivos con cambio de temperatura, no de todos.
    r2 = lambda g: np.corrcoef(g["vcd_d3"], g["dia"])[0, 1] ** 2
    dv = d.dropna(subset=["vcd_d3"])
    for val, col, nombre in ((0, NARANJA, "sin cambio de T"), (1, AZUL, "con cambio de T")):
        g = dv[dv["tshift"] == val]
        b.scatter(g["vcd_d3"], g["dia"] + rng.uniform(-0.12, 0.12, len(g)), s=22, color=col,
                  edgecolor="#fcfcfb", lw=0.8, label=f"{nombre} (n={len(g)}, R² {r2(g):.2f})")
    b.set_xlabel("VCD día 3 (normalizada)")
    b.set_ylabel("día del shift (jitter ±0.12)")
    b.legend(frameon=False, fontsize=8)
    b.set_title(f"VCD del día 3 y día del shift (todos: n={len(dv)}, R² {r2(dv):.2f})", fontsize=10, loc="left",
                color=TINTA)
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig4_temperatura.png")
    plt.close(fig)


def fig_modelos():
    r = pd.read_csv(TABLAS / "modelo2_resumen.csv")
    r = r[r["modelo"].isin(["perfil promedio", "M1 sin interruptor (script 17)", "M2 dos estados (F predicho)",
                            "M2 control: riesgo solo con el dia", "M2 oraculo (F observado; techo)"])]
    # orden de arriba a abajo: referencias, control, modelo, techo
    orden = ["perfil promedio", "M1 sin interruptor (script 17)", "M2 control: riesgo solo con el dia",
             "M2 dos estados (F predicho)", "M2 oraculo (F observado; techo)"]
    r = r.set_index("modelo").loc[[m for m in orden if m in set(r["modelo"])]].reset_index()
    etiquetas = {"perfil promedio": "perfil promedio", "M1 sin interruptor (script 17)": "modelo 1 (sin interruptor)",
                 "M2 dos estados (F predicho)": "modelo 2 (interruptor predicho)",
                 "M2 control: riesgo solo con el dia": "modelo 2, riesgo solo del día (control)",
                 "M2 oraculo (F observado; techo)": "techo (interruptor perfecto)"}
    col = {"perfil promedio": GRIS, "M1 sin interruptor (script 17)": GRIS,
           "M2 dos estados (F predicho)": AZUL, "M2 control: riesgo solo con el dia": "#a9a8a2",
           "M2 oraculo (F observado; techo)": "#86b6ef"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
    y = np.arange(len(r))[::-1]
    for yi, (_, f) in zip(y, r.iterrows()):
        a.plot([0, f["rmse_2dias"]], [yi, yi], color=col[f["modelo"]], lw=2)
        a.plot(f["rmse_2dias"], yi, "o", color=col[f["modelo"]], ms=8)
        a.text(f["rmse_2dias"] + 0.0008, yi, f"{f['rmse_2dias']:.4f}", va="center", fontsize=8, color=TINTA)
        if pd.notna(f["shift_pm1"]):
            b.plot([0, f["shift_pm1"] * 100], [yi, yi], color=col[f["modelo"]], lw=2)
            b.plot(f["shift_pm1"] * 100, yi, "o", color=col[f["modelo"]], ms=8)
            b.text(f["shift_pm1"] * 100 + 1.5, yi, f"{f['shift_pm1']:.0%}", va="center", fontsize=8, color=TINTA)
    for ax in (a, b):
        ax.set_yticks(y)
        ax.set_yticklabels([etiquetas[m] for m in r["modelo"]])
    b.set_yticklabels([])
    a.set_xlabel("error del pronóstico a 2 días (RMSE, menor es mejor)")
    b.set_xlabel("shift simulado a ±1 día del real (%)")
    a.set_xlim(0, 0.05); b.set_xlim(0, 110)
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig5_comparacion_modelos.png")
    plt.close(fig)


def main() -> None:
    raw = cargar("Raw Data")
    raw_l, _ = c15.qc_gases(raw)
    et = pd.read_csv(TABLAS / "etiquetas_evento_tasa.csv", index_col="cult")
    S = m1.series(raw_l, et, m1.CORTE_VCD)
    orden = ejemplos(et, S)
    print("cultivos de ejemplo:", orden)
    sims = simulaciones(raw_l, et, S, orden)
    fig_perfiles(sims, raw, et, orden)
    fig_riesgo(sims, et, orden)
    fig_asociaciones()
    fig_temperatura(raw, et)
    fig_modelos()
    # outputs/ no se sube a git; el README muestra dos figuras de resumen (sin series
    # de cultivos individuales), asi que se copian a docs/figuras/, que si se versiona
    import shutil
    destino = FIGURAS.parents[1] / "docs" / "figuras"
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in ("fig3_asociaciones.png", "fig5_comparacion_modelos.png"):
        shutil.copy(FIGURAS / nombre, destino / nombre)
    print("figuras en", FIGURAS, "| copias para el README en", destino)


if __name__ == "__main__":
    main()
