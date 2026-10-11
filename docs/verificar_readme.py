"""Comprueba que cada cifra del README coincida con las salidas del analisis.

Cada entrada dice de donde sale el numero y como debe aparecer escrito. Si
alguna no aparece, el script termina con codigo 1.

Requiere haber corrido antes los scripts de analysis/ y el ejemplo:
    python examples/validar_detector_sintetico.py
    python docs/verificar_readme.py

Desarrollado por Arturo Rodriguez.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
TAB = RAIZ / "outputs" / "tablas"
sys.path.insert(0, str(RAIZ / "analysis"))
sys.path.insert(0, str(RAIZ / "src"))


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def pm(x):
    return f"{x:+.3f}"


def leer(nombre, **kw):
    return pd.read_csv(TAB / nombre, **kw)


def cifras() -> list[tuple[str, str]]:
    c: list[tuple[str, str]] = []

    # --- etiquetas
    et = leer("etiquetas_evento.csv", index_col="cult")
    mod = et[~et["excluir"]]
    ev = mod[mod["occurred"]]
    dias = ev["day"].value_counts().sort_index()
    c += [
        ("cultivos con shift", f"**{int(et['occurred'].sum())} de {len(et)} cultivos hacen el shift**"),
        ("excluidos", f"{int(et['excluir'].sum())} cultivos hacen el shift **dentro**"),
        ("conjunto de modelado",
         f"**{len(mod)} cultivos: {len(ev)} con evento y {len(mod) - len(ev)} censurados**"),
        ("distribucion de dias",
         f"({int(dias[5])} cultivos en el dia 5, {int(dias[6])} en el 6,\n"
         f"{int(dias[7])} en el 7, {int(dias[8])} en el 8, {int(dias[9])} en el 9 y "
         f"{int(dias[11])} en el 11)"),
        ("eventos a un dia del cierre",
         f"{int((ev['day'] == 5).sum())} de {len(ev)} eventos ocurren a un"),
        ("eventos en tres dias", f"{int(dias[[5, 6, 7]].sum())} de\nlos {len(ev)} eventos caen en tres dias"),
    ]

    # --- regla del maximo global, mismos parametros
    from _comun import cargar
    from lactateshift.detect import regularize, smooth
    raw = cargar("Raw Data")
    n_global = 0
    for _, d in raw.groupby("cult"):
        d = d.sort_values("day")
        _, v = regularize(d["day"], d["[Lactate]"])
        s = smooth(v, 3)
        i = int(np.argmax(s))
        if (i < len(s) - 2 and np.all(np.diff(s)[i:i + 2] < 0)
                and (s[i] - s[i + 1:].min()) / s[i] >= 0.30):
            n_global += 1
    c.append(("regla del maximo global", f"detecta {n_global} de 106 cultivos"))

    # --- refinamiento en datos reales
    rr = leer("refine_real.csv").set_index("desplazamiento")["cultivos"]
    movidos = int(rr.drop(0.0, errors="ignore").sum())
    c.append(("refinamiento en datos reales",
              f"mueve el dia en {movidos} de {int(rr.sum())} cultivos:\n"
              f"  {int(rr.get(-1.0, 0))} hacia atras y {int(rr.get(1.0, 0))} hacia adelante"))

    # --- detector sintetico
    ds = leer("detector_sintetico.csv")
    s1f, s1t = ds[(ds.seed == 1) & ~ds.refine_peak].iloc[0], ds[(ds.seed == 1) & ds.refine_peak].iloc[0]
    agg = ds.groupby("refine_peak").agg(sesgo=("sesgo_medio", "mean"), ex=("exactos", "sum"),
                                        det=("detectados", "sum"), real=("con_shift_real", "sum"),
                                        fp=("falsos_positivos", "sum"))
    a0, a1 = agg.loc[False], agg.loc[True]
    c += [
        ("sesgo 10 semillas", f"de {a0.sesgo:.2f} a {a1.sesgo:.2f} dias".replace("-", "−")),
        ("exactos 10 semillas", f"del\n  {a0.ex / a0.det * 100:.1f}% al {a1.ex / a1.det * 100:.1f}%"),
        ("deteccion sintetica", f"detecta el {a1.det / a1.real * 100:.1f}% de los shifts"),
        ("falsos positivos", "no da falsos positivos" if a1.fp == 0 else f"{int(a1.fp)} falsos positivos"),
        ("semilla 1", f"de {s1f.sesgo_medio:.2f} a {s1t.sesgo_medio:.2f} dias, de "
         f"{int(s1f.exactos)}/{int(s1f.detectados)}\n  a {int(s1t.exactos)}/{int(s1t.detectados)}".replace("-", "−")),
    ]

    # --- regresion
    r = leer("resultados_regresion.csv").set_index(["conjunto", "modelo"])["MAE_dias"]
    c += [
        ("tabla nucleo", f"| Nucleo (28) | {f3(r['nucleo', 'baseline (mediana)'])} | "
         f"{f3(r['nucleo', 'ridge'])} | **{f3(r['nucleo', 'random forest'])}** |"),
        ("tabla extendidas", f"| + glutamina y osmolalidad (36) | {f3(r['nucleo+extendidas', 'baseline (mediana)'])} | "
         f"{f3(r['nucleo+extendidas', 'ridge'])} | {f3(r['nucleo+extendidas', 'random forest'])} |"),
        ("tabla control", f"| Solo la escala del reactor (control) | {f3(r['control: solo escala', 'baseline (mediana)'])} | "
         f"{f3(r['control: solo escala', 'ridge'])} | {f3(r['control: solo escala', 'random forest'])} |"),
        ("cifra destacada", f"**Sobre el {f3(r['nucleo', 'random forest'])}.**"),
    ]

    # --- permutacion
    p = leer("permutacion.csv").set_index("modelo")
    for nombre, etiqueta in [("ridge", "Ridge"), ("random forest", "Random Forest")]:
        x = p.loc[nombre]
        c.append((f"permutacion {nombre}",
                  f"| {etiqueta} | {f3(x.mae_observado)} | {f3(x.nulo_media)} ± {f3(x.nulo_sd)} | "
                  f"{f3(x.p_valor)} | {int(x.barajadas)} |"))

    # --- Cox
    cx = leer("cox.csv").iloc[0]
    c.append(("cox", f"c-index\n{f3(cx.c_index)}, contra un nulo permutado de {f3(cx.nulo_media)} ± "
                     f"{f3(cx.nulo_sd)} (p = {f3(cx.p_valor)}, {int(cx.barajadas)} barajadas) y\n"
                     f"un control de solo-escala de {f3(cx.control_solo_escala)}"))

    # --- LOSO
    lo = leer("loso.csv")
    tab = {(m, e): v for m, e, v in zip(lo.modelo, lo.escala.astype(str), lo.mae)}
    for esc, n in [("0.00202", 43), ("0.00181", 18), ("0.0", 16)]:
        e = esc if esc != "0.0" else [k for k in ("0.0", "0.00000") if ("ridge", k) in tab][0]
        etq = esc if esc != "0.0" else "0.00000"
        c.append((f"LOSO {etq}", f"| {etq} | {n} | {f3(tab['baseline (mediana)', e])} | "
                                 f"{f3(tab['ridge', e])} | {f3(tab['random forest', e])} |"))
    c.append(("LOSO ponderado", f"| **Ponderado** | | **{f3(tab['baseline (mediana)', 'ponderado'])}** | "
                                f"{f3(tab['ridge', 'ponderado'])} | {f3(tab['random forest', 'ponderado'])} |"))
    b0 = [k for k in ("0.00202",) if ("ridge", k) in tab][0]
    c.append(("Ridge gana en una escala", f"obtiene {f3(tab['ridge', b0])} frente a "
                                          f"{f3(tab['baseline (mediana)', b0])} del baseline"))
    rf_peor = sum(tab['random forest', k] > tab['baseline (mediana)', k]
                  for k in {k2 for (_, k2) in tab if k2 != 'ponderado'})
    c.append(("RF empeora en cuantas escalas", "lo empeora en las tres escalas" if rf_peor == 3
              else f"lo empeora en {rf_peor} de las tres escalas"))
    sb = leer("aud_b_loso_sin_escala.csv").set_index(["variables", "modelo"])["mae_ponderado"]
    c.append(("LOSO sin escala", f"(Random Forest {f3(sb['SIN ' + chr(39) + 'escala' + chr(39), 'random forest'])} frente a "
                                 f"{f3(sb['con ' + chr(39) + 'escala' + chr(39) + ' entre las variables', 'random forest'])})"))

    # --- baseline por escala
    bp = leer("baseline_por_escala.csv").set_index(["modelo", "baseline"])
    for (m, b), etq in [(("random forest", "mediana global"), "| Random Forest vs mediana global |"),
                        (("random forest", "mediana de la escala"), "| **Random Forest vs mediana de la escala** |"),
                        (("ridge", "mediana global"), "| Ridge vs mediana global |"),
                        (("ridge", "mediana de la escala"), "| Ridge vs mediana de la escala |")]:
        x = bp.loc[(m, b)]
        if "**" in etq:
            s = f"{etq} **{pm(x.diferencia)}** | **[{pm(x.ic_bajo)}, {pm(x.ic_alto)}]** |"
        else:
            s = f"{etq} {pm(x.diferencia)} | [{pm(x.ic_bajo)}, {pm(x.ic_alto)}] |"
        c.append((f"bootstrap {m} vs {b}", s))
    mae7 = leer("baseline_por_escala_mae.csv", index_col=0)["mae"]
    c.append(("mediana de la escala sola", f"que por si solo erra {f3(mae7['escala'])} dias"))
    c.append(("07 reproduce a 04", "ok" if abs(mae7["rf"] - r["nucleo", "random forest"]) < 1e-3
              else "07 y 04 dan cifras distintas para el mismo modelo"))

    de = leer("dentro_de_escala.csv").iloc[0]
    c += [
        ("dentro de escala mediana", f"| Mediana | {f3(de.mediana)} |"),
        ("dentro de escala ridge", f"| Ridge | {f3(de.ridge)} |"),
        ("dentro de escala RF", f"| Random Forest | {f3(de.random_forest)} |"),
        ("dentro de escala permutacion", f"{f3(de.nulo_media)} ± {f3(de.nulo_sd)} (p = {f3(de.p_valor)}, el minimo con "
                                         f"{int(de.barajadas)} barajadas)"),
    ]
    pdia = leer("baseline_por_escala_por_dia.csv")
    for _, x in pdia.iterrows():
        dia = int(x.dia)
        rf = f"**{f2(x.random_forest)}**" if dia in (5, 8, 9) else f2(x.random_forest)
        c.append((f"por dia {dia}", f"| {dia} | {int(x.n)} | {f2(x.mediana_escala)} | {rf} |"))

    # --- SHAP
    g = leer("shap_importancia_agrupada.csv", index_col=0).iloc[:, 0]
    c += [
        ("SHAP principal", f"(0.{int(round(g.iloc[0] * 100)):02d} de |SHAP| sumado, contra "
                           f"0.{int(round(g.iloc[1] * 100)):02d} de la siguiente, la glucosa)"
         if g.index[1] == "glucose" else f"segunda variable: {g.index[1]}"),
        ("rho SHAP-permutacion", f"(correlacion de rangos {f2(leer('shap_rho.csv').iloc[0, 0])})"),
    ]
    ab = leer("shap_ablacion.csv").set_index("conjunto")["MAE_dias"]
    c += [
        ("ablacion todas", f"| Todas las variables | {f3(ab['todas las variables'])} |"),
        ("ablacion solo", f"| Solo glutamato | {f3(ab['solo glutamate'])} |"),
        ("ablacion sin", f"| Todas **sin** glutamato | {f3(ab['todas SIN glutamate'])} |"),
    ]

    # division por la mediana del glutamato (recalculada como en 05)
    from sklearn.impute import SimpleImputer
    t = leer("features_d1_4.csv", index_col="cult")
    t = t[~t["excluir"]]
    con = t[t["evento"]]
    meta = ["evento", "dia_evento", "tiempo", "excluir"]
    ext = [x for x in t.columns if x.startswith(("glutamine", "osmolality"))]
    nucleo = [x for x in t.columns if x not in meta + ext]
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(con[nucleo]),
                     columns=nucleo, index=con.index)
    alto = X["glutamate_mean"] > X["glutamate_mean"].median()
    frac = pd.crosstab(alto, con["escala"].round(5)).max(axis=1) / alto.value_counts().sort_index()
    med = con["dia_evento"].groupby(alto).median()
    c.append(("glutamato y escala", f"el {frac[True] * 100:.0f}% del grupo alto cae\nen una sola escala"))
    c.append(("glutamato y dia", f"(mediana {med[True]:.0f} contra {med[False]:.0f})"))

    # --- controles adicionales (script 06)
    a = leer("aud_a_sin_lactato.csv").set_index("conjunto")["MAE"]
    c += [
        ("sin lactato: solo lactato", f"| Solo variables de lactato (6) | {f3(a['solo variables de lactato'])} |"),
        ("sin lactato: sin", f"| **Sin ninguna variable de lactato (22)** | **{f3(a['SIN ninguna variable de lactato'])}** |"),
        ("sin lactato: todas", f"| Todas (28 variables) | {f3(a['todas'])} |"),
    ]
    cd = leer("aud_cd_sensibilidad_desempeno.csv")
    sinref = cd[(cd.suav == 3) & (cd.consec == 2) & (cd.umbral == 0.3) & (~cd.refine.astype(bool))].iloc[0]
    c += [
        ("mejora rango", f"entre {f3(cd.mejora.min())} y {f3(cd.mejora.max())} dias (mediana {f3(cd.mejora.median())})"),
        ("variante sin refine", f"excluye {int(sinref.excluidos)} cultivos en vez de 16 (mejora\n{f3(sinref.mejora)})"),
    ]

    # --- validacion externa (validacion_externa/, archivos versionados)
    ve = RAIZ / "validacion_externa"
    r = pd.read_csv(ve / "resultado.csv")
    ex = pd.read_csv(ve / "exploratorio.csv")
    from scipy.stats import beta
    k, n = int(r["acierto"].sum()), len(r)
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    por = r.groupby("definicion")["acierto"].agg(["sum", "count"])
    pico = r[r["definicion"] == "pico"]
    assert (pico["diferencia"] == 0).all(), "algun 'pico' no fue exacto: revisar la tabla"
    fallos = ex[~ex["acierto"]].sort_values("caida_suavizada")
    c += [
        ("validacion externa: curvas y fuentes",
         f"Se probo con {n} curvas de {r['articulo'].nunique()} fuentes"),
        ("validacion externa: resultado",
         f"**Resultado: {k} de {n} aciertos ({k / n:.0%}, IC 95% {lo * 100:.0f}-{hi * 100:.0f}%)"),
        ("validacion externa: pico",
         f"| El dia del maximo | {por.loc['pico', 'count']} | {por.loc['pico', 'sum']}, todos en el dia exacto |"),
        ("validacion externa: sin shift",
         f"| Que no hubo shift | {por.loc['no_aplica', 'count']} | {por.loc['no_aplica', 'sum']} |"),
        ("validacion externa: inicio del consumo",
         f"| El inicio del consumo, con una caida suave | {por.loc['inicio_consumo', 'count']} | "
         f"{por.loc['inicio_consumo', 'sum']} |"),
        ("validacion externa: caidas de los fallos",
         f"({fallos['caida_suavizada'].iloc[0]:.0%} en una curva y "
         f"{fallos['caida_suavizada'].iloc[1]:.0%} en la otra"),
        ("limitacion: shifts suaves",
         f"fallo\n  en las {len(fallos)} curvas donde el lactato bajo menos del 30%"),
        ("limitacion: censurados",
         f"los {len(mod) - len(ev)} cultivos censurados pueden incluir"),
    ]

    # --- puntos bajos aislados (analysis/08_puntos_aislados.py)
    pa = leer("puntos_aislados.csv")
    k_pa = int(pa["cambia"].sum())
    efecto = ("quitarlo no cambia el resultado de ninguno" if k_pa == 0
              else f"quitarlo cambia el resultado de {k_pa}")
    c.append(("puntos aislados",
              f"{len(pa)} de {len(et)} cultivos\n  tienen un punto bajo aislado y {efecto}"))

    c += cifras_v2()

    # --- tests
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=RAIZ, capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONPATH": str(RAIZ / "src")})
    n_tests = sum(1 for linea in out.stdout.splitlines() if "::" in linea)
    c.append(("numero de tests", f"{n_tests} tests, ninguno depende del dataset"))
    return c


def cifras_v2() -> list[tuple[str, str]]:
    """Cifras del caso de estudio actual (scripts 10-19)."""
    c: list[tuple[str, str]] = []

    # --- etiquetas por tasa especifica (11)
    et = leer("etiquetas_evento_tasa.csv", index_col="cult")
    m = et[~et["excluir"]]
    ev = m[m["evento"]]
    n_ev, n_cens = len(ev), len(m) - len(ev)
    d = ev["dia"].value_counts().sort_index()
    c += [
        ("v2: conjunto", f"**{len(m)} cultivos, {n_ev} con shift y {n_cens} censurados**"),
        ("v2: excluidos", f"{' y '.join(et[et['excluir']].index)} se\nexcluyen"),
        ("v2: distribucion",
         f"mediana {ev['dia'].median():.0f} ({d[4]:.0f} cultivos\nen el dia 4, {d[5]:.0f} en el 5, "
         f"{d[6]:.0f} en el 6, {d[7]:.0f} en el 7, {d[8]:.0f} en el 8, {d[9]:.0f} en el 9 y {d[11]:.0f} en\nel 11)"),
    ]
    for D, txt in (("0.02", "2% diario,\n{} de {} dias del shift no cambian"), ("0.05", "5%, {} de {}")):
        k = int((m[f"dia_D{D}"] == m["dia_D0.00"]).sum())
        c.append((f"v2: dilucion {D}", txt.format(k, m["dia_D0.00"].notna().sum())))
    qc = leer("qc_gases_enmascarados.csv")
    ce = leer("qc_gases_ceros.csv", index_col="variable")
    c.append(("v2: ceros de gases",
              f"Hay {ce.loc['pCO2', 'lecturas_cero']} ceros de pCO2 en {ce.loc['pCO2', 'cultivos']} cultivos y "
              f"{ce.loc['pO2', 'lecturas_cero']} de pO2 en {ce.loc['pO2', 'cultivos']}"))
    c.append(("v2: QC gases", f"Son {len(qc)}: " +
              " y\n".join(f"{r.cult} dia {int(r.day)}" for r in qc.sort_values("cult", ascending=False).itertuples())))

    # --- modelo de riesgo (15)
    mc = leer("metricas_corregidas.csv", index_col="modelo")
    s, m3 = mc.loc["solo dia"], mc.loc["M3"]
    c += [
        ("v2: AUC solo dia", f"| Solo el dia | {s['auc']:.3f} |"),
        ("v2: AUC M3", f"| Dia + mediciones | **{m3['auc']:.3f}** (IC 95% {m3['auc_ic_bajo']:.3f}-{m3['auc_ic_alto']:.3f}) |"),
        ("v2: mejora", f"Mejora: {m3['mejora']:+.3f} (IC 95% {m3['mejora_ic_bajo']:+.3f} a {m3['mejora_ic_alto']:+.3f}"),
        ("v2: alarma", f"en {int(m3['exacto'])} de\n{n_ev} cultivos"),
    ]
    for r in leer("auc_por_dia.csv").itertuples():
        c.append((f"v2: AUC dia {r.dia}", f"| {r.dia} | {r.eventos} | {r.en_riesgo} | {r.auc_M3:.3f} |"))

    # --- asociaciones (15)
    orc = leer("or_por_iqr_conjunto.csv", index_col=0)
    # >= 10: sin decimales; el extremo del bootstrap varia ~0.1 entre plataformas
    fmt = lambda x: f"{x:.0f}" if x >= 10 else f"{x:.2f}"
    for v, nombre in (("vcd", "VCD"), ("ph", "pH"), ("pco2", "pCO2"), ("glutamina", "Glutamina"), ("amonio", "Amonio")):
        r = orc.loc[v]
        txt = f"| {nombre} | {fmt(r['or_por_iqr'])} | {fmt(r['ic_bajo'])}-{fmt(r['ic_alto'])} |"
        if v == "vcd":  # extremo superior inestable entre plataformas: se acepta +-1
            hi = round(r["ic_alto"])
            opciones = [f"| VCD | {fmt(r['or_por_iqr'])} | {fmt(r['ic_bajo'])}-{h} |" for h in (hi - 1, hi, hi + 1)]
            txt = next((o for o in opciones if o in (RAIZ / "README.md").read_text(encoding="utf-8")), txt)
        c.append((f"v2: OR {v}", txt))
    og = leer("or_por_iqr_por_grupo.csv")
    for v, txt in (("vcd", "| VCD |"), ("ph", "| pH |"), ("pco2", "| pCO2 |")):
        k = int(og[og["variable"] == v]["ic_excluye_1"].sum())
        c.append((f"v2: {v} en grupos", "Si |" if k == 3 else f"En {k} de 3 |"))
    ge = leer("glutamina_evento_auc.csv")
    incluye0 = bool(((ge["dif_ic_bajo"] <= 0) & (ge["dif_ic_alto"] >= 0)).all())
    c.append(("v2: glutamina evento",
              "(diferencias menores que 0.003, con IC que incluyen 0)"
              if incluye0 and ge["dif_vs_M3"].abs().max() < 0.003 else "FALTA: revisar diferencias de glutamina"))
    ga = leer("glutamina_agotamiento_vs_shift.csv", index_col=0)
    k = int(((ga["primer_agotamiento"] - ga["dia_shift"]).abs() <= 1).sum())
    c.append(("v2: glutamina coincidencia", f"shift en {k} de {len(ga)} cultivos"))
    c.append(("v2: glutamina temprana", f"{int((ga['primer_agotamiento'] <= 2).sum())} de esos {len(ga)} cultivos llegan al nivel bajo"))

    # --- VCD dia 3 (calculado aqui: el 0.66 de la tabla es solo de los 34 con cambio de T)
    from _comun import cargar
    raw = cargar("Raw Data")
    v3 = raw[raw["day"] == 3].set_index("cult")["VCD"]
    rec = leer("receta_temperatura.csv", index_col="cult")
    dd = ev.join(v3.rename("v3")).join(rec).dropna(subset=["v3"])
    r2 = lambda g: np.corrcoef(g["v3"], g["dia"])[0, 1] ** 2
    c.append(("v2: R2 VCD d3",
              f"R² {r2(dd):.2f} del dia del shift en los {len(dd)} cultivos con shift ({r2(dd[dd['tshift'] == 1]):.2f} en\n"
              f"  los {int((dd['tshift'] == 1).sum())} con cambio de temperatura y {r2(dd[dd['tshift'] == 0]):.2f} en los "
              f"{int((dd['tshift'] == 0).sum())} sin el)"))

    # --- temperatura (15)
    tc = leer("temperatura_corregida.csv").iloc[0]
    tr = leer("temperatura_regresion.csv", index_col="modelo")
    c += [
        ("v2: temperatura coincidencia",
         f"En los {int(tc['n'])} cultivos con cambio de temperatura, el shift cae a ±1 dia del\n  cambio en el "
         f"{tc['observado']:.0%}; por azar, permutando, se espera {tc['azar']:.0%} (p = {tc['p']:.3f})"),
        ("v2: temperatura por grupo", f"el azar ya da {tc['azar_dentro_grupo']:.0%} (p = {tc['p_grupo']:.3f})"),
        ("v2: temperatura R2", f"explica R² {tr.loc['vcd_d3', 'r2']:.3f}"),
        ("v2: temperatura R2 tshift", f"solo explica {tr.loc['tshift_day', 'r2']:.3f}"),
        ("v2: temperatura coef",
         f"{tr.loc['vcd_d3 + tshift_day', 'coef_tshift_day']:.3f} (IC {tc['coef_tshift_ajustado_ic_bajo']:.3f} a "
         f"{tc['coef_tshift_ajustado_ic_alto']:.3f})".replace("-", "−")),
    ]

    # --- reversibilidad (16)
    rv = leer("reversibilidad_shift.csv")
    rp = rv["reproduce_15"].notna()
    c.append(("v2: reversibilidad",
              f"{int(rp.sum())} de {len(rv)} cultivos vuelven a producir lactato"))
    c.append(("v2: reversibilidad dia", f"La mediana es\nel dia {rv.loc[rp, 'reproduce_15'].median():.0f}"))
    c.append(("v2: reversibilidad regla anterior",
              f"posterior al shift ({int(rv['reproduce_15_rel_min'].notna().sum())}\nde 100)"))
    c.append(("v2: reversibilidad sin filtro", f"Sin ningun filtro de subida son {int(rv['reproduce_0'].notna().sum())}"))
    c.append(("v2: reversibilidad rango", f"va de {int(rp.sum())} a {int(rv['reproduce_0'].notna().sum())} de 100"))
    c.append(("v2: reversibilidad VCD",
              f"{int((rv['vcd_relativa_al_max'] < 0.95).sum())} de esos {int(rp.sum())}"))

    cj = leer("cambios_junto_al_lactato.csv").set_index("variable")["rho_spearman"]
    c.append(("v2: pH y pCO2 con el lactato",
              f"(Spearman {cj['pH']:.2f} para pH y {cj['pCO2']:.2f} para pCO2)".replace("-", "−")))
    # --- modelos 1 y 2 (17, 18), alfa por cultivo
    neg = lambda x: f"{x:.4f}".replace("-", "−")
    neg3 = lambda x: f"{x:.3f}".replace("-", "−")
    pc = lambda x: f"{x * 100:.0f}%"
    ic = lambda f, a, b: f"IC {neg(f[a])} a {neg(f[b])}"
    r = leer("modelo2_resumen.csv", index_col="modelo")
    def fila(nombre, clave, negrita=False):
        f = r.loc[clave]
        b = "**" if negrita else ""
        return (f"v2: tabla {clave}", f"| {b}{nombre}{b} | {b}{f['rmse_perfil']:.4f}{b} | {b}{f['rmse_2dias']:.4f}{b} |")
    c += [fila("Perfil promedio (referencia)", "perfil promedio"),
          fila("Modelo 1, sin interruptor", "M1 sin interruptor (script 17)"),
          fila("Modelo 2, riesgo solo del dia (control)", "M2 control: riesgo solo con el dia"),
          fila("Modelo 2, dos estados", "M2 dos estados (F predicho)", True),
          fila("Modelo 2 con el shift real", "M2 con el shift real")]
    g = r.loc["M2 dos estados (F predicho)"]
    ctl = r.loc["M2 control: riesgo solo con el dia"]
    sp = r.loc["M2 sin pH, pCO2 ni osmolalidad en el riesgo"]
    c += [
        ("v2: M2 vs M1", f"{neg(g['dif_perfil_vs_M1'])} en el perfil (IC 95% {neg(g['ic_perfil_bajo'])} a "
                         f"{neg(g['ic_perfil_alto'])}) y {neg(g['dif_2dias_vs_M1'])} a 2 dias "
                         f"({ic(g, 'ic_bajo', 'ic_alto')})"),
        ("v2: control vs M1", f"{neg(ctl['dif_perfil_vs_M1'])} ({ic(ctl, 'ic_perfil_bajo', 'ic_perfil_alto')})"),
        ("v2: congelado", f"queda en {g['rmse_2dias_congelado']:.4f} y sigue mejor que el modelo 1 "
                          f"({neg(g['dif_congelado_vs_M1'])}, {ic(g, 'ic_congelado_bajo', 'ic_congelado_alto')})"),
        ("v2: sin proxies", f"el perfil queda en {sp['rmse_perfil']:.4f}"),
        ("v2: pH", f"{r.loc['M2 con pH solo en el interruptor', 'rmse_2dias']:.4f} y "
                   f"{r.loc['M2 con pH solo en la cinetica', 'rmse_2dias']:.4f} a 2 dias"),
    ]
    sc = leer("modelo2_serie_completa.csv", index_col="modelo").loc["M2 dos estados"]
    c.append(("v2: serie completa", f"(modelo 2 {neg(sc['dif_perfil_vs_promedio'])}, "
                                    f"{ic(sc, 'ic_perfil_bajo', 'ic_perfil_alto')})"))
    c.append(("v2: serie completa 2 dias", f"({neg(sc['dif_2dias_vs_promedio'])}, {ic(sc, 'ic_2dias_bajo', 'ic_2dias_alto')})"))
    pr = leer("modelo2_pronostico_2dias.csv", index_col="metodo")
    c += [("v2: pronostico incremento", f"| Lactato de hoy + incremento medio de entrenamiento | "
                                        f"{pr.loc['incremento medio de entrenamiento', 'rmse_2dias']:.4f} |"),
          ("v2: pronostico M1", f"| Modelo 1, VCD extrapolada | {pr.loc['M1, VCD extrapolada', 'rmse_2dias']:.4f} |"),
          ("v2: pronostico M2", f"| **Modelo 2, VCD extrapolada** | "
                                f"**{pr.loc['M2 interruptor congelado, VCD extrapolada', 'rmse_2dias']:.4f}** |")]
    f2 = pr.loc["M2 interruptor congelado, VCD extrapolada"]
    f1 = pr.loc["M1, VCD extrapolada"]
    c.append(("v2: pronostico M2 dif", f"({neg(f2['dif_vs_incremento'])}, {ic(f2, 'ic_bajo', 'ic_alto')})"))
    c.append(("v2: pronostico M1 dif", f"({neg(f1['dif_vs_incremento'])}, {ic(f1, 'ic_bajo', 'ic_alto')})"))
    tt = leer("modelo2_tiempo.csv", index_col="metodo")
    for clave, nombre in (("dia mediano de entrenamiento (constante)", "Siempre el dia mediano de entrenamiento"),
                          ("M1 sin interruptor", "Modelo 1, sin interruptor"),
                          ("M2 riesgo solo con el dia (control)", "Modelo 2, riesgo solo del dia (control)"),
                          ("riesgo solo (primer dia con F >= 0.5)", "Riesgo diario solo (primer dia con F ≥ 0.5)"),
                          ("M2 dos estados", "Modelo 2, dos estados")):
        f = tt.loc[clave]
        c.append((f"v2: tiempo {clave}", f"| {nombre} | {pc(f['exacto'])} | {pc(f['pm1'])} |"))
    por = leer("modelo2_tiempo_por_dia.csv")
    m2d = por[por["metodo"] == "M2 dos estados"].set_index("dia_real")["cultivos"]
    c.append(("v2: shifts dias 5-7", f"{int(m2d.loc[[5, 6, 7]].sum())} de los {int(m2d.sum())} shifts dentro de la ventana"))
    c.append(("v2: constante +-1", f"ya acierta {pc(tt.loc['dia mediano de entrenamiento (constante)', 'pm1'])}"))
    pm = pd.read_csv(TAB / "modelo2_por_cultivo.csv", index_col="cult")
    lab = pm["dia_shift_etiqueta"]
    coincide = (pm["M2_shift_pred"] == pm["M2_dia_riesgo"])[lab.notna()].mean()
    c.append(("v2: coincide con el riesgo", f"coincide con el del riesgo en {coincide:.0%} de los cultivos"))

    # --- script 20: consumo con piso
    cv = leer("consumo_vs_lactato.csv").set_index("tramo")["median"]
    c.append(("v20: consumo por tramos", f"(mediana {min(cv['25-50%'], cv['50-75%'], cv['75-100%']):.3f} a "
                                         f"{max(cv['25-50%'], cv['50-75%'], cv['75-100%']):.3f})"))
    c.append(("v20: consumo bajo", f"({neg3(cv['0-25%'])})"))
    fo = leer("modelo2_consumo_formas.csv")
    c.append(("v20: formas", f"Fue elegida sobre el primer orden en el ajuste de entrenamiento de los {int((~np.isinf(fo['K'])).sum())} folds, con L* entre {fo['piso'].min():.2f} "
                             f"y {fo['piso'].max():.2f} y K entre {fo['K'].min():.2f} y {fo['K'].max():.2f}"))
    cm = leer("modelo2_consumo_mezcla.csv")
    def f20(comp, met):
        return cm[(cm["comparacion"] == comp) & (cm["metrica"] == met)].iloc[0]
    a0 = f20("A saturable vs lineal (reinicio en el shift real)", "RMSE despues del shift")
    a2 = f20("A saturable vs lineal (F predicho)", "RMSE despues del shift")
    ap = f20("A saturable vs lineal (F predicho)", "RMSE del perfil")
    b1 = f20("B mezcla de trayectorias vs de tasas (saturable)", "RMSE despues del shift")
    cc = f20("control de estructura con consumo saturable (riesgo solo dia vs M2)", "RMSE del perfil")
    c += [("v20: reinicio", f"en {neg(a0['dif'])} (IC {neg(a0['ic_bajo'])} a {neg(a0['ic_alto'])})"),
          ("v20: pred", f"{neg(a2['dif'])} (IC {neg(a2['ic_bajo'])} a {neg(a2['ic_alto'])})"),
          ("v20: perfil", f"baja de {ap['referencia']:.4f} a {ap['modelo']:.4f}"),
          ("v20: control", f"(+{cc['dif']:.4f}, IC {neg(cc['ic_bajo'])} a {neg(cc['ic_alto'])})"),
          ("v20: B", f"(+{b1['dif']:.4f}, IC {neg(b1['ic_bajo'])} a {neg(b1['ic_alto'])})")]

    # --- script 21: forma tipo Kotidis
    dg = leer("modelo_kotidis_diagnostico.csv").iloc[0]
    rk = leer("modelo_kotidis_resumen.csv").set_index("modelo")
    ck = leer("modelo_kotidis_comparaciones.csv").set_index("comparacion")
    K = rk.loc["tipo Kotidis, sin interruptor (inhibe el lactato)"]
    c += [("v21: c entre", f"c sale negativo ({neg3(dg['c_una_alfa_comun'])})"),
          ("v21: c dentro", f"c = {dg['c_alfa_por_cultivo']:.3f}"),
          ("v21: sin modelo", f"(Spearman mediana {neg3(dg['rho_mediana_sin_modelo'])[:-1]}, negativo en "
                              f"{dg['frac_rho_negativo']:.0%} de {int(dg['cultivos_rho'])} cultivos)"),
          ("v21: Kotidis vs M2", f"{K['rmse_perfil']:.4f} contra {rk.loc['M2 interruptor externo (riesgo)', 'rmse_perfil']:.4f} "
                                 f"(diferencia {neg(ck.loc['Kotidis - M2 (perfil)', 'dif'])}, IC "
                                 f"{neg(ck.loc['Kotidis - M2 (perfil)', 'ic_bajo'])} a {neg(ck.loc['Kotidis - M2 (perfil)', 'ic_alto'])})"),
          ("v21: Kotidis vs M1", f"({neg(ck.loc['Kotidis - M1 (perfil)', 'dif'])}, IC "
                                 f"{neg(ck.loc['Kotidis - M1 (perfil)', 'ic_bajo'])} a {neg(ck.loc['Kotidis - M1 (perfil)', 'ic_alto'])})"),
          ("v21: Kotidis tiempo", f"({pc(K['exacto'])} exacto, {pc(K['pm1'])} a ±1 dia)"),
          ("v21: controles", f"queda en {rk.loc['control: inhibe el dia del cultivo', 'rmse_perfil']:.4f} y "
                             f"{rk.loc['control: inhibe la IVCD acumulada', 'rmse_perfil']:.4f}"),
          ("v21: M2 + inhibicion", f"llega a {rk.loc['M2 + inhibicion por lactato (exploratorio)', 'rmse_perfil']:.4f} "
                                   f"({neg(ck.loc['M2 + inhibicion - M2 (perfil)', 'dif'])} contra el modelo 2, IC "
                                   f"{neg(ck.loc['M2 + inhibicion - M2 (perfil)', 'ic_bajo'])} a "
                                   f"{neg(ck.loc['M2 + inhibicion - M2 (perfil)', 'ic_alto'])})")]
    return c


def main() -> int:
    # para consolas de Windows sin algunos caracteres
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    # las cifras de la version 1 del caso de estudio viven en docs/caso_estudio_v1.md
    readme = "\n".join((RAIZ / f).read_text(encoding="utf-8")
                        for f in ("README.md", "docs/caso_estudio_v1.md"))
    fallos = 0
    lista = cifras()
    # se compara ignorando saltos de linea y espacios repetidos, para que
    # reacomodar un parrafo no rompa la verificacion
    plano = lambda t: " ".join(t.split())
    readme_plano = plano(readme)
    for desc, texto in lista:
        if texto == "ok" or plano(texto) in readme_plano:
            print(f"  ok     {desc}")
        else:
            fallos += 1
            print(f"  FALLA  {desc}\n         esperado en el README:\n         {texto!r}")
    print(f"\n{len(lista) - fallos} de {len(lista)} cifras coinciden.")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
