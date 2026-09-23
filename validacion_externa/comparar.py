"""Validacion externa del detector contra articulos publicados.

Sigue validacion_externa/PROTOCOLO.md. Dos pasos, separados a proposito para
que git registre que las detecciones se hicieron antes de mirar las
respuestas de los autores:

    python validacion_externa/comparar.py detectar   # curvas.csv -> detecciones.csv
    python validacion_externa/comparar.py comparar   # + respuestas.csv -> resultado.csv

Formato de curvas.csv (una fila por muestra):
    articulo, curva, dia, lactato, empieza_en_cero (si/no)
Formato de respuestas.csv (una fila por curva):
    articulo, curva, dia_autor_min, dia_autor_max, sin_shift (si/no),
    definicion (pico/inicio_consumo/otra), cita, ubicacion
    (si los autores dan un solo dia, dia_autor_min = dia_autor_max)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from lactateshift import detect_shift  # noqa: E402

# --- Detector congelado (PROTOCOLO.md) ---------------------------------------
HUELLA = "19e46eaec3035c219dd934018b9bc60d5e60ca1bd2bd73e2d93145cbfd089e13"
PARAMETROS = dict(smooth_window=3, n_consecutive=2, drop_threshold=0.30,
                  refine_peak=True, min_peak=None)
TOLERANCIA = 1          # dias
EXITO = 0.80            # proporcion minima de aciertos


def verificar_congelado() -> None:
    real = hashlib.sha256((RAIZ / "src" / "lactateshift" / "detect.py").read_bytes()).hexdigest()
    if real != HUELLA:
        sys.exit("El detector cambio desde que se congelo el protocolo "
                 f"(huella {real[:12]}... en vez de {HUELLA[:12]}...). "
                 "La validacion externa no puede correr con un detector distinto.")


def preparar_dias(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Aplica las reglas de PROTOCOLO.md sobre el eje de tiempo.
    Devuelve dias enteros desde 1, valores y el desplazamiento aplicado."""
    desplazamiento = 1 if str(g["empieza_en_cero"].iloc[0]).strip().lower() == "si" else 0
    d = g["dia"].astype(float).round().astype(int) + desplazamiento
    t = pd.DataFrame({"dia": d, "lactato": g["lactato"].astype(float)})
    t = t.groupby("dia", as_index=False)["lactato"].mean()   # dias repetidos -> promedio
    return t["dia"].to_numpy(), t["lactato"].to_numpy(), desplazamiento


def detectar() -> None:
    verificar_congelado()
    curvas = pd.read_csv(AQUI / "curvas.csv")
    filas = []
    for (art, cur), g in curvas.groupby(["articulo", "curva"], sort=False):
        dias, lac, desp = preparar_dias(g)
        r = detect_shift(dias, lac, **PARAMETROS)
        filas.append({"articulo": art, "curva": cur, "n_puntos": len(dias),
                      "hubo_shift": r.occurred,
                      "dia_detectado": (r.day - desp) if r.occurred else np.nan,
                      "motivo": r.reason})
    out = pd.DataFrame(filas)
    out.to_csv(AQUI / "detecciones.csv", index=False)
    print(out.to_string(index=False))
    print(f"\n{len(out)} curvas. Detecciones guardadas; haz commit ANTES de llenar respuestas.csv.")


def clopper_pearson(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def comparar() -> None:
    verificar_congelado()
    det = pd.read_csv(AQUI / "detecciones.csv")
    resp = pd.read_csv(AQUI / "respuestas.csv")
    m = det.merge(resp, on=["articulo", "curva"], how="outer", indicator=True)
    huerfanas = m[m["_merge"] != "both"]
    if len(huerfanas):
        sys.exit("Curvas sin respuesta o respuestas sin curva:\n"
                 + huerfanas[["articulo", "curva", "_merge"]].to_string(index=False))

    def evaluar(f):
        sin_autor = str(f["sin_shift"]).strip().lower() == "si"
        if sin_autor:
            return (0.0 if not f["hubo_shift"] else np.inf), (not f["hubo_shift"])
        if not f["hubo_shift"]:
            return np.inf, False
        d, a, b = f["dia_detectado"], f["dia_autor_min"], f["dia_autor_max"]
        dif = 0.0 if a <= d <= b else float(min(abs(d - a), abs(d - b)))
        return dif, dif <= TOLERANCIA

    ev = m.apply(evaluar, axis=1, result_type="expand")
    m["diferencia"], m["acierto"] = ev[0], ev[1].astype(bool)
    con_dia = m[(m["sin_shift"].astype(str).str.lower() != "si") & m["hubo_shift"]]
    m["error_con_signo"] = np.where(
        m.index.isin(con_dia.index),
        m["dia_detectado"] - (m["dia_autor_min"] + m["dia_autor_max"]) / 2, np.nan)

    cols = ["articulo", "curva", "definicion", "dia_autor_min", "dia_autor_max",
            "dia_detectado", "diferencia", "acierto"]
    m[cols + ["error_con_signo", "cita", "ubicacion"]].to_csv(AQUI / "resultado.csv", index=False)
    print(m[cols].to_string(index=False))

    n, k = len(m), int(m["acierto"].sum())
    lo, hi = clopper_pearson(k, n)
    exactos = int((m["diferencia"] == 0).sum())
    print(f"\nAciertos (±{TOLERANCIA} dia): {k}/{n} = {k / n:.0%}  IC95% [{lo:.0%}, {hi:.0%}]")
    print(f"Coincidencia exacta: {exactos}/{n}")
    if m["error_con_signo"].notna().any():
        print(f"Sesgo medio (detectado - declarado): {np.nanmean(m['error_con_signo']):+.2f} dias")
    print("\nPor definicion de los autores:")
    print(m.groupby("definicion")["acierto"].agg(["sum", "count"]).to_string())
    veredicto = "CUMPLE" if k / n >= EXITO else "NO CUMPLE"
    print(f"\nCriterio del protocolo (≥{EXITO:.0%} de aciertos): {veredicto}")
    if lo < EXITO <= k / n:
        print("Ojo: la proporcion cumple, pero el intervalo de confianza incluye valores "
              "por debajo del criterio. Con estas pocas curvas no se puede afirmar con seguridad.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    if len(sys.argv) != 2 or sys.argv[1] not in ("detectar", "comparar"):
        sys.exit(__doc__)
    {"detectar": detectar, "comparar": comparar}[sys.argv[1]]()
