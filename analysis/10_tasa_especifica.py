"""
10 - El shift definido con la tasa especifica de lactato, no con la concentracion.

Por que: la Dra. Kontoravdi senalo que definir el shift con la concentracion
de lactato "probablemente no" es aceptable, y recomendo Gopalakrishnan et al.
(2024, COSMIC-dFBA), donde los cambios metabolicos se caracterizan con tasas
de consumo y secrecion, no con concentraciones.

La idea, con un balance de materia de lactato en fed-batch:

    dL/dt = q * X - D * L

    L  concentracion de lactato
    X  densidad de celulas viables (VCD)
    q  tasa especifica de lactato (por celula): q > 0 produce, q < 0 consume
    D  tasa de dilucion por la alimentacion (volumen alimentado / volumen / dia)

Despejando, la tasa por celula en un intervalo de un dia es:

    q = (delta L + D * L_medio) / (integral de X en el intervalo)

El denominador es la densidad celular integrada (IVCD) del intervalo. Como
siempre es positiva, SIN dilucion (D = 0) el signo de q es el mismo que el de
delta L: el momento en que q se vuelve negativa coincide con el de la
derivada de la concentracion. La dilucion es lo que puede separarlos: si la
alimentacion diluye el lactato, la concentracion baja aunque las celulas no
lo consuman todavia.

El problema: el dataset no trae el regimen de alimentacion, asi que D no se
conoce. Lo que si se puede hacer es un analisis de sensibilidad: repetir la
deteccion suponiendo diluciones de 0 a 5% del volumen por dia (un rango
generoso para fed-batch) y ver si el dia del shift cambia. Si no cambia, la
conclusion no depende de la dilucion. Si cambia, hay que decirlo.

Sobre la normalizacion min-max (x' = (x - min) / rango):
- La forma y el signo de delta L se conservan (el rango es positivo).
- El termino D * L necesita L real. Se usa L' suponiendo que el minimo real
  del lactato es cercano a cero (el lactato llega a casi 0 en muchos cultivos
  cuando se consume). Lo verifica el script.
- La magnitud de q queda en unidades relativas: sirve para comparar el
  momento del cambio, no para dar valores en pmol/celula/dia.

Regla de deteccion con la tasa (paralela a la de concentracion):
el shift es el inicio del primer intervalo en que q < 0 durante 2 intervalos
seguidos, despues de que hubo produccion neta (q > 0).

Por que no se pone un umbral de magnitud relativo a la mayor produccion
previa: los primeros dias la VCD es muy baja, asi que q sale inflada (se
divide entre casi nada). Cualquier umbral "X% de la produccion maxima"
queda fijado por ese artefacto y elimina shifts reales. Se probo: con 10%
se perdian 21 de 101 shifts. Por eso el umbral principal es 0 y los umbrales
de 2% y 5% se reportan solo como sensibilidad.

Ejecutar:  python analysis/10_tasa_especifica.py

Desarrollado por Arturo Rodriguez.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lactateshift import regularize, smooth  # noqa: E402

from _comun import TABLAS, VENTANA, cargar  # noqa: E402

DILUCIONES = [0.0, 0.01, 0.02, 0.03, 0.05]  # fraccion del volumen por dia
N_CONSEC = 2
UMBRAL_RELATIVO = 0.0
UMBRALES_SENSIBILIDAD = [0.02, 0.05]
SUAVIZADO = 3  # misma media movil que la definicion por concentracion


def tasas(dias, lac, vcd, D):
    """Tasa especifica de lactato por intervalo de un dia.

    Devuelve el dia de inicio de cada intervalo y q (unidades relativas).
    """
    d, L = regularize(dias, lac)
    _, X = regularize(dias, vcd)
    L = smooth(L, SUAVIZADO)
    dL = np.diff(L)
    L_medio = (L[:-1] + L[1:]) / 2
    ivcd = (X[:-1] + X[1:]) / 2  # trapecio, intervalo de 1 dia
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (dL + D * L_medio) / ivcd
    q[~np.isfinite(q) | (ivcd <= 0)] = np.nan
    return d[:-1], q


def detectar(dias_ini, q, umbral=UMBRAL_RELATIVO):
    """Primer intervalo con consumo sostenido y no trivial."""
    for i in range(len(q) - N_CONSEC + 1):
        previo = q[:i]
        if previo.size == 0 or not np.any(previo > 0):
            continue  # todavia no hubo produccion neta
        ref = np.nanmax(previo)
        tramo = q[i:i + N_CONSEC]
        if np.all(tramo < 0) and np.all(np.abs(tramo) >= umbral * ref):
            return float(dias_ini[i])
    return np.nan


def main() -> None:
    raw = cargar("Raw Data")
    et = pd.read_csv(TABLAS / "etiquetas_evento.csv", index_col="cult")

    print("=== Supuestos de la normalizacion ===")
    lac_min = raw.groupby("cult")["[Lactate]"].min()
    print(f"minimo de lactato por cultivo (normalizado): mediana {lac_min.median():.3f}, "
          f"cultivos que llegan a < 0.01: {(lac_min < 0.01).sum()} de {lac_min.notna().sum()}")
    vcd1 = raw[raw["day"] == 1]["VCD"]
    print(f"VCD del dia 1 (normalizada): mediana {vcd1.median():.3f}  "
          "(el desfase de la normalizacion de VCD no cambia el signo de q)")

    filas = []
    for cult, g in raw.sort_values("day").groupby("cult"):
        g = g.dropna(subset=["[Lactate]", "VCD"])
        if len(g) < 4:
            continue
        fila = {"cult": cult}
        for D in DILUCIONES:
            dias_ini, q = tasas(g["day"].to_numpy(), g["[Lactate]"].to_numpy(),
                                g["VCD"].to_numpy(), D)
            fila[f"dia_q_D{D:.2f}"] = detectar(dias_ini, q)
        filas.append(fila)
    res = pd.DataFrame(filas).set_index("cult")
    res = res.join(et[["occurred", "day", "excluir"]].rename(
        columns={"occurred": "evento_conc", "day": "dia_conc"}))
    res.to_csv(TABLAS / "shift_por_tasa.csv")

    print("\n=== Shift por tasa especifica vs por concentracion (106 cultivos) ===")
    print(f"por concentracion (regla actual): {int(res['evento_conc'].sum())} con shift")
    resumen = []
    for D in DILUCIONES:
        col = f"dia_q_D{D:.2f}"
        ambos = res["evento_conc"] & res[col].notna()
        dif = (res.loc[ambos, col] - res.loc[ambos, "dia_conc"])
        dentro = res[col] <= VENTANA
        resumen.append({
            "dilucion_por_dia": D,
            "con_shift": int(res[col].notna().sum()),
            "coinciden_en_ocurrencia": int((res[col].notna() == res["evento_conc"]).sum()),
            "mismo_dia": int((dif == 0).sum()),
            "a_1_dia": int((dif.abs() <= 1).sum()),
            "dif_mediana": float(dif.median()) if len(dif) else np.nan,
            "shift_dentro_dias_1_4": int(dentro.sum()),
        })
        print(f"\nD = {D:.2f}/dia: {int(res[col].notna().sum())} con shift | "
              f"dia igual al de concentracion en {int((dif == 0).sum())} de {len(dif)}, "
              f"a +-1 dia en {int((dif.abs() <= 1).sum())}")
        print("  diferencia (tasa - concentracion):", dif.value_counts().sort_index().to_dict())
    resumen = pd.DataFrame(resumen)

    # Sensibilidad al umbral de magnitud (ver la nota del docstring)
    print("\n=== Sensibilidad al umbral de magnitud ===")
    sens = []
    for u in UMBRALES_SENSIBILIDAD:
        for D in [0.0, 0.02, 0.05]:
            dias = {}
            for cult, g in raw.sort_values("day").groupby("cult"):
                g = g.dropna(subset=["[Lactate]", "VCD"])
                if len(g) < 4:
                    continue
                di, q = tasas(g["day"].to_numpy(), g["[Lactate]"].to_numpy(),
                              g["VCD"].to_numpy(), D)
                dias[cult] = detectar(di, q, u)
            s_ = pd.Series(dias)
            j = et.join(s_.rename("dq"))
            ambos = j["occurred"] & j["dq"].notna()
            dif = (j["dq"] - j["day"])[ambos]
            sens.append({"umbral": u, "dilucion_por_dia": D, "con_shift": int(s_.notna().sum()),
                         "a_1_dia": int((dif.abs() <= 1).sum()), "comparables": int(ambos.sum())})
    sens = pd.DataFrame(sens)
    sens.to_csv(TABLAS / "shift_por_tasa_umbral.csv", index=False)
    print(sens.to_string(index=False))
    resumen.to_csv(TABLAS / "shift_por_tasa_resumen.csv", index=False)
    print("\n=== Resumen ===")
    print(resumen.to_string(index=False))
    print("\nLectura: con D = 0 el signo de q es el de la derivada de la concentracion, "
          "asi que las diferencias vienen del refinamiento del pico. Lo que importa es "
          "cuanto cambia el dia al suponer mas dilucion.")


if __name__ == "__main__":
    main()
