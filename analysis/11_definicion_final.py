"""
11 - Definicion final del shift (por tasa especifica) y conjunto de modelado sin sesgo.

Este script cierra el paso 1 del plan acordado tras la asesoria de octubre:

A. DEFINICION DEL EVENTO POR TASA ESPECIFICA
   El shift es el ultimo dia de produccion neta de lactato por celula: el
   inicio del primer intervalo en que la tasa especifica q (script 10) es
   negativa durante 2 intervalos seguidos. Como la serie se suaviza con una
   media movil centrada (que desplaza el maximo), el dia se ajusta al maximo
   de la serie sin suavizar dentro de +-1 dia, igual que refine_peak en la
   regla por concentracion.
   Se calcula con dilucion 0 y, como sensibilidad, con 2% y 5% por dia.

   Filtro de ruido: con solo "q negativa 2 intervalos" se colaban mesetas con
   oscilaciones (C100: baja 9% y vuelve a subir). Por eso se exige ademas que
   el lactato (suavizado) baje al menos 15% desde ese maximo antes de volver a
   superarlo. 15% es unas 3 veces el error analitico tipico de una medicion de
   lactato (~5%); no es el 30% de antes, que dejaba fuera shifts suaves reales
   (la validacion externa fallo justo en caidas de 16% y 25%).

   Cultivos sin acumulacion: si el lactato nunca supera 3% del rango del
   dataset en todo el cultivo, no hay pico que definir y cualquier "shift" es
   ruido de medicion. Se reportan aparte y se excluyen (solo C33).

B. QUE CULTIVOS SE EXCLUYEN (y por que cambia)
   Antes se excluian los cultivos con dia del shift <= 4, con el argumento de
   que "el desenlace ya esta en los datos". Revisando los 16 excluidos, 15
   tienen su maximo de lactato justo en el dia 4: al cerrar el dia 4 todavia
   NO se ha visto ninguna bajada (la primera muestra mas baja es la del dia
   5). Predecirlos es anticipar un dia, que la Dra. Kontoravdi considera
   util (1-2 dias). Ademas, 11 de esos 16 tienen cambio de temperatura
   temprano: excluirlos sesgaba el conjunto contra una receta concreta.
   Criterio nuevo: solo se excluye un cultivo si la bajada ya es visible
   dentro de la ventana, es decir, si el dia del shift es <= 3.

C. EFECTO EN EL MODELADO
   Repite la comparacion del script 09 (mediana, solo receta, variables de los
   dias 1-4, ambas) con el conjunto nuevo y el mismo protocolo (RF 300
   arboles, RepeatedKFold 5x5, imputacion dentro del fold), para ver si las
   conclusiones cambian.

Ejecutar:  python analysis/11_definicion_final.py

Desarrollado por Arturo Rodriguez.
"""

import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lactateshift import regularize, smooth  # noqa: E402

from _comun import TABLAS, VENTANA, cargar  # noqa: E402

warnings.filterwarnings("ignore")
tasa = import_module("10_tasa_especifica")      # funcion tasas()
receta_mod = import_module("09_receta_temperatura")  # funcion mae_cv()
SEED = 0
N_CONSEC = 2
CAIDA_MIN = 0.15
LACTATO_MIN = 0.03


def dia_shift_por_tasa(dias, lac, vcd, D=0.0):
    """Ultimo dia de produccion neta por celula, ajustado al maximo crudo."""
    d_ini, q = tasa.tasas(dias, lac, vcd, D)
    _, L_crudo = regularize(dias, lac)
    L_suave = smooth(L_crudo, tasa.SUAVIZADO)
    for i in range(len(q) - N_CONSEC + 1):
        if not np.any(q[:i] > 0):
            continue  # aun no hubo produccion neta
        if np.all(q[i:i + N_CONSEC] < 0):
            pico, bajo = L_suave[i], L_suave[i]
            for k in range(i + 1, len(L_suave)):
                if L_suave[k] > pico:
                    break
                bajo = min(bajo, L_suave[k])
            if pico <= 0 or (pico - bajo) / pico < CAIDA_MIN:
                continue  # oscilacion, no consumo sostenido
            lo, hi = max(0, i - 1), min(len(L_crudo), i + 2)
            j = lo + int(np.nanargmax(L_crudo[lo:hi]))
            return float(d_ini[0] + j)
    return np.nan


def main() -> None:
    raw = cargar("Raw Data")
    viejo = pd.read_csv(TABLAS / "etiquetas_evento.csv", index_col="cult")

    # ---------------------------------------------------------------- A
    filas = []
    for cult, g in raw.sort_values("day").groupby("cult"):
        g = g.dropna(subset=["[Lactate]", "VCD"])
        fila = {"cult": cult, "ultimo_dia": float(raw.loc[raw["cult"] == cult, "day"].max())}
        for D in (0.0, 0.02, 0.05):
            fila[f"dia_D{D:.2f}"] = (dia_shift_por_tasa(g["day"].to_numpy(), g["[Lactate]"].to_numpy(),
                                                        g["VCD"].to_numpy(), D)
                                     if len(g) >= 4 else np.nan)
        filas.append(fila)
    ev = pd.DataFrame(filas).set_index("cult")
    ev["evento"] = ev["dia_D0.00"].notna()
    ev["dia"] = ev["dia_D0.00"]
    ev["tiempo"] = ev["dia"].fillna(ev["ultimo_dia"])
    lac_max = raw.groupby("cult")["[Lactate]"].max()
    ev["sin_acumulacion"] = lac_max.reindex(ev.index) < LACTATO_MIN
    ev["excluir"] = (ev["evento"] & (ev["dia"] <= VENTANA - 1)) | ev["sin_acumulacion"]
    ev.to_csv(TABLAS / "etiquetas_evento_tasa.csv")

    print("=== A. Definicion por tasa especifica (D = 0) ===")
    print(f"con shift: {int(ev['evento'].sum())} de {len(ev)}; sin shift: {list(ev.index[~ev['evento']])}")
    comp = ev.join(viejo[["occurred", "day"]])
    ambos = comp["evento"] & comp["occurred"]
    dif = comp.loc[ambos, "dia"] - comp.loc[ambos, "day"]
    print(f"vs regla por concentracion: mismo dia en {int((dif == 0).sum())} de {int(ambos.sum())}; "
          f"distinto: {dif[dif != 0].to_dict()}")
    for D in (0.02, 0.05):
        x = ev[f"dia_D{D:.2f}"] - ev["dia"]
        x = x.dropna()
        print(f"sensibilidad D = {D:.2f}/dia: mismo dia en {int((x == 0).sum())} de {len(x)}, "
              f"+-1 en {int((x.abs() <= 1).sum())}")

    # ---------------------------------------------------------------- B
    print("\n=== B. Exclusion ===")
    print(f"criterio viejo (dia <= {VENTANA}): {int((ev['evento'] & (ev['dia'] <= VENTANA)).sum())} excluidos")
    print(f"cultivos sin acumulacion de lactato (maximo < {LACTATO_MIN}): {list(ev.index[ev['sin_acumulacion']])}")
    print(f"criterio nuevo (dia <= {VENTANA - 1}, bajada ya visible, o sin acumulacion): "
          f"{int(ev['excluir'].sum())} excluidos -> {list(ev.index[ev['excluir']])}")
    mod = ev[~ev["excluir"]]
    print(f"conjunto de modelado: {len(mod)} cultivos, {int(mod['evento'].sum())} con evento, "
          f"{int((~mod['evento']).sum())} censurados")
    print("dias del evento:", mod.loc[mod["evento"], "dia"].value_counts().sort_index().to_dict())

    # ---------------------------------------------------------------- C
    f = pd.read_csv(TABLAS / "features_d1_4.csv", index_col="cult")
    meta = ["evento", "dia_evento", "tiempo", "excluir"]
    ext = [c for c in f.columns if c.startswith(("glutamine", "osmolality"))]
    nucleo = [c for c in f.columns if c not in meta + ext]
    rec = pd.read_csv(TABLAS / "receta_temperatura.csv", index_col="cult")
    c = mod[mod["evento"]].join(f[nucleo]).join(rec)
    c["tshift_day_f"] = c["tshift_day"].fillna(30)
    receta = ["T0", "tshift", "tshift_day_f", "escala"]
    y = c["dia"]
    rf = lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    ridge = lambda: RidgeCV(alphas=np.logspace(-3, 3, 25))

    print(f"\n=== C. MAE en dias, conjunto nuevo (n = {len(c)}) ===")
    res = [{"conjunto": "mediana", "ridge": receta_mod.mae_cv(DummyRegressor(strategy="median"), c[["escala"]], y),
            "random_forest": np.nan}]
    for nombre, cols in [("solo receta", receta), ("variables d1-4", nucleo),
                         ("d1-4 + receta", nucleo + receta)]:
        res.append({"conjunto": nombre, "ridge": receta_mod.mae_cv(ridge(), c[cols], y),
                    "random_forest": receta_mod.mae_cv(rf(), c[cols], y)})
    res = pd.DataFrame(res).round(3)
    res.to_csv(TABLAS / "modelado_conjunto_nuevo.csv", index=False)
    print(res.to_string(index=False))
    print("\nReferencia con el conjunto viejo (script 09, n = 85): mediana 0.835, "
          "receta RF 0.687, d1-4 RF 0.571, d1-4 + receta RF 0.554")


if __name__ == "__main__":
    main()
