"""Detecta el lactate shift en un CSV propio.

Formato esperado: una fila por cultivo y dia, con columnas
    culture   identificador del cultivo (texto o numero)
    day       dia de cultivo, entero, empezando en 1
    lactate   concentracion de lactato, en cualquier unidad

Uso:
    python examples/detectar_en_mi_csv.py mis_datos.csv
    python examples/detectar_en_mi_csv.py mis_datos.csv --suavizado 3 --umbral 0.3

Salidas: una tabla en pantalla y en <archivo>_shift.csv, y una figura por
cultivo en la carpeta <archivo>_figuras/.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from lactateshift import detect_shift_batch
from lactateshift.detect import regularize, smooth

COLUMNAS = {"culture", "day", "lactate"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Detecta el lactate shift en un CSV.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--suavizado", type=int, default=3, help="ventana de la media movil (dias)")
    ap.add_argument("--umbral", type=float, default=0.30, help="caida minima relativa")
    ap.add_argument("--consecutivos", type=int, default=2, help="dias seguidos de descenso")
    ap.add_argument("--sin-figuras", action="store_true")
    a = ap.parse_args()

    if not a.csv.exists():
        print(f"No existe el archivo {a.csv}")
        return 1
    datos = pd.read_csv(a.csv)
    faltan = COLUMNAS - set(datos.columns)
    if faltan:
        print(f"Al CSV le faltan columnas: {sorted(faltan)}. Necesita {sorted(COLUMNAS)}.")
        return 1

    try:
        r = detect_shift_batch(datos, id_col="culture", day_col="day", value_col="lactate",
                               smooth_window=a.suavizado, drop_threshold=a.umbral,
                               n_consecutive=a.consecutivos)
    except ValueError as e:
        # los errores de entrada (dias repetidos, no enteros...) se explican
        # en el mensaje; no hace falta la traza completa para corregirlos
        print(f"Problema con los datos: {e}")
        return 1

    tabla = r[["occurred", "day", "time", "drop_fraction", "reason"]].rename(columns={
        "occurred": "hubo_shift", "day": "dia_shift", "time": "tiempo",
        "drop_fraction": "caida_relativa", "reason": "motivo"})
    print(tabla.round(3).to_string())
    n = int(tabla["hubo_shift"].sum())
    print(f"\n{n} de {len(tabla)} cultivos con shift detectado.")
    salida = a.csv.with_name(a.csv.stem + "_shift.csv")
    tabla.to_csv(salida)
    print(f"Tabla guardada en {salida}")

    if not a.sin_figuras:
        # matplotlib no es dependencia del paquete (solo numpy y pandas). Si no
        # esta instalado, la tabla ya se guardo: se avisa en vez de fallar.
        try:
            import matplotlib
        except ImportError:
            print("\nPara las figuras falta matplotlib. Instalalo con:\n"
                  "    pip install matplotlib\n"
                  "o corre el script con --sin-figuras.")
            return 0
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        carpeta = a.csv.with_name(a.csv.stem + "_figuras")
        carpeta.mkdir(exist_ok=True)
        for cult, g in datos.groupby("culture"):
            g = g.sort_values("day")
            dias, lac = regularize(g["day"], g["lactate"])
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(g["day"], g["lactate"], "o", color="0.55", label="medido")
            ax.plot(dias, smooth(lac, a.suavizado), "-", lw=2, label="suavizado")
            fila = r.loc[cult]
            if fila["occurred"]:
                ax.axvline(fila["day"], color="tab:red", ls="--", label=f"shift dia {fila['day']:.0f}")
            ax.set_xlabel("dia"); ax.set_ylabel("lactato"); ax.set_title(str(cult))
            ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig(carpeta / f"{cult}.png", dpi=110); plt.close(fig)
        print(f"Figuras guardadas en {carpeta}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
