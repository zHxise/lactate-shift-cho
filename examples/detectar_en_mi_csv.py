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

Acepta tambien el CSV que guarda Excel en espanol (punto y coma entre
columnas y coma decimal). Los nombres de columna no distinguen mayusculas.

Desarrollado por Arturo Rodriguez.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from lactateshift import detect_shift_batch, isolated_low_points
from lactateshift.detect import regularize, smooth

COLUMNAS = {"culture", "day", "lactate"}


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee el CSV. Si el encabezado viene separado por punto y coma, como lo
    guarda Excel en espanol, se lee con ese separador y coma decimal."""
    with open(ruta, encoding="utf-8-sig") as f:
        encabezado = f.readline()
    if ";" in encabezado and "," not in encabezado:
        datos = pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig")
        print("Nota: el archivo usa punto y coma y coma decimal (formato de Excel "
              "en espanol); se leyo asi.\n")
    else:
        datos = pd.read_csv(ruta, encoding="utf-8-sig")
    datos.columns = [str(c).strip().lower() for c in datos.columns]
    return datos


def revisar(datos: pd.DataFrame) -> str | None:
    """Devuelve un mensaje si los datos no se pueden usar, o None.
    Los problemas de cada serie (dias repetidos, etc.) los revisa el detector."""
    faltan = COLUMNAS - set(datos.columns)
    if faltan:
        return (f"Al CSV le faltan columnas: {sorted(faltan)}. Necesita {sorted(COLUMNAS)}; "
                f"encontre {list(datos.columns)}.")
    if datos.empty:
        return "El CSV no tiene datos: solo trae el encabezado."
    for col in ("day", "lactate"):
        num = pd.to_numeric(datos[col], errors="coerce")
        malos = datos.index[num.isna() & datos[col].notna()]
        if len(malos):
            i = malos[0]
            return (f"En la columna {col}, fila {i + 2} del archivo: "
                    f"'{datos.at[i, col]}' no es un numero.")
        datos[col] = num
    if datos["day"].isna().any():
        return f"La fila {datos.index[datos['day'].isna()][0] + 2} del archivo no tiene dia."
    return None


def _orden_natural(s) -> list:
    # C2 antes que C10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


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
    datos = leer_csv(a.csv)
    problema = revisar(datos)
    if problema:
        print(problema)
        return 1
    negativos = sorted(datos.loc[datos["lactate"] < 0, "culture"].astype(str).unique())
    if negativos:
        # solo se avisa, puede haber datos con linea base restada
        print(f"Aviso: hay lactato negativo en {', '.join(negativos)}. Una concentracion "
              "no puede ser negativa; revisa esos datos.\n")

    try:
        r = detect_shift_batch(datos, id_col="culture", day_col="day", value_col="lactate",
                               smooth_window=a.suavizado, drop_threshold=a.umbral,
                               n_consecutive=a.consecutivos)
    except ValueError as e:
        print(f"Problema con los datos: {e}")
        return 1

    # aviso de puntos bajos aislados (pueden crear un shift falso)
    aislados = {c: isolated_low_points(g["day"], g["lactate"])
                for c, g in datos.groupby("culture")}
    aislados = {c: d for c, d in aislados.items() if d}
    if aislados:
        detalle = "; ".join(f"{c} (dia {', '.join(f'{x:.0f}' for x in d)})"
                            for c, d in sorted(aislados.items(), key=lambda kv: _orden_natural(kv[0])))
        print(f"Aviso: medicion muy baja y aislada en {detalle}. Si es un error de "
              "medicion, el resultado de ese cultivo puede estar mal: un solo punto asi "
              "puede crear un shift falso. Revisa su figura.\n")

    tabla = r[["occurred", "day", "time", "drop_fraction", "reason"]].rename(columns={
        "occurred": "hubo_shift", "day": "dia_shift", "time": "tiempo",
        "drop_fraction": "caida_relativa", "reason": "motivo"})
    for col in ("dia_shift", "caida_relativa"):
        tabla[col] = pd.to_numeric(tabla[col])
    tabla = tabla.loc[sorted(tabla.index, key=_orden_natural)]
    print(tabla.round(3).to_string())
    n = int(tabla["hubo_shift"].sum())
    print(f"\n{n} de {len(tabla)} cultivos con shift detectado.")
    salida = a.csv.with_name(a.csv.stem + "_shift.csv")
    tabla.to_csv(salida)
    print(f"Tabla guardada en {salida}")

    if not a.sin_figuras:
        # matplotlib es opcional
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
