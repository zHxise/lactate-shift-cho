"""
03 — Variables predictoras de los dias 1-4.

Una fila por cultivo, solo con informacion disponible al cerrar el dia 4.
La construccion vive en ``lactateshift.features``; aqui se eligen las
variables del caso de estudio y se justifica la eleccion.

POR QUE ESTAS VARIABLES Y NO OTRAS
El nucleo son las que tienen cobertura real en la ventana: lactato, glucosa,
VCD, amonio, pH y glutamato tienen al menos tres dias medidos en 95 o mas de
los 106 cultivos. Glutamina y osmolalidad no llegan (55 y 68 cultivos), y
ademas son las que mas imputo el gap-filling de los autores, asi que se
calculan aparte para poder medir si aportan algo o solo meten ruido.

Los cocientes tienen justificacion biologica: el lactato absoluto depende de
cuantas celulas hay, asi que lo que interesa es la intensidad glucolitica por
celula (lactato/VCD) y respecto al sustrato disponible (lactato/glucosa).

QUE NO PUEDE SER VARIABLE
La duracion del cultivo se conoce al cosechar, no al dia 4. ECT es el dia
mismo. Midpoint y Endpoint Set son posteriores por definicion. El volumen de
cultivo si entra: se conoce antes de inocular.

LIMITACION HEREDADA DEL DATASET
Los datos vienen normalizados 0-1 por columna sobre los 106 cultivos. Esa
normalizacion ya uso todo el conjunto, asi que hay una fuga leve e inevitable
en el dato de origen. No se puede deshacer (no hay unidades) y hay que
declararla. Es otra razon para preferir cocientes y pendientes sobre niveles.

Ejecutar:  python analysis/03_features.py
"""

import numpy as np
import pandas as pd

from lactateshift import early_window_features
from _comun import TABLAS, VENTANA, cargar

NUCLEO = ["[Lactate]", "[Glucose]", "VCD", "[NH3]", "pH", "[Glutamate]"]
EXTENDIDAS = ["[Glutamine]", "Osmolality"]
COCIENTES = [("[Lactate]", "[Glucose]"), ("[Lactate]", "VCD")]


def limpiar_nombres(df: pd.DataFrame) -> pd.DataFrame:
    """Nombres de columna utilizables como identificadores."""
    def f(c: str) -> str:
        return (c.replace("[", "").replace("]", "").replace("+", "")
                 .replace("-", "").replace(" ", "_").lower())
    return df.rename(columns=f)


def main() -> None:
    raw = cargar("Raw Data")
    et = pd.read_csv(TABLAS / "etiquetas_evento.csv", index_col="cult")

    X = early_window_features(
        raw, id_col="cult", day_col="day",
        value_cols=NUCLEO + EXTENDIDAS, window=VENTANA,
        ratios=COCIENTES, static_cols=["Culture Volume"],
    )
    X = limpiar_nombres(X).rename(columns={"culture_volume": "escala"})

    # Crecimiento relativo: un cultivo que crece mas rapido agota sustrato
    # antes y deberia cambiar de regimen antes. Es la hipotesis mas obvia y
    # hay que vencerla antes de contar historias metabolicas mas finas.
    v = raw[raw["day"] <= VENTANA].sort_values("day").groupby("cult")["VCD"]
    X["vcd_crec_rel"] = v.last() / v.first().replace(0, np.nan)

    tabla = X.join(et[["occurred", "day", "time", "excluir"]]
                   .rename(columns={"occurred": "evento", "day": "dia_evento",
                                    "time": "tiempo"}))
    tabla.to_csv(TABLAS / "features_d1_4.csv")

    mod = tabla[~tabla["excluir"]]
    print("=== Matriz de variables ===")
    print(f"cultivos: {len(tabla)} | de modelado: {len(mod)} | variables: {X.shape[1]}")

    falt = mod[X.columns].isna().mean().mul(100).round(1)
    falt = falt[falt > 0].sort_values(ascending=False)
    print("\nvariables con celdas faltantes (se imputan dentro del fold, script 04):")
    print(falt.to_string() if len(falt) else "  ninguna")

    # Vistazo exploratorio. NO es seleccion de variables: elegir por esta
    # tabla seria fuga, porque se calcula con todo el conjunto.
    con_ev = mod[mod["evento"]]
    cors = (con_ev[X.columns].corrwith(con_ev["dia_evento"], method="spearman")
            .dropna().sort_values(key=abs, ascending=False))
    cors.to_csv(TABLAS / "correlaciones_exploratorias.csv", header=["spearman"])
    print(f"\ncorrelacion de Spearman con el dia del evento (n={len(con_ev)}), top 10:")
    print(cors.head(10).round(3).to_string())
    print(f"\nCon n={len(con_ev)} y decenas de variables miradas a la vez, un |rho| "
          "de ~0.20 es ruido. Solo cuentan las que sobrevivan a la validacion.")


if __name__ == "__main__":
    main()
