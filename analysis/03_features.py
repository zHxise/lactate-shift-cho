"""
03 - Variables predictoras de los dias 1-4.

Una fila por cultivo, solo con informacion disponible al cerrar el dia 4.

Seleccion de variables:
- Nucleo: lactato, glucosa, VCD, amonio, pH y glutamato (al menos tres dias
  medidos en 95 o mas de los 106 cultivos).
- Glutamina y osmolalidad tienen menos cobertura (55 y 68 cultivos) y son las
  mas imputadas, por eso van aparte.
- Cocientes lactato/VCD y lactato/glucosa: lactato por celula y respecto al
  sustrato.
- No se usan la duracion, ECT ni las hojas Midpoint/Endpoint (se conocen
  despues del dia 4). El volumen si, se conoce desde el inicio.

Nota: los datos vienen normalizados 0-1 sobre los 106 cultivos, lo que es una
fuga leve que no se puede deshacer.

Ejecutar:  python analysis/03_features.py

Desarrollado por Arturo Rodriguez.
"""

import pandas as pd

from _comun import TABLAS, cargar, construir_variables

def main() -> None:
    raw = cargar("Raw Data")
    et = pd.read_csv(TABLAS / "etiquetas_evento.csv", index_col="cult")

    X = construir_variables(raw, extendidas=True)

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

    # Solo exploratorio, no se usa para elegir variables (seria fuga).
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
