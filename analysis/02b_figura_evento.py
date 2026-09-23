"""
02b — Verificacion visual de la definicion del evento.

Ningun numero sustituye a mirar las curvas. Fue esta figura la que tumbo la
primera version de la regla, anclada en el maximo global: al graficar los
cultivos marcados como "sin shift" se veia que muchos si lo hacian y despues
rebotaban.

Ejecutar:  python analysis/02b_figura_evento.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lactateshift.detect import regularize, smooth
from _comun import FIGURAS, TABLAS, VENTANA, cargar

SUAVIZADO = 3


def panel(raw, et, cults, titulo, archivo):
    cults = list(cults)[:12]
    if not cults:
        return
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for ax, c in zip(axes.ravel(), cults):
        d = raw[raw["cult"] == c].sort_values("day")
        dias, lac = regularize(d["day"], d["[Lactate]"])
        ax.plot(dias, lac, "o-", ms=3, lw=0.8, color="0.65")
        ax.plot(dias, smooth(lac, SUAVIZADO), "-", lw=1.8, color="tab:blue")
        r = et.loc[c]
        if r["occurred"]:
            ax.axvline(r["day"], color="tab:red", ls="--", lw=1.3)
        ax.axvspan(1, VENTANA, color="tab:green", alpha=0.12)
        ax.set_title(f"{c}  " + (f"shift d{int(r['day'])}" if r["occurred"] else "censurado"),
                     fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(cults):]:
        ax.axis("off")
    fig.suptitle(f"{titulo}   (banda verde = ventana de observacion, dias 1-{VENTANA})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURAS / archivo, dpi=130)
    plt.close(fig)


def main() -> None:
    raw = cargar("Raw Data")
    et = pd.read_csv(TABLAS / "etiquetas_evento.csv", index_col="cult")
    rng = np.random.default_rng(0)
    mod = et[~et["excluir"]]

    grupos = [
        (mod[mod["occurred"] & (mod["day"] <= 5)].index, "Shift temprano (dia 5)", "evento_temprano.png"),
        (mod[mod["occurred"] & (mod["day"] >= 8)].index, "Shift tardio (dia >= 8)", "evento_tardio.png"),
        (mod[~mod["occurred"]].index, "Censurados: nunca hacen shift", "evento_censurados.png"),
        (et[et["excluir"]].index, "Excluidos: shift dentro de la ventana", "evento_excluidos.png"),
    ]
    for idx, titulo, archivo in grupos:
        idx = np.asarray(idx)
        sel = rng.choice(idx, min(12, len(idx)), replace=False) if len(idx) else idx
        panel(raw, et, sel, titulo, archivo)
    print(f"Figuras en {FIGURAS}")


if __name__ == "__main__":
    main()
