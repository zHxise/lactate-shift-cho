"""Genera el diagrama del pipeline del proyecto (docs/pipeline.png).

Ejecutar:  python docs/diagrama_pipeline.py

Desarrollado por Arturo Rodriguez.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SALIDA = Path(__file__).resolve().parent / "pipeline.png"

ETAPAS = [
    "1. Datos crudos\n\n106 cultivos CHO\n5 a 500 L, 9 a 18 dias\n24 variables",
    "2. Deteccion del evento\n\nsuavizado de 3 dias\nderivada del lactato\ncaida >= 30%",
    "3. Variables dias 1-4\n\nultimo valor, pendiente,\npromedio, n medidos,\ncocientes (28 variables)",
    "4. Modelos\n\nRidge, Random Forest\nCox\nbaselines (mediana)",
    "5. Validacion\n\nCV repetida, permutacion,\nbootstrap, leave-one-\nscale-out, SHAP",
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1.6, 1]})

# curva de ejemplo
dias = np.arange(1, 16)
lac = np.where(dias <= 6, 1 - np.exp(-0.35 * dias),
               (1 - np.exp(-0.35 * 6)) * np.exp(-0.28 * (dias - 6)))
ax1.plot(dias, lac, "o-")
ax1.axvspan(1, 4, alpha=0.2, color="green", label="ventana de observacion (dias 1-4)")
ax1.axvline(6, color="red", ls="--", label="lactate shift (mediana: dia 6)")
ax1.set_xlabel("dia de cultivo")
ax1.set_ylabel("lactato (ejemplo)")
ax1.set_xticks(dias)
ax1.legend(loc="upper right")
ax1.set_title("Ejemplo: el modelo solo usa los dias 1-4 para predecir el dia del shift")

# pipeline
ax2.axis("off")
ax2.set_xlim(0, len(ETAPAS))
ax2.set_ylim(0, 1)
for i, texto in enumerate(ETAPAS):
    ax2.text(i + 0.5, 0.55, texto, ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="square,pad=0.6", facecolor="white", edgecolor="black"))
    if i < len(ETAPAS) - 1:
        ax2.annotate("", xy=(i + 1.08, 0.55), xytext=(i + 0.92, 0.55),
                     arrowprops=dict(arrowstyle="->"))

# las cifras se leen de las salidas del analisis
tablas = Path(__file__).resolve().parents[1] / "outputs" / "tablas"
try:
    import pandas as pd
    mae = pd.read_csv(tablas / "baseline_por_escala_mae.csv", index_col=0)["mae"]
    resultado = (f"MAE Random Forest: {mae['rf']:.2f} dias; mediana de la escala: "
                 f"{mae['escala']:.2f} dias")
except FileNotFoundError:
    resultado = "Correr los scripts de analysis/ para ver los resultados"
ax2.text(len(ETAPAS) / 2, 0.08, resultado, ha="center", va="center", fontsize=9)

fig.suptitle("Pipeline del proyecto")
fig.text(0.99, 0.01, "Desarrollado por Arturo Rodriguez", ha="right", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(SALIDA, dpi=120)
print(f"Diagrama en {SALIDA}")
