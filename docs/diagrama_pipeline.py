"""Genera el diagrama del pipeline del proyecto (docs/pipeline.png).

No es una red neuronal: el proyecto no tiene capas de neuronas. El equivalente
honesto es el recorrido de los datos, desde las series crudas del biorreactor
hasta la predicción y su verificación.

Ejecutar:  python docs/diagrama_pipeline.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SALIDA = Path(__file__).resolve().parent / "pipeline.png"

MORADO, ROSA, ROJO, NARANJA, AMBAR = "#6B4FBB", "#C94FA8", "#E8455F", "#F07A26", "#D99211"
GRIS, VERDE, TINTA = "#8A8A8A", "#3F8A4E", "#2B2B2B"

ETAPAS = [
    (MORADO, "1. Datos crudos",
     ["106 cultivos CHO", "5 a 500 litros", "9 a 18 días", "24 variables", "de proceso"]),
    (ROSA, "2. Detección del evento",
     ["suavizado de 3 días", "derivada del lactato", "primer máximo local", "caída ≥ 30 %"]),
    (ROJO, "3. Variables días 1-4",
     ["nivel al día 4", "pendiente", "promedio", "n.º de mediciones", "cocientes", "28 variables"]),
    (NARANJA, "4. Modelos",
     ["Random Forest", "Ridge", "Cox (supervivencia)", "baseline: la mediana"]),
    (AMBAR, "5. Validación e\ninterpretación",
     ["validación cruzada", "permutación", "leave-one-scale-out", "SHAP + ablación"]),
]

ENTRE = ["todo\nel cultivo", "día del shift\n+ censurados", "una fila\npor cultivo", "día estimado"]


def flecha(ax, x1, y, x2, color=GRIS, lw=2.0):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=3))


fig = plt.figure(figsize=(16.2, 8.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.06,
                      left=0.045, right=0.975, top=0.885, bottom=0.04)

# ------------------------------------------------------------- el fenómeno
ax = fig.add_subplot(gs[0])
dias = np.arange(1, 16)
lac = np.where(dias <= 6, 0.14 * (1 - np.exp(-0.35 * dias)),
               0.14 * (1 - np.exp(-0.35 * 6)) * np.exp(-0.28 * (dias - 6)))
ax.axvspan(0.6, 4.4, color=VERDE, alpha=0.15, zorder=0)
ax.plot(dias, lac, "o-", color=ROSA, lw=2.4, ms=5, zorder=3)
ax.axvline(6, color=ROJO, ls="--", lw=2, zorder=2)

ax.text(2.5, 0.150, "lo único que ve el modelo\ndías 1 a 4", ha="center", va="center",
        fontsize=10.5, color=VERDE, fontweight="bold")
ax.annotate("lactate shift\ndía 6 (mediana)", xy=(6.05, 0.112), xytext=(7.6, 0.150),
            fontsize=10.5, color=ROJO, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.5))
ax.annotate("el resto del cultivo solo sirve para saber CUÁNDO\nocurrió; nunca entra como variable predictora",
            xy=(11.6, 0.019), xytext=(10.6, 0.062), fontsize=9.8, color=GRIS,
            arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.2))

ax.set_xlim(0.4, 15.6); ax.set_ylim(-0.004, 0.176)
ax.set_xlabel("día de cultivo", fontsize=10.5, labelpad=2)
ax.set_ylabel("lactato", fontsize=10.5)
ax.set_title("El fenómeno: el lactato sube y, en algún momento, el cultivo empieza a consumirlo",
             fontsize=12, pad=8)
ax.set_yticks([]); ax.set_xticks(range(1, 16)); ax.tick_params(labelsize=9)
for lado in ("top", "right", "left"):
    ax.spines[lado].set_visible(False)

# ------------------------------------------------------------- el pipeline
ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

n, w, hueco = len(ETAPAS), 0.149, 0.052
x0 = (1 - (n * w + (n - 1) * hueco)) / 2
y_caja, h_caja = 0.34, 0.545
y_medio = y_caja + h_caja / 2

for i, (color, titulo, puntos) in enumerate(ETAPAS):
    x = x0 + i * (w + hueco)
    ax2.add_patch(FancyBboxPatch((x, y_caja), w, h_caja,
                                 boxstyle="round,pad=0.010,rounding_size=0.025",
                                 facecolor=color, alpha=0.08, edgecolor="none", zorder=1))
    ax2.add_patch(FancyBboxPatch((x, y_caja), w, h_caja,
                                 boxstyle="round,pad=0.010,rounding_size=0.025",
                                 facecolor="none", edgecolor=color, linewidth=2.0, zorder=2))
    n_lineas = titulo.count("\n") + 1
    ax2.text(x + w / 2, y_caja + h_caja - 0.045, titulo, ha="center", va="top",
             fontsize=11.5, fontweight="bold", color=color, zorder=4, linespacing=1.35)
    y_texto = y_caja + h_caja - 0.045 - n_lineas * 0.058 - 0.030
    for j, p in enumerate(puntos):
        ax2.text(x + w / 2, y_texto - j * 0.062, p, ha="center", va="top",
                 fontsize=10, color=TINTA, zorder=4)
    if i < n - 1:
        xa, xb = x + w + 0.020, x + w + hueco - 0.008
        flecha(ax2, xa, y_medio, xb)
        ax2.text((xa + xb) / 2, y_medio + 0.038, ENTRE[i], ha="center", va="bottom",
                 fontsize=8.3, color=GRIS, style="italic", zorder=5, linespacing=1.25,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor="none", alpha=0.92))

ancho_total = n * w + (n - 1) * hueco
ax2.add_patch(FancyBboxPatch((x0, 0.165), ancho_total, 0.105,
                             boxstyle="round,pad=0.008,rounding_size=0.025",
                             facecolor=VERDE, alpha=0.14, edgecolor=VERDE, lw=1.4, zorder=1))
ax2.text(0.5, 0.218,
         "Regla que gobierna todo: ninguna variable puede depender de un dato posterior al día 4, "
         "ni de otros cultivos del conjunto",
         ha="center", va="center", fontsize=10.5, color="#27632F", fontweight="bold", zorder=4)

ax2.text(0.5, 0.075,
         "Error de 0.58 días frente a 0.84 del baseline   ·   c-index de Cox 0.795   ·   "
         "pero el modelo no transfiere a una escala de reactor que no vio",
         ha="center", va="center", fontsize=10.5, color=TINTA)

fig.suptitle("Cómo funciona el proyecto: del biorreactor a la predicción",
             fontsize=15.5, fontweight="bold", y=0.962)
fig.savefig(SALIDA, dpi=140, facecolor="white")
print(f"Diagrama en {SALIDA}")
