"""Extrae las curvas de lactato de los articulos de la validacion externa.

Genera validacion_externa/curvas.csv, el archivo de entrada de comparar.py.
Metodo detallado y limitaciones en DIGITALIZACION.md.

Requiere los PDF en validacion_externa/fuentes/ (no estan en git por copyright;
LEEME.txt dice de donde bajarlos) y PyMuPDF:

    pip install pymupdf
    python validacion_externa/digitalizar.py

Tres tipos de extraccion, de mas a menos exacta:

- Lularevic 2019, Fig. 4.4: la figura es vectorial. Cada marcador es un objeto
  del PDF con coordenadas exactas, asi que no hay nada que "leer a ojo".
- Becker 2019, Fig. 2D: imagen. Los marcadores rellenos (circulos negros de NOB,
  cuadrados rojos de COP) se localizan por color y forma de manera automatica.
- Becker 2019 (REF, triangulos huecos) y Yin 2025, Fig. 2d: lectura manual
  sobre la figura ampliada, porque los marcadores se enciman con las barras de
  error y entre si. Los valores estan escritos abajo; la figura de control
  permite comprobarlos a simple vista.

Con --control ademas guarda en fuentes/ figuras que dibujan los puntos
extraidos encima de la figura original (requiere matplotlib). Se quedan en
fuentes/ porque contienen la figura del articulo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
FUENTES = AQUI / "fuentes"


def _pymupdf():
    try:
        import pymupdf
    except ImportError:
        sys.exit("Falta PyMuPDF: pip install pymupdf")
    return pymupdf


def _recta(pos, val):
    """Ajuste lineal valor = a*pos + b a partir de las marcas de un eje."""
    a, b = np.polyfit(np.asarray(pos, float), np.asarray(val, float), 1)
    return lambda p: a * np.asarray(p, float) + b


# ---------------------------------------------------------------------------
# Lularevic 2019 (tesis EngD, UCL), Figura 4.4, pagina impresa 110
# ---------------------------------------------------------------------------

def _pagina_fig44(doc):
    # El texto "Figure 4.4 A)" aparece tambien en el indice de figuras; la pagina
    # correcta es la que ademas contiene los trazos vectoriales de la grafica.
    for p in doc:
        if "Figure 4.4 A)" in p.get_text() and len(p.get_drawings()) > 300:
            return p
    sys.exit("No encontre la Figura 4.4 en lularevic2019.pdf")


def _eje_x(palabras, y_min, y_max, x_min, x_max):
    """Calibra el eje del tiempo con los centros de sus numeros."""
    marcas = [((w[0] + w[2]) / 2, float(w[4])) for w in palabras
              if y_min <= w[1] <= y_max and x_min <= w[0] <= x_max and w[4].isdigit()]
    return _recta([m[0] for m in marcas], [m[1] for m in marcas])


def _eje_y(palabras, lineas, x_min, x_max, y_min, y_max):
    """Calibra el eje del lactato.

    La pendiente sale de los centros de los numeros del eje; el cero se ancla
    en la linea del eje horizontal, que es exacta (el centro vertical de un
    numero queda unas decimas de punto por encima de su marca)."""
    marcas = [((w[1] + w[3]) / 2, float(w[4])) for w in palabras
              if x_min <= w[0] <= x_max and y_min <= w[1] <= y_max and w[4].isdigit()]
    pendiente = np.polyfit([m[0] for m in marcas], [m[1] for m in marcas], 1)[0]
    base = max(y for y in lineas if y_min <= y <= y_max + 10)   # linea del eje x
    return lambda y: pendiente * (np.asarray(y, float) - base)


def _es_leyenda(cx, cy, palabras):
    # Un simbolo de leyenda tiene su etiqueta de texto justo a la derecha.
    return any(abs((w[1] + w[3]) / 2 - cy) < 3 and 0 < w[0] - cx < 12
               and any(ch.isalpha() for ch in w[4])
               for w in palabras)


def _centro(dibujo):
    r = dibujo["rect"]
    tipos = "".join(it[0] for it in dibujo["items"])
    if tipos == "lll":
        # Triangulo: el dato esta en el centroide (verificado: los triangulos
        # con lactato 0 caen exactamente sobre el eje con esta regla).
        vertices = {(round(p.x, 3), round(p.y, 3)) for it in dibujo["items"] for p in it[1:3]}
        v = np.array(sorted(vertices))
        return float(v[:, 0].mean()), float(v[:, 1].mean())
    return (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2


def _marcadores(dibujos, palabras, color, tipos, zona):
    x0, y0, x1, y1 = zona
    pts = []
    for d in dibujos:
        f = d.get("fill")
        if f is None or tuple(round(c, 2) for c in f) != color:
            continue
        if "".join(it[0] for it in d["items"]) != tipos or d["rect"].width > 6:
            continue
        cx, cy = _centro(d)
        if x0 <= cx <= x1 and y0 <= cy <= y1 and not _es_leyenda(cx, cy, palabras):
            pts.append((cx, cy))
    return sorted(pts)


def lularevic() -> list[dict]:
    fitz = _pymupdf()
    pag = _pagina_fig44(fitz.open(FUENTES / "lularevic2019.pdf"))
    palabras = pag.get_text("words")
    dibujos = pag.get_drawings()
    lineas_h = [d["rect"].y0 for d in dibujos
                if d.get("color") == (0.0, 0.0, 0.0) and d["rect"].height < 0.01
                and d["rect"].width > 150]

    paneles = {
        # panel: (zona de datos, calibracion x, calibracion y, series)
        "B": ((322, 130, 493, 256),
              _eje_x(palabras, 255, 258, 330, 480),
              _eje_y(palabras, lineas_h, 310, 318, 128, 254),
              {"CY01": ((0.94, 0.25, 0.25), "cccc")}),          # circulos rojos
        "D": ((322, 280, 493, 406),
              _eje_x(palabras, 405, 408, 330, 480),
              _eje_y(palabras, lineas_h, 310, 318, 290, 404),
              {"3C12": ((0.0, 0.0, 0.0), "re"),                   # cuadrados negros
               "R33A": ((0.5, 0.5, 0.5), "lll"),                  # triangulos grises
               "EB7": ((0.75, 0.75, 0.75), "llll")}),             # rombos gris claro
    }
    filas = []
    for panel, (zona, a_horas, a_mM, series) in paneles.items():
        for curva, (color, tipos) in series.items():
            pts = _marcadores(dibujos, palabras, color, tipos, zona)
            for cx, cy in pts:
                # max(.., 0): la calibracion por centros de numeros deja la
                # primera muestra en -0.1 h; es el tiempo 0.
                h, mM = max(float(a_horas(cx)), 0.0), float(a_mM(cy))
                filas.append({"articulo": "lularevic2019", "curva": curva,
                              "dia": round(h / 24, 3), "lactato": round(max(mM, 0.0), 2),
                              "empieza_en_cero": "si", "_horas": h, "_panel": panel,
                              "_px": (cx, cy), "_metodo": "vectorial"})
    return filas


# ---------------------------------------------------------------------------
# Becker et al. 2019, Front. Bioeng. Biotechnol. 7:76, Figura 2D
# ---------------------------------------------------------------------------

def _imagen_becker():
    fitz = _pymupdf()
    doc = fitz.open(FUENTES / "becker2019.pdf")
    for p in doc:
        for im in p.get_images(full=True):
            if (im[2], im[3]) == (2008, 1242):          # la Figura 2 completa
                pix = fitz.Pixmap(doc, im[0])
                a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                return a[621:, 1004:, :3].astype(int)   # cuadrante inferior derecho = panel D
    sys.exit("No encontre la Figura 2 en becker2019.pdf")


def _centros_etiquetas(mascara, eje):
    """Centros de los grupos de pixeles oscuros (los numeros de un eje)."""
    perfil = mascara.any(axis=1 if eje == "y" else 0)
    lab, n = ndi.label(ndi.binary_closing(perfil, np.ones(13)))
    return [float(np.mean(np.where(lab == k)[0])) for k in range(1, n + 1)]


def _calibrar_becker(img):
    oscuro = img.max(axis=2) < 128
    # numeros del eje y (120 arriba ... 0 abajo), a la izquierda del eje
    cy = _centros_etiquetas(oscuro[20:520, 80:152], "y")
    a_mM = _recta(np.array(cy) + 20, [120, 100, 80, 60, 40, 20, 0])
    # numeros del eje x (0, 72, ..., 360 h), debajo del eje
    cx = _centros_etiquetas(oscuro[505:545, 140:], "x")
    a_horas = _recta(np.array(cx) + 140, [0, 72, 144, 216, 288, 360])
    return a_horas, a_mM


def _manchas(mascara, estructura, area_min):
    abierta = ndi.binary_opening(mascara, structure=estructura)
    lab, n = ndi.label(abierta)
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        area = int((lab[sl] == k).sum())
        if area >= area_min:
            cy, cx = ndi.center_of_mass(lab == k)
            out.append((float(cx), float(cy), area))
    return sorted(out)


def _disco(d):
    r = d // 2
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


# REF: lectura manual (triangulos huecos, casi siempre encimados con COP o con
# las barras de error). Horas y mmol/L, centro del triangulo. Antes de 142 h
# los triangulos quedan bajo los marcadores de NOB y COP; a 166 h, bajo COP.
# 285 h y 311 h: el triangulo esta tapado en parte por el cuadrado de COP y se
# ubico por su base visible.
BECKER_REF = [(142, 36.6), (154, 42.8), (178, 43.6), (190, 52.9), (202, 52.9),
              (214, 57.7), (225, 62.0), (238, 62.3), (250, 69.5), (262, 69.6),
              (273, 77.0), (285, 83.9), (296, 88.7), (311, 91.5)]


def becker() -> list[dict]:
    img = _imagen_becker()
    a_horas, a_mM = _calibrar_becker(img)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    datos = np.zeros(r.shape, bool)
    datos[48:494, 140:970] = True          # dentro de los ejes
    datos[:170, :330] = False              # fuera la leyenda

    series = {
        # NOB: circulos negros rellenos. La apertura con un disco de 13 px borra
        # lineas, barras de error y texto, y deja solo los circulos (~17 px).
        "NOB": (_manchas((r < 90) & (g < 90) & (b < 90) & datos, _disco(13), 150), "automatico"),
        # COP: cuadrados rojos. Solo se aceptan los que se ven completos o casi
        # (area >= 75% de un cuadrado aislado de 16x16); los que NOB tapa en
        # parte (antes de 142 h) se omiten en lugar de estimarse.
        "COP": (_manchas((r > 180) & (g < 90) & (b < 90) & datos, np.ones((7, 7)), 192), "automatico"),
    }
    filas = []
    for curva, (pts, metodo) in series.items():
        for cx, cy, _ in pts:
            h = float(a_horas(cx))
            filas.append({"articulo": "becker2019", "curva": curva, "dia": round(h / 24, 3),
                          "lactato": round(float(a_mM(cy)), 2), "empieza_en_cero": "si",
                          "_horas": h, "_px": (cx, cy), "_metodo": metodo})
    for h, mM in BECKER_REF:
        filas.append({"articulo": "becker2019", "curva": "REF", "dia": round(h / 24, 3),
                      "lactato": mM, "empieza_en_cero": "si", "_horas": float(h),
                      "_metodo": "manual"})
    return filas


# ---------------------------------------------------------------------------
# Yin et al. 2025, Cytotechnology 77:145, Figura 2d
# ---------------------------------------------------------------------------

# Lectura manual, g/L, dias 0-14, centro del marcador (media de n = 3).
# Hasta el dia 7 las dos condiciones son el mismo proceso (el cambio de
# difusor ocurre el dia 7) y sus marcadores se enciman: se asigna el mismo
# valor a ambas.
_YIN_COMUN = [0.30, 0.97, 2.05, 2.25, 2.43, 2.68, 2.97, 3.13]
YIN = {
    "vvm_bajo": _YIN_COMUN + [4.16, 4.69, 5.58, 5.40, 5.36, 5.60, 5.90],   # circulos huecos
    "vvm_alto": _YIN_COMUN + [3.40, 3.82, 4.03, 3.86, 3.57, 3.10, 2.74],   # triangulos
}


def yin() -> list[dict]:
    return [{"articulo": "yin2025", "curva": c, "dia": float(d), "lactato": v,
             "empieza_en_cero": "si", "_horas": 24.0 * d, "_metodo": "manual"}
            for c, vals in YIN.items() for d, v in enumerate(vals)]


def _imagen_yin():
    fitz = _pymupdf()
    doc = fitz.open(FUENTES / "yin2025.pdf")
    for p in doc:
        for im in p.get_images(full=True):
            if (im[2], im[3]) == (2032, 1125):          # la Figura 2 completa
                pix = fitz.Pixmap(doc, im[0])
                a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                a = a[562:1125, 0:677]                   # panel d
                return np.repeat(a[:, :, :1], 3, axis=2) if a.shape[2] == 1 else a[:, :, :3]
    sys.exit("No encontre la Figura 2 en yin2025.pdf")


# ---------------------------------------------------------------------------
# Figuras de control
# ---------------------------------------------------------------------------

def figuras_control(filas: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fitz = _pymupdf()
    colores = {"NOB": "cyan", "COP": "lime", "REF": "magenta", "CY01": "cyan",
               "3C12": "orange", "R33A": "lime", "EB7": "magenta",
               "vvm_bajo": "orange", "vvm_alto": "cyan"}

    def dibujar(fondo, puntos, nombre, titulo):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.imshow(fondo, cmap="gray" if fondo.ndim == 2 else None)
        for curva, (xs, ys) in puntos.items():
            ax.scatter(xs, ys, s=70, facecolors="none", edgecolors=colores[curva],
                       linewidths=1.6, label=curva)
        # fuera de la grafica, para que no tape ningun punto
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.set_title(titulo)
        ax.axis("off")
        fig.savefig(FUENTES / nombre, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  control: fuentes/{nombre}")

    # Lularevic: se renderiza la pagina a 4x y se dibujan las coordenadas exactas
    pag = _pagina_fig44(fitz.open(FUENTES / "lularevic2019.pdf"))
    z = 4
    pix = pag.get_pixmap(matrix=fitz.Matrix(z, z), clip=fitz.Rect(300, 125, 520, 430))
    fondo = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    pts = {}
    for f in filas:
        if f["articulo"] == "lularevic2019":
            xs, ys = pts.setdefault(f["curva"], ([], []))
            xs.append((f["_px"][0] - 300) * z)
            ys.append((f["_px"][1] - 125) * z)
    dibujar(fondo, pts, "control_lularevic2019.png", "Lularevic 2019, Fig. 4.4 B y D")

    # Becker: puntos automaticos en su pixel; REF, convertido con la calibracion
    img = _imagen_becker()
    a_horas, a_mM = _calibrar_becker(img)
    malla = np.linspace(0, img.shape[1], 2000), np.linspace(0, img.shape[0], 2000)
    h_de_px, mM_de_px = a_horas(malla[0]), a_mM(malla[1])
    pts = {}
    for f in filas:
        if f["articulo"] == "becker2019":
            xs, ys = pts.setdefault(f["curva"], ([], []))
            xs.append(np.interp(f["_horas"], h_de_px, malla[0]))
            ys.append(np.interp(f["lactato"], mM_de_px[::-1], malla[1][::-1]))
    dibujar(img.astype(np.uint8), pts, "control_becker2019.png", "Becker 2019, Fig. 2D")

    # Yin: calibrado con las marcas de los ejes del panel d (en pixeles del recorte)
    img = _imagen_yin()
    px_x = _recta([0, 14], [139.0, 627.5])        # dia -> pixel x (marcas 0 y 14)
    px_y = _recta([0, 8], [419.5, 9.0])           # g/L -> pixel y (marcas 0 y 8)
    pts = {}
    for f in filas:
        if f["articulo"] == "yin2025":
            xs, ys = pts.setdefault(f["curva"], ([], []))
            xs.append(float(px_x(f["dia"])))
            ys.append(float(px_y(f["lactato"])))
    dibujar(img, pts, "control_yin2025.png", "Yin 2025, Fig. 2d")


def main() -> None:
    faltan = [n for n in ("becker2019.pdf", "lularevic2019.pdf", "yin2025.pdf")
              if not (FUENTES / n).exists()]
    if faltan:
        sys.exit(f"Faltan en {FUENTES}: {', '.join(faltan)}. Ver fuentes/LEEME.txt.")
    filas = becker() + yin() + lularevic()
    t = pd.DataFrame(filas)
    publicas = ["articulo", "curva", "dia", "lactato", "empieza_en_cero"]
    t[publicas].to_csv(AQUI / "curvas.csv", index=False)

    resumen = (t.groupby(["articulo", "curva"], sort=False)
               .agg(puntos=("dia", "size"), metodo=("_metodo", "first"),
                    primera_h=("_horas", "min"), ultima_h=("_horas", "max")))
    print(resumen.round(1).to_string())
    print(f"\n{len(t)} puntos en {len(resumen)} curvas -> {AQUI / 'curvas.csv'}")
    if "--control" in sys.argv:
        figuras_control(filas)


if __name__ == "__main__":
    main()
