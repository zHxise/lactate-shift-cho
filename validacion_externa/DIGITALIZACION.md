# Como se extrajeron las curvas

Escrito antes de correr el detector sobre ellas. Complementa `PROTOCOLO.md`;
lo que aqui se aparta del protocolo esta marcado como **desviacion**.

`curvas.csv` se genera con `python validacion_externa/digitalizar.py`
(requiere los PDF en `fuentes/` y `pip install pymupdf`). Con `--control`
dibuja los puntos extraidos encima de cada figura original para revisarlos a
simple vista.

## Resumen

| Articulo | Figura | Curvas | Metodo | Puntos |
|---|---|---|---|---|
| Lularevic 2019 (tesis, UCL) | 4.4 B y D | CY01, 3C12, R33A, EB7 | vectorial, exacto | 9, 11, 11, 11 |
| Becker et al. 2019 | 2D | NOB, COP | automatico por color y forma | 26, 15 |
| Becker et al. 2019 | 2D | REF | lectura manual | 14 |
| Yin et al. 2025 | 2d | vvm_bajo, vvm_alto | lectura manual | 15, 15 |

9 curvas de 3 articulos independientes entre si y del caso de estudio.

## Lularevic 2019, Figura 4.4 (pagina impresa 110)

La figura es vectorial: cada marcador es un objeto del PDF con coordenadas
exactas. Se identifican por color de relleno y forma (circulos rojos = lactato
de CY01 en el panel B; cuadrados negros, triangulos grises y rombos gris claro
= 3C12, R33A y EB7 en el panel D). Los simbolos de la leyenda se descartan
porque tienen texto justo a la derecha.

- Eje del tiempo: recta ajustada a los centros de los numeros del eje (h).
- Eje del lactato: la pendiente sale de los numeros del eje y el cero se ancla
  en la linea del eje horizontal. Comprobacion: los marcadores del tiempo 0,
  que valen 0, caen sobre esa linea (error <= 0.01 mM).
- Triangulos: el dato esta en el centroide del triangulo (con esa regla los
  triangulos con lactato 0 caen exactamente en el eje; con el centro del
  rectangulo que lo contiene quedarian 0.6 puntos arriba).
- El muestreo no cae en multiplos exactos de 24 h (por ejemplo 21, 45, 71,
  94, 117 h); se convierte a dias dividiendo entre 24 y el protocolo redondea.

## Becker et al. 2019, Figura 2D

La figura es una imagen (2008 x 1242 px; el panel D es el cuadrante inferior
derecho). Calibracion con los centros de los numeros de los ejes; residuos del
ajuste menores de 0.5 h y 0.1 mM.

- **NOB** (circulos negros): se aislan con una apertura morfologica con un
  disco de 13 px, que borra lineas, barras de error y texto y deja solo los
  circulos (~17 px). Los 26 marcadores se ven completos.
- **COP** (cuadrados rojos): mismo procedimiento por color. Solo se aceptan los
  que se ven completos o casi (area >= 75% de un cuadrado aislado). Quedan 15,
  de 142 a 311 h. Antes de 142 h los cuadrados estan tapados en parte por los
  circulos de NOB y **se omiten en lugar de estimarse**.
- **REF** (triangulos huecos): no se pueden aislar de forma automatica porque
  se enciman con las barras de error y con COP. Lectura manual del centro del
  triangulo sobre la figura ampliada; comprobada con correlacion de plantilla
  (diferencias <= 0.6 mM). Antes de 142 h y a 166 h estan tapados y se omiten.
  A 285 y 311 h el cuadrado de COP tapa la mitad superior y el triangulo se
  ubico por su base visible.
- Consecuencia: REF y COP empiezan en el dia 6. Lo omitido es el tramo de 0 a
  130 h, donde las tres condiciones se enciman.
- El muestreo es cada ~12 h. Al pasar a dias, el protocolo redondea y promedia
  las dos muestras que caen en el mismo dia.

## Yin et al. 2025, Figura 2d

Imagen en escala de grises. Los marcadores (circulos huecos para vvm baja,
triangulos rellenos para vvm alta, segun la leyenda del panel a de la misma
figura)
se enciman con barras de error gruesas, asi que la lectura fue manual sobre la
figura ampliada, con la cuadricula de las marcas de los ejes (g/L, dias 0-14).
La figura de control usa esas mismas marcas para dibujar los valores leidos.

Del dia 0 al 7 las dos condiciones son el mismo proceso (el cambio de difusor
empieza el dia 7) y sus marcadores se superponen: se asigno el mismo valor a
ambas curvas.

## Desviaciones y limitaciones, registradas antes de detectar

1. **La extraccion no fue ciega.** Para aplicar el criterio 4 del protocolo
   (que los autores declaren el dia) hubo que leer su texto antes de
   digitalizar, asi que quien extrajo las curvas conocia las respuestas. Lo
   que limita el efecto: el detector esta congelado por su huella; Lularevic
   (vectorial) y NOB/COP (automaticos) no dependen de quien lee; los valores
   manuales (REF, Yin) se pueden revisar contra la figura con `--control`.
   Una replica ciega haria que otra persona digitalice sin leer el texto.
2. **Yin, vvm alta: intervalo por fase, no por dia.** Los autores dicen que con
   vvm alta el lactato paso a consumo en la fase estacionaria, que ellos
   mismos definen como dias 7-14. Se toma ese intervalo porque el protocolo
   admite intervalos, pero es de 8 dias y la prueba es poco exigente. El
   resultado se reporta con y sin esta curva (lo segundo, como exploratorio).
3. **Yin, Figura 1c excluida** por un motivo que el protocolo no previo: 16
   corridas con simbolos encimados que no se pueden separar.
4. **Objetivo no alcanzado:** el protocolo pedia al menos 10 curvas; hay 9.
   Brunner 2018 quedo fuera por no poder obtener el PDF.
5. **Definiciones distintas.** Lularevic declara el dia del maximo (`pico`).
   Becker y Yin declaran cuando empieza o ocurre el consumo
   (`inicio_consumo`), que por definicion cae despues del maximo. El detector
   reporta el maximo, asi que en esas curvas un sesgo negativo es esperable.
