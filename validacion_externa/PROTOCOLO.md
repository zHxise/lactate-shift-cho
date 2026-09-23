# Validacion externa del detector — protocolo

Escrito y guardado en git **antes** de buscar o extraer cualquier dato. La
fecha del commit que introduce este archivo es la prueba de que las reglas se
fijaron primero. Nada de lo que sigue se cambia despues de ver resultados; si
algo resulta mal pensado, se reporta como desviacion del protocolo, no se
corrige en silencio.

## Pregunta

Cuando un articulo publicado declara en que dia ocurrio el lactate shift de un
cultivo, ¿el detector encuentra el mismo dia a partir de la curva de lactato
de ese articulo?

Esto valida **solo el detector**. No valida el modelo predictivo del caso de
estudio: sus variables estan en una escala normalizada que no se puede
reproducir con datos de otros laboratorios.

## Detector congelado

- Codigo: `src/lactateshift/detect.py`, SHA-256
  `19e46eaec3035c219dd934018b9bc60d5e60ca1bd2bd73e2d93145cbfd089e13`
  (commit de partida `8e6d568`). `comparar.py` verifica esta huella y se
  niega a correr si el archivo cambio.
- Parametros: `smooth_window=3`, `n_consecutive=2`, `drop_threshold=0.30`,
  `refine_peak=True`, `min_peak=None`. Son los mismos del caso de estudio.

## Que articulos entran

Un articulo entra si cumple todo lo siguiente:

1. Cultivo de celulas CHO (otros mamiferos se registran aparte, como analisis
   secundario).
2. Muestra la concentracion de lactato en el tiempo, en tabla o en figura.
3. La curva tiene al menos 6 puntos y ningun intervalo entre muestras mayor de
   2 dias en el tramo que incluye el maximo.
4. **Los autores declaran el dia del shift**: en el texto, en una tabla o con
   una marca explicita en la figura (flecha, linea o etiqueta puesta por
   ellos). Frases como "el lactato alcanzo su maximo el dia 5 y despues se
   consumio" cuentan. Nuestra lectura del maximo en su grafica NO cuenta: eso
   seria nuestra respuesta, no la de ellos.

Cada articulo revisado se anota en `articulos.csv`, entre o no, con el motivo
de exclusion. Se incluyen todos los que cumplan, en el orden en que se
encuentran; no se descarta ninguno por el resultado.

Objetivo: al menos 10 curvas de al menos 3 articulos independientes. Si no se
alcanza, se reporta lo que haya.

## Como se extraen las curvas

- Si hay valores numericos (tabla o suplemento), se usan tal cual.
- Si solo hay figura, se digitaliza. Se anota el metodo y se guarda en
  `curvas.csv` un punto por muestra, en las unidades del articulo (el
  detector usa caidas relativas, asi que la unidad no importa).
- Si la figura muestra la media de replicas, se usa la media: una curva por
  condicion.
- Dias: si el eje empieza en el dia 0, se suma 1 a todos los dias antes de
  detectar y se resta 1 al dia detectado. Si el tiempo esta en horas, se pasa
  a dias. Si quedan dias no enteros, se redondean al entero mas cercano y, si
  dos muestras caen en el mismo dia, se promedian.
- Los dias sin muestra se rellenan por interpolacion lineal (el detector ya lo
  hace).

## Como se registra la respuesta de los autores

En `respuestas.csv`, separado de las curvas: el dia (o el intervalo) que dan
los autores, la cita textual, donde aparece (pagina, figura, tabla) y la
definicion que usan (`pico`, `inicio_consumo` u `otra`). Cada cita se
verifica contra el articulo.

## Orden de trabajo, para que git lo deje registrado

1. Commit de `curvas.csv` y `articulos.csv`.
2. `python validacion_externa/comparar.py detectar` → commit de
   `detecciones.csv`.
3. Solo entonces se llena `respuestas.csv` y se corre
   `python validacion_externa/comparar.py comparar`.

## Criterio de exito, fijado de antemano

- **Medida principal:** la diferencia absoluta entre el dia detectado y el
  dia declarado. Si los autores dan un intervalo [a, b], la diferencia es 0
  dentro del intervalo y la distancia al extremo mas cercano fuera de el.
- **Acierto:** diferencia de 1 dia o menos (la resolucion del muestreo
  diario). Si el detector no encuentra shift donde los autores si, es fallo.
  Si los autores declaran que no hubo shift y el detector tampoco lo
  encuentra, es acierto.
- **Criterio de exito:** al menos 80% de aciertos. Se reporta con su
  intervalo de confianza exacto (Clopper-Pearson), porque con pocas curvas la
  proporcion sola engana.
- Se reportan tambien: aciertos exactos, sesgo medio (detectado − declarado)
  y el desglose por definicion de los autores.

## Lo que no se hara

- Cambiar parametros o codigo del detector despues de ver resultados.
- Quitar curvas o articulos por su resultado.
- Presentar como confirmatorio cualquier analisis que no este en este
  protocolo; si se hace alguno, se marca como exploratorio.
