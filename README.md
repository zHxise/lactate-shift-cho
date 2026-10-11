# lactateshift

Desarrollado por Arturo Rodriguez.

Que determina el momento del *lactate shift* en cultivos CHO fed-batch, y
si se puede anticipar con un enfoque interpretable.

El **lactate shift** es el momento en que un cultivo deja de producir lactato
netamente y empieza a consumirlo. La acumulacion de lactato baja el pH, obliga a anadir base y sube la
osmolalidad, lo que afecta el crecimiento y la productividad (ver, por
ejemplo, Morrissey et al. 2026, doi:10.1002/bit.70224). Entender que
adelanta o retrasa el cambio es util para controlarla.

Este repositorio tiene dos partes:

1. **`lactateshift`**, un paquete de Python que detecta el shift en cualquier
   serie de lactato y construye variables de una ventana temprana sin dejar
   entrar informacion posterior a ella.
2. **`analysis/`**, un caso de estudio sobre 106 cultivos CHO industriales de
   5 a 500 L: una definicion del shift por celula, un modelo de riesgo diario
   y dos modelos mecanisticos del perfil de lactato.

En corto: el momento del shift se asocia sobre todo con el crecimiento
temprano, y ademas con pH y pCO2. Un modelo de riesgo diario sin variables de
lactato anticipa bien los shifts tempranos y mal los tardios. Usar ese riesgo
como interruptor de una cinetica de lactato mejora el perfil y permite un
pronostico a 2 dias mejor que una referencia simple. Una cinetica sin
interruptor, en la que el propio lactato frena la produccion (la forma del
grupo de Kontoravdi), describe el perfil igual de bien. Todo son
asociaciones en datos normalizados, con las limitaciones que se detallan
abajo.

---

## Instalacion

```bash
git clone https://github.com/zHxise/lactate-shift-cho
cd lactate-shift-cho
pip install -e ".[dev]"        # paquete + tests
pip install -e ".[analysis]"   # + lo necesario para el caso de estudio
```

Requiere Python 3.10 o superior. El paquete depende solo de numpy y pandas.

## Uso

No hace falta ningun dato externo: el paquete genera cultivos sinteticos con
el dia del shift conocido.

```python
from lactateshift import detect_shift, make_synthetic_cultures, detect_shift_batch

# una serie
r = detect_shift(days=[1,2,3,4,5,6,7,8], values=[1,2,4,7,9,6,3,2])
print(r.occurred, r.day)      # True 5.0

# muchas series a la vez, listo para un modelo de supervivencia
series, verdad = make_synthetic_cultures(40, seed=0)
tabla = detect_shift_batch(series, id_col="culture", day_col="day", value_col="lactate")
print(tabla[["occurred", "day", "time"]].head())
```

`time` es el dia del evento cuando ocurrio y el dia de censura cuando no, que
es exactamente lo que piden `lifelines` o `scikit-survival`.

Para probarlo con **tus propios datos**, guarda un CSV con columnas
`culture`, `day` y `lactate` (una fila por cultivo y dia, dias enteros desde 1)
y corre:

```bash
pip install matplotlib          # solo para las figuras
python examples/detectar_en_mi_csv.py examples/ejemplo_cultivos.csv
```

Imprime el dia del shift de cada cultivo y guarda una figura por cultivo con
la curva y el dia marcado. `examples/ejemplo_cultivos.csv` trae cultivos
sinteticos para ver el formato. Acepta tambien el CSV que guarda Excel en
espanol (punto y coma y coma decimal), y si el archivo tiene un problema
(una letra en lugar de un numero, dias repetidos, columnas con otro nombre)
dice cual y en que fila. Tambien avisa si un cultivo tiene una medicion muy
baja y aislada, que podria crear un shift falso (ver Limitaciones).

Variables predictoras de una ventana temprana:

```python
import pandas as pd
from lactateshift import early_window_features

# dos cultivos inventados, en formato largo: una fila por cultivo y dia
datos = pd.DataFrame({
    "culture": ["A"] * 5 + ["B"] * 5,
    "day":     [1, 2, 3, 4, 5] * 2,
    "lactate": [0.5, 1.0, 1.8, 2.6, 3.0,   0.4, 0.7, 1.1, 1.4, 1.6],
    "glucose": [5.0, 4.6, 4.1, 3.5, 3.0,   5.0, 4.8, 4.5, 4.2, 4.0],
    "vcd":     [0.3, 0.6, 1.1, 1.8, 2.4,   0.3, 0.5, 0.8, 1.1, 1.4],
    "scale":   [5] * 5 + [50] * 5,
})

X = early_window_features(
    datos, id_col="culture", day_col="day",
    value_cols=["lactate", "glucose", "vcd"],
    window=4,                          # solo los dias 1 a 4
    ratios=[("lactate", "vcd")],       # cocientes al cierre de la ventana
    static_cols=["scale"],             # constantes conocidas desde el inicio
)
print(X[["lactate_last", "lactate_slope", "lactate_over_vcd", "scale"]])
```

Nada de lo que devuelve depende de un dato posterior al dia 4 ni de otras
series del conjunto. Lo que falta por completo queda `NaN` a proposito: la
imputacion pertenece al pipeline de modelado, donde se ajusta solo con el
fold de entrenamiento.

## Como funciona la deteccion

Hay shift en el primer dia `t` tal que:

1. la derivada de la serie suavizada es negativa durante `n_consecutive` dias
   a partir de `t`, y
2. el valor cae desde `t` al menos `drop_threshold` veces su valor, **antes**
   de que la serie vuelva a superarlo.

La condicion 2, con esa restriccion, descarta las caidas que se recuperan
antes de alcanzar el umbral.

El punto `t` es siempre un **maximo local** de la serie suavizada (en una
meseta, su ultimo punto). No se impone: se sigue de la regla. Si `t`
estuviera en plena bajada, el punto anterior tambien cumpliria las dos
condiciones con una caida mayor y habria sido elegido primero. Hay un test que
lo comprueba sobre cientos de series aleatorias.

**Lo que la regla no hace:** una caida profunda que despues se revierte **si**
cuenta como shift. Es a proposito (en cultivos reales el cambio a consumo
suele ser reversible y el lactato vuelve a subir en fase tardia), pero
significa que la regla detecta *el inicio de un descenso sostenido y profundo*, no *un cambio
de regimen permanente*. Tampoco filtra por amplitud absoluta salvo que se le
pase `min_peak`: sin ese parametro, una serie que oscile cerca de cero puede
producir una caida relativa del 30% que es solo ruido analitico.

**Por que el primer maximo local y no el maximo global.** Una parte de los
cultivos hace el shift, consume lactato varios dias y despues vuelve a
producirlo al final hasta superar el pico inicial. Una regla anclada en el
maximo global marca esos cultivos como "sin shift", lo cual es falso: el shift
ocurrio, solo fue reversible. En el caso de estudio, con los mismos
parametros, la regla del maximo global detecta 76 de 106 cultivos y la del
primer maximo local 101. El problema se encontro al graficar las curvas.

**Correccion del desplazamiento del suavizado.** Una media movil centrada
corre el maximo hacia el lado donde la curva es mas suave. Con
`refine_peak=True` (por omision) el dia se reajusta al maximo de la serie sin
suavizar, dentro de un dia, lo que corrige en cualquiera de los dos sentidos.

- En los cultivos sinteticos de `lactateshift.datasets`, donde el dia real se
  conoce, el sesgo medio pasa de −0.93 a −0.21 dias y los aciertos exactos del
  9.6% al 82.7% (400 cultivos, 10 semillas; detecta el 97.2% de los shifts y
  no da falsos positivos). Con la semilla 1: de −0.91 a −0.18 dias, de 3/34
  a 29/34 aciertos exactos.
- En los datos reales el refinamiento mueve el dia en 40 de 101 cultivos:
  31 hacia atras y 9 hacia adelante.

Los sinteticos tienen una subida que se aplana y una caida brusca, asi que
ahi el suavizado corre el maximo hacia atras. En los cultivos reales pasa lo
contrario con mas frecuencia: la caida suele ser mas lenta que la subida.
**Los sinteticos no reproducen la forma de los cultivos reales**: sirven para
verificar el detector, no para describir cultivos. La validacion sintetica se
reproduce con `python examples/validar_detector_sintetico.py`.

## Validacion externa del detector

¿El detector encuentra el mismo dia que declaran los autores de articulos
publicados? Se probo con 9 curvas de 3 fuentes ajenas al dataset del caso de
estudio (Becker et al. 2019, Yin et al. 2025 y la tesis de Lularevic 2019),
con un protocolo fijado en git antes de buscar datos
(`validacion_externa/PROTOCOLO.md`): detector congelado por su huella,
criterio de exito de al menos 80% de aciertos con tolerancia de un dia, y
commits separados para las curvas, las detecciones y las respuestas de los
autores.

**Resultado: 7 de 9 aciertos (78%, IC 95% 40-97%). No cumple el criterio
fijado de antemano.**

| Lo que declaran los autores | Curvas | Aciertos |
|---|---|---|
| El dia del maximo | 4 | 4, todos en el dia exacto |
| Que no hubo shift | 3 | 3 |
| El inicio del consumo, con una caida suave | 2 | 0 |

Los dos fallos tienen la misma causa: el lactato bajo, pero menos del 30% que
exige la regla (16% en una curva y 25% en la otra, en la serie suavizada). El
detector es conservador: marca los shifts claros en el dia correcto y no marca
los suaves. No se cambio ningun parametro despues de ver esto.

Con 9 curvas el intervalo es ancho, cuatro de ellas vienen de la misma tesis,
y la extraccion no fue ciega: hubo que leer el texto de los autores para saber
si el articulo cumplia los criterios. Detalle, citas y desviaciones en
`validacion_externa/RESULTADO.md`.

## El caso de estudio

**Pregunta:** ¿que determina el momento del lactate shift y el perfil de
lactato en cultivos CHO fed-batch, y se puede anticipar 1-2 dias antes con un
enfoque interpretable?

Dataset: 106 cultivos CHO de AstraZeneca, de 5 a 500 L y 9 a 18 dias, con 24
variables de proceso normalizadas min-max (0-1) sobre todo el conjunto y sin
unidades, del material suplementario de Gangadharan et al. (2021). Se usa la
hoja *Raw Data*: la hoja *Gap-Filled* rellena huecos con informacion de dias
posteriores. **El archivo no se distribuye aqui: esta bajo copyright de
Elsevier.** Ver [`data/raw/README.md`](data/raw/README.md) para obtenerlo,
con verificacion por SHA-256.

```bash
pip install -r requirements-analisis.txt      # versiones fijas del analisis
python analysis/00_verificar_datos.py         # comprueba el archivo
python analysis/10_tasa_especifica.py         # tasa especifica de lactato por celula
python analysis/11_definicion_final.py        # etiqueta del shift (definicion vigente)
python analysis/12_estado_celular.py          # tabla persona-periodo y modelo de riesgo
python analysis/13_asociaciones_por_grupo.py  # asociaciones por grupo de volumen
python analysis/14_modelo_cinetico.py         # modelo cinetico con parametros globales
python analysis/15_correcciones_auditoria.py  # QC de gases, metricas, temperatura
python analysis/16_reversibilidad_glutamina.py
python analysis/17_modelo1_crecimiento.py     # modelo 1: sin interruptor
python analysis/18_modelo2_dos_estados.py     # modelo 2: dos estados
python analysis/19_figuras.py
python docs/verificar_readme.py               # cada cifra del README contra las salidas
```

Los scripts 02 a 09 son la primera version del analisis, con otra definicion
y otra pregunta. Se conservan como registro en
[`docs/caso_estudio_v1.md`](docs/caso_estudio_v1.md).

### Definicion del evento

Esta definicion no es la del paquete `lactateshift`: el paquete trabaja
sobre la concentracion (y es el que se valido externamente); la del caso de
estudio trabaja sobre la tasa por celula y vive en `analysis/10` y `11`.

El shift es **el ultimo dia de produccion neta de lactato por celula**. Para
cada intervalo entre mediciones se calcula la tasa especifica
q = ΔL / IVCD, donde IVCD es la integral de la densidad de celulas viables en
el intervalo. Hay shift cuando q es negativa dos intervalos seguidos y el
lactato suavizado cae al menos 15%; el dia se ajusta al maximo de la serie
cruda, dentro de un dia.

Por que por celula y no por concentracion: la concentracion mezcla cuanto
produce cada celula con cuantas celulas hay. Un cultivo que crece mucho puede
seguir acumulando lactato aunque cada celula ya haya cambiado de regimen.

La tasa no incluye dilucion por alimentacion, porque el regimen de
alimentacion no esta en el dataset. Con una dilucion supuesta de 2% diario,
98 de 100 dias del shift no cambian; con 5%, 90 de 100.

Resultado: **104 cultivos, 100 con shift y 4 censurados**. C33 y C34 se
excluyen porque nunca acumulan lactato. Dia del shift: mediana 6 (15 cultivos
en el dia 4, 21 en el 5, 31 en el 6, 22 en el 7, 6 en el 8, 4 en el 9 y 1 en
el 11).

Control de calidad de gases, con dos reglas que no usan la etiqueta:

- Las lecturas con pH a mas de 3 rangos intercuartilicos se enmascaran (pH,
  pCO2 y pO2 de ese dia). Son 2: C75 dia 2 y C101 dia 1.
- Un 0 exacto en pCO2 o pO2 se trata como lectura faltante. Hay 89 ceros de
  pCO2 en 40 cultivos y 156 de pO2 en 35, en bloques de dias seguidos,
  mientras que el valor no nulo mas bajo de pCO2 ronda 0.09: son lecturas
  fallidas, no minimos reales.

### Modelo de riesgo diario

Cada dia en que un cultivo todavia no ha hecho el shift es una fila; el
desenlace es si el shift ocurre ese dia. Una regresion logistica con el dia
como categorias estima la probabilidad diaria. Las variables son las
mediciones del propio dia (VCD, glutamina, glutamato, amonio, pH,
osmolalidad, glucosa, pO2, pCO2), el cambio de VCD y la temperatura.
**Ninguna variable de lactato entra al modelo**, para que no aprenda a
reconocer el shift en la propia curva que lo define. La validacion es
agrupada por cultivo: todas las filas de un cultivo caen en el mismo fold.

| Modelo | AUC fuera de fold |
|---|---|
| Solo el dia | 0.815 |
| Dia + mediciones | **0.943** (IC 95% 0.925-0.961) |

Mejora: +0.128 (IC 95% +0.099 a +0.159, bootstrap por cultivo). Con una
alarma en probabilidad ≥0.5, el modelo marca el dia exacto del shift en 56 de
100 cultivos.

El "solo el dia" es el rival correcto: el riesgo sube con los dias en
cualquier cultivo, y un AUC alto puede venir solo de eso. Contra ese rival,
las mediciones agregan informacion.

Donde funciona y donde no:

| Dia | Eventos | En riesgo | AUC dentro del dia |
|---|---|---|---|
| 4 | 15 | 104 | 0.958 |
| 5 | 21 | 89 | 0.961 |
| 6 | 31 | 68 | 0.879 |
| 7 | 22 | 37 | 0.676 |
| 8 | 6 | 15 | 0.407 |

**Anticipa bien los shifts tempranos y mal los tardios.** El AUC dentro de un
dia compara solo cultivos que siguen en riesgo ese dia, asi que no lo infla
el efecto del calendario.

### Que se asocia con el momento del shift

Razon de momios por rango intercuartilico, modelo conjunto (IC 95% por
bootstrap de cultivos; el extremo superior de la VCD es inestable y puede
variar en una unidad entre computadoras). Una razon mayor que 1 adelanta el shift; menor que 1
lo retrasa.

| Variable | Razon de momios | IC 95% | ¿Se sostiene en los 3 grupos de volumen? |
|---|---|---|---|
| VCD | 15 | 11-45 | Si |
| pH | 0.16 | 0.08-0.29 | Si |
| pCO2 | 0.22 | 0.08-0.60 | Si |
| Glutamina | 1.50 | 1.23-2.12 | No (cambia de signo) |
| Amonio | 0.97 | 0.45-0.99 | No |

![Asociaciones](docs/figuras/fig3_asociaciones.png)

- **El crecimiento es la asociacion mas fuerte.** La VCD del dia 3, sola,
  explica R² 0.46 del dia del shift en los 100 cultivos con shift (0.66 en
  los 34 con cambio de temperatura y 0.29 en los 65 sin el).
- **pH y pCO2 mas altos se asocian con un shift mas tardio**, pero no se
  pueden leer como causas. Dentro de cada cultivo, su cambio diario se mueve
  con el del lactato (Spearman −0.38 para pH y −0.49 para pCO2), quiza por el
  control de pH respondiendo al acido; el dataset no dice si el pH se
  controla con CO2 o con base. Aunque el lactato no entra al modelo, estas
  razones de momios pueden reflejar en parte la propia dinamica del
  lactato.
- **Glutamina y amonio no son robustos.** La glutamina sube con el tiempo en
  la mayoria de los cultivos, que no es el patron de agotamiento tipico (una
  linea GS-CHO lo explicaria, pero es inferencia). Un nivel bajo de
  glutamina, con el umbral fijado solo con datos de entrenamiento, no mejora
  el AUC (diferencias menores que 0.003, con IC que incluyen 0) y coincide con
  el shift en 8 de 31 cultivos, al nivel del azar. Ojo: no es una prueba de
  agotamiento: 20 de esos 31 cultivos llegan al nivel bajo en los dias 1-2,
  es decir, empiezan bajos; no caen.

Son asociaciones, no causas.

### El cambio de temperatura no explica el momento

Una version anterior de este analisis decia que el shift ocurre casi siempre
junto al cambio de temperatura (36.5 → 33 °C). **No se sostiene.**

- En los 34 cultivos con cambio de temperatura, el shift cae a ±1 dia del
  cambio en el 82%; por azar, permutando, se espera 67% (p = 0.018). Dentro
  de cada grupo de volumen el azar ya da 70% (p = 0.042).
- En esos mismos 34 cultivos, la VCD del dia 3 explica R² 0.663 del dia del
  shift. El dia del cambio de temperatura solo explica 0.261, y al sumarlo a
  la VCD su coeficiente es 0.023 (IC −0.177 a 0.173): no aporta nada.

En este dataset la receta, el crecimiento y el proceso van juntos y no se
pueden separar.

### El shift es reversible

43 de 100 cultivos vuelven a producir lactato de forma sostenida despues del
shift: q positiva dos intervalos seguidos y una subida de al menos 15% de la
amplitud del propio cultivo, confirmada en los valores crudos. La mediana es
el dia 11. En 31 de esos 43 la VCD ya esta por debajo del 95% de su maximo:
la re-produccion ocurre sobre todo en el declive del cultivo.

La cifra depende de la definicion: va de 43 a 65 de 100. Una version
anterior pedia una subida de 15% relativa al minimo posterior al shift (55
de 100), pero en datos min-max ese minimo es casi cero y 15% de el es ruido.
Sin ningun filtro de subida son 65. En cualquier caso la re-produccion
tardia es frecuente: un modelo con un cambio irreversible describe bien la
fase de crecimiento, pero no el cultivo completo.

### Modelos mecanisticos del perfil de lactato

Se comparan sobre la misma ventana (hasta que la VCD cae por debajo del 85%
de su maximo), con validacion agrupada por cultivo:

- **Modelo 1, sin interruptor.** El lactato se produce en proporcion al
  crecimiento, con un ajuste aditivo por pH y pCO2 del dia anterior, y se
  consume en proporcion al lactato y a las celulas:
  ΔL = α·ΔX⁺ + a_p·ΔX⁺·pH' + a_c·ΔX⁺·pCO2' + β·IVCD − k·L·IVCD.
- **Modelo 2, dos estados.** La misma cinetica, pero la produccion se apaga
  y el consumo se enciende segun F, la probabilidad acumulada de haber hecho
  el shift. F sale del modelo de riesgo diario ajustado solo con los
  cultivos de entrenamiento.
- **Tipo Kotidis, sin interruptor** (seccion siguiente). El propio lactato
  frena la produccion ligada al crecimiento.

En todos, α es propia de cada cultivo: en entrenamiento se ajusta una α por
cultivo, y en un cultivo nuevo se estima con sus dias 1-4 y se encoge hacia
la media (Bayes empirico). Una version anterior usaba una sola α para todos
los cultivos de entrenamiento; la auditoria de la tarea 7 mostro que asi los
parametros compartidos absorben diferencias entre cultivos y todos los
modelos quedaban en desventaja.

**Los modelos usan la VCD medida**, asi que el perfil completo es una
reconstruccion explicativa, no un pronostico. El pronostico verdadero a 2
dias se reporta aparte, con la VCD extrapolada.

| Modelo | RMSE del perfil | RMSE a 2 dias |
|---|---|---|
| Perfil promedio (referencia) | 0.0636 | 0.0428 |
| Modelo 1, sin interruptor | 0.0520 | 0.0366 |
| Modelo 2, riesgo solo del dia (control) | 0.0486 | 0.0358 |
| **Modelo 2, dos estados** | **0.0452** | **0.0326** |
| Modelo 2 con el shift real | 0.0426 | 0.0314 |

![Comparacion de modelos](docs/figuras/fig5_comparacion_modelos.png)

- **El interruptor informado por las mediciones mejora al modelo 1**:
  −0.0068 en el perfil (IC 95% −0.0090 a −0.0047) y −0.0039 a 2 dias (IC
  −0.0051 a −0.0029). Cumple la regla de decision fijada antes de correrlo.
- **Con un riesgo que solo conoce el dia**, la mejora sobre el modelo 1 es
  menor y no concluyente: −0.0034 (IC −0.0069 a 0.0003).
- **El interruptor no usa mediciones posteriores al dia de partida** (salvo
  la receta de temperatura programada del dia siguiente, valido solo si el
  dia del cambio de temperatura estaba fijado de antemano): con
  las mediciones del dia siguiente congeladas, el error a 2 dias queda en
  0.0340 y sigue mejor que el modelo 1 (−0.0026, IC −0.0037 a −0.0016). La
  cinetica si usa la VCD observada; para eso esta el pronostico de abajo.
- Sin pH, pCO2 ni osmolalidad en el riesgo, el perfil queda en 0.0442.
- Donde va el pH (interruptor o cinetica) no se puede decidir: 0.0334 y
  0.0326 a 2 dias.
- **En la serie completa**, incluido el declive y la re-produccion tardia,
  ninguna ventaja es clara en el perfil (modelo 2 −0.0067, IC −0.0146 a
  0.0005); a 2 dias el modelo 2 si mejora un poco (−0.0029, IC −0.0050 a
  −0.0009).

**Pronostico verdadero a 2 dias.** Desde el dia 4 en adelante, sin conocer
la VCD futura (se extrapola en linea recta con los dos ultimos dias) y con
el interruptor congelado:

| Metodo | RMSE a 2 dias |
|---|---|
| Lactato de hoy + incremento medio de entrenamiento | 0.0448 |
| Modelo 1, VCD extrapolada | 0.0427 |
| **Modelo 2, VCD extrapolada** | **0.0362** |
| Modelo 2, VCD observada (referencia) | 0.0358 |

El modelo 2 le gana a la referencia simple (−0.0086, IC −0.0119 a −0.0054)
y casi no pierde nada por no conocer la VCD futura. El modelo 1 no le gana
(−0.0021, IC −0.0055 a 0.0012). Esto es lo mas cercano al objetivo de
anticipar 1-2 dias, y vale desde el dia 4.

**El dia del shift.** Acertar a ±1 dia es un criterio flojo: 70 de los 95
shifts dentro de la ventana caen en los dias 5 a 7, y predecir siempre el
dia mediano ya acierta 74%. Un cultivo sin shift simulado cuenta como fallo:

| Metodo | Dia exacto | A ±1 dia |
|---|---|---|
| Siempre el dia mediano de entrenamiento | 29% | 74% |
| Modelo 1, sin interruptor | 44% | 83% |
| Modelo 2, riesgo solo del dia (control) | 48% | 92% |
| Riesgo diario solo (primer dia con F ≥ 0.5) | 59% | 94% |
| Modelo 2, dos estados | 59% | 93% |

El modelo 2 fecha el shift igual que el riesgo diario por si solo; su dia
simulado coincide con el del riesgo en 75% de los cultivos. El momento lo
aporta el modelo de riesgo; la cinetica aporta la forma del perfil.

**Forma del consumo (script 20).** Despues del shift, la tasa de consumo por
celula sube solo debilmente con el lactato dentro de cada cultivo (Spearman
mediana 0.26), y es parecida mientras el lactato esta por encima de un
cuarto de su pico (mediana 0.048 a 0.062); se detiene cerca del final
(−0.002). Se probo un consumo saturable con piso,
q·(L − L*)⁺/(K + (L − L*)⁺)·IVCD, eligiendo K y el piso L* solo con los
cultivos de entrenamiento (el primer orden compite en la misma rejilla):

- Fue elegida sobre el primer orden en el ajuste de entrenamiento de los 5
  folds, con L* entre 0.03 y 0.04 y K entre 0.02 y 0.05
  (unidades normalizadas; L* no es una concentracion: puede ser agotamiento
  o un limite del analisis).
- Con el shift real y reiniciando desde el lactato observado ese dia, mejora
  el error despues del shift en −0.0071 (IC −0.0090 a −0.0052). Con F
  predicho, −0.0044 (IC −0.0067 a −0.0018), y el perfil del modelo 2 baja de
  0.0452 a 0.0422. **Se adopta como variante exploratoria** (post hoc: el
  piso y la estimacion por cultivo se eligieron durante la auditoria de la
  tarea 7).
- Con este consumo, el control con riesgo solo del dia ya no queda
  claramente peor que el modelo 2 (+0.0026, IC −0.0017 a 0.0072).
- Promediar trayectorias con cambio brusco en lugar de mezclar tasas con F
  no cambia nada (+0.0007, IC 0.0004 a 0.0011), pero en esta clase de modelos
  las dos formas dan casi lo mismo por construccion: la prueba no sirve para
  decidir como interpretar F.

**Un disparador interno: la forma de Kotidis et al. (2019,
doi:10.1002/bit.26960) (script 21).** En
la cinetica del grupo de Kontoravdi no hay interruptor externo: el propio
lactato frena la produccion ligada al crecimiento, (1 − L/Lacmax1), y por
encima de Lacmax2 activa el consumo. En nuestro balance, el termino de
consumo de esa forma es el del modelo 1; lo nuevo es un parametro, c, que
mide cuanto frena el lactato la produccion ligada al crecimiento.

- **Entre cultivos y dentro de cultivos dan signos opuestos.** Con una sola
  α para todos, c sale negativo (−1.771): los cultivos que producen mas
  lactato por celula tambien acumulan mas. Con α propia de cada cultivo,
  c = 2.636: dentro de un cultivo, la produccion por celula nueva baja a
  medida que sube el lactato. Una prueba sin modelo es compatible con eso,
aunque tiene sesgo negativo por el ruido de medicion del lactato: antes del
  shift, el lactato producido por unidad de crecimiento baja con el lactato
  (Spearman mediana −0.51, negativo en 80% de 76 cultivos). Una version
  anterior de este analisis concluyo lo contrario por usar una sola α; la
  auditoria de la tarea 7 lo detecto.
- **Sin interruptor externo, la forma tipo Kotidis no se distingue del
  modelo 2 en el perfil** (las dos con consumo de primer orden): 0.0440
  contra 0.0452 (diferencia −0.0012, IC −0.0038
  a 0.0015), y mejor que el modelo 1 (−0.0079, IC −0.0103 a −0.0058). Fecha
  el shift algo peor (56% exacto, 86% a ±1 dia).
- **No es solo el paso del tiempo**: si el freno se pone con el dia del
  cultivo o con la IVCD acumulada en lugar del lactato, el perfil queda en
  0.0493 y 0.0511.
- **Las dos ideas se suman**: el modelo 2 con el freno por lactato llega a
  0.0372 (−0.0080 contra el modelo 2, IC −0.0108 a −0.0054). Es exploratorio.

**Cautela.** Lo de esta seccion es post hoc: el estimador por cultivo y las
variantes se eligieron durante una auditoria, despues de ver los datos, asi
que los intervalos son optimistas. Es una asociacion: el lactato puede estar
representando otras cosas que se acumulan con el, como la base anadida y la
osmolalidad o el amonio. No contradice a Kotidis ni lo confirma: su termino
de glucosa consumida no se puede usar aqui porque la glucosa esta controlada
y no hay datos de alimentacion. Hay que confirmarlo con otros datos.

**Trabajos cercanos.** La idea de que las covariables mejoran el momento de
un cambio de fase frente al tiempo solo ya esta publicada: Wang, Harcum y
Xie (2026, Biotechnol Bioeng, doi:10.1002/bit.70205) y COSMIC-dFBA
(Gopalakrishnan et al. 2024, Metab Eng, doi:10.1016/j.ymben.2024.02.012).
Lo de este trabajo es mas acotado: un conjunto industrial de 100 cultivos con
validacion agrupada, un control explicito de estructura y la formulacion como
riesgo diario. Ademas, una forma con disparador interno (el freno por
lactato de Kotidis et al. 2019, doi:10.1002/bit.26960) no se distingue del
interruptor externo en el perfil (post hoc; con consumo de primer orden en
ambos), aunque fecha el shift algo peor.

**Revision independiente.** Estos resultados incorporan dos auditorias
independientes (scripts 15 a 21). La primera encontro dos errores en la
estimacion de α, el segundo paso del error a 2 dias usando mediciones
futuras, el criterio de ±1 dia demasiado flojo, los ceros de pCO2 y pO2 y la
regla de re-produccion relativa a un minimo casi nulo. La segunda encontro
que ajustar una sola α para todos los cultivos invertia el signo del freno
por lactato y perjudicaba a todos los modelos, que el argumento de "los
datos prefieren desplazamiento 0" era circular (ahora el piso del consumo se
elige en entrenamiento) y que la prueba de mezcla de trayectorias no podia
discriminar. Todo esta corregido arriba. Ninguna encontro fuga de la
etiqueta.

## Limitaciones

- **Datos normalizados y sin unidades.** Todo es relativo dentro del
  dataset: no hay umbrales absolutos, y los cocientes entre variables
  normalizadas no tienen sentido fisico. La normalizacion se hizo sobre todo
  el conjunto, una fuga leve e inevitable en el dato de origen.
- **Receta, producto, escala y temperatura estan entrelazados.** No hay llave
  entre las series y la linea celular o el producto, asi que la validacion se
  agrupa por cultivo, no por producto. El desempeno es probablemente
  optimista.
- **104 cultivos heterogeneos.** Cada modelo extra es una oportunidad de
  sobreajuste; por eso las comparaciones se hacen contra rivales fuertes, con
  intervalos por bootstrap de cultivos y reglas de decision fijadas antes.
- **El modelo de riesgo falla en los shifts tardios** (dia 7 en adelante).
- **Los modelos mecanisticos son explicativos**: usan la VCD medida. El
  pronostico verdadero a 2 dias (VCD extrapolada) solo se evaluo desde el
  dia 4.
- **El freno por lactato (script 21) y el consumo con piso (script 20) son
  post hoc**: se eligieron durante una auditoria, despues de ver los datos.
  El freno es una asociacion: el lactato puede representar otras cosas que
  se acumulan con el, y el ruido de medicion puede inflar su tamano.
- **Sin regimen de alimentacion ni dilucion.** La tasa especifica la ignora;
  es robusta hasta una dilucion de alrededor de 3% diario.
- **Sin marcadores redox** (NAD+/NADH, piruvato): las asociaciones no
  identifican el mecanismo.
- **El fenomeno esta muy estudiado** y su mecanismo sigue en debate (ver,
  entre otros, Schmitt et al. 2019, doi:10.1002/bit.27015). Esto no es el primer trabajo que modela
  o anticipa el shift en CHO; lo que aporta es una definicion por celula, una
  comparacion honesta entre estructuras mecanisticas y la documentacion de lo
  que no funciono.

## Historia del analisis

La primera version del caso de estudio (scripts 02 a 09) usaba otra
definicion del shift (sobre la concentracion) y otra pregunta (predecir el
dia con los dias 1-4). Esta documentada completa, con lo que no funciono, en
[`docs/caso_estudio_v1.md`](docs/caso_estudio_v1.md). El analisis se reviso
varias veces con auditorias independientes; las correcciones que salieron de
ellas estan en los mensajes de commit y en `analysis/15_correcciones_auditoria.py`.

## Tests

```bash
pytest
```

63 tests, ninguno depende del dataset con copyright: todo lo que
verifican se construye en el momento. Cubren el rebote tardio, la caida
profunda reversible, dias faltantes, NaN internos, entradas invalidas (dias
repetidos, no enteros o en cero), la propiedad de maximo local, el
refinamiento del pico en los dos sentidos, que la cuenta de dias medidos no
incluya los rellenados, que la pendiente use solo mediciones reales, que las
variables no cambien al agregar dias posteriores, que los p-valores nunca
sean cero, que los ejemplos de codigo de este README corran tal cual, y que
el script del CSV explique en espanol lo que esta mal en un archivo, y que los
puntos bajos aislados se senalen sin confundir una bajada real ni el ruido
cerca de cero.

## Cita del dataset

> Gangadharan, N., Sewell, D., Turner, R., Field, R., Cheeks, M., Oliver, S.G., Slater,
> N.K.H., Dikicioglu, D. (2021). Data intelligence for process performance
> prediction in biologics manufacturing. *Computers & Chemical Engineering*,
> 146, 107226. https://doi.org/10.1016/j.compchemeng.2021.107226

## Autor

Desarrollado por Arturo Rodriguez (Braulio Arturo Rodriguez Angulo),
Ingenieria Bioquimica, ENCB-IPN.

## Licencia

MIT. El codigo es libre; el dataset del caso de estudio no es mio y no se
redistribuye.
