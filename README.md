# lactateshift

Deteccion del *lactate shift* en cultivos de celulas de mamifero, y un caso de
estudio sobre si ese cambio puede anticiparse desde los primeros dias del
cultivo.

El **lactate shift** es el momento en que un cultivo deja de producir lactato
netamente y empieza a consumirlo. Cuando ocurre y que tan sostenido es se
asocia con el desempeno del lote, asi que detectarlo de forma consistente es
un requisito previo para cualquier analisis de proceso que lo involucre.

Este repositorio tiene dos partes:

1. **`lactateshift`**, un paquete de Python que detecta el shift en cualquier
   serie de lactato y construye variables predictoras de una ventana temprana
   sin dejar entrar informacion posterior a ella.
2. **`analysis/`**, un caso de estudio sobre 106 cultivos CHO industriales de
   5 a 500 L, que intenta predecir el dia del shift usando solo los dias 1-4.

El caso de estudio llega a un **resultado mixto y eso es parte del punto**: hay
senal cuando entrenamiento y prueba mezclan escalas de reactor, pero el modelo
no transfiere a una escala que no vio. Esta documentado como hallazgo, no
escondido.

![Pipeline del proyecto](docs/pipeline.png)

---

## Instalacion

```bash
git clone https://github.com/USUARIO/lactate-shift-cho
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

Variables predictoras de una ventana temprana:

```python
from lactateshift import early_window_features

X = early_window_features(
    datos, id_col="culture", day_col="day",
    value_cols=["lactate", "glucose", "vcd"],
    window=4,                          # solo los dias 1 a 4
    ratios=[("lactate", "vcd")],       # cocientes al cierre de la ventana
    static_cols=["scale"],             # constantes conocidas desde el inicio
)
```

Nada de lo que devuelve depende de un dato posterior al dia 4 ni de otras
series del conjunto. Lo que falta por completo queda `NaN` a proposito: la
imputacion pertenece al pipeline de modelado, donde se ajusta solo con el
fold de entrenamiento.

## Como funciona la deteccion

Hay shift en el primer dia `t` tal que:

1. la derivada de la serie suavizada es negativa durante `n_consecutive` dias
   a partir de `t`, y
2. el valor cae desde el maximo local en `t` al menos `drop_threshold` veces
   ese maximo, **antes** de que la serie vuelva a superarlo.

La condicion 2, con esa restriccion, separa un cambio de regimen de un bache:
si la serie se recupera enseguida, la caida nunca alcanza el umbral.

**Por que el primer maximo local y no el maximo global.** Una parte de los
cultivos hace el shift, consume lactato varios dias y despues vuelve a
producirlo al final hasta superar el pico inicial. Una regla anclada en el
maximo global marca esos cultivos como "sin shift", lo cual es falso: el shift
ocurrio, solo fue reversible. En el caso de estudio eso movia el conteo de 76
a 101 cultivos de 106. El error se descubrio graficando las curvas, no
mirando numeros.

**Correccion del sesgo del suavizado.** Una media movil centrada desplaza el
maximo hacia atras cuando la caida es mas rapida que la subida, que es el caso
tipico de un cultivo. Sin corregirlo, el dia detectado salia sistematicamente
un dia antes del real. Con `refine_peak=True` (por omision) el dia se reajusta
al maximo de la serie sin suavizar en una vecindad estrecha. En los cultivos
sinteticos, donde el dia real se conoce, el sesgo medio pasa de −0.91 a −0.18
dias y los aciertos exactos de 3/34 a 29/34. Es la razon de ser de
`lactateshift.datasets`: un detector sin datos de verdad conocida no se puede
auditar.

## El caso de estudio

Dataset: 106 cultivos CHO de 5 a 500 L, 9 a 18 dias, 24 variables de proceso
normalizadas, del material suplementario de Gangadharan et al. (2021).
**El archivo no se distribuye aqui: esta bajo copyright de Elsevier.** Ver
[`data/raw/README.md`](data/raw/README.md) para obtenerlo, con verificacion
por SHA-256.

```bash
python analysis/00_verificar_datos.py        # comprueba el archivo
python analysis/01_exploracion.py            # estructura, faltantes, cobertura
python analysis/02_definicion_evento.py      # etiqueta + sensibilidad
python analysis/02b_figura_evento.py         # verificacion visual
python analysis/03_features.py               # variables de los dias 1-4
python analysis/04_modelado.py               # regresion, Cox y controles
python analysis/05_shap.py                   # interpretacion y sus limites
python analysis/06_respuesta_auditoria.py    # experimentos de la auditoria externa
```

### La pregunta que este trabajo responde, y la que no

Con la definicion congelada, **101 de 106 cultivos hacen el shift**. Preguntar
"lo hace o no" da 95 contra 5 y no tiene contenido: lo que varia es **cuando**,
entre los dias 4 y 11. Por eso el problema se planteo como tiempo-a-evento.

Ademas, 16 cultivos hacen el shift **dentro** de la ventana de observacion y se
excluyen: no hay nada que anticipar cuando el desenlace ya esta en las propias
variables. Eso deja **90 cultivos: 85 con evento y 5 censurados**.

La consecuencia hay que decirla sin rodeos, porque cambia el alcance de todo lo
demas: este trabajo **no** responde "se puede predecir el lactate shift en
cultivos CHO". Responde una pregunta condicional:

> Dado un cultivo que **todavia no ha hecho el shift** al cerrar el dia 4,
> .que dia lo hara?

Los 16 cultivos excluidos no se predicen, y el modelo tampoco resuelve la
decision previa de "ya ocurrio / aun no". Es una limitacion de alcance, no un
detalle de redaccion.

Dia del evento: mediana 6 (rango 5-11). La anticipacion efectiva sobre la
ventana es de **2 dias en mediana**, y 20 de 85 eventos ocurren a un solo dia
del cierre. Es poco, y decirlo cambia como se lee lo que sigue.

### Que salio

Prediccion del dia del evento, error absoluto medio en dias
(validacion cruzada repetida 5×5):

| Variables | Baseline (mediana) | Ridge | Random Forest |
|---|---|---|---|
| Nucleo (28) | 0.835 | 0.632 | **0.575** |
| + glutamina y osmolalidad (36) | 0.835 | 0.667 | 0.581 |
| Solo la escala del reactor (control) | 0.835 | 0.928 | 0.795 |

Prueba de permutacion (200 barajadas, seleccion de hiperparametro incluida
dentro de cada barajada): MAE observado 0.643 contra un nulo de 0.935 ± 0.023,
p < 0.005. La senal existe.

Modelo de Cox sobre los 90 cultivos, aprovechando los censurados:
**c-index 0.825**, contra un nulo permutado de 0.485 ± 0.051 y un control de
solo-escala de 0.542.

> **Sobre el 0.575.** Es la mejor de nueve combinaciones de modelo y conjunto
> de variables, elegida despues de verlas todas. Sin una validacion cruzada
> anidada, esa cifra es **exploratoria**, no una estimacion confirmatoria de
> desempeno futuro. El contraste que si aguanta es el orden: los tres modelos
> le ganan al baseline y el control de solo-escala no.

### El resultado negativo

Dejando fuera una escala de reactor completa y prediciendo sobre ella:

| Escala excluida | n | Baseline | Ridge | Random Forest |
|---|---|---|---|---|
| 0.00202 | 43 | 1.023 | 0.967 | 1.075 |
| 0.00181 | 18 | 0.778 | 1.344 | 1.609 |
| 0.00000 | 16 | 0.438 | 0.526 | 0.466 |
| **Ponderado** | | **0.844** | 0.964 | 1.073 |

**Ningun modelo le gana al baseline cuando la escala es nueva.** La senal que
se ve con validacion cruzada aleatoria es especifica del contexto de proceso.
Para alguien que quisiera llevar esto a planta, esa es la conclusion
operativa: un modelo asi habria que reentrenarlo por escala, no trasladarlo.

Con la precision que corresponde: son **tres** escalas (43, 18 y 16 cultivos).
Lo que los numeros permiten afirmar es que no transfiere entre estas tres, no
que no transfiera en general. Quitar `escala` de las variables no cambia el
resultado (Random Forest 1.073 frente a 1.080), asi que la degradacion no
viene de que el modelo conociera el volumen de la escala nueva.

### Que dice la interpretacion, y que no

Los valores SHAP se calculan **fuera del fold de entrenamiento**: SHAP explica
al modelo, y un modelo sobreajustado da explicaciones nitidas de su propio
sobreajuste.

La direccion de los efectos es coherente con lo que se esperaria: mas biomasa
y crecimiento mas rapido en los dias 1-4 adelantan el shift; un cociente
lactato/glucosa alto tambien. Mas glutamato, amonio o pH lo retrasan.

Pero la magnitud engana, y el repositorio incluye las comprobaciones que lo
demuestran.

**Importancia no es necesidad.** El glutamato domina el ranking de SHAP
(0.53 de |SHAP| sumado, contra 0.17 de la siguiente) y tambien encabeza la
importancia por permutacion. Aun asi:

| Conjunto | MAE (dias) |
|---|---|
| Baseline (mediana) | 0.835 |
| Todas las variables | 0.575 |
| Solo glutamato | 0.676 |
| Todas **sin** glutamato | 0.592 |

Quitarlo casi no empeora nada. SHAP y la permutacion miden cuanto **usa** el
modelo una variable, no cuanta informacion **unica** aporta: la del glutamato
tambien esta en las demas, y el modelo la recupera de ahi.

(La variable a ablacionar se eligio despues de ver los resultados de SHAP, lo
cual sesga el experimento. Conviene notar hacia donde: ese sesgo favorece que
quitarla empeore el modelo, y no empeora. La lectura es conservadora.)

**Parte de la senal es la identidad del proceso, no fisiologia.** El grafico
de dependencia del glutamato no muestra una relacion continua sino dos nubes
separadas, lo que no parece una respuesta metabolica graduada. Partiendo la
muestra por la mediana de esa variable, el 74% del grupo alto cae en una sola
escala de reactor, y los dos grupos difieren en el dia del evento (mediana 7
contra 6). Es decir, la variable esta funcionando en parte como marcador de a
que familia de cultivos pertenece el lote.

Eso encaja con el leave-one-scale-out: el modelo aprende el dia caracteristico
de cada familia de procesos y por eso no transfiere a una escala nueva. La
lectura honesta no es "el glutamato temprano controla el shift", sino "el
glutamato temprano ayuda a identificar de que proceso viene el lote, y cada
proceso tiene su momento tipico de shift".

### Auditoria externa

El codigo y el analisis se revisaron buscando errores.

Encontraron **tres errores reales**, todos corregidos:

1. `<col>_n` contaba como medidos los dias rellenados hacia atras. Habia
   incluso un test que afirmaba ese comportamiento, lo que lo volvia
   permanente. La cuenta se hace ahora sobre la mascara previa al relleno.
2. La prueba de permutacion congelaba un `alpha` de Ridge elegido con las
   etiquetas reales. El sesgo iba a favor del resultado: con etiquetas
   barajadas, `RidgeCV` elegiria mucha mas regularizacion y erraria menos, asi
   que fijar un alpha pequeno inflaba el error nulo. Ahora el hiperparametro se
   reajusta dentro de cada barajada.
3. La agrupacion de variables para SHAP colapsaba lactato/glucosa y
   lactato/VCD en una sola categoria, mezclando dos senales distintas.

Una objecion resulto incorrecta al verificarla: la permutacion del modelo de
Cox si mantiene unido el par (tiempo, evento), porque el mismo indice se aplica
a las dos series.

Tres objeciones no tenian respuesta con el analisis existente y se convirtieron
en los experimentos de `analysis/06_respuesta_auditoria.py`:

**.El modelo solo extrapola la curva de lactato que ya empezo?** Era la
objecion mas fuerte: la etiqueta se construye de la trayectoria futura del
lactato, y el lactato temprano esta entre las variables.

| Conjunto | MAE (dias) |
|---|---|
| Baseline | 0.835 |
| Todas (28 variables) | 0.575 |
| Solo variables de lactato (6) | 0.692 |
| **Sin ninguna variable de lactato (22)** | **0.594** |

Quitar todo el lactato deja el desempeno casi igual. El modelo no esta
continuando una curva.

**.Las conclusiones dependen de la definicion del evento, que se ajusto
mirando los datos?** Se rehizo el modelo completo con ocho definiciones
alternativas, cada una comparada contra su propio baseline. La mejora sobre el
baseline es positiva en las ocho, entre 0.176 y 0.264 dias (mediana 0.249),
incluida la variante sin la correccion del pico, que cambia el numero de
cultivos excluidos de 16 a 6.

**.El leave-one-scale-out mide transferencia real?** Repetido sin la variable
`escala`: sin cambios relevantes (ver arriba).

## Limitaciones

- **El alcance es condicional**: cultivos que no han hecho el shift al cerrar
  el dia 4. Los 16 excluidos no se predicen.
- **El 0.575 es una cifra seleccionada** entre nueve combinaciones, sin
  validacion cruzada anidada. Exploratoria, no confirmatoria.
- **La imputacion del dataset de origen no es causal.** Los autores rellenaron
  huecos con interpolacion de Stineman mas SVR sobre series completas;
  Stineman usa el punto anterior y el posterior. Un valor "del dia 3" puede
  contener informacion del dia 10. Por eso todo el analisis parte de la hoja
  sin rellenar, y la imputacion se hace aqui, hacia adelante y dentro de cada
  fold.
- **No se puede agrupar la validacion por linea celular.** El dataset trae
  linea celular y lote para 45 cultivos, pero sin llave a las series de tiempo.
  Dos cultivos de la misma linea pueden caer uno en entrenamiento y otro en
  prueba, asi que el desempeno reportado es probablemente optimista.
- **Los datos vienen normalizados 0-1 por columna sobre todo el conjunto.** Esa
  normalizacion ya uso todos los cultivos: hay una fuga leve e inevitable en el
  dato de origen, que no se puede deshacer porque no hay unidades.
- **El c-index de 0.825 convive con empates masivos** (32 eventos en el dia 6,
  22 en el 7, 20 en el 5) y solo 5 censurados. Es calculable e informativo,
  pero no soporta una interpretacion fuerte.
- **Sin marcadores redox** (NAD+, piruvato) en el dataset, esto describe
  senales de proceso, no mecanismo. Asociacion, no causalidad.
- **El muestreo esparso degrada la precision del dia.** Con 30% de dias
  ausentes el error en el dia detectado puede llegar a 2 dias; el evento se
  sigue detectando, el dia no es igual de confiable. Hay un test que lo
  caracteriza.
- El fenomeno esta muy estudiado y su mecanismo sigue en debate. Esto no es el
  primer trabajo que predice comportamiento de lactato en CHO; lo que aporta
  es una deteccion auditable y una validacion que incluye lo que no funciono.

## Tests

```bash
pytest
```

34 tests, ninguno depende del dataset con copyright: todo lo que verifican se
construye en el momento. Cubren el caso del rebote tardio, dias faltantes,
NaN internos, series demasiado cortas, parametros invalidos, el sesgo del
suavizado y su correccion, que la cuenta de dias medidos no incluya los dias
rellenados, y que las variables de la ventana temprana no cambien cuando se
agregan dias posteriores.

## Cita del dataset

> Gangadharan, N., Turner, R., Field, R., Cheeks, M., Oliver, S.G., Slater,
> N.K.H., Dikicioglu, D. (2021). Data intelligence for process performance
> prediction in biologics manufacturing. *Computers & Chemical Engineering*,
> 146, 107226. https://doi.org/10.1016/j.compchemeng.2021.107226

## Licencia

MIT. El codigo es libre; el dataset del caso de estudio no es mio y no se
redistribuye.
