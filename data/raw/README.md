# Como obtener el dataset del caso de estudio

Este directorio esta vacio a proposito.

El caso de estudio usa el material suplementario `mmc3.xlsx` de:

> Gangadharan, N., Turner, R., Field, R., Cheeks, M., Oliver, S.G., Slater, N.K.H.,
> Dikicioglu, D. (2021). *Data intelligence for process performance prediction in
> biologics manufacturing.* Computers & Chemical Engineering, 146, 107226.
> https://doi.org/10.1016/j.compchemeng.2021.107226

**El articulo y su material suplementario estan bajo copyright de Elsevier y no
son de acceso abierto.** Por eso el archivo no se redistribuye aqui. Para
reproducir el caso de estudio hay que descargarlo por cuenta propia desde la
pagina del articulo (se necesita acceso institucional o compra) y colocarlo en
este directorio.

El manuscrito aceptado si esta disponible en abierto en el repositorio de UCL:
https://discovery.ucl.ac.uk/id/eprint/10119588/ — util para leer el metodo,
aunque no incluye el suplemento de datos.

## Verificacion

Una vez descargado, deberia quedar como:

```
data/raw/1-s2_0-S0098135421000041-mmc3.xlsx
```

con este SHA-256:

```
d572a1aa4a74bbaecb10046e759364d1740dc8731de86661d7646ad14758e839
```

Para comprobarlo:

```bash
python analysis/00_verificar_datos.py
```

## Si no tienes acceso al archivo

El paquete `lactateshift` no lo necesita. Los tests y el ejemplo de uso
funcionan con cultivos sinteticos generados en el momento:

```python
from lactateshift import make_synthetic_cultures, detect_shift_batch
series, verdad = make_synthetic_cultures(40, seed=0)
detect_shift_batch(series, id_col="culture", day_col="day", value_col="lactate")
```
