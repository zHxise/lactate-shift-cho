# Validacion externa del detector — resultado

Fecha: 2026-09-24. Protocolo: `PROTOCOLO.md` (commit `b48eaa8`, fijado antes de
buscar datos). Orden registrado en git: curvas (`d530162`) → detecciones
(`54b4293`) → respuestas y este resultado.

## Resultado confirmatorio

**7 de 9 aciertos (78%), IC 95% exacto 40-97%. No cumple el criterio fijado de
antemano (80%).**

| Articulo | Curva | Que declaran los autores | Detector | Acierto |
|---|---|---|---|---|
| Lularevic 2019 | CY01 | maximo el dia 4 | dia 4 | si |
| Lularevic 2019 | 3C12 | maximo el dia 5 | dia 5 | si |
| Lularevic 2019 | R33A | maximo el dia 5 | dia 5 | si |
| Lularevic 2019 | EB7 | maximo el dia 5 | dia 5 | si |
| Becker 2019 | REF | sin shift (sube todo el cultivo) | sin shift | si |
| Becker 2019 | COP | sin shift (sube todo el cultivo) | sin shift | si |
| Yin 2025 | vvm baja | sin shift (acumula en fase estacionaria) | sin shift | si |
| Becker 2019 | NOB | pasa a consumo entre los dias 6 y 10 | sin shift | **no** |
| Yin 2025 | vvm alta | pasa a consumo en la fase de dias 7-14 | sin shift | **no** |

Las citas textuales y su ubicacion estan en `respuestas.csv`. **Verificadas**
el 2026-09-24 por Braulio Rodriguez contra las paginas originales (Becker p. 5,
Yin pp. 5-6, Lularevic p. 109): las siete frases aparecen tal cual y se
refieren a las curvas asignadas. En la verificacion se agrego a las citas de
Lularevic la frase anterior del mismo parrafo ("All four experiments showed a
shift in lactate metabolism"), que hace explicito que el maximo es el shift;
no cambia ningun dia ni resultado.

Detalle menor: Lularevic escribe que los maximos del fed-batch van de 20 a
42 mM, pero en su figura van de 19 a 38 mM (EB7, 3C12). El dia coincide; el
rango de concentraciones del texto es aproximado.

## Por que fallo donde fallo

Las dos curvas falladas son las unicas donde los autores declaran el
**inicio del consumo** y la caida de concentracion es suave
(`exploratorio.csv`, descriptivo):

- Becker NOB: el lactato baja 19% del dia 5 al 10 (16% en la serie
  suavizada) y despues vuelve a subir. Los autores detectan el consumo con
  tasas especificas y balances de carbono, no con la concentracion: en
  fed-batch la alimentacion puede sostener la concentracion aunque las celulas
  consuman.
- Yin vvm alta: baja 32% del dia 10 al 14, pero el cultivo termina a media
  bajada y la media movil del ultimo dia solo promedia dos puntos, asi que en
  la serie suavizada la caida queda en 25%. Los autores se apoyan tambien en
  la tasa especifica de produccion de lactato de los dias 7-14 (su Fig. 2f),
  que es negativa con vvm alta.

En las cuatro curvas donde los autores declaran el **maximo**, las caidas son
de 53% a 100% (suavizadas) y el detector da el mismo dia exacto.

Lectura: el detector es conservador. Marca en el dia correcto los shifts
claros (caida de al menos 30%) y no marca los suaves. Es la definicion que se
eligio para el caso de estudio, no un error de codigo, pero ahora hay
evidencia externa de su costo. No se cambio ningun parametro.

## Exploratorio anunciado antes de detectar

Sin la curva de Yin vvm alta, cuya respuesta es una fase de 8 dias y no un
dia: 7 de 8 (88%), IC 95% 47-100%. Se anuncio en `DIGITALIZACION.md` antes de
correr el detector. No cambia el veredicto: el criterio confirmatorio se
aplica a las 9 curvas, y el intervalo sigue siendo demasiado ancho para
afirmar nada.

## Lo que este resultado no dice

- **Pocas curvas.** Con 9, el intervalo va de 40% a 97%. Ni un 9 de 9 habria
  sido concluyente.
- **No son 9 datos independientes.** Cuatro vienen de la misma tesis (misma
  plataforma de proceso) y los cuatro aciertos de "maximo" son todos de ahi.
- **La extraccion no fue ciega** (ver `DIGITALIZACION.md`, desviacion 1).
- **Definiciones mezcladas.** "Maximo" e "inicio del consumo" no son el mismo
  evento. El detector reporta el maximo.
- **Solo valida el detector**, no el modelo predictivo del caso de estudio.

## Consecuencia para el caso de estudio

En el caso de estudio "shift" significa una caida de al menos 30%. Los 5
cultivos censurados (sin shift) pueden incluir shifts suaves como los dos que
aqui no se detectaron. El README lo agrega a las limitaciones.

## Reproducir

```bash
pip install pymupdf                              # solo para digitalizar
python validacion_externa/digitalizar.py         # PDF de fuentes/ -> curvas.csv
python validacion_externa/comparar.py detectar   # -> detecciones.csv
python validacion_externa/comparar.py comparar   # + respuestas.csv -> resultado.csv
python validacion_externa/exploratorio.py        # -> exploratorio.csv
```
