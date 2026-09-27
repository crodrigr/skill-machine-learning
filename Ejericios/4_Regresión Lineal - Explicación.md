# Regresión Lineal: Predicción del coste de un incidente de seguridad

*Explicación para quien no tiene conocimientos previos de programación ni de Machine Learning.*

## ¿De qué trata este ejercicio?

Imagina que en una empresa se producen incidentes de seguridad (ataques, virus, fugas de datos...) y cada vez que ocurre uno, se ve afectado un número distinto de equipos (ordenadores, servidores...). La empresa quiere saber algo muy práctico:

> **Si sé cuántos equipos se han visto afectados, ¿puedo estimar cuánto me va a costar arreglarlo?**

Para responder a esa pregunta, el notebook construye un modelo matemático muy sencillo que aprende, a partir de datos pasados, la relación entre "equipos afectados" y "coste del incidente". Una vez que el modelo ha aprendido esa relación, se le puede preguntar por un caso nuevo (por ejemplo, "¿cuánto costará un incidente con 1.800 equipos afectados?") y él dará una estimación.

La técnica que se usa para esto se llama **regresión lineal**.

## ¿Qué es la regresión lineal? (la idea, sin fórmulas)

Piensa en una gráfica con dos ejes:
- El eje horizontal representa el número de equipos afectados.
- El eje vertical representa el coste del incidente.

Si colocamos en esa gráfica todos los incidentes que ha tenido la empresa en el pasado, obtenemos una nube de puntos. La regresión lineal busca **la línea recta que mejor atraviesa esa nube de puntos**, es decir, la recta que se queda "en medio" de todos ellos con el menor error posible.

Una vez encontrada esa recta, se puede usar para dos cosas:
1. **Entender la tendencia**: por ejemplo, saber que "cada equipo afectado adicional suele añadir tantos euros de coste".
2. **Predecir casos nuevos**: dado un número de equipos afectados que aún no se ha visto, calcular en qué punto de la recta caería y así estimar su coste.

Es la misma idea que cuando decimos, de forma intuitiva, "cuantos más equipos se ven afectados, más caro sale el incidente, y más o menos sube de esta manera". La regresión lineal simplemente hace esa intuición precisa y calculable.

## ¿Por qué se usa este algoritmo y no otro?

- **Es el punto de partida natural en Machine Learning.** Es el algoritmo más simple que existe para predecir un número (el coste) a partir de otro número (los equipos afectados), y por eso se usa aquí como ejercicio introductorio.
- **Es fácil de interpretar.** El resultado es literalmente una recta con una fórmula tipo "coste = un valor base + un incremento por cada equipo afectado". Cualquier persona, sin ser experta, puede entender qué significa esa fórmula.
- **Encaja con el problema.** Si al dibujar los datos reales se observa que, a grandes rasgos, cuantos más equipos se afectan más sube el coste de forma bastante constante (ni con saltos raros ni con curvas complicadas), una línea recta es una aproximación razonable. Si la relación fuera muy distinta (por ejemplo, que el coste se disparara exponencialmente), habría que usar otro tipo de modelo.
- **Es rápido y barato de entrenar.** No necesita muchos datos ni mucha potencia de cálculo, a diferencia de modelos más complejos (redes neuronales, etc.), que aquí serían innecesarios para un problema tan simple.

## Recorrido del notebook, paso a paso

### 0. Imports (herramientas que se van a usar)

```python
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install scikit-learn
```

Antes de poder trabajar, hay que instalar las "cajas de herramientas" (librerías) que se van a necesitar. Cada una sirve para algo distinto:

| Librería | Para qué se usa aquí |
|---|---|
| **numpy** | Trabajar con números y tablas de números (matrices) de forma eficiente. |
| **pandas** | Organizar los datos en tablas parecidas a una hoja de Excel. |
| **matplotlib** | Dibujar gráficas. |
| **scikit-learn** | Contiene el algoritmo de regresión lineal ya implementado, listo para usar. |

### 1. Generación del conjunto de datos

```python
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```

Como en este ejercicio no se dispone de datos reales de incidentes de una empresa, se **generan datos de ejemplo de forma aleatoria**, pero simulando la relación que queremos que exista:

- `X` representa el número de equipos afectados (100 incidentes de ejemplo).
- `y` representa el coste de cada incidente, calculado como *"un valor base (4) + 3 veces el número de equipos afectados"*, más un poco de aleatoriedad (`np.random.randn`) para que no sea una recta perfecta, sino que se parezca a datos reales, que siempre tienen algo de "ruido" o variación.

Esto es importante: **se está fabricando el ejemplo a propósito de forma que sepamos de antemano que la relación es lineal**, para poder comprobar después que el algoritmo es capaz de "descubrir" esa relación por sí solo a partir de los datos.

### 2. Visualización del conjunto de datos

```python
plt.plot(X, y, "b.")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Coste del incidente (u/10000)")
plt.show()
```

Se dibuja la nube de puntos generada. Cada punto azul es un incidente simulado: su posición horizontal indica cuántos equipos se vieron afectados y su posición vertical, cuánto costó. Al mirar la gráfica se aprecia a simple vista una tendencia: a más equipos afectados, más coste, siguiendo aproximadamente una línea recta.

### 3. Modificación del conjunto de datos (ajuste de escala)

```python
data = {'n_equipos_afectados': X.flatten(), 'coste': y.flatten()}
df = pd.DataFrame(data)
```

Aquí se convierten los números anteriores en una **tabla** (llamada `DataFrame`, el formato típico de pandas), con dos columnas: `n_equipos_afectados` y `coste`. Es literalmente como pasar los datos a una hoja de cálculo con dos columnas y 100 filas (una por incidente).

**¿Qué hace `.flatten()`?**

`X` e `y` no son listas simples de números: son matrices de 2 dimensiones con forma `(100, 1)`, es decir, 100 filas y **1 columna** (como una columna de Excel). Para poder meter esos datos en el diccionario que arma el `DataFrame`, hace falta una lista "plana" de una sola dimensión, no una columna 2D.

`.flatten()` toma esa matriz de forma `(100, 1)` y la "aplasta" en un array simple de 1 dimensión con 100 elementos, sin perder ni cambiar ningún dato, solo reorganizando su forma:

```
Antes (forma 100x1):        Después de .flatten() (forma 100,):
[[0.37],                    [0.37, 0.95, 0.73, ...]
 [0.95],
 [0.73],
 ...]
```

Es el paso inverso al `.reshape(-1, 1)` que se usa más abajo en el paso 4: `.flatten()` convierte una columna en una lista plana, y `.reshape(-1, 1)` convierte una lista plana de vuelta en columna. Ninguno de los dos cambia los valores, solo la forma en que están organizados.

```python
df['n_equipos_afectados'] = df['n_equipos_afectados'] * 1000
df['n_equipos_afectados'] = df['n_equipos_afectados'].astype('int')
df['coste'] = df['coste'] * 10000
df['coste'] = df['coste'].astype('int')
```

Los números generados aleatoriamente eran muy pequeños (decimales entre 0 y 2, por ejemplo). Para que el ejercicio se parezca más a una situación real —donde hablamos de "300 equipos afectados" y "45.000 € de coste", y no de "0,3" y "4,5"— se multiplican esos números por 1.000 y por 10.000 respectivamente, y se convierten en números enteros (`astype('int')`, es decir, sin decimales). Es solo un cambio de escala para que los datos "se lean" de forma más natural, no cambia la relación de fondo entre las dos variables.

### 4. Construcción del modelo

Esta es la parte central del ejercicio: enseñarle al ordenador la relación entre las dos columnas.

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)
```

- `LinearRegression()` crea un modelo de regresión lineal "en blanco", todavía sin entrenar. Es como coger una regla vacía, sin haberla apoyado aún sobre los puntos.
- `.fit(...)` es el paso de **entrenamiento**: aquí es donde el algoritmo mira todos los puntos (equipos afectados → coste) y calcula matemáticamente cuál es la recta que mejor se ajusta a ellos, minimizando el error total entre la recta y los puntos reales.
  - `df['n_equipos_afectados'].values` son los datos de entrada (equipos afectados).
  - `df['coste'].values` son los datos que queremos predecir (el coste).
  - El `.reshape(-1, 1)` es un detalle técnico que exige la librería: los datos de entrada tienen que entregarse con un formato de tabla (columna), aunque solo haya una columna. Es simplemente un requisito de formato, no cambia el contenido de los datos.

Tras ejecutar `.fit()`, el modelo ya "sabe" cuál es la recta.

```python
lin_reg.intercept_   # Parámetro theta 0
lin_reg.coef_        # Parámetro theta 1
```

Toda recta se puede describir con dos números:
- **`intercept_` (theta 0):** el punto de partida, es decir, el coste estimado cuando el número de equipos afectados es 0. Sería como el "coste fijo" de gestionar un incidente aunque no afecte a ningún equipo (por ejemplo, tiempo del equipo de respuesta, investigación, etc.).
- **`coef_` (theta 1):** cuánto sube el coste por cada equipo afectado adicional. Es la "pendiente" de la recta.

Con estos dos números, la fórmula final queda así (en palabras):

> **Coste estimado = coste base + (coste por equipo × número de equipos afectados)**

### Predicción sobre los extremos de los datos (para poder dibujar la recta)

```python
X_min_max = np.array([[df["n_equipos_afectados"].min()], [df["n_equipos_afectados"].max()]])
y_train_pred = lin_reg.predict(X_min_max)
```

Para dibujar una línea recta en una gráfica basta con conocer dos puntos (sus dos extremos) y unirlos. Por eso aquí se coge el valor **mínimo** y el valor **máximo** de equipos afectados que hay en los datos, y se le pide al modelo (`.predict()`) que calcule el coste estimado para esos dos casos. Con esos dos puntos ya se puede trazar la recta completa.

### Representación gráfica de la recta aprendida

```python
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()
```

Aquí se ve el resultado final del entrenamiento en una sola imagen:
- Los **puntos azules** son los incidentes reales (los datos con los que se entrenó el modelo).
- La **línea verde** es la recta que el modelo ha aprendido.

Si la línea verde pasa "por en medio" de la nube de puntos azules, es una señal de que el modelo ha aprendido bien la tendencia general de los datos.

### 5. Predicción de un incidente nuevo

```python
x_new = np.array([[1800]]) # equipos afectados
coste = lin_reg.predict(x_new)
print("El coste del incidente sería:", int(coste[0]), "€")
```

Este es el objetivo final de todo el ejercicio: usar el modelo ya entrenado para un caso real y nuevo que no estaba en los datos originales. Se le indica al modelo "imagina un incidente con 1.800 equipos afectados" y el modelo devuelve el coste que estima según la recta que aprendió.

```python
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(x_new, coste, "rx")
plt.show()
```

Por último, se dibuja de nuevo toda la información junta:
- Puntos azules: incidentes históricos.
- Línea verde: la recta aprendida.
- Una **cruz roja**: la nueva predicción, marcando exactamente sobre la recta el punto correspondiente a "1.800 equipos afectados".

## Resumen para quedarse con la idea clave

1. Se recopilan (aquí, se simulan) datos históricos de incidentes: equipos afectados y su coste.
2. Se dibujan para comprobar que existe una tendencia razonablemente lineal.
3. Se usa un algoritmo de **regresión lineal**, que encuentra automáticamente la recta que mejor resume esa tendencia.
4. Esa recta se puede leer en dos números: un coste base y un incremento por cada equipo afectado.
5. Con la recta ya calculada, se puede **predecir el coste de incidentes futuros** solo con saber cuántos equipos afectará.

Es un ejemplo deliberadamente sencillo para introducir la idea de Machine Learning: aprender un patrón a partir de datos pasados para hacer predicciones sobre casos nuevos. La regresión lineal es el ejemplo más básico de ese principio, y por eso es el punto de partida habitual antes de pasar a algoritmos más complejos.

## Otro caso real y común: precio de un piso en alquiler según sus metros cuadrados

Para terminar de fijar la idea, veamos otro ejemplo clásico donde se aplica exactamente el mismo procedimiento, cambiando solo el problema de negocio: **estimar el precio mensual de alquiler de un piso a partir de sus metros cuadrados**.

> **Si sé cuántos metros cuadrados tiene un piso, ¿puedo estimar cuánto costará alquilarlo?**

Es un caso muy habitual en el sector inmobiliario: una inmobiliaria tiene el histórico de pisos que ha alquilado, con sus metros cuadrados y el precio al que finalmente se alquilaron. Con esos datos, quiere poder darle a un cliente una estimación de precio en cuanto conozca los metros del piso nuevo que quiere sacar al mercado.

### 1. Generación (simulada) de los datos

```python
X = 2 * np.random.rand(100, 1)
y = 300 + 8 * X + np.random.randn(100, 1)
```

De nuevo se simulan datos porque no se dispone de un histórico real, pero fabricados con una relación conocida:

- `X` representa (a escala reducida) los metros cuadrados de cada piso.
- `y` representa el precio mensual de alquiler, calculado como *"un precio base (300) + 8 veces los metros cuadrados"*, con algo de ruido aleatorio para que se parezca a datos reales (no todos los pisos con los mismos metros se alquilan exactamente por el mismo precio: influye la zona, el estado, la planta...).

### 2. Ajuste de escala y tabla de datos

```python
data = {'metros_cuadrados': X.flatten(), 'precio_alquiler': y.flatten()}
df = pd.DataFrame(data)

df['metros_cuadrados'] = (df['metros_cuadrados'] * 50 + 30).astype('int')
df['precio_alquiler'] = df['precio_alquiler'].astype('int')
```

Igual que en el ejemplo anterior, se pasan los datos a una tabla y se reescalan para que se parezcan a un caso real: pisos de entre 30 y 130 m² aproximadamente, con precios de alquiler en euros.

### 3. Entrenamiento del modelo

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(df['metros_cuadrados'].values.reshape(-1, 1), df['precio_alquiler'].values)

lin_reg.intercept_   # precio base, para 0 m² (theta 0)
lin_reg.coef_        # cuánto sube el precio por cada m² adicional (theta 1)
```

El modelo aprende, a partir de los pisos históricos, la recta:

> **Precio estimado = precio base + (precio por m² × metros cuadrados del piso)**

### 4. Predicción de un piso nuevo

```python
x_new = np.array([[90]])  # piso de 90 m²
precio = lin_reg.predict(x_new)
print("El precio de alquiler estimado sería:", int(precio[0]), "€/mes")
```

Con el modelo ya entrenado, basta con darle los metros cuadrados de un piso que aún no se ha alquilado (en este caso, 90 m²) para obtener una estimación de su precio de alquiler mensual, del mismo modo que antes se estimaba el coste de un incidente a partir de los equipos afectados.

Este ejemplo deja ver algo importante: **el procedimiento es siempre el mismo**, cambie lo que cambie el problema (coste de incidentes, precio de alquiler, o cualquier otra magnitud que dependa de una sola variable de forma aproximadamente lineal). Lo único que varía es qué representan `X` e `y` en cada caso.

## Un paso más allá: regresión lineal múltiple

En los dos ejemplos anteriores el coste (o el precio) dependía de **una sola cosa**: el número de equipos afectados, o los metros cuadrados. Pero en la vida real, casi ningún resultado depende de una única causa. El coste de un incidente de seguridad, por ejemplo, probablemente no depende solo de cuántos equipos se vieron afectados, sino también de:

- **Cuánto tiempo se tardó en responder** al incidente (cuantas más horas pasen sin contenerlo, más se suele encarecer).
- **Si se filtraron datos sensibles** o no (un incidente con fuga de datos personales suele implicar costes legales y de notificación mucho mayores).

Cuando un modelo tiene que aprender a partir de **varias variables de entrada a la vez** en lugar de una sola, ya no hablamos de regresión lineal "simple", sino de **regresión lineal múltiple**. La idea de fondo es la misma (buscar la combinación que mejor explica los datos), pero en vez de ajustar una recta en un plano de 2 dimensiones, el modelo ajusta un "plano" (o hiperplano, si hay muchas variables) en un espacio con más dimensiones, una por cada variable de entrada más una para el resultado.

> **Si sé cuántos equipos se vieron afectados, cuántas horas se tardó en responder y si hubo fuga de datos sensibles, ¿puedo estimar cuánto va a costar el incidente?**

### ¿Por qué esta variante es más compleja (y por qué sigue mereciendo la pena)?

- **Hay más de una "palanca" que mover.** El modelo ya no busca un único incremento (theta 1), sino uno por cada variable: cuánto sube el coste por cada equipo afectado, cuánto sube por cada hora de retraso en la respuesta, y cuánto sube (de golpe) si hay fuga de datos.
- **Ya no se puede dibujar en una gráfica sencilla de 2 ejes.** Con una sola variable de entrada, la relación se veía como una línea en un plano. Con tres variables de entrada, haría falta un espacio de 4 dimensiones para dibujarlo todo junto, algo que no se puede representar directamente en una pantalla. Por eso, en este caso, se renuncia a la representación gráfica de "todo a la vez" y se analizan los coeficientes de forma numérica, o se representan las variables de dos en dos.
- **Las variables no siempre son números "naturales".** `datos_sensibles_filtrados` no es una cantidad (como los equipos afectados), sino una condición de sí/no. Para que el modelo pueda usarla, se representa como `0` (no hubo fuga) o `1` (sí hubo fuga). Es una técnica muy habitual llamada **variable binaria (o dummy)**.
- **Sigue siendo interpretable**, que es la gran ventaja de la regresión lineal frente a modelos más complejos: al final se obtiene una fórmula con un número por cada variable, y esos números se pueden explicar en una frase, como se verá más abajo.

### 1. Generación (simulada) de los datos

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

n_muestras = 200

equipos_afectados = np.random.randint(1, 500, n_muestras)
horas_respuesta = np.random.randint(1, 72, n_muestras)
datos_sensibles_filtrados = np.random.randint(0, 2, n_muestras)  # 0 = no, 1 = sí

ruido = np.random.randn(n_muestras) * 3000

coste = (
    5000
    + 80 * equipos_afectados
    + 600 * horas_respuesta
    + 40000 * datos_sensibles_filtrados
    + ruido
)
```

Como antes, no se dispone de un histórico real de incidentes, así que se simulan **200 incidentes** con una relación conocida de antemano, para poder comprobar después que el modelo es capaz de "descubrirla":

- `equipos_afectados`: entre 1 y 500 equipos.
- `horas_respuesta`: entre 1 y 72 horas hasta que se contuvo el incidente.
- `datos_sensibles_filtrados`: 0 si no hubo fuga de datos, 1 si la hubo.
- `coste`: se calcula a propósito como *"un coste fijo (5.000) + 80 € por cada equipo afectado + 600 € por cada hora de retraso en la respuesta + 40.000 € extra si hubo fuga de datos sensibles"*, más algo de ruido aleatorio para que no sea una relación perfecta.

### 2. Construcción de la tabla de datos

```python
df = pd.DataFrame({
    'equipos_afectados': equipos_afectados,
    'horas_respuesta': horas_respuesta,
    'datos_sensibles_filtrados': datos_sensibles_filtrados,
    'coste': coste.astype('int')
})

df.head()
```

Igual que antes, se organiza todo en una tabla (`DataFrame`), pero ahora con **tres columnas de entrada** en lugar de una, más la columna con el resultado (`coste`) que se quiere predecir.

| equipos_afectados | horas_respuesta | datos_sensibles_filtrados | coste |
|---|---|---|---|
| 312 | 40 | 1 | 90 340 |
| 47 | 5 | 0 | 11 870 |
| 198 | 60 | 0 | 57 210 |
| ... | ... | ... | ... |

(Los valores son solo un ejemplo ilustrativo; al ser datos aleatorios, cada ejecución del notebook generará números distintos.)

### 3. Entrenamiento del modelo

```python
X = df[['equipos_afectados', 'horas_respuesta', 'datos_sensibles_filtrados']].values
y = df['coste'].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.intercept_   # theta 0: coste base
lin_reg.coef_        # theta 1, theta 2, theta 3: un coeficiente por cada variable
```

La diferencia clave respecto a los ejemplos anteriores está aquí: en vez de pasarle al modelo **una sola columna** como entrada, se le pasan **las tres columnas a la vez** (`X` ahora es una tabla de 200 filas × 3 columnas, no una única lista de números). El resto del proceso es idéntico: `.fit()` sigue siendo el paso de entrenamiento, solo que ahora el modelo tiene que encontrar cuatro números en lugar de dos:

- `intercept_` (theta 0): el coste base, cuando todas las variables valen 0 (0 equipos, 0 horas, sin fuga de datos).
- `coef_` ahora es una **lista de tres números**, uno por cada columna de entrada, en el mismo orden en que se pasaron:
  - theta 1 → cuánto sube el coste por cada equipo afectado adicional.
  - theta 2 → cuánto sube el coste por cada hora adicional de retraso en la respuesta.
  - theta 3 → cuánto sube el coste, de golpe, si hay fuga de datos sensibles (al ser una variable de 0 o 1, este coeficiente representa directamente "el extra que cuesta que el incidente incluya una fuga").

Con estos cuatro números, la fórmula completa queda así (en palabras):

> **Coste estimado = coste base + (coste por equipo × equipos afectados) + (coste por hora × horas de respuesta) + (coste extra × fuga de datos)**

### 4. Predicción de un incidente nuevo

```python
incidente_nuevo = np.array([[250, 30, 1]])  # 250 equipos, 30 horas de respuesta, con fuga de datos
coste_estimado = lin_reg.predict(incidente_nuevo)

print("El coste del incidente sería:", int(coste_estimado[0]), "€")
```

Aquí se ve la utilidad práctica de haber añadido más variables: ahora se puede describir un incidente de forma mucho más realista (no solo "cuántos equipos", sino también "cuánto se tardó en reaccionar" y "si hubo fuga de datos") y obtener una estimación de coste que tiene en cuenta los tres factores a la vez, en lugar de fijarse solo en uno.

Si se comparan dos incidentes con los mismos equipos afectados y el mismo tiempo de respuesta, pero uno con fuga de datos y otro sin ella, el modelo mostrará que el primero sale bastante más caro, exactamente por el peso que aprendió para la variable `datos_sensibles_filtrados`.

### 5. Cómo comparar el peso de cada variable

```python
for nombre, coeficiente in zip(['equipos_afectados', 'horas_respuesta', 'datos_sensibles_filtrados'], lin_reg.coef_):
    print(f"Cada unidad de '{nombre}' añade aproximadamente {coeficiente:.2f} € al coste")
```

Una ventaja de que el modelo siga siendo una regresión lineal (y no, por ejemplo, una red neuronal) es que se puede leer directamente, variable por variable, cuánto "pesa" cada una en el resultado final. Esto permite responder a preguntas de negocio muy concretas, como: *"¿nos conviene más invertir en reducir el tiempo de respuesta o en tener menos equipos expuestos?"*, simplemente comparando los coeficientes obtenidos.

## Resumen de esta segunda parte

1. Cuando un resultado depende de **varias causas a la vez** (no solo una), se usa **regresión lineal múltiple** en lugar de la simple.
2. El procedimiento es una extensión natural del caso simple: en vez de pasarle al modelo una columna de entrada, se le pasan varias a la vez.
3. El modelo aprende **un coeficiente por cada variable de entrada**, además del coste base, en lugar de un único coeficiente.
4. Las variables que no son numéricas por naturaleza (como "¿hubo fuga de datos, sí o no?") se pueden incorporar convirtiéndolas en `0` y `1`.
5. Se pierde la posibilidad de visualizar todo en una única gráfica sencilla, pero se gana en realismo: el modelo puede tener en cuenta varios factores relevantes a la vez, y sigue siendo tan interpretable como el caso simple, ya que cada coeficiente se puede explicar en una frase.

## Un ejemplo de complejidad alta: precio de venta de una vivienda

Los ejemplos anteriores usaban variables que ya venían "listas para usar": números que se podían meter directamente en el modelo. En la vida real, muchos problemas tienen un ingrediente adicional que los complica bastante: **mezclan variables numéricas con variables categóricas** (cosas que no son un número, sino una etiqueta, como el nombre de un barrio o una letra de certificado energético), y encima hay que **conseguir los datos reales**, no simularlos.

Vamos a ver un caso muy habitual en el sector inmobiliario, pero llevado a un nivel más realista que el ejemplo del alquiler visto antes: **estimar el precio de venta de una vivienda** a partir de sus características.

> **Si conozco los metros cuadrados, las habitaciones, el barrio, la antigüedad y el certificado energético de un piso, ¿puedo estimar por cuánto se vendería?**

### ¿Por qué este ejemplo es más complejo que los anteriores?

- **Hay muchas más variables a la vez** (siete u ocho, en vez de una o tres), y no todas influyen igual en el precio.
- **Aparecen variables categóricas con más de dos valores posibles.** Antes, "¿hubo fuga de datos?" solo podía ser sí/no (0 o 1). Ahora, el "barrio" puede ser Centro, Ensanche, Periferia... (más de dos opciones) y el "certificado energético" puede ser A, B, C, D, E, F o G. Estas variables no se pueden convertir en un único número 0/1: hace falta una técnica distinta, llamada **one-hot encoding**, que se explica más abajo.
- **Las variables pueden estar relacionadas entre sí** (por ejemplo, los pisos más grandes suelen tener también más habitaciones), lo que hace que interpretar cada coeficiente por separado sea algo más delicado que en los ejemplos anteriores.
- **Los datos reales no vienen limpios.** A diferencia de los ejemplos anteriores, donde los datos se generaban a propósito sin errores, un dataset real de viviendas suele traer huecos (pisos sin dato de antigüedad), errores de escritura, valores extremos poco creíbles (un piso de 12 m² por 900.000 €) y formatos inconsistentes, todo lo cual hay que revisar antes de entrenar el modelo.

### 1. Las variables del dataset

| Variable | Tipo | Ejemplo |
|---|---|---|
| `metros_cuadrados` | Numérica | 85 |
| `habitaciones` | Numérica | 3 |
| `banos` | Numérica | 2 |
| `planta` | Numérica | 4 |
| `ascensor` | Categórica binaria (0/1) | 1 (sí tiene) |
| `antiguedad_anios` | Numérica | 15 |
| `certificado_energetico` | Categórica (A–G) | "C" |
| `barrio` | Categórica (varios valores) | "Centro" |
| `distancia_centro_km` | Numérica | 2.3 |
| `precio` (lo que se quiere predecir) | Numérica | 245 000 |

### 2. Generación (simulada) de los datos, para practicar el procedimiento

```python
import numpy as np
import pandas as pd

n_muestras = 500
barrios = ['Centro', 'Ensanche', 'Periferia', 'Zona Norte']
certificados = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

df = pd.DataFrame({
    'metros_cuadrados': np.random.randint(40, 200, n_muestras),
    'habitaciones': np.random.randint(1, 5, n_muestras),
    'banos': np.random.randint(1, 3, n_muestras),
    'planta': np.random.randint(0, 10, n_muestras),
    'ascensor': np.random.randint(0, 2, n_muestras),
    'antiguedad_anios': np.random.randint(0, 60, n_muestras),
    'certificado_energetico': np.random.choice(certificados, n_muestras),
    'barrio': np.random.choice(barrios, n_muestras),
    'distancia_centro_km': np.round(np.random.uniform(0.5, 15, n_muestras), 1),
})

# Precio "de verdad" fabricado a propósito, para comprobar que el modelo lo aprende
ajuste_barrio = df['barrio'].map({'Centro': 60000, 'Ensanche': 30000, 'Periferia': 0, 'Zona Norte': 15000})
ajuste_certificado = df['certificado_energetico'].map(
    {'A': 15000, 'B': 10000, 'C': 5000, 'D': 0, 'E': -5000, 'F': -10000, 'G': -15000}
)

ruido = np.random.randn(n_muestras) * 8000

df['precio'] = (
    50000
    + 1800 * df['metros_cuadrados']
    + 5000 * df['habitaciones']
    + 3000 * df['banos']
    - 1200 * df['antiguedad_anios']
    + 10000 * df['ascensor']
    - 2000 * df['distancia_centro_km']
    + ajuste_barrio
    + ajuste_certificado
    + ruido
).astype('int')

df.head()
```

Como en los ejemplos anteriores, se fabrica el precio "a propósito" combinando todas las variables con unos pesos conocidos, para poder comprobar después que el modelo es capaz de recuperarlos a partir de los datos. En un caso real, esta columna `precio` sería el dato observado (lo que realmente pagó el comprador), no algo calculado con una fórmula.

### 3. Codificación de las variables categóricas (one-hot encoding)

```python
df_codificado = pd.get_dummies(df, columns=['certificado_energetico', 'barrio'], drop_first=True)
```

Un modelo de regresión lineal solo entiende números, así que no se le puede pasar directamente la palabra `"Centro"` o la letra `"C"`. La técnica de **one-hot encoding** resuelve esto convirtiendo cada categoría en varias columnas nuevas de 0 y 1, una por cada valor posible (menos uno, que sirve de referencia):

```
Antes:                    Después (one-hot):
barrio                    barrio_Ensanche  barrio_Periferia  barrio_Zona_Norte
Centro          →         0                0                 0
Ensanche        →         1                0                 0
Periferia       →         0                1                 0
```

Cada fila queda descrita por ceros y unos: un piso en "Centro" tiene todas esas columnas a 0 (es la categoría de referencia, implícita), y un piso en "Ensanche" tiene un 1 solo en la columna `barrio_Ensanche`. El parámetro `drop_first=True` es lo que evita crear una columna redundante para la primera categoría, ya que "no ser ninguna de las demás" ya identifica esa categoría.

Con esto, cada barrio y cada letra de certificado energético pasan a tener su propio coeficiente en el modelo, igual que ocurría con `datos_sensibles_filtrados` en el ejemplo anterior, pero ahora con más de dos posibilidades.

### 4. Entrenamiento del modelo

```python
from sklearn.linear_model import LinearRegression

columnas_entrada = [c for c in df_codificado.columns if c != 'precio']
X = df_codificado[columnas_entrada].values
y = df_codificado['precio'].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

for nombre, coeficiente in zip(columnas_entrada, lin_reg.coef_):
    print(f"{nombre}: {coeficiente:.0f} €")
```

El procedimiento es exactamente el mismo que en el ejemplo con tres variables: se le pasan todas las columnas de entrada a la vez y `.fit()` calcula un coeficiente por cada una. La única diferencia real es que ahora hay más columnas (porque las categorías se han "desdoblado" en varias columnas 0/1), pero el algoritmo no distingue entre una columna numérica normal y una columna que viene de un one-hot encoding: para él, todo son números.

### 5. Predicción de una vivienda nueva

```python
piso_nuevo = pd.DataFrame([{
    'metros_cuadrados': 95,
    'habitaciones': 3,
    'banos': 2,
    'planta': 3,
    'ascensor': 1,
    'antiguedad_anios': 10,
    'distancia_centro_km': 1.8,
    'certificado_energetico': 'B',
    'barrio': 'Ensanche',
}])

piso_nuevo_codificado = pd.get_dummies(piso_nuevo, columns=['certificado_energetico', 'barrio'], drop_first=True)
piso_nuevo_codificado = piso_nuevo_codificado.reindex(columns=columnas_entrada, fill_value=0)

precio_estimado = lin_reg.predict(piso_nuevo_codificado.values)
print("Precio estimado:", int(precio_estimado[0]), "€")
```

Aquí aparece un detalle importante que no salía en los ejemplos anteriores: el piso nuevo hay que codificarlo **exactamente con las mismas columnas** que se usaron para entrenar (por eso se usa `.reindex(columns=columnas_entrada, fill_value=0)`). Si el piso nuevo es del barrio "Centro", por ejemplo, todas las columnas de barrio deben quedar a 0, igual que en los datos de entrenamiento.

### 6. Cómo se conseguirían los datos reales para este problema

En todos los ejemplos anteriores los datos se generaban con `np.random`. En un caso real, montar este dataset implicaría reunir información de fuentes externas y limpiarla. Estas son las vías más habituales:

**a) Portales inmobiliarios (web scraping)**

Sitios como Idealista, Fotocasa o pisos.com publican miles de anuncios con casi todas estas variables (metros, habitaciones, planta, certificado energético, barrio, precio). Se pueden extraer con herramientas como `BeautifulSoup`, `Scrapy` o `Selenium` (para páginas que cargan contenido dinámicamente). Aspectos a tener en cuenta:
- Revisar el archivo `robots.txt` y los términos de servicio del portal antes de scrapear: muchos limitan o prohíben la extracción automatizada, y algunos ofrecen una **API oficial de pago** como alternativa legal (por ejemplo, Idealista tiene una API para desarrolladores/empresas).
- El scraping da datos de **precio de oferta** (lo que se pide), no necesariamente el precio final de venta, que suele ser algo menor.

**b) Fuentes públicas y datos abiertos (gratuitas y sin problemas legales)**

- **Sede Electrónica del Catastro**: datos oficiales de superficie construida, año de construcción y uso del inmueble, consultables por referencia catastral.
- **Instituto Nacional de Estadística (INE)** y los **Registradores de la Propiedad**: publican índices de precios de vivienda y estadísticas de compraventas, útiles para contextualizar o calibrar el modelo (no dan el precio de un piso concreto, pero sí la tendencia media por zona).
- **Portales de datos abiertos** (datos.gob.es, y los portales open data de ayuntamientos como Madrid o Barcelona): a veces publican precios medios por barrio, transacciones inmobiliarias agregadas, o el callejero y equipamientos (colegios, metro, parques), útiles para calcular variables como `distancia_centro_km`.

**c) Datasets ya preparados por terceros**

Plataformas como Kaggle tienen datasets de vivienda ya limpios y listos para usar (algunos de España, muchos de otros países como el clásico "House Prices" de EE.UU.). Son ideales para practicar el modelado sin tener que resolver primero el problema de la recogida de datos, aunque no sirven si lo que se necesita es un modelo ajustado a un mercado local muy concreto.

**d) Proveedores de datos especializados (de pago)**

Empresas como Tinsa, idealista Data o Fotocasa ofrecen informes y datasets con precios de venta contrastados (no solo de oferta), pensados para tasadoras, bancos e inmobiliarias. Es la opción más fiable cuando el modelo se va a usar en un contexto profesional real, aunque tiene coste económico.

**e) Enriquecer los datos con información geográfica**

Variables como `distancia_centro_km` normalmente no vienen dadas: hay que calcularlas. Para eso se usan APIs de geocodificación como **OpenStreetMap Nominatim** (gratuita) o **Google Maps Geocoding API** (de pago a partir de cierto volumen), que convierten una dirección en coordenadas y permiten calcular distancias a puntos de interés (centro de la ciudad, estaciones de metro, colegios...).

**f) Limpieza de los datos reales antes de entrenar**

A diferencia de los datos simulados, un dataset real necesitaría un trabajo previo de limpieza:
- Rellenar o descartar filas con datos faltantes (pisos sin certificado energético informado, por ejemplo).
- Detectar y revisar valores atípicos poco creíbles (errores de tecleo como un piso de "1000 m²" que en realidad eran "100").
- Unificar formatos (que "Centro" y "centro" no se traten como dos barrios distintos, que los precios no mezclen textos como "245.000 €" con números puros).

### Resumen de este ejemplo

1. Los problemas reales suelen mezclar **variables numéricas y categóricas**, y estas últimas necesitan una codificación previa (**one-hot encoding**) antes de poder entrenar el modelo.
2. El entrenamiento y la predicción siguen el mismo patrón que en los ejemplos anteriores: la diferencia está en la preparación de los datos, no en el algoritmo en sí.
3. Conseguir un dataset real implica normalmente **combinar varias fuentes**: portales con datos de oferta (scraping o API), fuentes públicas oficiales (Catastro, INE, datos abiertos) para contexto y variables auxiliares, y opcionalmente proveedores de pago si se necesita precisión profesional.
4. Antes de entrenar cualquier modelo con datos reales, hace falta un trabajo de **limpieza y validación** que no existe cuando los datos están simulados, y que en la práctica suele llevar más tiempo que el propio entrenamiento del modelo.
