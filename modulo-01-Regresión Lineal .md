# Curso: Regresión Lineal

*Curso estructurado a partir de "4_Regresión Lineal - Explicación.md" y de los ejemplos del notebook "4_Regresión Lineal - Predicción del coste de un incidente de seguridad". Pensado para alguien sin conocimientos previos de programación ni de Machine Learning.*

## Índice

1. [Introducción: ¿qué problema resuelve la regresión lineal?](#1-introducción-qué-problema-resuelve-la-regresión-lineal)
2. [Módulo 1 — La intuición: una recta que resume una nube de puntos](#módulo-1--la-intuición-una-recta-que-resume-una-nube-de-puntos)
3. [Módulo 2 — Las matemáticas detrás de la recta](#módulo-2--las-matemáticas-detrás-de-la-recta)
4. [Módulo 3 — De la teoría a la práctica: coste de un incidente de seguridad](#módulo-3--de-la-teoría-a-la-práctica-coste-de-un-incidente-de-seguridad)
5. [Módulo 4 — Repaso del patrón: precio de alquiler de un piso](#módulo-4--repaso-del-patrón-precio-de-alquiler-de-un-piso)
6. [Módulo 5 — Regresión lineal múltiple](#módulo-5--regresión-lineal-múltiple)
7. [Módulo 6 — Variables categóricas y one-hot encoding](#módulo-6--variables-categóricas-y-one-hot-encoding)
8. [Módulo 7 — Cómo evaluar si el modelo es bueno](#módulo-7--cómo-evaluar-si-el-modelo-es-bueno)
9. [Módulo 8 — De los datos simulados a los datos reales](#módulo-8--de-los-datos-simulados-a-los-datos-reales)
10. [Módulo 9 — Límites de la regresión lineal: cuándo no usarla](#módulo-9--límites-de-la-regresión-lineal-cuándo-no-usarla)
11. [Resumen final y checklist de repaso](#resumen-final-y-checklist-de-repaso)
12. [Glosario](#glosario)

---

## 1. Introducción: ¿qué problema resuelve la regresión lineal?

En Machine Learning, uno de los problemas más comunes es este: **tengo datos del pasado y quiero predecir un número para un caso nuevo.**

Ejemplos de esa misma pregunta, con distinto disfraz:

- *"Si sé cuántos equipos se han visto afectados por un incidente, ¿puedo estimar cuánto me va a costar arreglarlo?"*
- *"Si sé cuántos metros cuadrados tiene un piso, ¿puedo estimar cuánto costará alquilarlo?"*
- *"Si conozco los metros, las habitaciones, el barrio y el certificado energético de una vivienda, ¿puedo estimar por cuánto se vendería?"*

Todas estas preguntas comparten la misma estructura: **a partir de una o varias variables de entrada, quiero predecir una variable numérica de salida.** Este tipo de problema se llama **regresión** (se diferencia de la *clasificación*, donde lo que se predice no es un número sino una categoría, como "spam / no spam").

La **regresión lineal** es el algoritmo más simple para resolver un problema de regresión, y por eso es el punto de partida obligado de cualquier curso de Machine Learning. Este curso usa como hilo conductor los tres ejemplos ya trabajados en el notebook de referencia:

| Ejemplo | Entrada(s) | Salida a predecir |
|---|---|---|
| Incidente de seguridad | Equipos afectados | Coste del incidente |
| Alquiler de un piso | Metros cuadrados | Precio de alquiler mensual |
| Incidente de seguridad (versión múltiple) | Equipos afectados, horas de respuesta, fuga de datos | Coste del incidente |
| Venta de una vivienda | Metros, habitaciones, baños, planta, ascensor, antigüedad, certificado energético, barrio, distancia al centro | Precio de venta |

Cada módulo de este curso retoma uno de estos ejemplos para ilustrar un concepto nuevo, siempre siguiendo el mismo patrón de trabajo.

---

## Módulo 1 — La intuición: una recta que resume una nube de puntos

**Objetivo del módulo:** entender qué hace la regresión lineal sin usar ni una sola fórmula.

Imagina una gráfica con dos ejes:

- Eje horizontal: número de equipos afectados en un incidente.
- Eje vertical: coste de ese incidente.

Si colocamos en esa gráfica todos los incidentes que ha tenido una empresa en el pasado, aparece una nube de puntos. La regresión lineal busca **la línea recta que mejor atraviesa esa nube de puntos**, es decir, la que se queda "en medio" de todos ellos con el menor error posible.

Con esa recta ya calculada se pueden hacer dos cosas:

1. **Entender la tendencia.** Por ejemplo, saber que "cada equipo afectado adicional añade aproximadamente tantos euros de coste".
2. **Predecir casos nuevos.** Dado un número de equipos afectados que aún no se ha visto, calcular en qué punto de la recta caería y así estimar su coste.

Es la misma idea que la intuición de "cuantos más equipos se ven afectados, más caro sale el incidente, y sube de forma más o menos constante". La regresión lineal convierte esa intuición en algo preciso y calculable.

### ¿Por qué empezar por este algoritmo y no por otro?

- **Es el punto de partida natural.** Es el algoritmo más simple que existe para predecir un número a partir de otro número.
- **Es fácil de interpretar.** El resultado es literalmente una recta con una fórmula del tipo "coste = un valor base + un incremento por cada equipo afectado". No hace falta ser experto para entenderla.
- **Encaja cuando la relación es aproximadamente lineal.** Si al dibujar los datos reales se ve que, a grandes rasgos, el coste sube de forma bastante constante según suben los equipos afectados (sin saltos raros ni curvas complicadas), una recta es una aproximación razonable. Si la relación fuera muy distinta (por ejemplo, un coste que se dispara exponencialmente), haría falta otro tipo de modelo (ver [Módulo 9](#módulo-9--límites-de-la-regresión-lineal-cuándo-no-usarla)).
- **Es rápido y barato de entrenar.** No necesita muchos datos ni mucha potencia de cálculo, a diferencia de modelos más complejos (como redes neuronales), que aquí serían innecesarios.

### Checkpoint

> Antes de seguir, asegúrate de poder responder: ¿qué representa cada eje en la gráfica de "equipos afectados vs. coste"? ¿Qué busca exactamente la regresión lineal en esa nube de puntos?

---

## Módulo 2 — Las matemáticas detrás de la recta

**Objetivo del módulo:** poner nombre y símbolo a los dos números que definen la recta, y entender —a alto nivel— cómo se calculan.

Toda línea recta en un plano se puede describir con dos números:

> **h(x) = θ₀ + θ₁ · x**

Donde:

- **x** es la variable de entrada (por ejemplo, equipos afectados).
- **h(x)** ("hipótesis") es el valor que predice el modelo (por ejemplo, el coste estimado).
- **θ₀ (theta 0)** es el **intercepto**: el valor de salida cuando x = 0. En el ejemplo del incidente, sería el "coste fijo" de gestionar un incidente aunque no afecte a ningún equipo (tiempo del equipo de respuesta, investigación, etc.).
- **θ₁ (theta 1)** es la **pendiente**: cuánto sube la salida por cada unidad que sube x. En el ejemplo, cuánto se encarece el incidente por cada equipo afectado adicional.

En scikit-learn, estos dos números se obtienen así después de entrenar el modelo:

```python
lin_reg.intercept_   # theta 0
lin_reg.coef_        # theta 1
```

### ¿Cómo se calculan θ₀ y θ₁?

El modelo no se inventa la recta: la calcula minimizando el **error** entre lo que predice y lo que realmente ocurrió en los datos de entrenamiento. Ese error se mide con una **función de coste**, y la más habitual en regresión lineal es el **error cuadrático medio** (a menudo llamado *MSE*, de *Mean Squared Error*):

> Para cada punto de los datos, se calcula la diferencia entre el valor real y el valor que predice la recta, se eleva al cuadrado (para que los errores positivos y negativos no se cancelen entre sí, y para penalizar más los errores grandes) y se promedia entre todos los puntos.

El algoritmo busca los valores de θ₀ y θ₁ que hacen que ese error promedio sea **lo más pequeño posible**. Existen dos formas habituales de encontrar esos valores:

- **La ecuación normal**: una fórmula matemática cerrada que calcula θ₀ y θ₁ directamente, resolviendo el problema de golpe. Es la que usa `LinearRegression` de scikit-learn quantiando el dataset no es enorme.
- **El descenso de gradiente**: un método iterativo que va ajustando θ₀ y θ₁ poco a poco, dando pequeños pasos en la dirección que reduce el error, hasta converger a la mejor solución. Es más habitual en modelos más complejos o con datasets muy grandes.

Para este curso no hace falta programar ninguna de las dos a mano: `scikit-learn` ya las tiene implementadas dentro de `LinearRegression()`. Lo importante es entender la idea de fondo: **"mejor recta" significa "la recta que minimiza el error cuadrático medio entre las predicciones y los datos reales"**.

### Checkpoint

> ¿Qué significa que θ₀ valga, por ejemplo, 38.267? ¿Y que θ₁ valga 30,35? (Pista: relee la tabla del Módulo 3 más abajo, donde aparecen estos valores reales del ejemplo).

---

## Módulo 3 — De la teoría a la práctica: coste de un incidente de seguridad

**Objetivo del módulo:** recorrer, paso a paso, el ejemplo completo de regresión lineal simple, con código real y resultados reales del notebook de referencia.

### 3.0 Herramientas necesarias

```python
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install scikit-learn
```

| Librería | Para qué se usa aquí |
|---|---|
| **numpy** | Trabajar con números y matrices de forma eficiente. |
| **pandas** | Organizar los datos en tablas parecidas a una hoja de Excel. |
| **matplotlib** | Dibujar gráficas. |
| **scikit-learn** | Contiene el algoritmo de regresión lineal ya implementado. |

### 3.1 Generación del conjunto de datos

```python
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```

Como no se dispone de datos reales de incidentes, se **simulan 100 incidentes de ejemplo**, pero fabricados a propósito con una relación lineal conocida:

- `X`: número de equipos afectados (a escala reducida).
- `y`: coste del incidente, calculado como *"un valor base (4) + 3 veces el número de equipos afectados"*, más algo de ruido aleatorio (`np.random.randn`) para que no sea una recta perfecta y se parezca a datos reales.

Fabricar el ejemplo así permite comprobar después que el algoritmo es capaz de "descubrir" esa relación lineal por sí solo, solo a partir de los datos.

### 3.2 Visualización del conjunto de datos

```python
plt.plot(X, y, "b.")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Coste del incidente (u/10000)")
plt.show()
```

Cada punto azul es un incidente simulado. A simple vista ya se aprecia la tendencia: a más equipos afectados, más coste, siguiendo aproximadamente una línea recta.

### 3.3 Ajuste de escala de los datos

```python
data = {'n_equipos_afectados': X.flatten(), 'coste': y.flatten()}
df = pd.DataFrame(data)
```

`X` e `y` son matrices de forma `(100, 1)` (100 filas, 1 columna). `.flatten()` las "aplasta" a una lista de una sola dimensión, sin cambiar ningún valor, solo la forma:

```
Antes (100x1):        Después de .flatten() (100,):
[[0.37],               [0.37, 0.95, 0.73, ...]
 [0.95],
 [0.73], ...]
```

```python
df['n_equipos_afectados'] = df['n_equipos_afectados'] * 1000
df['n_equipos_afectados'] = df['n_equipos_afectados'].astype('int')
df['coste'] = df['coste'] * 10000
df['coste'] = df['coste'].astype('int')
```

Los números generados eran decimales muy pequeños. Se multiplican y convierten a enteros solo para que el ejercicio "se lea" como una situación real (equipos afectados en cientos, costes en decenas de miles de euros), sin cambiar la relación de fondo entre las dos variables.

### 3.4 Entrenamiento del modelo

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)
```

- `LinearRegression()` crea un modelo "en blanco", todavía sin entrenar.
- `.fit(...)` es el paso de **entrenamiento**: el algoritmo mira todos los puntos (equipos afectados → coste) y calcula la recta que minimiza el error total, tal como se explicó en el Módulo 2.
- `.reshape(-1, 1)` es un requisito de formato de scikit-learn: la entrada debe tener forma de tabla (columna), aunque solo haya una columna. No cambia el contenido de los datos.

Resultado real de esta ejecución:

```python
lin_reg.intercept_   # 38267.707392012584   (theta 0)
lin_reg.coef_        # [30.35465234]        (theta 1)
```

En palabras:

> **Coste estimado ≈ 38.268 € + 30,35 € × número de equipos afectados**

### 3.5 Dibujar la recta aprendida

Para dibujar una recta basta con dos puntos. Se usan el mínimo y el máximo de `n_equipos_afectados` en los datos:

```python
X_min_max = np.array([[df["n_equipos_afectados"].min()], [df["n_equipos_afectados"].max()]])
y_train_pred = lin_reg.predict(X_min_max)
# X_min_max      -> [[2], [1964]]
# y_train_pred   -> [38328.42, 97884.24]
```

```python
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()
```

- Puntos azules: incidentes reales usados para entrenar.
- Línea verde: recta aprendida por el modelo.

Si la línea verde pasa "por en medio" de los puntos azules, el modelo ha aprendido bien la tendencia general.

### 3.6 Predicción de un incidente nuevo

```python
x_new = np.array([[1800]])  # equipos afectados
coste = lin_reg.predict(x_new)
print("El coste del incidente sería:", int(coste[0]), "€")
# -> El coste del incidente sería: 92906 €
```

Este es el objetivo final: usar el modelo ya entrenado para un caso nuevo, que no estaba en los datos originales. Se le pregunta "¿cuánto costaría un incidente con 1.800 equipos afectados?" y el modelo responde según la recta que aprendió.

```python
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(x_new, coste, "rx")
plt.show()
```

- Puntos azules: incidentes históricos.
- Línea verde: recta aprendida.
- Cruz roja: la nueva predicción, sobre la recta, en el punto correspondiente a 1.800 equipos.

### Checkpoint

> Con `intercept_ = 38267.7` y `coef_ = 30.35`, calcula a mano el coste estimado para un incidente con **500** equipos afectados. Después compáralo con lo que daría `lin_reg.predict([[500]])`.

---

## Módulo 4 — Repaso del patrón: precio de alquiler de un piso

**Objetivo del módulo:** confirmar que el procedimiento es siempre el mismo, cambiando solo qué representan `X` e `y`.

> **Pregunta de negocio:** *si sé cuántos metros cuadrados tiene un piso, ¿puedo estimar cuánto costará alquilarlo?*

Una inmobiliaria tiene el histórico de pisos alquilados, con sus metros cuadrados y el precio al que finalmente se alquilaron, y quiere estimar el precio de un piso nuevo en cuanto sepa sus metros.

### 4.1 Datos (simulados)

```python
X = 2 * np.random.rand(100, 1)
y = 300 + 8 * X + np.random.randn(100, 1)
```

- `X`: metros cuadrados (a escala reducida).
- `y`: precio mensual de alquiler, como *"un precio base (300) + 8 veces los metros cuadrados"*, con ruido aleatorio (zona, estado, planta... hacen que dos pisos con los mismos metros no se alquilen exactamente por el mismo precio).

### 4.2 Tabla y reescalado

```python
data = {'metros_cuadrados': X.flatten(), 'precio_alquiler': y.flatten()}
df = pd.DataFrame(data)

df['metros_cuadrados'] = (df['metros_cuadrados'] * 50 + 30).astype('int')
df['precio_alquiler'] = df['precio_alquiler'].astype('int')
```

Resultado: pisos de entre 30 y 130 m² aproximadamente, con precios de alquiler en euros.

### 4.3 Entrenamiento

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(df['metros_cuadrados'].values.reshape(-1, 1), df['precio_alquiler'].values)

lin_reg.intercept_   # precio base, para 0 m² (theta 0)
lin_reg.coef_        # cuánto sube el precio por cada m² adicional (theta 1)
```

> **Precio estimado = precio base + (precio por m² × metros cuadrados del piso)**

### 4.4 Predicción de un piso nuevo

```python
x_new = np.array([[90]])  # piso de 90 m²
precio = lin_reg.predict(x_new)
print("El precio de alquiler estimado sería:", int(precio[0]), "€/mes")
```

### Idea clave de este módulo

**El procedimiento es siempre el mismo**, cambie lo que cambie el problema de negocio (coste de incidentes, precio de alquiler, o cualquier otra magnitud que dependa de una variable de forma aproximadamente lineal). Lo único que varía es qué representan `X` e `y` en cada caso: los cinco pasos —generar/cargar datos, visualizar, ajustar escala, entrenar con `.fit()`, predecir con `.predict()`— no cambian.

---

## Módulo 5 — Regresión lineal múltiple

**Objetivo del módulo:** pasar de una sola variable de entrada a varias a la vez, y entender por qué hace falta.

En los módulos anteriores, el coste (o el precio) dependía de **una sola cosa**. Pero en la vida real, casi ningún resultado depende de una única causa. El coste de un incidente de seguridad probablemente no depende solo de los equipos afectados, sino también de:

- **Cuánto tiempo se tardó en responder** (más horas sin contener el incidente, más se encarece).
- **Si se filtraron datos sensibles** (una fuga de datos personales implica costes legales y de notificación mucho mayores).

> **Pregunta de negocio:** *si sé cuántos equipos se vieron afectados, cuántas horas se tardó en responder y si hubo fuga de datos sensibles, ¿puedo estimar cuánto va a costar el incidente?*

Cuando el modelo aprende a partir de **varias variables de entrada a la vez**, ya no hablamos de regresión lineal "simple" sino de **regresión lineal múltiple**. La idea de fondo es la misma —buscar la combinación que mejor explica los datos—, pero en vez de ajustar una recta en un plano de 2 dimensiones, el modelo ajusta un "plano" (o hiperplano, con muchas variables) en un espacio con más dimensiones.

### ¿Qué cambia respecto al caso simple?

- **Hay más de una "palanca" que mover.** En vez de un único incremento (θ₁), el modelo busca uno por cada variable.
- **Ya no se puede dibujar en una gráfica sencilla de 2 ejes.** Con 3 variables de entrada haría falta un espacio de 4 dimensiones. Por eso, aquí se renuncia a la representación gráfica de "todo a la vez" y se analizan los coeficientes numéricamente (o se representan las variables de dos en dos).
- **Las variables no siempre son números "naturales".** `datos_sensibles_filtrados` no es una cantidad, sino una condición de sí/no. Se representa como `0` (no hubo fuga) o `1` (sí hubo fuga): una **variable binaria (o dummy)**.
- **Sigue siendo interpretable**: al final se obtiene una fórmula con un número por cada variable, explicable en una frase.

### 5.1 Datos (simulados)

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

Se simulan **200 incidentes** con una relación conocida de antemano: *"un coste fijo (5.000) + 80 € por cada equipo afectado + 600 € por cada hora de retraso en la respuesta + 40.000 € extra si hubo fuga de datos sensibles"*, más ruido aleatorio.

### 5.2 Tabla de datos

```python
df = pd.DataFrame({
    'equipos_afectados': equipos_afectados,
    'horas_respuesta': horas_respuesta,
    'datos_sensibles_filtrados': datos_sensibles_filtrados,
    'coste': coste.astype('int')
})
```

Ahora hay **tres columnas de entrada** en lugar de una, más la columna `coste` que se quiere predecir.

### 5.3 Entrenamiento

```python
X = df[['equipos_afectados', 'horas_respuesta', 'datos_sensibles_filtrados']].values
y = df['coste'].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.intercept_   # theta 0: coste base
lin_reg.coef_        # theta 1, theta 2, theta 3: un coeficiente por cada variable
```

La diferencia clave: en vez de pasar **una sola columna**, se pasan **las tres a la vez** (`X` es ahora una tabla de 200 filas × 3 columnas). El resto es idéntico: `.fit()` sigue siendo el entrenamiento, solo que ahora busca cuatro números en lugar de dos.

> **Coste estimado = coste base + (coste por equipo × equipos afectados) + (coste por hora × horas de respuesta) + (coste extra × fuga de datos)**

### 5.4 Predicción de un incidente nuevo

```python
incidente_nuevo = np.array([[250, 30, 1]])  # 250 equipos, 30 horas, con fuga de datos
coste_estimado = lin_reg.predict(incidente_nuevo)
print("El coste del incidente sería:", int(coste_estimado[0]), "€")
```

Ahora se puede describir un incidente de forma mucho más realista y obtener una estimación que tiene en cuenta los tres factores a la vez, no solo uno.

### 5.5 Comparar el peso de cada variable

```python
for nombre, coeficiente in zip(
    ['equipos_afectados', 'horas_respuesta', 'datos_sensibles_filtrados'],
    lin_reg.coef_
):
    print(f"Cada unidad de '{nombre}' añade aproximadamente {coeficiente:.2f} € al coste")
```

Una gran ventaja de que el modelo siga siendo lineal (y no, por ejemplo, una red neuronal) es que se puede leer directamente, variable por variable, cuánto "pesa" cada una en el resultado. Esto responde preguntas de negocio como: *"¿nos conviene más invertir en reducir el tiempo de respuesta o en tener menos equipos expuestos?"*, comparando coeficientes.

### Checkpoint

> Si dos incidentes tienen los mismos equipos afectados y el mismo tiempo de respuesta, pero uno tuvo fuga de datos y el otro no, ¿qué diferencia de coste predice el modelo entre ambos? (Pista: mira el coeficiente de `datos_sensibles_filtrados`).

---

## Módulo 6 — Variables categóricas y one-hot encoding

**Objetivo del módulo:** incorporar variables que no son números "de forma natural" (barrios, letras de certificado energético) a un modelo que solo entiende números.

Los ejemplos anteriores usaban variables que ya venían "listas para usar". En la vida real, muchos problemas mezclan **variables numéricas** con **variables categóricas** (etiquetas, no números: un barrio, una letra de certificado energético), y además hay que **conseguir los datos reales**, no simularlos.

> **Pregunta de negocio:** *si conozco los metros cuadrados, las habitaciones, el barrio, la antigüedad y el certificado energético de un piso, ¿puedo estimar por cuánto se vendería?*

### ¿Por qué es más complejo que los módulos anteriores?

- **Hay muchas más variables a la vez** (siete u ocho, en vez de una o tres), y no todas influyen igual.
- **Aparecen variables categóricas con más de dos valores posibles.** Antes, "¿hubo fuga de datos?" era sí/no (0 o 1). Ahora, `barrio` puede ser Centro, Ensanche, Periferia... y `certificado_energetico` puede ser A–G. No se pueden convertir en un único número 0/1: hace falta **one-hot encoding**.
- **Las variables pueden estar relacionadas entre sí** (pisos más grandes suelen tener también más habitaciones), lo que hace más delicada la interpretación de cada coeficiente por separado.
- **Los datos reales no vienen limpios**: huecos, errores de escritura, valores extremos poco creíbles, formatos inconsistentes.

### 6.1 Variables del dataset

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
| `precio` (a predecir) | Numérica | 245 000 |

### 6.2 Datos (simulados, para practicar el procedimiento)

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
```

El precio "de verdad" se fabrica combinando todas las variables con pesos conocidos, para comprobar después que el modelo los recupera. En un caso real, `precio` sería el dato observado (lo que pagó el comprador), no algo calculado con una fórmula.

### 6.3 One-hot encoding

```python
df_codificado = pd.get_dummies(df, columns=['certificado_energetico', 'barrio'], drop_first=True)
```

Un modelo de regresión lineal solo entiende números: no se le puede pasar directamente `"Centro"` o `"C"`. El **one-hot encoding** convierte cada categoría en varias columnas nuevas de 0 y 1, una por cada valor posible (menos uno, que sirve de referencia):

```
Antes:                    Después (one-hot):
barrio                    barrio_Ensanche  barrio_Periferia  barrio_Zona_Norte
Centro          →         0                0                 0
Ensanche        →         1                0                 0
Periferia       →         0                1                 0
```

Un piso en "Centro" tiene todas esas columnas a 0 (categoría de referencia implícita); un piso en "Ensanche" tiene un 1 solo en `barrio_Ensanche`. `drop_first=True` evita crear una columna redundante para la primera categoría, porque "no ser ninguna de las demás" ya identifica esa categoría.

### 6.4 Entrenamiento

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

El procedimiento es idéntico al del Módulo 5: se pasan todas las columnas a la vez y `.fit()` calcula un coeficiente por cada una. La única diferencia es que ahora hay más columnas (las categorías se "desdoblan"), pero el algoritmo no distingue entre una columna numérica normal y una que viene de un one-hot encoding: para él, todo son números.

### 6.5 Predicción de una vivienda nueva

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

**Detalle importante:** el piso nuevo hay que codificarlo **exactamente con las mismas columnas** que se usaron para entrenar (por eso `.reindex(columns=columnas_entrada, fill_value=0)`). Si el piso nuevo es del barrio "Centro", todas las columnas de barrio deben quedar a 0, igual que en los datos de entrenamiento.

### Checkpoint

> ¿Por qué `drop_first=True` no hace que se pierda información sobre la categoría "Centro"? ¿Qué pasaría si al predecir un piso nuevo se te olvidara el `.reindex(...)` y le faltara una columna que el modelo sí vio al entrenar?

---

## Módulo 7 — Cómo evaluar si el modelo es bueno

**Objetivo del módulo:** ir más allá de "mirar si la línea pasa por en medio de los puntos" y tener una forma numérica de medir qué tan bueno es el modelo.

En los módulos anteriores, la calidad del modelo se valoraba "a ojo", mirando la gráfica. En la práctica hace falta una medida numérica, sobre todo cuando hay más de una variable de entrada y ya no se puede dibujar la recta. Las métricas más habituales para regresión son:

- **MSE (Error Cuadrático Medio):** el promedio de los errores al cuadrado (la misma cantidad que el modelo minimiza al entrenar, explicada en el Módulo 2). Cuanto más bajo, mejor. Su problema es que no se interpreta en las mismas unidades que la variable predicha (si el coste está en euros, el MSE está en "euros al cuadrado").
- **RMSE (Raíz del Error Cuadrático Medio):** la raíz cuadrada del MSE. Se recupera la unidad original (euros), lo que lo hace más fácil de interpretar: *"de media, el modelo se equivoca en unos X euros"*.
- **R² (coeficiente de determinación):** un número entre 0 y 1 (a veces negativo, si el modelo es peor que simplemente predecir siempre la media) que indica qué proporción de la variación de los datos es capaz de explicar el modelo. Un R² de 0,85 se suele leer como *"el modelo explica el 85 % de la variabilidad del coste"*.

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = lin_reg.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
```

### Un matiz importante: entrenar y evaluar con los mismos datos no es suficiente

En todos los ejemplos de este curso, el modelo se entrena y se evalúa (a ojo o con métricas) **sobre los mismos datos**. Esto sirve para aprender el procedimiento, pero en un proyecto real hay que separar los datos en:

- **Conjunto de entrenamiento (train):** con el que el modelo aprende θ₀, θ₁... (típicamente el 70-80 % de los datos).
- **Conjunto de prueba (test):** datos que el modelo nunca vio durante el entrenamiento, usados solo para comprobar si de verdad generaliza bien a casos nuevos (el 20-30 % restante).

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg.fit(X_train, y_train)
r2_test = lin_reg.score(X_test, y_test)  # R² sobre datos nunca vistos
```

Si el modelo funciona muy bien en `train` pero mucho peor en `test`, es una señal de **sobreajuste** (overfitting): ha memorizado el ruido de los datos de entrenamiento en vez de aprender la tendencia real.

### Checkpoint

> Si un modelo tiene R² = 0,95 en entrenamiento pero R² = 0,40 en test, ¿qué probablemente está pasando?

---

## Módulo 8 — De los datos simulados a los datos reales

**Objetivo del módulo:** entender que, en todos los módulos anteriores, los datos se generaron con `np.random`, y ver de dónde saldrían en un proyecto real.

En un caso real (por ejemplo, el del Módulo 6 sobre precio de vivienda), montar el dataset implica reunir información de fuentes externas y limpiarla:

**a) Portales inmobiliarios (web scraping)**

Sitios como Idealista, Fotocasa o pisos.com publican anuncios con casi todas estas variables. Se pueden extraer con `BeautifulSoup`, `Scrapy` o `Selenium` (para páginas con contenido dinámico). Hay que:
- Revisar el `robots.txt` y los términos de servicio antes de scrapear (muchos portales lo limitan o lo prohíben; algunos ofrecen una API oficial de pago como alternativa legal).
- Tener en cuenta que el scraping da datos de **precio de oferta**, no necesariamente el precio final de venta (que suele ser algo menor).

**b) Fuentes públicas y datos abiertos (gratuitas)**

- **Sede Electrónica del Catastro**: superficie construida, año de construcción y uso del inmueble, por referencia catastral.
- **INE** y **Registradores de la Propiedad**: índices de precios de vivienda y estadísticas de compraventas (tendencia media por zona, no precios de un piso concreto).
- **Portales de datos abiertos** (datos.gob.es, portales open data de ayuntamientos): precios medios por barrio, callejero, equipamientos (colegios, metro, parques), útiles para variables como `distancia_centro_km`.

**c) Datasets ya preparados por terceros**

Plataformas como Kaggle tienen datasets de vivienda ya limpios (algunos de España, muchos de otros países, como el clásico "House Prices" de EE. UU.). Ideales para practicar el modelado sin resolver antes el problema de la recogida de datos, aunque no sirven si se necesita un modelo ajustado a un mercado local muy concreto.

**d) Proveedores de datos especializados (de pago)**

Empresas como Tinsa, idealista Data o Fotocasa ofrecen precios de venta contrastados (no solo de oferta), pensados para tasadoras, bancos e inmobiliarias. Es la opción más fiable para un contexto profesional real, aunque tiene coste económico.

**e) Enriquecer los datos con información geográfica**

Variables como `distancia_centro_km` normalmente no vienen dadas: hay que calcularlas con APIs de geocodificación como **OpenStreetMap Nominatim** (gratuita) o **Google Maps Geocoding API** (de pago a partir de cierto volumen).

**f) Limpieza de los datos reales antes de entrenar**

A diferencia de los datos simulados, un dataset real necesita trabajo previo:
- Rellenar o descartar filas con datos faltantes (pisos sin certificado energético informado).
- Detectar y revisar valores atípicos poco creíbles (un "1000 m²" que en realidad era "100").
- Unificar formatos ("Centro" y "centro" no deben tratarse como dos barrios distintos; los precios no deben mezclar textos como "245.000 €" con números puros).

Este trabajo de limpieza y validación, en la práctica, suele llevar más tiempo que el propio entrenamiento del modelo.

---

## Módulo 9 — Límites de la regresión lineal: cuándo no usarla

**Objetivo del módulo:** saber reconocer cuándo este algoritmo deja de ser la herramienta adecuada, para no forzarlo fuera de su terreno.

La regresión lineal es potente por su simplicidad, pero esa misma simplicidad es también su límite. Algunas señales de que quizá haga falta otro enfoque:

- **La relación no es lineal.** Si al dibujar los datos aparece una curva (por ejemplo, el coste se dispara exponencialmente a partir de cierto número de equipos afectados), una recta no la va a capturar bien. En esos casos se puede probar con regresión polinómica u otros modelos no lineales.
- **Lo que se quiere predecir es una categoría, no un número.** Si en vez de "cuánto costará" la pregunta fuera "¿es spam o no?" o "¿es phishing o no?", ya no es un problema de regresión sino de **clasificación**, y hace falta otro algoritmo (regresión logística, árboles de decisión, etc.).
- **Hay muchas variables muy correlacionadas entre sí (multicolinealidad).** Cuando dos variables de entrada están muy relacionadas (por ejemplo, metros cuadrados y número de habitaciones), puede volverse difícil interpretar el peso individual de cada coeficiente con confianza.
- **Hay valores atípicos (outliers) extremos.** La regresión lineal es sensible a puntos muy alejados del resto: unos pocos valores extremos pueden desplazar bastante la recta aprendida.
- **Se necesita capturar interacciones complejas entre variables** que no se explican bien sumando el efecto de cada una por separado (por ejemplo, que el efecto de la antigüedad en el precio dependa también del barrio). Para eso existen modelos más flexibles (árboles, random forests, redes neuronales), a costa de perder parte de la interpretabilidad tan directa que tiene la regresión lineal.

La buena noticia es que, incluso cuando se termina usando otro algoritmo, **la regresión lineal sigue siendo el punto de partida útil**: sirve como referencia ("baseline") para saber si un modelo más complejo realmente aporta una mejora, y el vocabulario que se aprende aquí (variable de entrada/salida, entrenamiento, coeficientes, error, overfitting) se reutiliza en prácticamente todos los algoritmos de Machine Learning que vienen después.

---

## Resumen final y checklist de repaso

1. La regresión lineal predice un **número** a partir de una o varias variables de entrada, buscando la recta (o el hiperplano) que mejor resume la relación entre ellas.
2. Toda recta se describe con dos números: **θ₀** (intercepto, el valor base) y **θ₁** (pendiente, cuánto sube la salida por cada unidad de entrada). Con varias variables, hay un coeficiente por cada una.
3. El modelo se entrena minimizando el **error cuadrático medio** entre sus predicciones y los datos reales; scikit-learn hace ese cálculo por dentro con `.fit()`.
4. El flujo de trabajo es siempre el mismo, cambie lo que cambie el problema: **cargar/generar datos → visualizar → preparar/escalar → `.fit()` → `.predict()`**.
5. Cuando el resultado depende de **varias causas**, se usa **regresión lineal múltiple**: se le pasa al modelo una tabla con varias columnas de entrada en vez de una sola.
6. Las variables categóricas (barrio, certificado energético...) no se pueden pasar directamente al modelo: hace falta **one-hot encoding** para convertirlas en columnas de 0 y 1.
7. Para saber si el modelo es bueno de verdad (no solo "a ojo"), se usan métricas como **RMSE** y **R²**, y se evalúa sobre datos de **test** que el modelo no vio al entrenar.
8. Los datos reales no vienen limpios ni ya preparados: conseguirlos y limpiarlos (scraping, fuentes públicas, datasets de terceros, proveedores de pago, geocodificación) suele ser el trabajo más largo del proyecto.
9. La regresión lineal tiene límites claros (relaciones no lineales, clasificación, multicolinealidad, outliers, interacciones complejas) y sirve como punto de partida y referencia antes de pasar a modelos más avanzados.

### Checklist de repaso

- [ ] Puedo explicar con mis palabras qué busca la regresión lineal en una nube de puntos.
- [ ] Sé leer `intercept_` y `coef_` y traducirlos a una frase en español.
- [ ] Puedo repetir el flujo `.fit()` / `.predict()` con un ejemplo distinto (una sola variable).
- [ ] Entiendo por qué hace falta `.reshape(-1, 1)` y `.flatten()`, y cuándo se usa cada uno.
- [ ] Sé cuándo pasar de regresión simple a regresión múltiple.
- [ ] Sé aplicar one-hot encoding a una variable categórica y por qué se usa `drop_first=True`.
- [ ] Sé calcular RMSE y R², y entiendo la diferencia entre evaluar en train y en test.
- [ ] Puedo nombrar al menos tres situaciones en las que la regresión lineal no sería la mejor opción.

---

## Glosario

| Término | Significado |
|---|---|
| **Variable de entrada (feature / X)** | El dato que se conoce y se usa para predecir (equipos afectados, metros cuadrados...). |
| **Variable de salida (target / y)** | El dato que se quiere predecir (coste, precio...). |
| **Hipótesis (h(x))** | La fórmula que usa el modelo para predecir, en función de las variables de entrada. |
| **θ₀ (intercepto)** | Valor de salida cuando todas las variables de entrada valen 0. |
| **θ₁, θ₂...(coeficientes / pendientes)** | Cuánto cambia la salida por cada unidad que cambia la variable de entrada correspondiente. |
| **Entrenamiento (`.fit()`)** | Proceso por el que el modelo calcula los valores de θ que mejor ajustan los datos. |
| **Predicción (`.predict()`)** | Uso del modelo ya entrenado para estimar la salida de un caso nuevo. |
| **Error cuadrático medio (MSE)** | Medida del error del modelo: promedio de los errores al cuadrado entre predicción y valor real. |
| **RMSE** | Raíz cuadrada del MSE; expresa el error en las mismas unidades que la variable predicha. |
| **R² (coeficiente de determinación)** | Proporción de la variabilidad de los datos que el modelo es capaz de explicar (0 a 1). |
| **Regresión lineal simple** | Regresión lineal con una sola variable de entrada. |
| **Regresión lineal múltiple** | Regresión lineal con varias variables de entrada a la vez. |
| **Variable categórica** | Variable cuyo valor es una etiqueta (barrio, certificado energético), no un número. |
| **Variable binaria / dummy** | Variable categórica de solo dos valores, representada como 0 o 1. |
| **One-hot encoding** | Técnica para convertir una variable categórica de varios valores en varias columnas de 0 y 1. |
| **Overfitting (sobreajuste)** | Cuando el modelo aprende demasiado bien el ruido de los datos de entrenamiento y generaliza mal a datos nuevos. |
| **Conjunto de entrenamiento / test** | División de los datos en una parte para entrenar el modelo y otra, nunca vista, para evaluarlo de forma honesta. |

---

*Este curso toma como base y referencia los ejemplos desarrollados en "4_Regresión Lineal - Explicación.md" y en el notebook "Ejericios/4_Regresión Lineal - Predicción del coste de un incidente de seguridad.ipynb".*
