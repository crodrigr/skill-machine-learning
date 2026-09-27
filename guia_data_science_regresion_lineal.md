# Del Problema a la Predicción
## Guía práctica de Ciencia de Datos con Regresión Lineal

> **Para quién es esta guía:** estudiantes que quieren entender, de principio a fin, cómo se desarrolla un proyecto real de Ciencia de Datos, y dónde encaja el Machine Learning dentro de ese proceso.
>
> **Qué necesitas:** Python 3, y las librerías `pandas`, `seaborn`, `matplotlib` y `scikit-learn`.
>
> ```bash
> pip install pandas seaborn matplotlib scikit-learn
> ```

---

## Contenido

0. [Antes de empezar: ¿qué es la Ciencia de Datos?](#0-antes-de-empezar-qué-es-la-ciencia-de-datos)
1. [Fase 1 – Planteamiento del problema](#fase-1--planteamiento-del-problema)
2. [Fase 2 – Obtención y entendimiento de los datos](#fase-2--obtención-y-entendimiento-de-los-datos)
3. [Fase 3 – Limpieza de datos](#fase-3--limpieza-de-datos)
4. [Fase 4 – Análisis exploratorio (EDA)](#fase-4--análisis-exploratorio-eda)
5. [Fase 5 – Preparación de datos para el modelo](#fase-5--preparación-de-datos-para-el-modelo)
6. [Fase 6 – Modelado con Regresión Lineal](#fase-6--modelado-con-regresión-lineal)
7. [Fase 7 – Evaluación del modelo](#fase-7--evaluación-del-modelo)
8. [Fase 8 – Interpretación y comunicación de resultados](#fase-8--interpretación-y-comunicación-de-resultados)
9. [Fase 9 – Despliegue y mejora continua](#fase-9--despliegue-y-mejora-continua)
10. [Código completo](#10-código-completo)
11. [Ejercicios para el estudiante](#11-ejercicios-para-el-estudiante)
12. [Glosario](#12-glosario)
13. [Lista de verificación final](#13-lista-de-verificación-final)

---

## 0. Antes de empezar: ¿qué es la Ciencia de Datos?

La **Ciencia de Datos (Data Science)** es la disciplina que convierte datos en **conocimiento y decisiones**. Incluye todo el recorrido: desde entender qué pregunta queremos responder, hasta comunicar la respuesta a quien debe tomar una decisión.

El **Machine Learning (Aprendizaje Automático)** es **una herramienta dentro** de la Ciencia de Datos. Es la parte donde un algoritmo aprende patrones de los datos para **predecir** o **clasificar**.

> **Idea clave:** el Data Science es todo el viaje, desde la pregunta hasta la decisión. El Machine Learning es uno de los vehículos que puedes usar en ese viaje, pero no el único.

### El ciclo de un proyecto de Ciencia de Datos

```
 ┌─────────────────────┐
 │ 1. Planteamiento    │◄──────────────────────────────┐
 │    del problema     │                               │
 └─────────┬───────────┘                               │
           ▼                                           │
 ┌─────────────────────┐    ┌─────────────────────┐    │
 │ 2. Obtención de     │───►│ 3. Limpieza         │    │
 │    datos            │    │                     │    │
 └─────────────────────┘    └─────────┬───────────┘    │
                                      ▼                │
 ┌─────────────────────┐    ┌─────────────────────┐    │
 │ 5. Preparación      │◄───│ 4. Análisis         │    │
 │    para el modelo   │    │    exploratorio     │    │
 └─────────┬───────────┘    └─────────────────────┘    │
           ▼                                           │
 ┌─────────────────────┐    ┌─────────────────────┐    │
 │ 6. Modelado (ML)    │───►│ 7. Evaluación       │    │
 └─────────────────────┘    └─────────┬───────────┘    │
                                      ▼                │
 ┌─────────────────────┐    ┌─────────────────────┐    │
 │ 9. Despliegue y     │◄───│ 8. Interpretación   │    │
 │    mejora continua  │    │    y comunicación   │    │
 └─────────┬───────────┘    └─────────────────────┘    │
           └───────────────────────────────────────────┘
```

Fíjate en que el ciclo **vuelve al inicio**. En proyectos reales casi nunca se avanza en línea recta: a veces en el análisis descubres que te faltan datos, o en la evaluación descubres que la pregunta estaba mal planteada.

Solo la **Fase 6** es Machine Learning. Todas las demás también son Ciencia de Datos, y son las que ocupan la mayor parte del tiempo.

---

## Fase 1 – Planteamiento del problema

### ¿Qué se hace en esta fase?

Se traduce una **necesidad del negocio** en una **pregunta que los datos puedan responder**. Es la fase más importante: un modelo perfecto que responde la pregunta equivocada no sirve para nada.

### Nuestro caso de estudio

El administrador de un restaurante quiere entender el comportamiento de las **propinas** para:

- Estimar cuánto recibirán los meseros en un turno.
- Saber qué factores influyen en que una mesa deje más o menos propina.
- Decidir si vale la pena asignar a los meseros con más experiencia a ciertos días u horarios.

> **Nota sobre los datos:** usaremos el dataset real **`tips`**, que viene incluido en la librería `seaborn`. Contiene 244 cuentas registradas por un mesero en un restaurante de Estados Unidos, por eso los valores están en **dólares**.

### De la necesidad a la pregunta

| Necesidad del negocio | Pregunta de datos | Tipo de problema |
|---|---|---|
| "Quiero saber cuánto dejarán de propina" | ¿Podemos **predecir el valor de la propina** de una mesa a partir de sus características? | Regresión (predecir un número) |
| "Quiero saber qué influye" | ¿Qué variables están **más relacionadas** con la propina? | Análisis descriptivo |

### Elementos que siempre debes definir

- **Variable objetivo (y):** lo que queremos predecir → `tip` (propina).
- **Variables predictoras (X):** lo que usaremos para predecir → total de la cuenta, número de personas, día, horario, etc.
- **Métrica de éxito:** ¿cómo sabremos si el modelo es bueno? → que el error promedio sea **claramente menor** que simplemente adivinar el promedio de propinas.
- **Restricciones:** los datos son pocos (244 registros) y de un solo restaurante; las conclusiones no se pueden generalizar a cualquier restaurante.

> **Pregunta para reflexionar:** ¿por qué este es un problema de **regresión** y no de **clasificación**?
> Porque la respuesta es un **número continuo** (por ejemplo, $3.06), no una categoría como "sí/no".

---

## Fase 2 – Obtención y entendimiento de los datos

### ¿Qué se hace en esta fase?

Se consiguen los datos (bases de datos, archivos CSV, APIs, encuestas…) y se entiende **qué significa cada columna**.

### Cargar los datos

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset("tips")
print(df.shape)     # (filas, columnas)
print(df.head())    # primeras 5 filas
```

**Resultado:**

```
(244, 7)

   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
```

### Diccionario de datos

Un **diccionario de datos** describe cada columna. Siempre deberías construir uno.

| Columna | Descripción | Tipo |
|---|---|---|
| `total_bill` | Total de la cuenta en dólares | Numérica continua |
| `tip` | Propina en dólares (**variable objetivo**) | Numérica continua |
| `sex` | Sexo de quien pagó (`Male`, `Female`) | Categórica |
| `smoker` | ¿Había fumadores en la mesa? (`Yes`, `No`) | Categórica |
| `day` | Día de la semana (`Thur`, `Fri`, `Sat`, `Sun`) | Categórica |
| `time` | Horario (`Lunch`, `Dinner`) | Categórica |
| `size` | Número de personas en la mesa | Numérica discreta |

### Revisar tipos de datos

```python
print(df.dtypes)
```

```
total_bill     float64
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
```

> **Concepto:** distinguir entre variables **numéricas** y **categóricas** es fundamental, porque los modelos de Machine Learning solo entienden números. Las categóricas tendrán que transformarse (lo veremos en la Fase 5).

---

## Fase 3 – Limpieza de datos

### ¿Qué se hace en esta fase?

Se detectan y corrigen problemas en los datos: valores faltantes, duplicados, errores de digitación, valores imposibles y valores atípicos. En proyectos reales, **esta fase puede ocupar entre el 60% y el 80% del tiempo**.

### 3.1 Valores faltantes

```python
print(df.isnull().sum())
```

En este dataset hay **0 valores faltantes**. En la vida real casi nunca es así. Las opciones típicas cuando faltan datos son:

- **Eliminar** las filas (si son pocas).
- **Imputar**: rellenar con el promedio, la mediana o el valor más frecuente.
- **Investigar** por qué faltan (a veces el hecho de que falte un dato ya es información).

### 3.2 Duplicados

```python
print(df.duplicated().sum())   # → 1
```

Hay **1 fila duplicada**. ¿La borramos?

> **Lección importante:** no todo duplicado es un error. Dos mesas distintas pueden tener exactamente la misma cuenta, propina, día y número de personas. Como no tenemos un identificador único de cada mesa, **no podemos asegurar que sea un error**, así que en este caso decidimos **conservarla**. Lo importante es que la decisión sea **consciente y documentada**.

### 3.3 Valores imposibles

Revisamos que no haya cuentas negativas, propinas negativas o mesas con 0 personas:

```python
print(df.describe().round(2))
```

```
       total_bill     tip    size
count      244.00  244.00  244.00
mean        19.79    3.00    2.57
std          8.90    1.38    0.95
min          3.07    1.00    1.00
25%         13.35    2.00    2.00
50%         17.80    2.90    2.00
75%         24.13    3.56    3.00
max         50.81   10.00    6.00
```

Todos los mínimos y máximos tienen sentido. No hay valores imposibles.

### 3.4 Valores atípicos (outliers)

Un valor atípico es un dato muy alejado del resto. Una regla común es la del **rango intercuartílico (IQR)**:

```python
q1 = df["tip"].quantile(0.25)
q3 = df["tip"].quantile(0.75)
iqr = q3 - q1
limite_superior = q3 + 1.5 * iqr

atipicos = df[df["tip"] > limite_superior]
print(len(atipicos), limite_superior)   # → 9 propinas mayores a ~$5.91
```

Hay **9 propinas** por encima de $5.91 (la máxima es $10). ¿Son errores? Probablemente no: corresponden a cuentas grandes, y una propina de $10 en una cuenta de $50 es completamente razonable. **Las conservamos.**

> **Regla práctica:** un outlier se elimina solo si es un **error** (por ejemplo, una propina de $1.000 en una cuenta de $20). Si es un valor real, aunque extremo, normalmente se conserva.

---

## Fase 4 – Análisis exploratorio (EDA)

### ¿Qué se hace en esta fase?

Se **exploran** los datos con estadísticas y gráficas para entender patrones, relaciones y posibles hipótesis. Aquí todavía **no hay Machine Learning**, y muchas veces esta fase por sí sola ya responde preguntas del negocio.

### 4.1 ¿Cómo se distribuye la propina?

```python
import matplotlib.pyplot as plt

sns.histplot(df["tip"], bins=20)
plt.title("Distribución de las propinas")
plt.show()
```

**Hallazgo:** la mayoría de las propinas están entre $2 y $3.56, con un promedio de **$3.00**.

### 4.2 Creando una nueva variable: porcentaje de propina

```python
df["pct_propina"] = df["tip"] / df["total_bill"] * 100
print(df["pct_propina"].describe().round(2))
```

**Hallazgo:** en promedio las mesas dejan un **16%** de propina (mediana 15.5%).

### 4.3 Relación entre la cuenta y la propina

```python
sns.scatterplot(data=df, x="total_bill", y="tip")
plt.title("Total de la cuenta vs. propina")
plt.show()
```

**Hallazgo:** los puntos forman una nube inclinada hacia arriba: **a mayor cuenta, mayor propina**. Esto sugiere que una **recta** podría describir bien la relación → buena candidata para **regresión lineal**.

### 4.4 Correlación

La **correlación** mide qué tan fuerte es la relación lineal entre dos variables, de -1 a 1.

```python
print(df[["total_bill", "tip", "size"]].corr().round(2))
```

```
            total_bill   tip  size
total_bill        1.00  0.68  0.60
tip               0.68  1.00  0.49
size              0.60  0.49  1.00
```

| Valor | Interpretación |
|---|---|
| Cerca de **1** | Relación positiva fuerte (cuando una sube, la otra sube) |
| Cerca de **0** | No hay relación lineal |
| Cerca de **-1** | Relación negativa fuerte (cuando una sube, la otra baja) |

**Hallazgos:**
- `total_bill` y `tip`: **0.68** → relación positiva fuerte. Es nuestra mejor variable predictora.
- `size` y `tip`: **0.49** → relación moderada.
- `total_bill` y `size`: **0.60** → ¡ojo! Las dos predictoras están relacionadas entre sí (más personas → cuenta más alta). Esto se llama **multicolinealidad** y lo veremos de nuevo en la evaluación.

### 4.5 Propina promedio por categoría

```python
print(df.groupby("day", observed=True)["tip"].mean().round(2))
print(df.groupby("time", observed=True)["tip"].mean().round(2))
print(df.groupby("smoker", observed=True)["tip"].mean().round(2))
```

| Variable | Resultado |
|---|---|
| Día | Jue $2.77 · Vie $2.73 · Sáb $2.99 · **Dom $3.26** |
| Horario | Almuerzo $2.73 · **Cena $3.10** |
| Fumadores | Sí $3.01 · No $2.99 (prácticamente igual) |

**Hallazgos:**
- Los **domingos** y las **cenas** tienen propinas más altas.
- Ser fumador **no parece** influir en la propina.

> **Pregunta crítica:** ¿las cenas dejan más propina porque la gente es más generosa en la noche, o simplemente porque **en la cena se gasta más**? Correlación no es causalidad. Esta duda es exactamente el tipo de pensamiento que distingue a un buen científico de datos.

---

## Fase 5 – Preparación de datos para el modelo

### ¿Qué se hace en esta fase?

Se transforman los datos al formato que el algoritmo necesita y se separan en conjuntos de **entrenamiento** y **prueba**.

### 5.1 Definir X (entradas) y y (salida)

Empezamos con el modelo más simple posible: **una sola variable**.

```python
X = df[["total_bill"]]   # doble corchete: X debe ser una tabla
y = df["tip"]            # un corchete: y es una columna
```

### 5.2 Separar en entrenamiento y prueba

Es como estudiar para un examen: si el profesor pone en el examen **exactamente** los mismos ejercicios del taller, no sabemos si aprendiste o si memorizaste. Por eso guardamos datos que el modelo **nunca verá** durante el entrenamiento.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(len(X_train), len(X_test))   # → 195 y 49
```

- **Entrenamiento (80% = 195 filas):** el modelo aprende de aquí.
- **Prueba (20% = 49 filas):** con esto lo evaluamos.
- `random_state=42`: fija la aleatoriedad para que todos obtengan los mismos resultados.

### 5.3 Variables categóricas → números (para el modelo avanzado)

Los modelos no entienden texto como `"Sun"` o `"Dinner"`. La técnica más común es **One-Hot Encoding**: crear una columna de 0/1 por cada categoría.

```python
X_completo = pd.get_dummies(
    df[["total_bill", "size", "sex", "smoker", "day", "time"]],
    drop_first=True
)
```

Por ejemplo, la columna `time` se convierte en `time_Dinner` (1 si fue cena, 0 si fue almuerzo). Usamos `drop_first=True` para no crear columnas redundantes: si no es cena, ya sabemos que es almuerzo.

---

## Fase 6 – Modelado con Regresión Lineal

### ¿Qué se hace en esta fase?

**Aquí entra el Machine Learning.** Elegimos un algoritmo y lo entrenamos con los datos.

### 6.1 ¿Qué es la regresión lineal?

Es un algoritmo que busca la **recta que mejor se ajusta** a los datos para predecir un número.

$$
\hat{y} = \theta_0 + \theta_1 \cdot x
$$

| Símbolo | Nombre | En nuestro caso |
|---|---|---|
| $\hat{y}$ | Predicción | Propina estimada |
| $x$ | Variable de entrada | Total de la cuenta |
| $\theta_0$ (theta cero) | **Intercepto** | Valor base cuando x = 0 |
| $\theta_1$ (theta uno) | **Pendiente** | Cuánto sube la propina por cada dólar extra de cuenta |

> También la verás escrita como `y = mx + b`. Es exactamente lo mismo: $m = \theta_1$ y $b = \theta_0$.

### 6.2 ¿Cómo aprende el modelo?

1. Prueba una recta.
2. Mide el **error** de cada punto: la distancia vertical entre el valor real y la recta.
3. Eleva esos errores al cuadrado (para que los negativos no cancelen a los positivos) y los suma.
4. Busca los valores de $\theta_0$ y $\theta_1$ que hacen esa suma **lo más pequeña posible**.

Este método se llama **Mínimos Cuadrados Ordinarios (OLS)**. Los valores de $\theta$ que encuentra se llaman **parámetros** del modelo: son "lo que aprendió".

### 6.3 Entrenar el modelo

```python
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)      # ← aquí ocurre el aprendizaje

print("θ₀ (intercepto):", round(modelo.intercept_, 3))
print("θ₁ (pendiente):", round(modelo.coef_[0], 3))
```

```
θ₀ (intercepto): 0.925
θ₁ (pendiente): 0.107
```

El modelo aprendió:

$$
\text{propina} = 0.925 + 0.107 \times \text{total\_cuenta}
$$

### 6.4 Hacer predicciones

```python
nuevas_mesas = pd.DataFrame({"total_bill": [20, 50]})
print(modelo.predict(nuevas_mesas).round(2))   # → [3.06 6.27]
```

| Cuenta | Propina predicha |
|---|---|
| $20 | **$3.06** |
| $50 | **$6.27** |

---

## Fase 7 – Evaluación del modelo

### ¿Qué se hace en esta fase?

Se mide **qué tan bueno es el modelo** con datos que nunca vio. Sin esta fase, no sabemos si el modelo sirve.

### 7.1 Siempre compara contra un modelo base (baseline)

Antes de celebrar, pregúntate: ¿mi modelo es mejor que **no usar ningún modelo**? El baseline más simple es **predecir siempre el promedio**.

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

pred_base = np.full(len(y_test), y_train.mean())
print(mean_absolute_error(y_test, pred_base))   # → 1.05
```

Si siempre adivináramos el promedio, nos equivocaríamos en **$1.05** en promedio.

### 7.2 Métricas de nuestro modelo

```python
from sklearn.metrics import r2_score, mean_squared_error

pred = modelo.predict(X_test)

print("R²:  ", round(r2_score(y_test, pred), 2))
print("MAE: ", round(mean_absolute_error(y_test, pred), 2))
print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 2))
```

```
R²:   0.54
MAE:  0.62
RMSE: 0.75
```

| Métrica | Qué mide | Nuestro resultado | Interpretación |
|---|---|---|---|
| **R²** | Qué porcentaje de la variación de la propina explica el modelo (0 a 1) | **0.54** | El total de la cuenta explica un 54% de las diferencias entre propinas |
| **MAE** | Error promedio, en las mismas unidades | **$0.62** | En promedio nos equivocamos por 62 centavos |
| **RMSE** | Parecido al MAE pero castiga más los errores grandes | **$0.75** | Hay algunos errores grandes que suben la métrica |

**Conclusión:** el modelo reduce el error de **$1.05 a $0.62** frente al baseline, es decir, **cerca de un 40% menos de error**. ✅ Cumple la métrica de éxito que definimos en la Fase 1.

> ¿Y el 46% que no explica? Depende de cosas que no están en los datos: la calidad del servicio, el estado de ánimo del cliente, la costumbre personal de cada uno. **Ningún modelo puede aprender de información que no tiene.**

### 7.3 Experimento: ¿más variables = mejor modelo?

Probamos un modelo con **todas** las variables (usando `X_completo` de la Fase 5):

```python
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_completo, y, test_size=0.2, random_state=42
)
modelo2 = LinearRegression().fit(X_train2, y_train2)
pred2 = modelo2.predict(X_test2)

print("R²: ", round(r2_score(y_test2, pred2), 2))
print("MAE:", round(mean_absolute_error(y_test2, pred2), 2))
```

| Modelo | Variables | R² (prueba) | MAE (prueba) |
|---|---|---|---|
| Simple | Solo `total_bill` | **0.54** | **$0.62** |
| Completo | Las 6 variables | 0.44 | $0.67 |

**¡El modelo con más variables fue peor!** ¿Por qué?

- Las variables extra (día, sexo, fumador) tienen **poca relación real** con la propina, como vimos en el EDA. Añaden "ruido".
- `size` está muy relacionada con `total_bill` (**multicolinealidad**): aporta poca información nueva.
- Con solo 195 datos de entrenamiento, más variables hacen que el modelo aprenda **coincidencias** de esos datos que no se repiten en los datos nuevos. Esto es una forma de **sobreajuste (overfitting)**.

> **Lección clave:** más columnas no significa mejor modelo. **Siempre evalúa con datos de prueba** y prefiere el modelo más simple cuando el complejo no mejora claramente. Este principio se conoce como la **Navaja de Ockham**.

### 7.4 Una advertencia sobre datos pequeños

Con solo 49 datos de prueba, los resultados pueden cambiar bastante si la división es diferente (prueba cambiando `random_state`). En proyectos serios se usa **validación cruzada (cross-validation)**, que repite la división varias veces y promedia los resultados:

```python
from sklearn.model_selection import cross_val_score

puntajes = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
print(puntajes.round(2), puntajes.mean().round(2))
```

---

## Fase 8 – Interpretación y comunicación de resultados

### ¿Qué se hace en esta fase?

Se traducen los resultados técnicos a un **lenguaje que entienda quien toma la decisión**. Un gerente no necesita saber qué es θ₁; necesita saber **qué hacer**.

### Del lenguaje técnico al lenguaje de negocio

| Lo que dice el modelo | Lo que le dices al administrador |
|---|---|
| θ₁ = 0.107 | "Por cada $10 adicionales en la cuenta, la propina sube cerca de **$1.07**." |
| MAE = 0.62 | "Nuestras estimaciones de propina por mesa se equivocan, en promedio, por unos **60 centavos**." |
| R² = 0.54 | "El valor de la cuenta explica un poco más de la mitad de la propina; el resto depende de factores que no medimos, como el servicio." |
| Domingos y cenas con mayor propina | "Los domingos en la noche son el turno con mayor propina esperada: conviene asignar ahí a los meseros con mejor servicio." |
| Fumador ≈ no fumador | "Separar las mesas por fumadores **no** afecta las propinas." |

### Estructura recomendada para un informe

1. **Pregunta:** ¿qué queríamos responder?
2. **Respuesta corta:** en una o dos frases.
3. **Evidencia:** una o dos gráficas claras (la de dispersión con la recta es ideal).
4. **Recomendaciones:** acciones concretas.
5. **Limitaciones:** qué no podemos afirmar.

### Limitaciones que debes comunicar siempre

- Solo 244 registros de **un solo** restaurante y **un solo** mesero.
- Correlación no implica causalidad: no sabemos si la cena *causa* propinas más altas o si solo coincide con cuentas más altas.
- Faltan variables importantes (calidad del servicio, tiempo de espera).

### Gráfica final para el informe

```python
sns.regplot(data=df, x="total_bill", y="tip", line_kws={"color": "red"})
plt.title("A mayor cuenta, mayor propina")
plt.xlabel("Total de la cuenta (USD)")
plt.ylabel("Propina (USD)")
plt.show()
```

---

## Fase 9 – Despliegue y mejora continua

### ¿Qué se hace en esta fase?

Se pone el modelo **a funcionar en el mundo real** y se vigila que siga funcionando bien con el tiempo.

### Formas de desplegar un modelo

- **Guardarlo en un archivo** para reutilizarlo sin volver a entrenar:
  ```python
  import joblib
  joblib.dump(modelo, "modelo_propinas.pkl")
  modelo_cargado = joblib.load("modelo_propinas.pkl")
  ```
- **Exponerlo en una API** (por ejemplo con FastAPI o Flask) para que un sistema de caja lo consulte.
- **Integrarlo en un dashboard** que muestre la propina esperada por turno.

### Mejora continua

- **Recolectar más datos** y nuevas variables (calificación del servicio, tiempo de atención).
- **Reentrenar** el modelo periódicamente.
- **Monitorear** el error: si los hábitos de los clientes cambian (por ejemplo, tras una subida de precios), el modelo se "desactualiza". Esto se llama **deriva de datos (data drift)**.
- **Volver a la Fase 1** si surge una nueva pregunta: ¿podríamos predecir si una mesa dejará una propina **alta o baja**? → eso ya sería **regresión logística** (clasificación).

---

## 10. Código completo

```python
# ============================================
# PROYECTO: Predicción de propinas
# ============================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Fase 2: Obtención de datos ---
df = sns.load_dataset("tips")
print(df.shape)
print(df.head())

# --- Fase 3: Limpieza ---
print("Faltantes:", df.isnull().sum().sum())
print("Duplicados:", df.duplicated().sum())
print(df.describe().round(2))

# --- Fase 4: Análisis exploratorio ---
df["pct_propina"] = df["tip"] / df["total_bill"] * 100
print(df[["total_bill", "tip", "size"]].corr().round(2))
print(df.groupby("day", observed=True)["tip"].mean().round(2))
print(df.groupby("time", observed=True)["tip"].mean().round(2))

sns.scatterplot(data=df, x="total_bill", y="tip")
plt.title("Total de la cuenta vs. propina")
plt.show()

# --- Fase 5: Preparación ---
X = df[["total_bill"]]
y = df["tip"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Fase 6: Modelado ---
modelo = LinearRegression()
modelo.fit(X_train, y_train)
print(f"propina = {modelo.intercept_:.3f} + {modelo.coef_[0]:.3f} × cuenta")

# --- Fase 7: Evaluación ---
pred_base = np.full(len(y_test), y_train.mean())
pred = modelo.predict(X_test)

print("MAE baseline:", round(mean_absolute_error(y_test, pred_base), 2))
print("R²:  ", round(r2_score(y_test, pred), 2))
print("MAE: ", round(mean_absolute_error(y_test, pred), 2))
print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 2))

# --- Fase 8: Comunicación ---
sns.regplot(data=df, x="total_bill", y="tip", line_kws={"color": "red"})
plt.title("A mayor cuenta, mayor propina")
plt.xlabel("Total de la cuenta (USD)")
plt.ylabel("Propina (USD)")
plt.show()

# Predicción para nuevas mesas
nuevas = pd.DataFrame({"total_bill": [20, 50]})
print(modelo.predict(nuevas).round(2))
```

---

## 11. Ejercicios para el estudiante

### Nivel básico

1. Cambia `random_state=42` por otros valores (0, 7, 100). ¿Cambian el R² y el MAE? ¿Por qué crees que pasa?
2. Usa el modelo para predecir la propina de cuentas de $10, $30 y $45. ¿Los resultados te parecen razonables?
3. Entrena un modelo usando **solo** `size` como variable. Compara su R² y MAE con el modelo de `total_bill`. ¿Cuál es mejor y por qué?

### Nivel intermedio

4. Entrena un modelo con `total_bill` y `time` (usa `pd.get_dummies`). ¿Mejora frente al modelo simple?
5. Crea una gráfica de **residuos** (`y_test - pred` contra `pred`). ¿Los errores están repartidos al azar o ves algún patrón?
6. Usa `cross_val_score` con 5 particiones para comparar el modelo simple y el completo. ¿Se mantiene la conclusión de la Fase 7?

### Nivel avanzado

7. En lugar de predecir `tip`, predice `pct_propina`. ¿Qué variables importan ahora? ¿Cambian las conclusiones del negocio?
8. Replantea el problema como **clasificación**: crea una variable `propina_alta = tip > 3` y entrena una `LogisticRegression`. Evalúala con *accuracy* y matriz de confusión.
9. **Proyecto propio:** busca un dataset de tu interés (por ejemplo, en [Kaggle](https://www.kaggle.com/datasets) o [datos.gov.co](https://www.datos.gov.co)) y recorre **las 9 fases** de esta guía. Entrega un informe de máximo 2 páginas siguiendo la estructura de la Fase 8.

---

## 12. Glosario

| Término | Definición sencilla |
|---|---|
| **Ciencia de Datos** | Disciplina que convierte datos en conocimiento y decisiones. |
| **Machine Learning** | Algoritmos que aprenden patrones de los datos para predecir o clasificar. |
| **Variable objetivo (y)** | Lo que queremos predecir. |
| **Variables predictoras (X)** | Los datos que usamos para hacer la predicción. |
| **Regresión** | Problema donde se predice un número continuo. |
| **Clasificación** | Problema donde se predice una categoría. |
| **Regresión lineal** | Algoritmo que ajusta una recta a los datos para predecir un número. |
| **θ₀ (intercepto)** | Valor de la predicción cuando todas las entradas son 0. |
| **θ₁ (pendiente/coeficiente)** | Cuánto cambia la predicción por cada unidad que aumenta la entrada. |
| **Mínimos cuadrados** | Método que busca la recta con la menor suma de errores al cuadrado. |
| **Entrenamiento / Prueba** | Datos para que el modelo aprenda / datos para evaluarlo. |
| **Baseline** | Modelo muy simple (por ejemplo, predecir siempre el promedio) usado como punto de comparación. |
| **R²** | Proporción de la variación que explica el modelo (de 0 a 1). |
| **MAE** | Error absoluto medio: cuánto se equivoca el modelo en promedio. |
| **RMSE** | Raíz del error cuadrático medio: como el MAE, pero castiga más los errores grandes. |
| **Correlación** | Medida de -1 a 1 de qué tan relacionadas linealmente están dos variables. |
| **Multicolinealidad** | Cuando dos variables predictoras están muy relacionadas entre sí. |
| **Outlier** | Valor muy alejado del resto de los datos. |
| **One-Hot Encoding** | Técnica para convertir categorías en columnas de 0 y 1. |
| **Sobreajuste (overfitting)** | Cuando el modelo memoriza los datos de entrenamiento y falla con datos nuevos. |
| **Validación cruzada** | Evaluar el modelo repitiendo varias divisiones de los datos y promediando. |
| **Data drift** | Cambio en los datos del mundo real que hace que un modelo pierda precisión con el tiempo. |

---

## 13. Lista de verificación final

Antes de entregar cualquier proyecto de Ciencia de Datos, verifica:

- [ ] ¿Definí claramente la **pregunta de negocio** y la **variable objetivo**?
- [ ] ¿Construí un **diccionario de datos**?
- [ ] ¿Revisé **faltantes, duplicados, valores imposibles y outliers**, y documenté mis decisiones?
- [ ] ¿Hice un **análisis exploratorio** con gráficas y correlaciones?
- [ ] ¿Separé los datos en **entrenamiento y prueba** antes de entrenar?
- [ ] ¿Comparé mi modelo contra un **baseline**?
- [ ] ¿Reporté las métricas con datos de **prueba** (no de entrenamiento)?
- [ ] ¿Probé si un modelo **más simple** funciona igual o mejor?
- [ ] ¿Traduje los resultados a **lenguaje de negocio**, con recomendaciones concretas?
- [ ] ¿Comuniqué las **limitaciones** del análisis?

---

> **Recuerda:** el objetivo de la Ciencia de Datos no es construir el modelo más complejo, sino **responder bien la pregunta correcta**. A veces la mejor respuesta está en una gráfica; otras veces, en una recta bien entrenada.
