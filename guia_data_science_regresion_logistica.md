# De Predecir un Número a Predecir una Categoría
## Guía práctica de Ciencia de Datos con Regresión Logística

> **Para quién es esta guía:** estudiantes que ya conocen el flujo de trabajo de la [Ciencia de Datos con Regresión Lineal](guia_data_science_regresion_lineal.md) y quieren dar el salto a problemas de **clasificación**: predecir una categoría ("sobrevive"/"no sobrevive", "spam"/"no spam", "paga"/"no paga") en vez de un número.
>
> **Qué necesitas:** Python 3, y las librerías `pandas`, `seaborn`, `matplotlib` y `scikit-learn`.
>
> ```bash
> pip install pandas seaborn matplotlib scikit-learn
> ```

---

## Contenido

0. [Antes de empezar: de predecir un número a predecir una categoría](#0-antes-de-empezar-de-predecir-un-número-a-predecir-una-categoría)
1. [Fase 1 – Planteamiento del problema](#fase-1--planteamiento-del-problema)
2. [Fase 2 – Obtención y entendimiento de los datos](#fase-2--obtención-y-entendimiento-de-los-datos)
3. [Fase 3 – Limpieza de datos](#fase-3--limpieza-de-datos)
4. [Fase 4 – Análisis exploratorio (EDA)](#fase-4--análisis-exploratorio-eda)
5. [Fase 5 – Preparación de datos para el modelo](#fase-5--preparación-de-datos-para-el-modelo)
6. [Fase 6 – Modelado con Regresión Logística](#fase-6--modelado-con-regresión-logística)
7. [Fase 7 – Evaluación del modelo](#fase-7--evaluación-del-modelo)
8. [Fase 8 – Interpretación y comunicación de resultados](#fase-8--interpretación-y-comunicación-de-resultados)
9. [Fase 9 – Despliegue y mejora continua](#fase-9--despliegue-y-mejora-continua)
10. [Código completo](#10-código-completo)
11. [Ejercicios para el estudiante](#11-ejercicios-para-el-estudiante)
12. [Glosario](#12-glosario)
13. [Lista de verificación final](#13-lista-de-verificación-final)

---

## 0. Antes de empezar: de predecir un número a predecir una categoría

En la guía de regresión lineal predijimos la **propina** (un número: $3.06, $6.27...). Pero muchas preguntas de negocio no se responden con un número, sino con una **categoría**:

- ¿Este cliente **se dará de baja** o no? (sí/no)
- ¿Esta transacción **es fraude** o no? (sí/no)
- ¿Este pasajero **sobrevivió** al naufragio o no? (sí/no)

A esto se le llama un problema de **clasificación**, y cuando la categoría tiene solo dos posibles valores (sí/no, 1/0), se llama **clasificación binaria**. El algoritmo más simple e importante para este tipo de problema es la **Regresión Logística**.

> **Idea clave:** a pesar de su nombre, la regresión logística **no predice un número continuo, predice una probabilidad** (entre 0 y 1) de que algo pertenezca a una categoría. Esa probabilidad luego se convierte en una decisión (sí/no) usando un umbral.

### ¿Por qué no usar regresión lineal para clasificar?

Podrías pensar: "si `survived` es 0 o 1, ¿por qué no le hago un `LinearRegression` normal?". El problema es que una recta:

- Puede predecir valores como -0.3 o 1.8, que **no tienen sentido** como probabilidad.
- No está diseñada para "aplastar" sus salidas dentro del rango [0, 1].

La regresión logística resuelve esto pasando la recta por una función especial, la **función sigmoide**, que veremos en la Fase 6.

El ciclo de las 9 fases del proyecto de Ciencia de Datos **es exactamente el mismo** que en la guía de regresión lineal — lo único que cambia es el algoritmo (Fase 6) y cómo evaluamos el modelo (Fase 7). Si no lo has leído, es buena idea revisar primero el [ciclo completo del proyecto](guia_data_science_regresion_lineal.md#el-ciclo-de-un-proyecto-de-ciencia-de-datos) antes de continuar.

---

## Fase 1 – Planteamiento del problema

### ¿Qué se hace en esta fase?

Se traduce una **necesidad** en una **pregunta que los datos puedan responder**. Igual que en regresión lineal, es la fase más importante.

### Nuestro caso de estudio

Vamos a trabajar con el dataset histórico del **Titanic**: 891 pasajeros con su información personal (edad, sexo, clase del boleto, tarifa pagada...) y si **sobrevivieron o no** al naufragio. Es el "hola mundo" de la clasificación en Ciencia de Datos, porque tiene variables numéricas, categóricas, valores faltantes y una pregunta binaria muy clara.

Imagina que eres analista de datos y te piden entender **qué factores estaban asociados a sobrevivir**, y construir un modelo capaz de **estimar la probabilidad de supervivencia** de un pasajero a partir de su perfil.

### De la necesidad a la pregunta

| Necesidad | Pregunta de datos | Tipo de problema |
|---|---|---|
| "Quiero saber si un pasajero sobrevivió" | ¿Podemos **predecir si sobrevive o no** a partir de su perfil? | Clasificación binaria |
| "Quiero saber qué factores importan" | ¿Qué variables están **más asociadas** a la supervivencia? | Análisis descriptivo |

### Elementos que siempre debes definir

- **Variable objetivo (y):** `survived` → 1 si sobrevivió, 0 si no.
- **Variables predictoras (X):** clase del boleto, sexo, edad, número de familiares a bordo, tarifa pagada, puerto de embarque.
- **Métrica de éxito:** que el modelo clasifique claramente mejor que el **baseline** (predecir siempre "no sobrevivió", que acierta el 61.6% de las veces porque es la clase mayoritaria).
- **Restricciones:** son datos históricos de un solo naufragio; no se pueden generalizar a "quién sobrevive a un accidente" en general.

> **Pregunta para reflexionar:** ¿por qué este es un problema de **clasificación** y no de **regresión**?
> Porque la respuesta es una **categoría** (sobrevivió / no sobrevivió), no un número continuo.

---

## Fase 2 – Obtención y entendimiento de los datos

### ¿Qué se hace en esta fase?

Se consiguen los datos y se entiende **qué significa cada columna**.

### Cargar los datos

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
print(df.shape)     # (filas, columnas)
print(df.head())    # primeras 5 filas
```

**Resultado:**

```
(891, 15)

   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third
1         1       1  female  38.0      1      0  71.2833        C  First
2         1       3  female  26.0      0      0   7.9250        S  Third
3         1       1  female  35.0      1      0  53.1000        S  First
4         0       3    male  35.0      0      0   8.0500        S  Third
```

### Diccionario de datos

| Columna | Descripción | Tipo |
|---|---|---|
| `survived` | ¿Sobrevivió? (1 = sí, 0 = no) (**variable objetivo**) | Categórica binaria |
| `pclass` | Clase del boleto (1 = primera, 2 = segunda, 3 = tercera) | Categórica ordinal |
| `sex` | Sexo del pasajero (`male`, `female`) | Categórica |
| `age` | Edad en años | Numérica continua |
| `sibsp` | Hermanos/cónyuges a bordo | Numérica discreta |
| `parch` | Padres/hijos a bordo | Numérica discreta |
| `fare` | Tarifa pagada (libras) | Numérica continua |
| `embarked` | Puerto de embarque (`C`, `Q`, `S`) | Categórica |

> Usaremos solo estas 8 columnas más `survived`. El dataset trae otras (`deck`, `alive`, `class`, `who`...) que son redundantes o casi todo valores faltantes, y las descartamos desde ya para simplificar.

### Revisar tipos y balance de la variable objetivo

```python
print(df["survived"].value_counts(normalize=True).round(3))
```

```
0    0.616
1    0.384
```

> **Concepto:** el **61.6% no sobrevivió** y el **38.4% sí**. Esto es un **desbalance de clases moderado**: no es 50/50, así que un modelo perezoso que siempre diga "no sobrevivió" ya acertaría un 61.6% de las veces. Lo tendremos en cuenta en la Fase 7.

---

## Fase 3 – Limpieza de datos

### ¿Qué se hace en esta fase?

Se detectan y corrigen valores faltantes, duplicados y valores imposibles.

### 3.1 Valores faltantes

```python
print(df[["survived","pclass","sex","age","sibsp","parch","fare","embarked"]].isnull().sum())
```

```
survived      0
pclass        0
sex           0
age         177
sibsp         0
parch         0
fare          0
embarked      2
```

- **`age`: 177 faltantes (19.9%).** Demasiados para eliminar las filas sin perder mucha información. La estrategia más simple es **imputar con la mediana** (28 años), que es más robusta a valores extremos que el promedio.
- **`embarked`: 2 faltantes.** Con tan pocos, imputamos con el valor **más frecuente** (la moda, puerto "S" = Southampton).

```python
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
```

> **Regla práctica:** cuando faltan **pocos** valores, imputa con la moda/mediana o elimina las filas. Cuando faltan **muchos** (como en la columna `deck`, con 688 de 891 faltantes = 77%), normalmente es mejor **descartar la columna completa**: no hay suficiente información real para inventar el resto de forma confiable.

### 3.2 Duplicados

```python
print(df.duplicated().sum())   # → 107
```

Aparecen **107 filas "duplicadas"**. Pero antes de borrarlas, piensa: el dataset no tiene un identificador único de pasajero, y es perfectamente posible que dos pasajeros compartan clase, sexo, edad, tarifa y puerto por coincidencia (por ejemplo, dos hombres de 24 años en tercera clase que pagaron la misma tarifa). Como en la guía anterior, **no asumimos que sea un error** y las conservamos, documentando la decisión.

### 3.3 Valores imposibles

```python
print(df[["age","fare","pclass"]].describe().round(2))
```

```
          age    fare  pclass
count  891.00  891.00  891.00
mean    29.36   32.20    2.31
std     13.02   49.69    0.84
min      0.42    0.00    1.00
25%     22.00    7.91    2.00
50%     28.00   14.45    3.00
max     80.00  512.33    3.00
```

Edades entre 0.42 (un bebé) y 80 años, tarifas entre 0 y 512 libras: todo tiene sentido histórico (algunos boletos de tripulación o cortesía cuestan $0). No hay valores imposibles.

---

## Fase 4 – Análisis exploratorio (EDA)

### ¿Qué se hace en esta fase?

Se exploran los datos para encontrar patrones relacionados con la variable objetivo. En clasificación, la pregunta que más se repite es: **¿esta variable separa bien las dos categorías?**

### 4.1 Supervivencia por sexo

```python
print(df.groupby("sex")["survived"].mean().round(3))
```

```
sex
female    0.742
male      0.189
```

**Hallazgo:** el **74.2%** de las mujeres sobrevivió, contra solo **18.9%** de los hombres. Es la variable con la relación más fuerte de todo el dataset — la política de evacuación "mujeres y niños primero" queda clarísima en los datos.

### 4.2 Supervivencia por clase del boleto

```python
print(df.groupby("pclass")["survived"].mean().round(3))
```

```
pclass
1    0.630
2    0.473
3    0.242
```

**Hallazgo:** a mejor clase, mayor supervivencia (**63% en primera** vs. **24.2% en tercera**). Probablemente por la ubicación de los camarotes respecto a los botes salvavidas, y por el acceso preferente durante la evacuación.

### 4.3 Distribución de la edad

```python
import matplotlib.pyplot as plt

sns.histplot(data=df, x="age", hue="survived", bins=30, multiple="stack")
plt.title("Edad vs. supervivencia")
plt.show()
```

**Hallazgo:** hay un pico de supervivencia entre los más pequeños (niños), pero la edad por sí sola separa poco a las dos clases — es una relación más débil que sexo o clase.

### 4.4 Correlación con la variable objetivo

```python
print(df[["survived","pclass","age","sibsp","parch","fare"]].corr()["survived"].round(2))
```

```
survived    1.00
pclass     -0.34
age        -0.08
sibsp      -0.04
parch       0.08
fare        0.26
```

| Variable | Correlación con `survived` | Interpretación |
|---|---|---|
| `pclass` | **-0.34** | A mayor número de clase (peor clase), menor supervivencia |
| `fare` | **0.26** | A mayor tarifa pagada, mayor supervivencia (relacionado con la clase) |
| `age`, `sibsp`, `parch` | cerca de 0 | Relación lineal débil por sí solas |

> **Ojo:** la correlación aquí solo mide relación **lineal** con una variable 0/1, y no captura relaciones como la de `sex` (categórica). Por eso el `groupby` de las secciones 4.1 y 4.2 fue tan revelador — en clasificación, agrupar por categoría suele contar más de la historia que la correlación numérica.

> **Pregunta crítica:** ¿la tarifa (`fare`) tiene un efecto propio en la supervivencia, o solo es un indicador indirecto de la clase (`pclass`)? Casi seguro es lo segundo: quien pagó más, viajaba en mejor clase. Esto es la misma **multicolinealidad** que vimos con `total_bill` y `size` en la guía de regresión lineal.

---

## Fase 5 – Preparación de datos para el modelo

### ¿Qué se hace en esta fase?

Se transforman los datos al formato numérico que el algoritmo necesita, y se separan en entrenamiento y prueba.

### 5.1 Variables categóricas → números

`sex` y `embarked` son texto. Igual que en la guía de regresión lineal, usamos **One-Hot Encoding**:

```python
df_modelo = df[["survived","pclass","sex","age","sibsp","parch","fare","embarked"]].copy()

X = pd.get_dummies(
    df_modelo.drop(columns="survived"),
    columns=["sex", "embarked"],
    drop_first=True,
)
y = df_modelo["survived"]

print(X.columns.tolist())
```

```
['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male', 'embarked_Q', 'embarked_S']
```

`sex` se convirtió en `sex_male` (1 si es hombre, 0 si es mujer), y `embarked` en dos columnas (`embarked_Q`, `embarked_S`), dejando "C" (Cherbourg) como categoría base.

### 5.2 Separar en entrenamiento y prueba

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(len(X_train), len(X_test))   # → 712 y 179
```

- **Entrenamiento (80% = 712 filas):** el modelo aprende de aquí.
- **Prueba (20% = 179 filas):** con esto lo evaluamos.
- **`stratify=y`:** a diferencia de la regresión lineal, en clasificación conviene asegurarte de que el **porcentaje de cada categoría se mantenga igual** en entrenamiento y prueba (61.6%/38.4% en ambos). Sin `stratify`, una partición aleatoria mala podría dejar un test set con muy pocos sobrevivientes y distorsionar la evaluación.

> **Nota:** en un proyecto real, la mediana de `age` con la que imputamos en la Fase 3 debería calcularse **solo con los datos de entrenamiento** y aplicarse igual al test, para evitar que información del test se filtre al entrenamiento (esto se llama **data leakage**). Aquí lo simplificamos calculándola sobre todo el dataset; en el ejercicio 6 lo vas a corregir.

---

## Fase 6 – Modelado con Regresión Logística

### ¿Qué se hace en esta fase?

**Aquí entra el Machine Learning.** Entrenamos un algoritmo que aprende a estimar la **probabilidad** de que `survived = 1`.

### 6.1 La función sigmoide

La regresión logística empieza igual que la lineal, calculando una combinación de las variables de entrada:

$$
z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
$$

Pero en vez de usar $z$ directamente como predicción (que podría ser cualquier número), lo pasa por la **función sigmoide**, que lo "aplasta" entre 0 y 1:

$$
\hat{p} = \frac{1}{1 + e^{-z}}
$$

| Valor de $z$ | $\hat{p}$ (probabilidad) |
|---|---|
| Muy negativo (-10) | Cerca de 0 |
| 0 | Exactamente 0.5 |
| Muy positivo (+10) | Cerca de 1 |

$\hat{p}$ es la **probabilidad estimada** de que el pasajero sobreviva. Para convertirla en una decisión sí/no, se aplica un **umbral** (por defecto 0.5):

$$
\text{predicción} = \begin{cases} 1 \text{ (sobrevive)} & \text{si } \hat{p} \geq 0.5 \\ 0 \text{ (no sobrevive)} & \text{si } \hat{p} < 0.5 \end{cases}
$$

### 6.2 ¿Cómo aprende el modelo?

En regresión lineal, el modelo minimizaba la suma de errores al cuadrado. En regresión logística, minimiza una métrica distinta llamada **log-loss** (o *entropía cruzada*), que penaliza mucho más fuerte estar **muy seguro y equivocado** (por ejemplo, predecir $\hat{p}=0.99$ para un pasajero que no sobrevivió) que estar dudoso y equivocado ($\hat{p}=0.51$).

El proceso de búsqueda de los mejores $\theta$ ya no tiene una fórmula cerrada como en mínimos cuadrados: se hace de forma iterativa con un algoritmo de optimización (por defecto, scikit-learn usa uno llamado `lbfgs`). No necesitas programarlo tú: `scikit-learn` lo hace internamente.

### 6.3 Entrenar el modelo

```python
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)      # ← aquí ocurre el aprendizaje

print("Intercepto (θ₀):", modelo.intercept_.round(3))
print("Coeficientes:", dict(zip(X.columns, modelo.coef_[0].round(3))))
```

```
Intercepto (θ₀): [4.981]
Coeficientes: {'pclass': -1.09, 'age': -0.038, 'sibsp': -0.244, 'parch': -0.071,
               'fare': 0.002, 'sex_male': -2.556, 'embarked_Q': 0.278, 'embarked_S': -0.382}
```

### 6.4 Interpretar los coeficientes: odds ratio

En regresión lineal, un coeficiente se leía directo ("por cada dólar más, sube $0.107"). En regresión logística, los coeficientes están en la escala del **log-odds**, así que para interpretarlos en términos de probabilidad se usa el **odds ratio**: $e^{\theta}$.

```python
import numpy as np
print("Odds ratio sex_male:", round(np.exp(modelo.coef_[0][5]), 3))   # → 0.078
print("Odds ratio pclass:  ", round(np.exp(modelo.coef_[0][0]), 3))   # → 0.336
```

| Variable | Coeficiente | Odds ratio | Interpretación |
|---|---|---|---|
| `sex_male` | -2.556 | **0.078** | Ser hombre multiplica las probabilidades (*odds*) de sobrevivir por 0.078 frente a ser mujer, manteniendo lo demás igual → **caen drásticamente**. |
| `pclass` | -1.090 | **0.336** | Por cada clase peor (de 1ª a 2ª, o de 2ª a 3ª), las probabilidades de sobrevivir se multiplican por 0.336 → **se reducen a un tercio**. |

> **Regla práctica:** odds ratio > 1 → la variable **aumenta** la probabilidad de la clase positiva. Odds ratio < 1 → la **disminuye**. Odds ratio = 1 → no tiene efecto.

### 6.5 Hacer predicciones

```python
nuevos_pasajeros = pd.DataFrame({
    "pclass": [1, 3],
    "age": [28, 28],
    "sibsp": [0, 0],
    "parch": [0, 0],
    "fare": [80, 8],
    "sex_male": [0, 1],
    "embarked_Q": [0, 0],
    "embarked_S": [1, 1],
})

print(modelo.predict(nuevos_pasajeros))                       # → [1 0]  (clase predicha)
print(modelo.predict_proba(nuevos_pasajeros)[:, 1].round(3))  # → [0.905 0.077] (probabilidad de sobrevivir)
```

| Perfil | Probabilidad de sobrevivir | Predicción |
|---|---|---|
| Mujer, 1ª clase, tarifa alta | **90.5%** | Sobrevive |
| Hombre, 3ª clase, tarifa baja | **7.7%** | No sobrevive |

> **Diferencia clave con regresión lineal:** `.predict()` te da la **clase** (0 o 1), pero `.predict_proba()` te da la **probabilidad**. Casi siempre vale la pena mirar la probabilidad, no solo la clase — no es lo mismo un 51% que un 99%, aunque ambos se redondeen a "sobrevive".

---

## Fase 7 – Evaluación del modelo

### ¿Qué se hace en esta fase?

Se mide qué tan bueno es el modelo con datos que nunca vio. **En clasificación, "accuracy" no basta por sí sola** — hay que mirar varias métricas a la vez.

### 7.1 Siempre compara contra un baseline

```python
import numpy as np
from sklearn.metrics import accuracy_score

pred_base = np.zeros(len(y_test))   # predecir siempre "no sobrevivió"
print(accuracy_score(y_test, pred_base))   # → 0.615
```

Si siempre dijéramos "no sobrevivió", acertaríamos el **61.5%** de las veces — porque esa es la clase mayoritaria. Cualquier modelo que reporte una accuracy cercana a esa debe encender una alarma.

### 7.2 La matriz de confusión

```python
from sklearn.metrics import confusion_matrix

pred = modelo.predict(X_test)
print(confusion_matrix(y_test, pred))
```

```
[[98 12]
 [23 46]]
```

|  | Predicho: No sobrevive | Predicho: Sobrevive |
|---|---|---|
| **Real: No sobrevive** | 98 (✅ Verdadero Negativo) | 12 (❌ Falso Positivo) |
| **Real: Sobrevive** | 23 (❌ Falso Negativo) | 46 (✅ Verdadero Positivo) |

- **Verdaderos Positivos (VP) = 46:** dijimos "sobrevive" y sobrevivió.
- **Verdaderos Negativos (VN) = 98:** dijimos "no sobrevive" y no sobrevivió.
- **Falsos Positivos (FP) = 12:** dijimos "sobrevive" pero no sobrevivió.
- **Falsos Negativos (FN) = 23:** dijimos "no sobrevive" pero sí sobrevivió — el modelo se le escaparon 23 sobrevivientes reales.

### 7.3 Métricas de clasificación

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

proba = modelo.predict_proba(X_test)[:, 1]

print("Accuracy: ", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall:   ", round(recall_score(y_test, pred), 3))
print("F1:       ", round(f1_score(y_test, pred), 3))
print("ROC-AUC:  ", round(roc_auc_score(y_test, proba), 3))
```

```
Accuracy:  0.804
Precision: 0.793
Recall:    0.667
F1:        0.724
ROC-AUC:   0.844
```

| Métrica | Qué mide | Nuestro resultado | Interpretación |
|---|---|---|---|
| **Accuracy** | % total de aciertos | **80.4%** | Acierta 4 de cada 5 pasajeros — bastante mejor que el 61.5% del baseline. |
| **Precision** | De los que predije "sobrevive", ¿cuántos sí sobrevivieron? | **79.3%** | Cuando el modelo dice "sobrevive", acierta casi 4 de cada 5 veces. |
| **Recall** | De los que **sí** sobrevivieron, ¿cuántos detecté? | **66.7%** | El modelo se pierde 1 de cada 3 sobrevivientes reales (los 23 falsos negativos). |
| **F1** | Balance entre precision y recall | **0.724** | Resumen único cuando te importan ambas cosas por igual. |
| **ROC-AUC** | Qué tan bien separa las dos clases el modelo, para *cualquier* umbral | **0.844** | 0.5 sería "adivinar al azar", 1.0 sería perfecto — 0.844 es una separación buena. |

> **Conclusión:** el modelo reduce el error de forma clara frente al baseline (80.4% vs. 61.5% de accuracy) y separa bien las dos clases (ROC-AUC 0.844). ✅ Cumple la métrica de éxito de la Fase 1.

### 7.4 ¿Por qué "accuracy" no siempre basta?

Imagina un dataset de **fraude bancario** donde solo el 1% de las transacciones son fraude. Un modelo que **siempre** dice "no es fraude" tendría **99% de accuracy** — y sería completamente inútil, porque nunca detecta ni un solo fraude (recall = 0%).

> **Regla práctica:** cuando las clases están desbalanceadas (como el fraude, o el cáncer, o el churn), mira siempre **precision, recall y F1** además de accuracy. Y pregúntate: ¿qué error es más costoso, un falso positivo o un falso negativo? En un diagnóstico médico, un falso negativo (decir "sano" a alguien enfermo) suele ser mucho peor que un falso positivo.

### 7.5 Ajustar el umbral de decisión

Por defecto el umbral es 0.5, pero no tiene que serlo. Si quieres **detectar más sobrevivientes** (subir el recall) a costa de equivocarte más seguido con falsos positivos, puedes bajar el umbral:

```python
umbral = 0.35
pred_umbral = (proba >= umbral).astype(int)

print("Recall con umbral 0.35:", round(recall_score(y_test, pred_umbral), 3))
print("Precision con umbral 0.35:", round(precision_score(y_test, pred_umbral), 3))
```

> **Lección clave:** no existe un modelo "perfecto" en clasificación — hay un **balance** entre precision y recall que depende de qué error le cuesta más caro a tu negocio. Esa decisión no la toma el algoritmo, la tomas tú (o el negocio).

### 7.6 Validación cruzada

Igual que en regresión lineal, con un test set de 179 filas los resultados pueden variar según la partición. Es buena práctica validar con **validación cruzada**:

```python
from sklearn.model_selection import cross_val_score

puntajes = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring="roc_auc")
print(puntajes.round(3), puntajes.mean().round(3))
```

---

## Fase 8 – Interpretación y comunicación de resultados

### ¿Qué se hace en esta fase?

Se traducen los resultados técnicos a un lenguaje que entienda quien toma la decisión.

### Del lenguaje técnico al lenguaje de negocio

| Lo que dice el modelo | Lo que comunicas |
|---|---|
| Odds ratio `sex_male` = 0.078 | "Ser hombre reducía drásticamente las probabilidades de sobrevivir frente a ser mujer, manteniendo igual la clase y la edad — consistente con el protocolo de 'mujeres y niños primero'." |
| Odds ratio `pclass` = 0.336 | "Viajar en una clase peor reducía las probabilidades de sobrevivir a cerca de un tercio por cada nivel de clase." |
| Accuracy = 80.4% | "El modelo acierta en 4 de cada 5 pasajeros al predecir si sobrevivió o no." |
| Recall = 66.7% | "De cada 3 sobrevivientes reales, el modelo identifica correctamente a 2." |
| ROC-AUC = 0.844 | "El modelo distingue bastante bien entre quién sobrevivió y quién no, mejor que adivinar al azar." |

### Estructura recomendada para un informe

1. **Pregunta:** ¿qué queríamos responder?
2. **Respuesta corta:** en una o dos frases.
3. **Evidencia:** la matriz de confusión y/o la curva ROC.
4. **Recomendaciones:** acciones concretas (o, en un caso histórico como este, conclusiones).
5. **Limitaciones:** qué no podemos afirmar.

### Limitaciones que debes comunicar siempre

- Es un solo evento histórico (891 pasajeros de un naufragio); no generaliza a "supervivencia en accidentes" en general.
- Correlación no implica causalidad: el modelo describe **asociaciones** del histórico, no *garantiza* que cambiar el sexo o la clase de una persona cambiaría su destino de forma causal.
- Faltan variables que probablemente importaban mucho (ubicación exacta del camarote, si sabía nadar, en qué momento se enteró de la emergencia) y que no están en los datos.

### Gráfica final para el informe: la curva ROC

```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(modelo, X_test, y_test)
plt.title("Curva ROC — Predicción de supervivencia en el Titanic")
plt.show()
```

---

## Fase 9 – Despliegue y mejora continua

### ¿Qué se hace en esta fase?

Se pone el modelo a funcionar en el mundo real y se vigila que siga funcionando bien.

### Formas de desplegar un modelo

- **Guardarlo en un archivo:**
  ```python
  import joblib
  joblib.dump(modelo, "modelo_titanic.pkl")
  modelo_cargado = joblib.load("modelo_titanic.pkl")
  ```
- **Exponerlo en una API** (FastAPI, Flask) para que otro sistema consulte la probabilidad en tiempo real.
- **Integrarlo en un dashboard** de riesgo.

### Mejora continua

- **Recolectar más variables** (en un caso real de negocio: historial del cliente, comportamiento reciente...).
- **Reentrenar** el modelo periódicamente con datos nuevos.
- **Monitorear el desbalance de clases**: si el porcentaje de la clase positiva cambia con el tiempo (por ejemplo, la tasa de fraude sube), el modelo puede necesitar recalibrarse. Es otra forma de **deriva de datos (data drift)**.
- **Volver a la Fase 1** si la pregunta cambia: ¿y si en vez de sí/no quisiéramos predecir **cuánto tiempo** sobrevivió alguien en el agua? Eso ya sería otro tipo de problema (regresión, o incluso análisis de supervivencia).

---

## 10. Código completo

```python
# ============================================
# PROYECTO: Predicción de supervivencia — Titanic
# ============================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay,
)

# --- Fase 2: Obtención de datos ---
df = sns.load_dataset("titanic")
print(df.shape)
print(df["survived"].value_counts(normalize=True).round(3))

# --- Fase 3: Limpieza ---
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
print("Duplicados:", df.duplicated().sum())
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# --- Fase 4: Análisis exploratorio ---
print(df.groupby("sex")["survived"].mean().round(3))
print(df.groupby("pclass")["survived"].mean().round(3))
print(df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].corr()["survived"].round(2))

# --- Fase 5: Preparación ---
X = pd.get_dummies(df.drop(columns="survived"), columns=["sex", "embarked"], drop_first=True)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Fase 6: Modelado ---
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)
print("Coeficientes:", dict(zip(X.columns, modelo.coef_[0].round(3))))
print("Odds ratios:", dict(zip(X.columns, np.exp(modelo.coef_[0]).round(3))))

# --- Fase 7: Evaluación ---
pred = modelo.predict(X_test)
proba = modelo.predict_proba(X_test)[:, 1]

print("Baseline accuracy:", round(accuracy_score(y_test, np.zeros(len(y_test))), 3))
print("Accuracy: ", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall:   ", round(recall_score(y_test, pred), 3))
print("F1:       ", round(f1_score(y_test, pred), 3))
print("ROC-AUC:  ", round(roc_auc_score(y_test, proba), 3))
print(confusion_matrix(y_test, pred))

puntajes = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring="roc_auc")
print("ROC-AUC (cross-val):", puntajes.round(3), puntajes.mean().round(3))

# --- Fase 8: Comunicación ---
RocCurveDisplay.from_estimator(modelo, X_test, y_test)
plt.title("Curva ROC — Predicción de supervivencia en el Titanic")
plt.show()

# Predicción para nuevos pasajeros
nuevos = pd.DataFrame({
    "pclass": [1, 3], "age": [28, 28], "sibsp": [0, 0], "parch": [0, 0],
    "fare": [80, 8], "sex_male": [0, 1], "embarked_Q": [0, 0], "embarked_S": [1, 1],
})
print(modelo.predict_proba(nuevos)[:, 1].round(3))
```

---

## 11. Ejercicios para el estudiante

### Nivel básico

1. Cambia el umbral de decisión de 0.5 a 0.3 y a 0.7. ¿Cómo cambian precision y recall en cada caso? ¿Cuál usarías si el "negocio" fuera un sistema de rescate que prefiere equivocarse de más antes que dejar a alguien sin ayuda?
2. Entrena un modelo usando **solo** `sex_male` y `pclass` como variables. Compara su accuracy y ROC-AUC con el modelo completo. ¿Cuánto se pierde por usar menos variables?
3. Calcula la matriz de confusión y explica con tus palabras qué significan sus 4 números para este problema específico.

### Nivel intermedio

4. Agrega la variable `age` en categorías (`niño` si `age < 12`, `adulto` en otro caso) en vez de usarla como número continuo. ¿Mejora el recall?
5. Corrige el **data leakage** de la Fase 5: calcula la mediana de `age` **solo con `X_train`**, y úsala para imputar tanto `X_train` como `X_test`. ¿Cambian mucho las métricas?
6. Usa `cross_val_score` con `scoring="f1"` y `scoring="roc_auc"` para comparar el modelo simple del ejercicio 2 contra el modelo completo. ¿Se sostiene la conclusión?

### Nivel avanzado

7. Compara la regresión logística contra un `RandomForestClassifier` o un `DecisionTreeClassifier` de scikit-learn, con las mismas variables. ¿Cuál da mejor ROC-AUC? ¿Cuál es más fácil de interpretar para explicarle a alguien de negocio?
8. Retoma el dataset de propinas (`tips`) de la guía anterior: crea una variable `propina_alta = tip > tip.median()` y entrena una `LogisticRegression` para predecirla. Evalúala con las mismas métricas de esta guía.
9. **Proyecto propio:** busca un dataset de clasificación binaria de tu interés (por ejemplo, en [Kaggle](https://www.kaggle.com/datasets)) y recorre las **9 fases** de esta guía, incluyendo matriz de confusión, curva ROC y ajuste de umbral. Entrega un informe de máximo 2 páginas siguiendo la estructura de la Fase 8.

---

## 12. Glosario

| Término | Definición sencilla |
|---|---|
| **Clasificación** | Problema donde se predice una categoría en vez de un número. |
| **Clasificación binaria** | Clasificación donde solo hay dos categorías posibles (por ejemplo, 0/1). |
| **Regresión logística** | Algoritmo que estima la probabilidad de pertenecer a una clase, pasando una combinación lineal de las variables por la función sigmoide. |
| **Función sigmoide** | Función que convierte cualquier número real en un valor entre 0 y 1. |
| **Log-odds (logit)** | La escala en la que "vive" la combinación lineal $z$ antes de pasar por la sigmoide. |
| **Odds ratio** | $e^{\theta}$: cuánto se multiplican las probabilidades (*odds*) de la clase positiva por cada unidad que sube una variable. |
| **Umbral de decisión** | Punto de corte (por defecto 0.5) sobre la probabilidad estimada, que decide si se predice la clase 0 o la 1. |
| **Matriz de confusión** | Tabla 2×2 que cruza la clase real contra la clase predicha (VP, VN, FP, FN). |
| **Verdadero/Falso Positivo/Negativo (VP, VN, FP, FN)** | Las 4 combinaciones posibles entre lo real y lo predicho en una matriz de confusión. |
| **Accuracy** | Porcentaje total de predicciones correctas. |
| **Precision** | De lo que el modelo predijo como positivo, qué porcentaje era realmente positivo. |
| **Recall (sensibilidad)** | De todo lo que realmente era positivo, qué porcentaje detectó el modelo. |
| **F1-score** | Media armónica entre precision y recall; resume ambas en un solo número. |
| **Curva ROC** | Gráfica que muestra el balance entre verdaderos positivos y falsos positivos para todos los umbrales posibles. |
| **ROC-AUC** | Área bajo la curva ROC (0.5 = azar, 1.0 = perfecto); mide qué tan bien separa el modelo las dos clases, sin depender de un umbral fijo. |
| **Desbalance de clases** | Cuando una categoría es mucho más frecuente que la otra en los datos. |
| **`stratify`** | Opción de `train_test_split` que mantiene la misma proporción de clases en entrenamiento y prueba. |
| **Data leakage** | Cuando información del conjunto de prueba se filtra (a propósito o por error) al proceso de entrenamiento, inflando las métricas de forma poco realista. |
| **Baseline** | Modelo muy simple (por ejemplo, predecir siempre la clase mayoritaria) usado como punto de comparación. |

---

## 13. Lista de verificación final

Antes de entregar cualquier proyecto de clasificación, verifica:

- [ ] ¿Definí claramente la **pregunta de negocio** y la **variable objetivo binaria**?
- [ ] ¿Revisé qué tan **desbalanceadas** están las clases?
- [ ] ¿Revisé **faltantes, duplicados y valores imposibles**, y documenté mis decisiones?
- [ ] ¿Hice un **análisis exploratorio** comparando la variable objetivo contra las predictoras (con `groupby`, no solo correlación)?
- [ ] ¿Separé entrenamiento y prueba con **`stratify`**?
- [ ] ¿Comparé mi modelo contra un **baseline** (predecir siempre la clase mayoritaria)?
- [ ] ¿Reporté **accuracy, precision, recall, F1 y ROC-AUC**, no solo accuracy?
- [ ] ¿Miré la **matriz de confusión** y entendí qué tipo de error (falso positivo o falso negativo) es más costoso para el negocio?
- [ ] ¿Consideré si el **umbral de 0.5** es el adecuado para este problema?
- [ ] ¿Traduje los coeficientes a **odds ratios** en lenguaje de negocio?
- [ ] ¿Comuniqué las **limitaciones** del análisis?

---

> **Recuerda:** en clasificación, un modelo con 95% de accuracy puede ser pésimo, y uno con 75% puede ser excelente — todo depende de qué tan desbalanceadas estén las clases y qué error le cuesta más caro a quien toma la decisión. Mirar una sola métrica es la forma más fácil de engañarte a ti mismo.
