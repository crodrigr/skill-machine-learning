# Guía paso a paso: Evaluación de resultados en Machine Learning
## Basada en el notebook `10_Evaluación de resultados.ipynb`

> **Objetivo:** comprender, paso a paso, cómo se entrena un modelo de Machine Learning, cómo se realizan predicciones y, sobre todo, cómo se evalúa si esas predicciones son buenas.
>
> Esta guía está construida directamente a partir del notebook **`10_Evaluación de resultados.ipynb`**, cuyo caso práctico utiliza el conjunto de datos **NSL-KDD** para detección de intrusiones en redes.

---

# 1. ¿Qué vamos a aprender?

El notebook no se centra solamente en entrenar un algoritmo.

Su objetivo principal es aprender a responder:

> **¿Cómo sabemos si un modelo de Machine Learning está funcionando bien?**

Para responder esta pregunta aprenderemos:

1. Qué es un problema de clasificación.
2. Qué son las características (`X`) y la etiqueta (`y`).
3. Cómo dividir los datos en entrenamiento, validación y prueba.
4. Cómo preparar variables numéricas y categóricas.
5. Qué es un pipeline de procesamiento.
6. Cómo entrenar una **Regresión Logística**.
7. Cómo realizar predicciones.
8. Qué es una **matriz de confusión**.
9. Qué significan **Precision, Recall y F1-score**.
10. Qué son las curvas **ROC** y **Precision-Recall**.
11. Por qué debemos utilizar un conjunto de prueba separado.
12. Cómo interpretar los resultados de un modelo.

Al final aplicaremos todo esto al caso real del notebook: **detectar anomalías o ataques en tráfico de red**.

---

# 2. Antes de comenzar: ¿qué es Machine Learning supervisado?

En Machine Learning supervisado tenemos datos históricos donde conocemos la respuesta correcta.

Por ejemplo:

| Duración | Protocolo | Bytes | Clase |
|---:|---|---:|---|
| 10 | TCP | 500 | normal |
| 20 | TCP | 900 | normal |
| 5 | UDP | 3000 | anomaly |
| 3 | ICMP | 5000 | anomaly |

Queremos que el modelo aprenda:

```text
Características
      ↓
Modelo
      ↓
Predicción
```

Por ejemplo:

```text
duración = 4
protocolo = UDP
bytes = 4000

        ↓

Modelo

        ↓

anomaly
```

La diferencia fundamental es que durante el entrenamiento conocemos la respuesta correcta.

---

# 3. Primer ejemplo sencillo

Antes de estudiar el NSL-KDD, vamos a utilizar un ejemplo pequeño para comprender el proceso.

Supongamos que queremos saber si un estudiante **aprueba o no aprueba**.

Tenemos:

| Horas de estudio | Asistencia | Resultado |
|---:|---:|---|
| 1 | 60 | no |
| 2 | 65 | no |
| 2 | 70 | no |
| 3 | 75 | no |
| 5 | 85 | sí |
| 6 | 90 | sí |
| 7 | 95 | sí |

Tenemos:

```text
X = características
y = resultado
```

En Python:

```python
X = [
    [1, 60],
    [2, 65],
    [2, 70],
    [3, 75],
    [5, 85],
    [6, 90],
    [7, 95]
]

y = [
    0,
    0,
    0,
    0,
    1,
    1,
    1
]
```

Donde:

```text
0 = no aprobado
1 = aprobado
```

---

# 4. ¿Qué representa X?

`X` contiene las variables que utilizamos para hacer la predicción.

En nuestro ejemplo:

```text
X
│
├── horas de estudio
└── asistencia
```

Estas variables también se llaman:

- características;
- variables predictoras;
- features.

Por ejemplo:

```python
[5, 85]
```

significa:

```text
5 horas de estudio
85% de asistencia
```

---

# 5. ¿Qué representa y?

`y` contiene la respuesta que queremos aprender a predecir.

```text
0 → no aprobado
1 → aprobado
```

Por eso:

```python
X[4] = [5, 85]
y[4] = 1
```

El quinto estudiante tiene:

```text
5 horas de estudio
85% de asistencia
aprobado
```

---

# 6. Dividir los datos

No debemos entrenar y evaluar con exactamente los mismos datos.

Queremos separar:

```text
DATOS
  │
  ├──────── TRAIN
  │
  └──────── TEST
```

El conjunto **TRAIN** sirve para aprender.

El conjunto **TEST** sirve para comprobar cómo funciona el modelo con datos que no utilizó durante el aprendizaje.

En el notebook real utilizaremos tres conjuntos:

```text
TRAIN
VALIDATION
TEST
```

---

# 7. ¿Por qué tres conjuntos?

Esta es una idea muy importante.

## Training Set

Se utiliza para:

> **Entrenar el modelo.**

## Validation Set

Se utiliza para:

> **Evaluar y ajustar decisiones durante el desarrollo del modelo.**

## Test Set

Se utiliza para:

> **Realizar la evaluación final con datos que se mantienen separados.**

Visualmente:

```text
                 DATOS
                   │
          ┌────────┴────────┐
          ↓                 ↓
        TRAIN          VALIDATION + TEST
                           │
                    ┌──────┴──────┐
                    ↓             ↓
               VALIDATION       TEST
```

---

# 8. Entrenar un modelo sencillo

Podríamos utilizar Regresión Logística:

```python
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()

modelo.fit(X_train, y_train)
```

La instrucción:

```python
fit()
```

significa:

> **Aprende los patrones de los datos de entrenamiento.**

Después podemos hacer:

```python
predicciones = modelo.predict(X_test)
```

Y obtenemos las predicciones.

---

# 9. Ahora aparece la pregunta realmente importante

Supongamos que obtenemos:

```text
Real:
[0, 0, 1, 1]

Predicción:
[0, 0, 1, 0]
```

El modelo acertó tres de cuatro.

Pero necesitamos medirlo de forma sistemática.

Aquí aparecen las **métricas de evaluación**.

---

# 10. Matriz de confusión

La matriz de confusión permite observar qué ocurrió con las predicciones.

Para un problema binario:

```text
                         PREDICCIÓN
                      0             1

REAL       0         TN            FP

           1         FN            TP
```

Tenemos cuatro posibilidades.

## TN — True Negative

Era:

```text
0
```

y el modelo predijo:

```text
0
```

Correcto.

---

## TP — True Positive

Era:

```text
1
```

y el modelo predijo:

```text
1
```

Correcto.

---

## FP — False Positive

Era:

```text
0
```

pero el modelo predijo:

```text
1
```

Es un falso positivo.

---

## FN — False Negative

Era:

```text
1
```

pero el modelo predijo:

```text
0
```

Es un falso negativo.

---

# 11. ¿Por qué importan tanto FP y FN?

Porque en un problema real los errores no necesariamente cuestan lo mismo.

En nuestro ejemplo de estudiantes:

```text
FP:
El modelo dice "aprobado"
pero realmente no aprobó.
```

Y:

```text
FN:
El modelo dice "no aprobado"
pero realmente aprobó.
```

En otro problema, como detección de ataques informáticos, la interpretación cambia.

Por eso siempre debemos conocer el contexto.

---

# 12. Ahora pasemos al problema real del notebook

El notebook `10_Evaluación de resultados.ipynb` utiliza el conjunto de datos:

# NSL-KDD

NSL-KDD es un conjunto de datos utilizado como referencia para estudiar sistemas de **detección de intrusiones de red**.

El objetivo es utilizar características del tráfico de red para identificar si una conexión es:

```text
normal
```

o:

```text
anomaly
```

En otras palabras:

> Tenemos información de conexiones de red y queremos determinar si corresponden a tráfico normal o a una anomalía.

---

# 13. Los archivos del dataset

El notebook menciona archivos como:

```text
KDDTrain+.ARFF
KDDTrain+.TXT

KDDTest+.ARFF
KDDTest+.TXT

KDDTest-21.ARFF
KDDTest-21.TXT
```

Para este notebook se utiliza:

```text
KDDTrain+.arff
```

---

# 14. ¿Qué significa ARFF?

ARFF significa:

> **Attribute-Relation File Format**

Es un formato utilizado especialmente en trabajos de Machine Learning y originalmente asociado con herramientas como Weka.

El notebook utiliza la librería `arff` para leerlo.

---

# 15. Importar las librerías

El notebook comienza importando:

```python
import arff
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin
```

Vamos a entenderlas.

---

# 16. Pandas

```python
import pandas as pd
```

Pandas nos permite trabajar con tablas de datos.

Por ejemplo:

```text
fila 1
fila 2
fila 3
...
```

Podemos imaginarlo como una hoja de Excel que podemos manipular mediante Python.

---

# 17. NumPy

```python
import numpy as np
```

NumPy proporciona estructuras y operaciones numéricas.

Es una librería fundamental para trabajar con datos numéricos en Python.

---

# 18. Scikit-learn

El notebook utiliza `sklearn` para:

- dividir datos;
- transformar datos;
- entrenar modelos;
- calcular métricas;
- construir pipelines;
- visualizar resultados.

Es una de las librerías más utilizadas para Machine Learning clásico en Python.

---

# 19. Leer el dataset

El notebook crea una función:

```python
def load_kdd_dataset(data_path):
    """Lectura del conjunto de datos NSL-KDD."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)

    attributes = [attr[0] for attr in dataset["attributes"]]

    return pd.DataFrame(
        dataset["data"],
        columns=attributes
    )
```

Vamos a entenderla.

---

# 20. La función `load_kdd_dataset`

La función recibe:

```python
data_path
```

que representa la ruta del archivo.

Por ejemplo:

```text
datasets/NSL-KDD/KDDTrain+.arff
```

Después:

```python
with open(data_path, 'r') as train_set:
```

abre el archivo.

Luego:

```python
dataset = arff.load(train_set)
```

lee el contenido ARFF.

Finalmente se obtienen los nombres de las columnas:

```python
attributes = [attr[0] for attr in dataset["attributes"]]
```

y se crea un DataFrame:

```python
pd.DataFrame(
    dataset["data"],
    columns=attributes
)
```

---

# 21. Cargar los datos

El notebook hace:

```python
df = load_kdd_dataset(
    "datasets/NSL-KDD/KDDTrain+.arff"
)
```

Ahora:

```text
df
```

contiene el dataset completo cargado en memoria.

---

# 22. Inspeccionar los datos

El notebook utiliza:

```python
df.head(10)
```

Esto muestra las primeras diez filas.

¿Por qué hacerlo?

Porque antes de entrenar un modelo debemos conocer nuestros datos.

Queremos saber:

- qué columnas existen;
- qué tipo de variables tenemos;
- cómo están representados los valores;
- cuál es nuestra variable objetivo.

---

# 23. ¿Cuál es la variable objetivo?

En este dataset la columna que queremos predecir es:

```text
class
```

Por eso el notebook hace:

```python
X_df = df.drop("class", axis=1)

y_df = df["class"].copy()
```

Esto es fundamental.

Estamos separando:

```text
X → características de la conexión
y → clase que queremos predecir
```

---

# 24. ¿Qué significa `drop("class", axis=1)`?

Tenemos:

```python
df.drop("class", axis=1)
```

Significa:

> Elimina la columna `class`.

¿Por qué?

Porque `class` es justamente la respuesta que queremos que el modelo aprenda a predecir.

No queremos entregarle la respuesta como una de las características.

---

# 25. La estructura mental correcta

Piensa siempre:

```text
DATASET
   │
   ├──────── X
   │         características
   │
   └──────── y
             respuesta
```

En el notebook:

```text
X → características de tráfico de red

y → class
```

Y `class` representa la clasificación de la conexión.

---

# 26. Dividir el dataset

El notebook crea:

```python
def train_val_test_split(
    df,
    rstate=42,
    shuffle=True,
    stratify=None
):
    strat = df[stratify] if stratify else None

    train_set, test_set = train_test_split(
        df,
        test_size=0.4,
        random_state=rstate,
        shuffle=shuffle,
        stratify=strat
    )

    strat = test_set[stratify] if stratify else None

    val_set, test_set = train_test_split(
        test_set,
        test_size=0.5,
        random_state=rstate,
        shuffle=shuffle,
        stratify=strat
    )

    return (
        train_set,
        val_set,
        test_set
    )
```

Esta función divide los datos en tres grupos.

---

# 27. ¿Cómo se dividen?

Primero:

```text
100% de los datos
        │
        ├── 60% TRAIN
        │
        └── 40% restante
```

Después ese 40% se divide por la mitad:

```text
40%
 │
 ├── 20% VALIDATION
 │
 └── 20% TEST
```

Por tanto, aproximadamente:

```text
TRAIN      = 60%
VALIDATION = 20%
TEST       = 20%
```

Visualmente:

```text
┌───────────────────────────────────────────┐
│                 DATASET                   │
├─────────────────────┬──────────┬──────────┤
│       TRAIN 60%     │ VAL 20%  │ TEST 20% │
└─────────────────────┴──────────┴──────────┘
```

---

# 28. ¿Por qué `random_state=42`?

El notebook utiliza:

```python
rstate=42
```

Esto permite que la división aleatoria sea reproducible.

Es decir, al ejecutar nuevamente el proceso, podemos obtener la misma división.

El número `42` no tiene un significado matemático especial aquí.

Podría utilizarse otro número.

---

# 29. ¿Qué hace `shuffle=True`?

Significa que los registros se mezclan antes de dividirlos.

Esto ayuda a evitar que la división dependa del orden original de los registros.

---

# 30. Crear los conjuntos

El notebook ejecuta:

```python
train_set, val_set, test_set = train_val_test_split(df)
```

Ahora tenemos:

```text
train_set
val_set
test_set
```

Podemos comprobar su tamaño:

```python
print(
    "Longitud del Training Set:",
    len(train_set)
)

print(
    "Longitud del Validation Set:",
    len(val_set)
)

print(
    "Longitud del Test Set:",
    len(test_set)
)
```

---

# 31. Separar X e y para cada conjunto

Para entrenamiento:

```python
X_train = train_set.drop(
    "class",
    axis=1
)

y_train = train_set["class"].copy()
```

Para validación:

```python
X_val = val_set.drop(
    "class",
    axis=1
)

y_val = val_set["class"].copy()
```

Para prueba:

```python
X_test = test_set.drop(
    "class",
    axis=1
)

y_test = test_set["class"].copy()
```

Tenemos:

```text
TRAIN
X_train → características
y_train → respuestas

VALIDATION
X_val → características
y_val → respuestas

TEST
X_test → características
y_test → respuestas
```

---

# 32. Ahora viene un problema real: los datos no están todos en el mismo formato

Una de las dificultades de un dataset real es que podemos tener:

```text
variables numéricas
```

y:

```text
variables categóricas
```

Por ejemplo:

```text
duration → numérica
protocol_type → categórica
service → categórica
flag → categórica
```

No podemos tratar todos estos datos exactamente de la misma manera.

Por eso necesitamos **preprocesamiento**.

---

# 33. ¿Qué es preprocesamiento?

Preprocesar significa preparar los datos antes de entregárselos al algoritmo.

Podemos tener:

```text
DATOS CRUDOS
     ↓
limpiar
     ↓
rellenar valores faltantes
     ↓
escalar números
     ↓
convertir categorías
     ↓
DATOS LISTOS PARA ML
```

---

# 34. Valores faltantes

El notebook utiliza:

```python
SimpleImputer(strategy="median")
```

Esto sirve para reemplazar valores numéricos faltantes utilizando la mediana.

Ejemplo:

```text
10
20
?
30
40
```

La mediana sería:

```text
25
```

y podríamos reemplazar:

```text
?
```

por:

```text
25
```

---

# 35. ¿Por qué utilizar la mediana?

La mediana puede ser más resistente a valores extremos que la media.

Por ejemplo:

```text
10, 11, 12, 13, 1000
```

La media se ve fuertemente afectada por `1000`.

La mediana no tanto.

Por eso puede ser útil para datos donde existen valores extremos.

---

# 36. RobustScaler

El notebook utiliza:

```python
RobustScaler()
```

Su función es escalar las variables numéricas.

Esto es especialmente relevante para algoritmos sensibles a la escala.

La idea es llevar variables con magnitudes diferentes a una escala comparable.

Por ejemplo:

```text
duración → 0 - 1000
bytes    → 0 - 1000000
```

Después del escalamiento, sus magnitudes quedan transformadas a una escala más manejable.

---

# 37. Variables categóricas

También tenemos variables como:

```text
TCP
UDP
ICMP
```

Un algoritmo matemático no puede utilizar directamente estos textos como números.

Necesitamos convertirlos.

Aquí aparece:

```python
OneHotEncoder()
```

---

# 38. ¿Qué hace One-Hot Encoding?

Supongamos:

```text
protocolo

TCP
UDP
ICMP
```

Podemos convertirlo en:

| TCP | UDP | ICMP |
|---:|---:|---:|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

Esto permite que el algoritmo trabaje con números.

---

# 39. Pipeline numérico

El notebook crea:

```python
num_pipeline = Pipeline([
    (
        'imputer',
        SimpleImputer(strategy='median')
    ),
    (
        'rbst_scaler',
        RobustScaler()
    ),
])
```

Aquí tenemos dos pasos:

```text
Datos numéricos
      ↓
Imputer
      ↓
RobustScaler
      ↓
Datos numéricos preparados
```

---

# 40. ¿Por qué utilizar un Pipeline?

Un Pipeline permite organizar varios pasos de transformación.

En lugar de hacer manualmente:

```text
paso 1
paso 2
paso 3
paso 4
```

podemos crear un proceso organizado:

```text
Pipeline
   ↓
transformación 1
   ↓
transformación 2
```

Esto reduce errores y facilita reutilizar exactamente las mismas transformaciones.

---

# 41. El transformador personalizado

El notebook crea:

```python
class CustomOneHotEncoder(
    BaseEstimator,
    TransformerMixin
):
```

Este componente se encarga de procesar las variables categóricas.

En particular identifica:

```python
X.select_dtypes(
    include=['object']
)
```

como variables categóricas.

Y:

```python
X.select_dtypes(
    exclude=['object']
)
```

como variables no categóricas.

---

# 42. ¿Qué hace `fit()`?

Dentro del transformador:

```python
def fit(self, X, y=None):
```

se aprende la estructura necesaria para transformar posteriormente los datos.

Por ejemplo:

```text
¿Qué categorías existen?
¿Qué columnas existen?
¿Cómo debo transformarlas?
```

---

# 43. ¿Qué hace `transform()`?

Después:

```python
def transform(self, X, y=None):
```

aplica las transformaciones aprendidas.

Conceptualmente:

```text
fit()
  ↓
aprender cómo transformar

transform()
  ↓
aplicar la transformación
```

---

# 44. DataFramePreparer

El notebook crea otro transformador:

```python
class DataFramePreparer(
    BaseEstimator,
    TransformerMixin
):
```

Este componente coordina el procesamiento de:

```text
variables numéricas
+
variables categóricas
```

Utiliza:

```python
ColumnTransformer
```

para aplicar diferentes procesos según el tipo de columna.

---

# 45. ¿Qué hace ColumnTransformer?

Conceptualmente:

```text
                    DATAFRAME
                        │
          ┌─────────────┴─────────────┐
          ↓                           ↓
      NUMÉRICAS                  CATEGÓRICAS
          ↓                           ↓
      Imputer                    OneHotEncoder
          ↓
   RobustScaler
          │                           │
          └─────────────┬─────────────┘
                        ↓
                 DATOS PREPARADOS
```

Esto es muy importante en proyectos reales.

---

# 46. Crear el preparador

El notebook hace:

```python
data_preparer = DataFramePreparer()
```

Ahora tenemos nuestro objeto encargado de preparar los datos.

---

# 47. `fit` del preparador

El notebook hace:

```python
data_preparer.fit(X_df)
```

Después puede transformar:

```python
X_train_prep = data_preparer.transform(
    X_train
)
```

Y:

```python
X_val_prep = data_preparer.transform(
    X_val
)
```

Finalmente:

```python
X_test_prep = data_preparer.transform(
    X_test
)
```

---

# 48. Una observación importante sobre este paso

El notebook original hace:

```python
data_preparer.fit(X_df)
```

donde:

```text
X_df = todo el conjunto de características
```

antes de transformar `train`, `validation` y `test`.

Para aprender el notebook debemos entender exactamente lo que hace.

Sin embargo, en un proyecto real es preferible evitar que información de validación o prueba influya en el aprendizaje de los transformadores.

Una práctica más segura es:

```text
fit → TRAIN

transform → TRAIN
transform → VALIDATION
transform → TEST
```

Es decir:

```python
data_preparer.fit(X_train)

X_train_prep = data_preparer.transform(X_train)
X_val_prep = data_preparer.transform(X_val)
X_test_prep = data_preparer.transform(X_test)
```

Esto ayuda a evitar **data leakage**.

---

# 49. ¿Qué es Data Leakage?

Data leakage significa que información que debería estar fuera del aprendizaje termina influyendo en el modelo o en el procesamiento.

La idea es:

```text
TRAIN
  ↓
puede enseñar al modelo

VALIDATION
  ↓
sirve para evaluar durante el desarrollo

TEST
  ↓
debe permanecer separado para la evaluación final
```

El conjunto de prueba debe comportarse como si fueran datos nuevos que el modelo nunca hubiera visto.

---

# 50. Entrenamiento de Regresión Logística

Ahora llegamos al algoritmo principal utilizado en el notebook.

Se importa:

```python
from sklearn.linear_model import LogisticRegression
```

Se crea:

```python
clf = LogisticRegression(
    solver="newton-cg",
    max_iter=1000
)
```

---

# 51. ¿Qué significa `solver`?

El `solver` es el método numérico utilizado por `scikit-learn` para encontrar los parámetros del modelo durante el entrenamiento.

En este notebook:

```python
solver="newton-cg"
```

No necesitas memorizar todavía los detalles matemáticos internos del solver.

Lo importante es entender:

```text
LogisticRegression
        ↓
algoritmo de clasificación
        ↓
solver
        ↓
método utilizado para optimizar el modelo
```

---

# 52. ¿Qué significa `max_iter=1000`?

Significa que el algoritmo puede realizar hasta:

```text
1000 iteraciones
```

durante el proceso de optimización.

¿Por qué?

Porque con datasets complejos puede necesitar más iteraciones para converger.

El notebook explícitamente aumenta este valor a 1000.

---

# 53. Entrenar

Ahora aparece:

```python
clf.fit(
    X_train_prep,
    y_train
)
```

Esta línea es fundamental.

Estamos diciendo:

> **Entrena la Regresión Logística utilizando los datos preparados de entrenamiento y sus respuestas conocidas.**

Visualmente:

```text
X_train_prep
     +
y_train
     ↓
Regresión Logística
     ↓
MODELO ENTRENADO
```

---

# 54. ¿Qué aprende el modelo?

El modelo intenta encontrar una relación entre:

```text
características de la conexión
```

y:

```text
class
```

para poder realizar una clasificación posteriormente.

---

# 55. Realizar una predicción

El notebook utiliza el conjunto de validación:

```python
y_pred = clf.predict(
    X_val_prep
)
```

Tenemos:

```text
X_val_prep
      ↓
modelo entrenado
      ↓
y_pred
```

Ahora:

```text
y_val
```

contiene la respuesta real.

Y:

```text
y_pred
```

contiene la predicción.

Por eso podemos compararlos.

---

# 56. La pregunta clave

Tenemos:

```text
y_val   → realidad
y_pred  → predicción
```

Queremos saber:

> **¿Qué tan parecidas son las predicciones a la realidad?**

Aquí comienza la evaluación.

---

# 57. Primera herramienta: matriz de confusión

El notebook importa:

```python
from sklearn.metrics import confusion_matrix
```

Y ejecuta:

```python
confusion_matrix(
    y_val,
    y_pred
)
```

Esto compara:

```text
REAL
vs.
PREDICCIÓN
```

---

# 58. Visualización de la matriz

También utiliza:

```python
from sklearn.metrics import ConfusionMatrixDisplay
```

y:

```python
ConfusionMatrixDisplay.from_estimator(
    clf,
    X_val_prep,
    y_val,
    values_format='d'
)
```

Esto permite visualizar la matriz.

La visualización ayuda a responder:

```text
¿Cuántos casos clasificó correctamente?
¿Cuántos clasificó incorrectamente?
¿Qué tipo de error cometió?
```

---

# 59. Precision

El notebook utiliza:

```python
from sklearn.metrics import precision_score

print(
    "Precisión:",
    precision_score(
        y_val,
        y_pred,
        pos_label='anomaly'
    )
)
```

Aquí hay un detalle muy importante:

```python
pos_label='anomaly'
```

Estamos indicando que la clase positiva que nos interesa es:

```text
anomaly
```

---

# 60. ¿Qué significa Precision?

Precision responde:

> **De todos los casos que el modelo clasificó como anomalía, ¿cuántos realmente eran anomalías?**

Fórmula:

```text
Precision = TP / (TP + FP)
```

Ejemplo:

El modelo dice:

```text
100 conexiones → anomaly
```

Pero realmente:

```text
80 eran anomaly
20 eran normales
```

Entonces:

```text
Precision = 80 / 100
          = 0.80
          = 80%
```

---

# 61. ¿Por qué Precision es importante en detección de intrusiones?

Imaginemos que el sistema genera una alerta cada vez que detecta una anomalía.

Si tenemos una Precision baja:

```text
muchas alertas
      ↓
muchas son falsas
      ↓
fatiga de alertas
```

Por eso Precision puede ser importante.

---

# 62. Recall

El notebook calcula:

```python
from sklearn.metrics import recall_score

print(
    "Recall:",
    recall_score(
        y_val,
        y_pred,
        pos_label='anomaly'
    )
)
```

Recall responde:

> **De todas las anomalías que realmente existían, ¿cuántas conseguimos detectar?**

Fórmula:

```text
Recall = TP / (TP + FN)
```

---

# 63. Ejemplo de Recall

Supongamos que realmente existen:

```text
100 anomalías
```

El modelo detecta:

```text
90
```

y no detecta:

```text
10
```

Entonces:

```text
Recall = 90 / 100
       = 90%
```

---

# 64. ¿Por qué Recall es muy importante en seguridad?

Porque un falso negativo significa:

```text
Había un ataque
      ↓
el modelo dijo "normal"
      ↓
ataque no detectado
```

Este tipo de error puede ser especialmente importante en un sistema de detección de intrusiones.

---

# 65. F1-score

El notebook calcula:

```python
from sklearn.metrics import f1_score

print(
    "F1 score:",
    f1_score(
        y_val,
        y_pred,
        pos_label='anomaly'
    )
)
```

F1 combina:

```text
Precision
+
Recall
```

La fórmula es:

```text
F1 = 2 × (Precision × Recall)
         ----------------------
         Precision + Recall
```

---

# 66. ¿Por qué utilizar F1?

Porque podemos tener situaciones como:

```text
Precision = 95%
Recall = 50%
```

El modelo es muy preciso cuando alerta, pero está dejando escapar muchas anomalías.

O:

```text
Precision = 50%
Recall = 95%
```

Detecta casi todas las anomalías, pero genera muchas falsas alarmas.

F1 busca un equilibrio entre ambas métricas.

---

# 67. Segunda parte: Curva ROC

El notebook utiliza:

```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(
    clf,
    X_val_prep,
    y_val
)
```

La curva ROC permite analizar el comportamiento del clasificador cuando cambiamos el umbral de decisión.

No debemos verla simplemente como:

> "Una gráfica bonita."

Su objetivo es estudiar el equilibrio entre:

```text
True Positive Rate
```

y:

```text
False Positive Rate
```

---

# 68. ¿Qué es True Positive Rate?

Es esencialmente:

```text
TPR = Recall
```

Es decir:

> ¿Qué proporción de las anomalías reales estamos detectando?

---

# 69. ¿Qué es False Positive Rate?

Es:

```text
FPR = FP / (FP + TN)
```

Responde:

> De todas las conexiones que realmente eran normales, ¿cuántas fueron marcadas incorrectamente como anomalías?

---

# 70. Curva Precision-Recall

El notebook también utiliza:

```python
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_estimator(
    clf,
    X_val_prep,
    y_val
)
```

Esta curva permite estudiar la relación entre:

```text
Precision
```

y:

```text
Recall
```

Es especialmente útil cuando la clase positiva es poco frecuente.

En problemas de detección de anomalías esto puede ser muy relevante.

---

# 71. ¿Por qué no quedarnos únicamente con Accuracy?

Imagina:

```text
1000 conexiones
```

y:

```text
950 normales
50 anomalías
```

Un modelo que diga:

```text
TODO ES NORMAL
```

acertaría:

```text
950 / 1000 = 95%
```

Tendría una Accuracy del 95%.

Pero:

```text
Recall de anomalías = 0%
```

No detectó ningún ataque.

Por eso:

> **Accuracy por sí sola puede ser engañosa.**

---

# 72. El conjunto de validación

Hasta este momento el notebook ha utilizado:

```text
TRAIN
```

para entrenar y:

```text
VALIDATION
```

para analizar el comportamiento del modelo.

Por ejemplo:

```text
X_train_prep
     ↓
     clf
     ↓
aprendizaje

X_val_prep
     ↓
     clf
     ↓
predicciones
     ↓
métricas
```

---

# 73. Ahora llega el conjunto TEST

Esta parte es especialmente importante.

El notebook transforma el conjunto de prueba:

```python
X_test_prep = data_preparer.transform(
    X_test
)
```

Después realiza:

```python
y_pred = clf.predict(
    X_test_prep
)
```

Ahora tenemos una predicción sobre el conjunto que se había reservado para la evaluación final.

---

# 74. Evaluar TEST

El notebook visualiza:

```python
ConfusionMatrixDisplay.from_estimator(
    clf,
    X_test_prep,
    y_test,
    values_format='d'
)
```

Y finalmente calcula:

```python
print(
    "F1 score:",
    f1_score(
        y_test,
        y_pred,
        pos_label='anomaly'
    )
)
```

Este resultado tiene especial importancia porque estamos evaluando sobre datos separados de los utilizados para entrenar.

---

# 75. Flujo completo del notebook

Ahora podemos entender todo el notebook como una cadena:

```text
                 NSL-KDD
                    ↓
              Cargar ARFF
                    ↓
             DataFrame
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
       X                        y
características               class
        ↓
Dividir datos
        ↓
┌───────┼──────────────┐
↓       ↓              ↓
TRAIN   VALIDATION     TEST
 ↓          ↓           ↓
Preparar datos          ↓
 ↓                      ↓
Regresión Logística     ↓
 ↓                      ↓
Predicción              ↓
 ↓                      ↓
Matriz de confusión     ↓
Precision               ↓
Recall                  ↓
F1                      ↓
ROC                     ↓
PR                      ↓
                        ↓
                 Evaluación TEST
                        ↓
                    F1-score
```

---

# 76. ¿Dónde está Machine Learning en todo esto?

Es importante distinguir dos cosas.

## Entrenamiento

Aquí el modelo aprende:

```python
clf.fit(
    X_train_prep,
    y_train
)
```

## Predicción

Aquí el modelo utiliza lo aprendido:

```python
clf.predict(
    X_val_prep
)
```

o:

```python
clf.predict(
    X_test_prep
)
```

## Evaluación

Aquí comparamos:

```text
predicción
vs.
respuesta real
```

---

# 77. El error de un principiante

Un estudiante puede pensar:

> "Si el modelo predice, ya terminé."

No.

El proceso real es:

```text
ENTRENAR
   ↓
PREDICIR
   ↓
EVALUAR
   ↓
ANALIZAR ERRORES
   ↓
MEJORAR
```

La evaluación es una parte fundamental del proyecto.

---

# 78. ¿Qué significa realmente "un modelo funciona bien"?

No significa necesariamente:

```text
Accuracy = 99%
```

Debemos preguntar:

- ¿Qué clase estamos intentando detectar?
- ¿Hay desbalance?
- ¿Cuántos falsos positivos tenemos?
- ¿Cuántos falsos negativos?
- ¿Cuál es Precision?
- ¿Cuál es Recall?
- ¿Cuál es F1?
- ¿Cómo se comporta la curva ROC?
- ¿Cómo se comporta Precision-Recall?
- ¿Cómo funciona en TEST?

---

# 79. Un ejemplo de interpretación

Supongamos que obtenemos:

```text
Precision = 0.90
Recall    = 0.80
F1        = 0.85
```

Podemos decir:

> El modelo tiene una Precision del 90%, lo que significa que una alta proporción de las conexiones clasificadas como anomalías realmente son anomalías. Su Recall del 80% indica que detecta aproximadamente el 80% de las anomalías reales. El F1-score de 85% representa un equilibrio entre Precision y Recall.

Esta es una interpretación mucho más útil que simplemente decir:

```text
F1 = 0.85
```

---

# 80. ¿Qué pasa si Recall es bajo?

Supongamos:

```text
Precision = 95%
Recall = 40%
```

El modelo es muy preciso cuando genera una alerta.

Pero está dejando pasar muchas anomalías.

En seguridad informática esto puede ser un problema importante.

---

# 81. ¿Qué pasa si Precision es baja?

Supongamos:

```text
Precision = 40%
Recall = 95%
```

El modelo encuentra casi todas las anomalías.

Pero genera muchas falsas alarmas.

En un centro de operaciones de seguridad esto podría generar una gran cantidad de trabajo innecesario.

---

# 82. ¿Qué métrica sería "la mejor"?

No existe una métrica universalmente mejor.

Depende del objetivo.

```text
Problema
   ↓
¿Qué error es más costoso?
   ↓
Elegir métricas
   ↓
Evaluar
```

En detección de intrusiones, por ejemplo, normalmente queremos prestar especial atención a:

```text
Recall
Precision
F1
Precision-Recall
```

y analizar cuidadosamente la matriz de confusión.

---

# 83. ¿Y Regresión Logística vs SVM?

El notebook proporcionado utiliza **Regresión Logística** como algoritmo de clasificación.

Si quisiéramos ampliar el experimento podríamos añadir SVM:

```python
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf"
)

svm.fit(
    X_train_prep,
    y_train
)

y_pred_svm = svm.predict(
    X_val_prep
)
```

Después utilizaríamos las mismas métricas:

```text
Regresión Logística
       ↓
Precision
Recall
F1
Matriz de confusión

SVM
       ↓
Precision
Recall
F1
Matriz de confusión
```

---

# 84. Comparación correcta

No debemos decir:

> "SVM es mejor."

o:

> "Regresión Logística es mejor."

Debemos decir algo como:

> "En este conjunto de datos, con esta preparación, esta configuración y estas métricas, el modelo X obtuvo determinados resultados."

Por ejemplo:

| Modelo | Precision | Recall | F1 |
|---|---:|---:|---:|
| Regresión Logística | 0.90 | 0.80 | 0.85 |
| SVM | 0.92 | 0.84 | 0.88 |

La interpretación sería:

> En este experimento, SVM obtuvo valores superiores en las tres métricas mostradas.

Pero eso no significa que SVM sea siempre superior.

---

# 85. Algo muy importante: comparar con el mismo TEST

Si comparamos dos modelos:

```text
Regresión Logística
```

y:

```text
SVM
```

ambos deben evaluarse sobre el mismo conjunto de prueba.

Correcto:

```text
                 MISMO TRAIN
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
   Regresión Logística         SVM
          ↓                     ↓
          └──────────┬──────────┘
                     ↓
                MISMO TEST
                     ↓
                 comparar
```

Así la comparación es mucho más justa.

---

# 86. ¿Por qué el notebook utiliza `pos_label='anomaly'`?

Esta parte es muy importante para entender el contexto.

El dataset tiene una clase:

```text
anomaly
```

Cuando calculamos:

```python
precision_score(
    y_val,
    y_pred,
    pos_label='anomaly'
)
```

le estamos diciendo a `sklearn`:

> "Considera `anomaly` como la clase positiva."

Entonces:

```text
TP
```

significa:

> El modelo detectó `anomaly` y realmente era `anomaly`.

Y:

```text
FN
```

significa:

> Era `anomaly`, pero el modelo no la detectó.

---

# 87. ¿Por qué esto cambia la interpretación?

Porque Precision, Recall y F1 dependen de cuál clase consideremos positiva.

En este problema queremos estudiar especialmente:

```text
ANOMALÍA
```

porque nuestro interés es detectar posibles intrusiones.

---

# 88. Lo que realmente estamos haciendo

Todo el notebook puede resumirse en una pregunta:

> **Dado el comportamiento de una conexión de red, ¿podemos identificar si corresponde a una conexión normal o a una anomalía?**

Y el proceso es:

```text
Características de red
        ↓
Preparación de datos
        ↓
Regresión Logística
        ↓
Predicción
        ↓
¿normal o anomaly?
        ↓
Comparar con realidad
        ↓
Medir errores
        ↓
Evaluar modelo
```

---

# 89. Resumen de cada función importante

| Función | ¿Para qué sirve? |
|---|---|
| `arff.load()` | Lee el archivo ARFF |
| `pd.DataFrame()` | Construye una tabla |
| `train_test_split()` | Divide los datos |
| `SimpleImputer()` | Trata valores faltantes |
| `RobustScaler()` | Escala variables numéricas |
| `OneHotEncoder()` | Convierte categorías a variables numéricas |
| `ColumnTransformer()` | Aplica transformaciones diferentes según columnas |
| `Pipeline()` | Encadena transformaciones |
| `LogisticRegression()` | Crea el clasificador |
| `.fit()` | Entrena/aprende |
| `.predict()` | Genera predicciones |
| `confusion_matrix()` | Calcula la matriz de confusión |
| `precision_score()` | Calcula Precision |
| `recall_score()` | Calcula Recall |
| `f1_score()` | Calcula F1 |
| `RocCurveDisplay` | Visualiza ROC |
| `PrecisionRecallDisplay` | Visualiza Precision-Recall |

---

# 90. Resumen conceptual para memorizar

## Paso 1 — Datos

```text
Tengo datos históricos
```

## Paso 2 — X e y

```text
X = características
y = respuesta
```

## Paso 3 — Dividir

```text
TRAIN
VALIDATION
TEST
```

## Paso 4 — Preparar

```text
faltantes
+
variables numéricas
+
variables categóricas
```

## Paso 5 — Entrenar

```python
clf.fit(X_train_prep, y_train)
```

## Paso 6 — Predecir

```python
y_pred = clf.predict(X_val_prep)
```

## Paso 7 — Evaluar

```text
Matriz de confusión
Precision
Recall
F1
ROC
Precision-Recall
```

## Paso 8 — Evaluación final

```python
clf.predict(X_test_prep)
```

## Paso 9 — Concluir

```text
¿Qué tan bien funciona?
¿Qué errores comete?
¿Detecta las anomalías?
```

---

# 91. Código esencial del notebook

Una vez entendido todo, el flujo principal puede verse así:

```python
# 1. Cargar datos
df = load_kdd_dataset(
    "datasets/NSL-KDD/KDDTrain+.arff"
)

# 2. Dividir
train_set, val_set, test_set = train_val_test_split(df)

# 3. Separar características y objetivo
X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

X_val = val_set.drop("class", axis=1)
y_val = val_set["class"].copy()

X_test = test_set.drop("class", axis=1)
y_test = test_set["class"].copy()

# 4. Preparar datos
data_preparer = DataFramePreparer()

data_preparer.fit(X_train)

X_train_prep = data_preparer.transform(X_train)
X_val_prep = data_preparer.transform(X_val)
X_test_prep = data_preparer.transform(X_test)

# 5. Crear modelo
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    solver="newton-cg",
    max_iter=1000
)

# 6. Entrenar
clf.fit(
    X_train_prep,
    y_train
)

# 7. Predecir validation
y_pred = clf.predict(
    X_val_prep
)

# 8. Evaluar
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(confusion_matrix(y_val, y_pred))

print(
    "Precision:",
    precision_score(
        y_val,
        y_pred,
        pos_label="anomaly"
    )
)

print(
    "Recall:",
    recall_score(
        y_val,
        y_pred,
        pos_label="anomaly"
    )
)

print(
    "F1:",
    f1_score(
        y_val,
        y_pred,
        pos_label="anomaly"
    )
)

# 9. Evaluación final sobre TEST
y_pred_test = clf.predict(
    X_test_prep
)

print(
    "F1 TEST:",
    f1_score(
        y_test,
        y_pred_test,
        pos_label="anomaly"
    )
)
```

---

# 92. Preguntas para comprobar el aprendizaje

## Pregunta 1

¿Qué representa `X`?

**Respuesta:** las características utilizadas para realizar la predicción.

## Pregunta 2

¿Qué representa `y`?

**Respuesta:** la variable objetivo o etiqueta que queremos predecir.

## Pregunta 3

¿Qué representa `class` en el notebook?

**Respuesta:** la clase de la conexión, que permite distinguir el tipo de tráfico que se está clasificando.

## Pregunta 4

¿Por qué separamos TRAIN, VALIDATION y TEST?

**Respuesta:** porque cada conjunto cumple una función diferente durante el desarrollo y evaluación del modelo.

## Pregunta 5

¿Qué hace `fit()`?

**Respuesta:** aprende los parámetros necesarios a partir de los datos de entrenamiento.

## Pregunta 6

¿Qué hace `predict()`?

**Respuesta:** utiliza el modelo entrenado para generar predicciones sobre nuevos datos.

## Pregunta 7

¿Qué muestra una matriz de confusión?

**Respuesta:** muestra los aciertos y errores de clasificación, incluyendo verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.

## Pregunta 8

¿Qué significa Precision?

**Respuesta:** de las observaciones que el modelo predijo como positivas, qué proporción realmente era positiva.

## Pregunta 9

¿Qué significa Recall?

**Respuesta:** de todas las observaciones realmente positivas, qué proporción consiguió detectar el modelo.

## Pregunta 10

¿Por qué utilizamos `pos_label='anomaly'`?

**Respuesta:** para indicar que `anomaly` es la clase positiva sobre la que queremos calcular Precision, Recall y F1.

## Pregunta 11

¿Por qué F1 combina Precision y Recall?

**Respuesta:** porque permite evaluar conjuntamente ambos aspectos mediante su media armónica.

## Pregunta 12

¿Por qué no debemos utilizar solamente Accuracy?

**Respuesta:** porque una Accuracy elevada puede ocultar un mal comportamiento sobre una clase minoritaria.

---

# 93. La idea más importante de todo el notebook

No memorices solamente:

```python
f1_score(...)
```

Comprende el razonamiento:

```text
Tengo datos
   ↓
quiero hacer una predicción
   ↓
separo X e y
   ↓
divido los datos
   ↓
preparo los datos
   ↓
entreno un modelo
   ↓
realizo predicciones
   ↓
comparo predicción vs realidad
   ↓
analizo los errores
   ↓
calculo métricas
   ↓
evalúo
```

Eso es **evaluar un modelo de Machine Learning**.

---

# 94. Conclusión

El notebook `10_Evaluación de resultados.ipynb` utiliza un problema realista de **detección de intrusiones de red** para enseñar un concepto fundamental de Data Science:

> **Entrenar un modelo no es suficiente. Tenemos que medir cómo se comporta.**

En el ejemplo:

```text
NSL-KDD
   ↓
Características de conexiones
   ↓
Preparación de datos
   ↓
Regresión Logística
   ↓
Predicción
   ↓
Matriz de confusión
   ↓
Precision
Recall
F1
ROC
Precision-Recall
   ↓
Evaluación sobre TEST
```

La parte más importante que debe aprender un estudiante es distinguir:

```text
ENTRENAR
```

de:

```text
EVALUAR
```

El modelo aprende con:

```text
TRAIN
```

Se analiza durante el desarrollo con:

```text
VALIDATION
```

Y se realiza la comprobación final con:

```text
TEST
```

Finalmente, una buena evaluación no pregunta solamente:

> "¿Cuántos acertó?"

También pregunta:

> **¿Qué tipo de errores cometió? ¿Cuántas anomalías detectó? ¿Cuántas dejó pasar? ¿Cuántas falsas alarmas generó?**

Ese cambio de perspectiva es fundamental para pasar de **aprender Machine Learning** a **hacer Data Science**.
