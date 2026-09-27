# Guía: ejecutar Jupyter Notebooks (`.ipynb`) en VS Code con `.venv`

Esta guía explica cómo trabajar con notebooks de Jupyter en Visual Studio Code usando un entorno virtual de Python.

## 1. Estructura esperada del proyecto

Ejemplo:

```text
casos_practicos_machine_learning/
├── .venv/
├── notebooks/
│   └── 01_introduccion_numpy.ipynb
├── *.py
└── requirements.txt
```

> La carpeta `.venv` contiene Python y las librerías del proyecto. No se debe subir a Git.

---

# PARTE A — Primera configuración del proyecto

Estos pasos se hacen **una sola vez**.

## 2. Abrir el directorio en VS Code

Desde una terminal:

```bash
cd ~/Documents/MachineLearning/casos_practicos_machine_learning
code .
```

O desde VS Code:

**File → Open Folder**

Seleccionar:

```text
~/Documents/MachineLearning/casos_practicos_machine_learning
```

---

## 3. Comprobar Python

En la terminal integrada de VS Code:

```bash
python3 --version
```

En este proyecto esperamos:

```text
Python 3.10.12
```

También podemos comprobar dónde está:

```bash
which python3
```

Ejemplo:

```text
/usr/bin/python3
```

---

# PARTE B — Crear el entorno virtual

## 4. Si ya existe un `.venv` defectuoso

Primero salir del entorno:

```bash
deactivate
```

Si aparece:

```text
bash: deactivate: command not found
```

no pasa nada: significa que no había un entorno activado.

Eliminar el entorno anterior:

```bash
rm -rf .venv
```

> Esto solamente elimina el entorno virtual y sus paquetes. No elimina los archivos `.py` ni `.ipynb`.

---

## 5. Crear `.venv`

Usar Python 3.10:

```bash
python3 -m venv .venv
```

Comprobar que se creó:

```bash
ls -la .venv/bin/python*
```

---

## 6. Activar `.venv`

```bash
source .venv/bin/activate
```

La terminal debe mostrar algo parecido a:

```text
(.venv) usuario@pc:~/Documents/MachineLearning/casos_practicos_machine_learning$
```

El `(.venv)` indica que el entorno está activo.

---

## 7. Comprobar que el Python es el del entorno

```bash
which python
```

Debe mostrar una ruta parecida a:

```text
/home/usuario/Documents/MachineLearning/casos_practicos_machine_learning/.venv/bin/python
```

Después:

```bash
python --version
```

Debe mostrar:

```text
Python 3.10.12
```

---

# PARTE C — Instalar las herramientas necesarias

## 8. Actualizar pip

Con `.venv` activado:

```bash
python -m pip install --upgrade pip
```

---

## 9. Instalar Jupyter, ipykernel y librerías

```bash
python -m pip install jupyter ipykernel numpy matplotlib pandas
```

Si el proyecto necesita otras librerías, instalarlas también en este mismo entorno.

Por ejemplo:

```bash
python -m pip install scikit-learn
```

---

## 10. Comprobar `ipykernel`

```bash
python -m pip show ipykernel
```

Debe mostrar una ubicación dentro de:

```text
.../.venv/lib/python3.10/site-packages
```

---

## 11. Comprobar NumPy

```bash
python -m pip show numpy
```

También debe aparecer dentro de:

```text
.../.venv/lib/python3.10/site-packages
```

---

# PARTE D — Registrar `.venv` como Kernel de Jupyter

## 12. Registrar el Kernel

Con `.venv` activado:

```bash
python -m ipykernel install --user \
  --name machine_learning \
  --display-name "Python 3.10.12 - Machine Learning"
```

Esto permite que VS Code/Jupyter vea el entorno como un Kernel.

---

# PARTE E — Configurar VS Code

## 13. Instalar extensiones

En VS Code instalar:

- **Python** — Microsoft
- **Jupyter** — Microsoft

Opcional:

- Python Debugger

---

## 14. Seleccionar el intérprete de Python

En VS Code:

**Ctrl + Shift + P**

Buscar:

```text
Python: Select Interpreter
```

Seleccionar:

```text
.venv/bin/python
```

Debe corresponder a:

```text
~/Documents/MachineLearning/casos_practicos_machine_learning/.venv/bin/python
```

---

## 15. Abrir el Notebook

Abrir un archivo:

```text
.ipynb
```

Por ejemplo:

```text
01_introduccion_numpy.ipynb
```

En la esquina superior derecha de Jupyter aparecerá:

```text
Select Kernel
```

Seleccionar:

```text
Python 3.10.12 - Machine Learning
```

o el Kernel que registramos anteriormente.

### Importante

No seleccionar un Python que apunte a:

```text
/usr/bin/python3
```

si el Notebook no está utilizando el Kernel del proyecto.

La idea es que Jupyter termine usando:

```text
.venv/bin/python
```

---

# PARTE F — Verificar que Jupyter usa el `.venv`

## 16. Primera celda del Notebook

Antes de importar NumPy, ejecutar:

```python
import sys

print(sys.executable)
print(sys.version)
```

La primera línea debe mostrar algo parecido a:

```text
/home/usuario/Documents/MachineLearning/casos_practicos_machine_learning/.venv/bin/python
```

La segunda debe comenzar con:

```text
3.10.12
```

---

## 17. Probar NumPy

```python
import numpy as np

print(np.__version__)
```

Después:

```python
numeros = np.array([10, 20, 30, 40, 50])

print(numeros)
```

Resultado esperado:

```text
[10 20 30 40 50]
```

---

# PARTE G — Qué hacer cada vez que abras el proyecto

## 18. Abrir el proyecto

Cada vez que vayas a trabajar:

```bash
cd ~/Documents/MachineLearning/casos_practicos_machine_learning
code .
```

---

## 19. Activar el entorno virtual

Cuando abras una nueva terminal:

```bash
source .venv/bin/activate
```

La terminal debe quedar:

```text
(.venv) usuario@pc:~/Documents/MachineLearning/casos_practicos_machine_learning$
```

---

## 20. Comprobar el entorno

Opcional, pero recomendable:

```bash
which python
```

Debe apuntar a:

```text
.../casos_practicos_machine_learning/.venv/bin/python
```

Y:

```bash
python --version
```

Debe ser:

```text
Python 3.10.12
```

---

## 21. Abrir el `.ipynb`

Abrir el Notebook y comprobar el Kernel.

Debe estar seleccionado:

```text
Python 3.10.12 - Machine Learning
```

Después ejecutar las celdas con:

```text
Shift + Enter
```

---

# PARTE H — Si aparece "Running cells requires ipykernel"

Si VS Code muestra:

```text
Running cells with 'Python 3.10.12' requires the ipykernel package.
```

NO instalar inmediatamente usando:

```text
/usr/bin/python3 -m pip ...
```

Primero comprobar qué Kernel está seleccionado.

Seleccionar:

```text
Python 3.10.12 - Machine Learning
```

Después comprobar desde una celda:

```python
import sys

print(sys.executable)
```

Debe apuntar a:

```text
.../.venv/bin/python
```

---

# PARTE I — Si el `.venv` se daña

Si ocurre algo extraño, por ejemplo:

```text
python: command not found
```

aunque aparezca:

```text
(.venv)
```

es posible que el entorno esté dañado.

Recrear:

```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

Después instalar nuevamente:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter ipykernel numpy matplotlib pandas
```

Y registrar nuevamente:

```bash
python -m ipykernel install --user \
  --name machine_learning \
  --display-name "Python 3.10.12 - Machine Learning"
```

---

# PARTE J — Guardar las dependencias del proyecto

Cuando el entorno esté funcionando:

```bash
python -m pip freeze > requirements.txt
```

Esto crea:

```text
requirements.txt
```

Por ejemplo:

```text
numpy==...
pandas==...
matplotlib==...
ipykernel==...
jupyter==...
```

---

# PARTE K — Instalar las dependencias en otro equipo

Si el proyecto ya tiene:

```text
requirements.txt
```

crear el entorno:

```bash
python3 -m venv .venv
```

Activarlo:

```bash
source .venv/bin/activate
```

Instalar:

```bash
python -m pip install -r requirements.txt
```

Registrar el Kernel:

```bash
python -m ipykernel install --user \
  --name machine_learning \
  --display-name "Python 3.10.12 - Machine Learning"
```

---

# PARTE L — Comandos esenciales

## Abrir proyecto

```bash
cd ~/Documents/MachineLearning/casos_practicos_machine_learning
code .
```

## Activar entorno

```bash
source .venv/bin/activate
```

## Salir del entorno

```bash
deactivate
```

## Ver Python utilizado

```bash
which python
```

## Ver versión

```bash
python --version
```

## Instalar paquete

```bash
python -m pip install nombre_paquete
```

## Instalar NumPy

```bash
python -m pip install numpy
```

## Instalar Jupyter

```bash
python -m pip install jupyter ipykernel
```

## Registrar Kernel

```bash
python -m ipykernel install --user \
  --name machine_learning \
  --display-name "Python 3.10.12 - Machine Learning"
```

## Guardar dependencias

```bash
python -m pip freeze > requirements.txt
```

---

# Flujo diario recomendado

Una vez configurado el proyecto, el flujo normal es simplemente:

```bash
cd ~/Documents/MachineLearning/casos_practicos_machine_learning
```

```bash
source .venv/bin/activate
```

```bash
code .
```

Luego:

1. Abrir el archivo `.ipynb`.
2. Seleccionar el Kernel `Python 3.10.12 - Machine Learning`.
3. Ejecutar las celdas con `Shift + Enter`.
4. Si se instala una nueva librería, hacerlo desde la terminal con:
   ```bash
   python -m pip install nombre_paquete
   ```

## Regla principal

Siempre que trabajes con este proyecto, procura que:

```text
Terminal
    ↓
(.venv)
    ↓
.venv/bin/python
    ↓
Jupyter Kernel
    ↓
.venv/bin/python
```

Es decir, **la terminal, VS Code y Jupyter deben utilizar el mismo entorno virtual `.venv`**.
