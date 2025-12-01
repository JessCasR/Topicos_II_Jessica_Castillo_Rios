# WiDS Datathon 2024 - Pipeline de Machine Learning

Este proyecto implementa un pipeline completo de Machine Learning basado en el dataset de la competencia WiDS Datathon 2024.
El Pipeline está organizado en módulos para que sea claro, ordenado y reproducible.

## Objetivos del proyecto

- Cargar y limpiar datos automáticamente  
- Manejar valores nulos y columnas irrelevantes  
- Transformar variables numéricas y categóricas  
- Aplicar **One-Hot Encoding** y escalado  
- Balancear las clases con **SMOTE**  
- Entrenar varios modelos de clasificación  
- Evaluarlos con métricas estándar  
- Comparar su desempeño  
- Registrar todos los experimentos usando **MLflow**


## Componentes Principales

### Preprocesamiento (`module_data.py`)

Incluye:

- Carga del dataset  
- Limpieza básica  
- Imputación  
- Eliminación de columnas con alto porcentaje de nulos  
- One-Hot Encoding  
- Escalado de variables  
- Split Train/Validation  
- Aplicación de **SMOTE**  
- Construcción del `ColumnTransformer`

---

### Modelos y Evaluación (`module_ml.py`)

Modelos utilizados:

- Logistic Regression  
- Random Forest  
- Naive Bayes  
- KNN  
- MLPClassifier  
- DecisionTreeClassifier  
- XGBoost  
- LightGBM  

Métricas calculadas:

- Accuracy  
- F1-score  
- Precision  
- Recall  
- ROC-AUC  
- Tiempo de entrenamiento  

Todos los modelos se registran automáticamente en MLflow.

### Orquestación (`main.py`)

El archivo `main.py`:

- Ejecuta el preprocesamiento  
- Entrena todos los modelos  
- Genera una tabla comparativa  
- Guarda una gráfica y un csv con los resultados  
- Registra cada experimento con MLflow  
- Exporta los resultados a la carpeta `outputs/`

## Resultados

El pipeline genera:

- Una tabla comparativa de métricas en consola  
- Un archivo CSV con los resultados  
- Una gráfica comparativa

## Estructura

- `data/`
- `src/`:
  - `module_data.py`: carga, limpieza y preprocesamiento de datos.
  - `module_ml.py`: entrenamiento, evaluación y tracking de modelos.
  - `__init__.py`
- `mlruns/`: carpeta donde MLflow guarda los experimentos.
- `main.py`: script principal del pipeline.
- `requirements.txt`: dependencias del proyecto.
- `README.md`


## Instrucciones

1. Clona el repositorio:
    a. `git clone https://github.com/JessCasR/Topicos_II_Jessica_Castillo_Rios`
    b.  cd Pipeline_ML_WIDS_2024
2. Instala las dependencias: `pip install -r requirements.txt`.
3. Corre el siguiente comando en consola para ejecutar el pipeline: `python main.py`.

## Instrucciones para visualizar los resultado en MLflow

1. Ejecuta el siguiente comando en consola: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
2. Abre el siguiente link http://localhost:5000 para visualizar los experimentos y modelos registrados.