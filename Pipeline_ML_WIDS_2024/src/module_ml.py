# module_ml.py
"""
El módulo realiza lo siguiente:
- Definir modelos de clasificación (clásicos y avanzados).
- Aplicar SMOTE para balancear la clase minoritaria.
- Cálculo de métricas de evaluación sobre el set de validación.
- Registro automático de cada run en MLflow mediante un decorador.

"""

# Importamos las librerias
import time
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Clase de configuración general para la parte de modelos y MLflow
class MLConfig:

    def __init__(self, experiment_name: str, tracking_uri: str, random_state: int = 42):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.random_state = random_state

# Función que se encarga de configurar MLflow (tracking_uri + experimento), crear un run por cada llamada a la función decorada y cerrar el run automáticamente
def mlflow_logger(func):

    def wrapper(self, *args, **kwargs):
        # Configuro a dónde y bajo qué experimento se va a registrar
        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        mlflow.set_experiment(self.cfg.experiment_name)

        # Cada llamada a evaluate() será un run independiente
        with mlflow.start_run():
            return func(self, *args, **kwargs)

    return wrapper

# Clase que se encarga de aplicar SMOTE sobre X_train / y_train
# Guardar los datos de validación sin modificar
# Definir el diccionario de modelos a entrenar
# Evaluar cada modelo y registrar sus métricas en MLflow
class Model:

    def __init__(self, X_train, X_val, y_train, y_val, cfg: MLConfig) -> None:
        self.cfg = cfg


        # Balanceo con SMOTE que solo se aplica al conjunto de entrenamiento
        sm = SMOTE(random_state=cfg.random_state)
        self.X_train, self.y_train = sm.fit_resample(X_train, y_train)

        # El conjunto de validación se deja intacto para evaluar el modelo en una distribución lo más parecida posible a la real
        self.X_val = X_val
        self.y_val = y_val

        # Usamos modelos "clásicos" con otros más potentes como XGBoost y LightGBM
        # Los hiperparámetros son configuraciones razonables para un primer experimento
        self.modelos = {
            "LogisticRegression": LogisticRegression(max_iter=400),
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=cfg.random_state,
            ),
            "NaiveBayes": GaussianNB(),
            "KNN": KNeighborsClassifier(),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=600,
                random_state=cfg.random_state,
            ),
            "DecisionTree": DecisionTreeClassifier(random_state=cfg.random_state),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=cfg.random_state,
                # Evitamos warnings con XGBoost moderno
                eval_metric="logloss",  
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                random_state=cfg.random_state,
            ),
        }

    @mlflow_logger
    # Función para entrenar un modelo, calcular sus métricas sobre el set de validación y registrar todo en MLflow.
    def evaluate(self, model):
        # Entrenamiento + timing
        start = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        # Predicciones y métricas
        preds = model.predict(self.X_val)

        metrics = {
            "accuracy": accuracy_score(self.y_val, preds),
            # Para roc_auc aqui usamos las predicciones de clase.
            "roc_auc": roc_auc_score(self.y_val, preds),
            "f1": f1_score(self.y_val, preds),
            "precision": precision_score(self.y_val, preds),
            "recall": recall_score(self.y_val, preds),
            "train_time": train_time,
        }

        # Registro en MLflow, guardo el modelo como artefacto del run
        mlflow.sklearn.log_model(model, artifact_path=model.__class__.__name__)

        # Registro cada métrica por separado
        for nombre_metric, valor in metrics.items():
            mlflow.log_metric(nombre_metric, valor)

        return metrics

    # Función que recorre el diccionario de modelos, llama a evaluate() por cada uno
    def evaluate_all(self):
        resultados = {}

        for nombre, modelo in self.modelos.items():
            print(f"\nEntrenando modelo: {nombre}...")
            resultados[nombre] = self.evaluate(modelo)

        return resultados
