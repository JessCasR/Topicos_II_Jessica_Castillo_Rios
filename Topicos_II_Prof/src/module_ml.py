# Librerías
import pandas as pd

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# mlflow
import mlflow
from mlflow.models.signature import infer_signature

def mlflow_logger(func):
    def wrapper(*args, **kwargs):
        # creamos un nuevo experimento
        mlruns_path = "../mlruns"
        mlflow.set_tracking_uri(mlruns_path)
        experiment_name = "WIDS2024"

        try:
            exp_id = mlflow.create_experiment(name=experiment_name)
        except Exception as e:
            exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        with mlflow.start_run(experiment_id=exp_id):
            return func(*args, **kwargs)
    return wrapper


class Model():

    def __init__(self, X:pd.DataFrame, y:pd.Series, seed:int=42):
        self.X = X
        self.y = y
        self.seed = seed
    
    def split(self, train_size:float=0.8):
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            train_size=train_size,
                                                            random_state=self.seed
                                                            )
        return X_train, X_test, y_train, y_test

    @mlflow_logger # decorator
    def evaluate(self, model):
        print(f"Método: {type(model).__name__}")
        X_train, X_test, y_train, y_test = self.split()
        model.fit(X_train, y_train)
        print("Entrenamiento completado")
        y_pred = model.predict(X_test)
        print("Metricas relevantes")
        accuracy = accuracy_score(y_test, y_pred)
        roc_score = roc_auc_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Roc_Score: {roc_score}")

        # Registros con mlflow
        mlflow.log_param("Model Type", type(model).__name__)
        mlflow.log_metric("accuracy", accuracy)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)