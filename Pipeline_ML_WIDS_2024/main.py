# main.py
"""
Archivo principal del proyecto WIDS 2024.

Ejecuta lo siguiente:
- Configuración de parámetros de datos.
- Configuración del tracking en MLflow.
- Preprocesamiento completo (train/validation).
- Entrenamiento y evaluación de todos los modelos.
- Impresión de resultados en una tabla.
- Guardado del CSV y gráfica comparativa en la carpeta outputs/.
"""

# Importamos las librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.module_data import Dataset, DataConfig
from src.module_ml import Model, MLConfig

# Función principal que ejecuta paso por paso todo el flujo del pipeline.
def main():
    # Definimos como queremos que se comporte el módulo de preprocesamiento.
    data_cfg = DataConfig(
        target_col="DiagPeriodL90D",
        null_thresh=0.30,
        scaler="standard",
        test_size=0.20,
        random_state=42
    )

    # Configuración de MLflow
    ml_cfg = MLConfig(
        # Nombre del experimento
        experiment_name="Pipeline_ML_WIDS_2024",
        # Elijo sqlite para que MLflow guarde los runs en un archivo local llamado mlflow.db
        tracking_uri="sqlite:///mlflow.db",
        random_state=42
    )

    # Carga y preprocesamiento
    dataset = Dataset(config=data_cfg)
    X_train, X_val, y_train, y_val = dataset.get_train_val()

    # Entrenamiento y evaluación de modelos
    runner = Model(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        cfg=ml_cfg
    )

    # Registramos los resultados en MLflow
    resultados = runner.evaluate_all()

    # Mostramos resultados en consola
    # Paso el diccionario a DataFrame para tener una tabla ordenada en filas/columnas.
    df = pd.DataFrame(resultados).T.round(4)

    print("\nResultados\n")
    print(tabulate(df, headers="keys", tablefmt="psql"))

    # Guardo un CSV con la tabla de resultados en la carpeta outputs
    df.to_csv("outputs/results_summary.csv")


    # Generamos una gráfica por metrica
    metricas_grafica = ["accuracy", "roc_auc", "f1", "precision", "recall", "train_time"]

    for metrica in metricas_grafica:
        plt.figure(figsize=(8, 5))
        # Creo la gráfica de barras
        ax = df[metrica].plot(kind="bar", color="skyblue")

        # Título y ejes
        plt.title(f"Métrica - {metrica}")
        plt.xticks(rotation=45)

        # Agregamos labels sobre cada barra
        for i, valor in enumerate(df[metrica]):
            ax.text(
                i,
                valor + (valor * 0.01),
                f"{valor:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

        plt.tight_layout()

        # Nombre de archivo
        filename = f"outputs/comparación_modelos_{metrica}.png"
        plt.savefig(filename)
        plt.close()

    print("\nResultados almacenados en la carpeta 'outputs'\n")


# Ejecuta el pipeline
# Esto asegura que main() solo se ejecuta cuando corro este archivo directamente
if __name__ == "__main__":
    main()
