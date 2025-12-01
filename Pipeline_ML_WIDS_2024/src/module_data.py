# module_data.py
"""
El modulo realiza lo siguiente:
- Carga de los datasets de entrenamiento y prueba
- Limpieza básica de columnas
- Creación de categorías de BMI
- Manejo de valores nulos
- División Train / Validation
- Definición del preprocesador (imputación + escalado + OneHotEncoder)

"""

# Importamos las librerias
from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Variable objetivo en el dataset de WIDS
TARGET_COL = "DiagPeriodL90D"


@dataclass
# Clase de configuración para controlar el comportamiento del módulo de datos
class DataConfig:
    target_col: str = TARGET_COL
    null_thresh: float = 0.30
    scaler: str = "standard"
    test_size: float = 0.20
    random_state: int = 42
    num_samples: Optional[int] = None

# Clase principal para el manejo de datos
class Dataset:
    def __init__(self, config: Optional[DataConfig] = None) -> None:
        # Si no se pasa configuración usamos default
        self.cfg = config or DataConfig()

        self.logger = logging.getLogger(__name__)

        # Aquí se guardará el ColumnTransformer una vez construido
        self.preprocessor: Optional[ColumnTransformer] = None

        # Listas para recordar qué columnas son numéricas y cuáles categóricas
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []


    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Cargamos los datos
        train_df = pd.read_csv("Dataset/training.csv")
        test_df = pd.read_csv("Dataset/test.csv")

        # Si queremos trabajar con una muestra podemos cambiar aqui la configuración
        if self.cfg.num_samples:
            train_df = train_df.sample(
                n=self.cfg.num_samples,
                random_state=self.cfg.random_state
            )

        return train_df, test_df

    # Función que elimina columnas cuyo porcentaje de valores nulos supere el umbral
    def drop_high_null_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # Porcentaje de nulos por columna
        pct = df.isna().mean()  
        cols = pct[pct > self.cfg.null_thresh].index.tolist()

        if cols:
            self.logger.info(f"Eliminando columnas con muchos nulos: {cols}")
            df = df.drop(columns=cols)

        return df

    # Función que aplica una limpieza básica sobre train y test:
    def basic_cleaning(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
       
        # Convertimos BMI a categorías
        if "bmi" in train_df.columns:
            # Defino los rangos y etiquetas
            bins = [0, 18.5, 25, 30, 35, 40, 100]
            labels = [
                "Underweight", "Normal", "Overweight",
                "Obesity I", "Obesity II", "Extreme"
            ]

            # pd.cut crea la variable categórica a partir del valor de bmi
            train_df["bmi_category"] = pd.cut(train_df["bmi"], bins=bins, labels=labels)
            test_df["bmi_category"] = pd.cut(test_df["bmi"], bins=bins, labels=labels)

            # Agregamos la categoría "Unknown" y relleno NaN con ella
            train_df["bmi_category"] = (
                train_df["bmi_category"]
                .cat.add_categories(["Unknown"])
                .fillna("Unknown")
            )
            test_df["bmi_category"] = (
                test_df["bmi_category"]
                .cat.add_categories(["Unknown"])
                .fillna("Unknown")
            )

            # Eliminamos la versión numérica original ya que no es necesaria
            train_df.drop(columns=["bmi"], inplace=True)
            test_df.drop(columns=["bmi"], inplace=True)

        if "patient_race" in train_df.columns:
            train_df["patient_race"] = train_df["patient_race"].fillna("Unknown")
            test_df["patient_race"] = test_df["patient_race"].fillna("Unknown")

        if "payer_type" in train_df.columns:
            train_df["payer_type"] = train_df["payer_type"].fillna("Unknown")
            test_df["payer_type"] = test_df["payer_type"].fillna("Unknown")

        # Imputación para el test set
        num_cols = test_df.select_dtypes(include="number").columns
        cat_cols = test_df.select_dtypes(include="object").columns

        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy="mean")
            test_df[num_cols] = num_imputer.fit_transform(test_df[num_cols])

        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            test_df[cat_cols] = cat_imputer.fit_transform(test_df[cat_cols])

        return train_df, test_df

    # Función que construye el ColumnTransformer que se usará para preprocesar tanto X_train como X_val
    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        # Identifico qué columnas son numéricas y cuáles categóricas
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Selecciono el escalador según la configuración
        scaler = MinMaxScaler() if self.cfg.scaler == "minmax" else StandardScaler()

        # Pipeline para variables numéricas: imputo con mediana y escalo
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ])

        # Para variables categóricas imputamos con el valor más frecuente y aplicamos OneHotEncoder
        # Uso sparse_output=False para obtener un array denso que es mejor para modelos como Naive Bayes
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        # ColumnTransformer que aplica num_pipe y cat_pipe a sus columnas respectivas
        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.num_cols_),
                ("cat", cat_pipe, self.cat_cols_),
            ]
        )

        return transformer

    # Función que ejecuta todo el flujo de datos:
    def get_train_val(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Cargo los datos
        train_df, test_df = self.load_raw()

        # Limpieza básica (creación de BMI, nulos básicos, etc.)
        train_df, test_df = self.basic_cleaning(train_df, test_df)

        # Elimino columnas con demasiados nulos solo en el train
        train_df = self.drop_high_null_cols(train_df)

        # Separo target (y) y features (X)
        y = train_df[self.cfg.target_col].astype(int)
        X = train_df.drop(columns=[self.cfg.target_col])

        # Split Train / Validation estratificado para respetar la proporción de clases
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )

        # Construyo el preprocesador a partir de X_train
        self.preprocessor = self.build_preprocessor(X_train)

        # Ajusto y transformo
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_val_proc = self.preprocessor.transform(X_val)

        # Devuelvo los arrays ya listos para alimentar a los modelos
        return X_train_proc, X_val_proc, y_train.values, y_val.values
