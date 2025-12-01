# Librería estándar
import os
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Módulos propios
from module_path import train_data_path, test_data_path

COL_PATIENT_ID = "patient_id"
COL_PATIENT_RACE = "patient_race"
COL_PAYER_TYPE = "payer_type"
COL_PATIENT_STATE = "patient_state"
COL_PATIENT_ZIP3 = "patient_zip3"
COL_PATIENT_AGE = "patient_age"
COL_PATIENT_GENDER = "patient_gender"
COL_BMI = "bmi"
COL_BREAST_CANCER_DIAGNOSIS_CODE = "breast_cancer_diagnosis_code"
COL_BREAST_CANCER_DIAGNOSIS_DESC = "breast_cancer_diagnosis_desc"
COL_METASTATIC_CANCER_DIAGNOSIS_CODE = "metastatic_cancer_diagnosis_code"
COL_METASTATIC_FIRST_NOVEL_TREATMENT = "metastatic_first_novel_treatment"
COL_METASTATIC_FIRST_NOVEL_TREATMENT_TYPE = "metastatic_first_novel_treatment_type"
COL_REGION = "Region"
COL_DIVISION = "Division"
COL_POPULATION = "population"
COL_DENSITY = "density"
COL_AGE_MEDIAN = "age_median"
COL_AGE_UNDER_10 = "age_under_10"
COL_AGE_10_TO_19 = "age_10_to_19"
COL_AGE_20S = "age_20s"
COL_AGE_30S = "age_30s"
COL_AGE_40S = "age_40s"
COL_AGE_50S = "age_50s"
COL_AGE_60S = "age_60s"
COL_AGE_70S = "age_70s"
COL_AGE_OVER_80 = "age_over_80"
COL_MALE = "male"
COL_FEMALE = "female"
COL_MARRIED = "married"
COL_DIVORCED = "divorced"
COL_NEVER_MARRIED = "never_married"
COL_WIDOWED = "widowed"
COL_FAMILY_SIZE = "family_size"
COL_FAMILY_DUAL_INCOME = "family_dual_income"
COL_INCOME_HOUSEHOLD_MEDIAN = "income_household_median"
COL_INCOME_HOUSEHOLD_UNDER_5 = "income_household_under_5"
COL_INCOME_HOUSEHOLD_5_TO_10 = "income_household_5_to_10"
COL_INCOME_HOUSEHOLD_10_TO_15 = "income_household_10_to_15"
COL_INCOME_HOUSEHOLD_15_TO_20 = "income_household_15_to_20"
COL_INCOME_HOUSEHOLD_20_TO_25 = "income_household_20_to_25"
COL_INCOME_HOUSEHOLD_25_TO_35 = "income_household_25_to_35"
COL_INCOME_HOUSEHOLD_35_TO_50 = "income_household_35_to_50"
COL_INCOME_HOUSEHOLD_50_TO_75 = "income_household_50_to_75"
COL_INCOME_HOUSEHOLD_75_TO_100 = "income_household_75_to_100"
COL_INCOME_HOUSEHOLD_100_TO_150 = "income_household_100_to_150"
COL_INCOME_HOUSEHOLD_150_OVER = "income_household_150_over"
COL_INCOME_HOUSEHOLD_SIX_FIGURE = "income_household_six_figure"
COL_INCOME_INDIVIDUAL_MEDIAN = "income_individual_median"
COL_HOME_OWNERSHIP = "home_ownership"
COL_HOUSING_UNITS = "housing_units"
COL_HOME_VALUE = "home_value"
COL_RENT_MEDIAN = "rent_median"
COL_RENT_BURDEN = "rent_burden"
COL_EDUCATION_LESS_HIGHSCHOOL = "education_less_highschool"
COL_EDUCATION_HIGHSCHOOL = "education_highschool"
COL_EDUCATION_SOME_COLLEGE = "education_some_college"
COL_EDUCATION_BACHELORS = "education_bachelors"
COL_EDUCATION_GRADUATE = "education_graduate"
COL_EDUCATION_COLLEGE_OR_ABOVE = "education_college_or_above"
COL_EDUCATION_STEM_DEGREE = "education_stem_degree"
COL_LABOR_FORCE_PARTICIPATION = "labor_force_participation"
COL_UNEMPLOYMENT_RATE = "unemployment_rate"
COL_SELF_EMPLOYED = "self_employed"
COL_FARMER = "farmer"
COL_RACE_WHITE = "race_white"
COL_RACE_BLACK = "race_black"
COL_RACE_ASIAN = "race_asian"
COL_RACE_NATIVE = "race_native"
COL_RACE_PACIFIC = "race_pacific"
COL_RACE_OTHER = "race_other"
COL_RACE_MULTIPLE = "race_multiple"
COL_HISPANIC = "hispanic"
COL_DISABLED = "disabled"
COL_POVERTY = "poverty"
COL_LIMITED_ENGLISH = "limited_english"
COL_COMMUTE_TIME = "commute_time"
COL_HEALTH_UNINSURED = "health_uninsured"
COL_VETERAN = "veteran"
COL_OZONE = "Ozone"
COL_PM25 = "PM25"
COL_N02 = "N02"
COL_DIAGPERIODL90D = "DiagPeriodL90D"

class Dataset():

    def __init__(self, num_samples:int=None, seed:int=100):
        self.num_samples = num_samples
        self.seed = seed

    def load_data(self):
        
        train_path = train_data_path()
        test_path = test_data_path()

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        cols=['patient_id','breast_cancer_diagnosis_desc',
                 'metastatic_first_novel_treatment',
                 'metastatic_first_novel_treatment_type']
        df_train = df_train.drop(columns=cols)
        df_test = df_test.drop(columns=cols)

        if self.num_samples is not None:
            df_train = df_train.sample(n=self.num_samples,
                                       random_state=self.seed)
            df_test = df_test.sample(n=self.num_samples,
                                     random_state=self.seed)

        return df_train, df_test
    
    def load_data_clean(self):
        df_train, df_test = self.load_data()

        # BMI
        df_train['bmi_category'] = pd.cut(df_train['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9,39.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II','Extreme']).astype(object)
        df_test['bmi_category'] = pd.cut(df_test['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9,39.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II','Extreme']).astype(object)
        df_train['bmi_category'] = df_train['bmi_category'].fillna('Unknown')
        df_test['bmi_category'] = df_test['bmi_category'].fillna('Unknown')
        df_train.drop(columns=['bmi'], inplace=True)
        df_test.drop(columns=['bmi'], inplace=True)

        # Race
        df_train['patient_race'] = df_train['patient_race'].fillna('Unknown')
        df_test['patient_race'] = df_test['patient_race'].fillna('Unknown')

        # Payment
        df_train['payer_type'] = df_train['payer_type'].fillna('Unknown')
        df_test['payer_type'] = df_test['payer_type'].fillna('Unknown')
        
        # Drop na (only training)
        df_train = df_train.dropna()

        # Fill test data
        test_categories = df_test.select_dtypes(include=['object']).columns.tolist()
        test_numeric = df_test.select_dtypes(include=['number']).columns.tolist()
        num_imputer = SimpleImputer(strategy='mean')
        df_test[test_numeric] = num_imputer.fit_transform(df_test[test_numeric])
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_test[test_categories] = cat_imputer.fit_transform(df_test[test_categories])
       
        return df_train, df_test
    
    def load_data_clean_encoded(self, method:int="std"):
        df_train, df_test = self.load_data_clean()

        train_categories = df_train.select_dtypes(include=['object']).columns.tolist()
        train_numeric = df_train.select_dtypes(include=['number']).columns.tolist()
        train_numeric.remove(COL_DIAGPERIODL90D)

        # Escalamiento de variables numéricas

        if method == 'std':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(df_train[train_numeric]) # array de numpy
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=train_numeric, index=df_train.index)
        
        # Reescribir mi dataframe original
        target = df_train[COL_DIAGPERIODL90D]
        df_train = pd.merge(df_train[train_categories], X_train_scaled_df, left_index=True, right_index=True)
        df_train = pd.merge(df_train, target, left_index=True, right_index=True)
        train_numeric.append(COL_DIAGPERIODL90D)

        # Codificación de variables categóricas
        # Instanciamos el codificador OneHotEncoder
        ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=int)

        # Aplicamos la codificación con ColumnTransformer
        column_transformer = ColumnTransformer(
            transformers=[
           ('cat', ohe, train_categories)
            ],
            remainder='passthrough'  # deja pasar las columnas numéricas sin cambios
        )

        # Aplicamos el transformador
        X_train_encoded = column_transformer.fit_transform(df_train)
        #X_test_encoded = column_transformer.fit_transform(df_test)

        # Recuperamos los nombres de las columnas dummy codificadas
        encoded_col_names = column_transformer.named_transformers_['cat'].get_feature_names_out(train_categories)

        # Combinamos nombres codificados y los que pasaron directamente
        final_columns = list(encoded_col_names) + train_numeric  # agregar nombres de las columnas que pasaron sin transformación

        # Convertimos a DataFrame para inspección
        df_train_encoded = pd.DataFrame(X_train_encoded, columns=final_columns)
        #df_test_encoded = pd.DataFrame(X_test_encoded, columns=final_columns)

        # Mostramos las primeras filas
        print(df_train_encoded.head())

        #df_train_encoded = pd.get_dummies(df_train, columns=test_categories, drop_first=False, dtype=int)
        #df_test_encoded = pd.get_dummies(df_test, columns=test_categories, drop_first=False, dtype=int)

        return df_train_encoded

    
    def load_xy(self, method:str='std'):
        df_train = self.load_data_clean_encoded(method=method)

        X = df_train.drop(columns=[COL_DIAGPERIODL90D])
        y = df_train[COL_DIAGPERIODL90D]

        return X, y
