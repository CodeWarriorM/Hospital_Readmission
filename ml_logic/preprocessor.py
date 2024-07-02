import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from ml_logic.data import DataCleaner

def preprocessor() -> Pipeline:
    project_path = os.path.dirname(os.path.dirname(__file__))

    with open(project_path + '/preprocessor/preprocessing_pipeline.pkl', 'rb') as file:
        pipe_preproc = pickle.load(file)

    return pipe_preproc

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    print(f'Preprocessing {X.shape[0]} rows of {X.shape[1]} features...')

    X_proc = preprocessor().transform(X)
    print(f'Preprocessing done. Final shape: {X_proc.shape}')

    return X_proc
