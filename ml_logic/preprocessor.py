import os
import pickle
import numpy as np
import pandas as pd

from ml_logic.data import clean_training_data

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    def preprocessor() -> Pipeline:
        project_path = os.path.dirname(os.path.dirname(__file__))

        with open(project_path + '/preprocessor/preprocessor.pkl', 'rb') as file:
            pipe_preproc = pickle.load(file)

        return pipe_preproc

    print()
    print(f'Preprocessing {X.shape[0]} rows of {X.shape[1]} features...')
    X_proc = preprocessor().transform(X)
    print(f'Preprocessing done. Final shape: {X_proc.shape}')

    return X_proc
