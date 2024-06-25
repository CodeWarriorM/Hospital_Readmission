import numpy as np
import pandas as pd

from ml_logic.data import clean_data

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    def preprocessor() -> ColumnTransformer:
        data_cleaner = FunctionTransformer(clean_data)

        num_preproc = Pipeline([
            ('scaler', MinMaxScaler()),
        ])

        cat_preproc = Pipeline([
            ('ohe', OneHotEncoder(sparse_output=False, drop="if_binary")),
        ])

        preproc = ColumnTransformer([
            ('num_transf', num_preproc, make_column_selector(dtype_include='number')),
            ('cat_transf', cat_preproc, make_column_selector(dtype_include='object')),
        ], verbose_feature_names_out=False).set_output(transform='pandas')

        pipe_preproc = Pipeline([
            ('data_cleaner', data_cleaner),
            ('preprocessor', preproc),
        ])

        return pipe_preproc

    print()
    print(f'Preprocessing {X.shape[0]} rows of {X.shape[1]} features...')
    X_proc = preprocessor().fit_transform(X)
    print(f'Preprocessing done. Final shape: {X_proc.shape}')

    return X_proc
