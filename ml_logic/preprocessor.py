import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    def make_clean_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['diag_1'] != 'Missing']
        df = df[df['diag_2'] != 'Missing']
        df = df[df['diag_3'] != 'Missing']

        df['age'] = df['age'].map({'[0-10]': 0.0,
                                    '[10-20)': 0.1,
                                    '[20-30)': 0.2,
                                    '[30-40)': 0.3,
                                    '[40-50)': 0.4,
                                    '[50-60)': 0.5,
                                    '[60-70)': 0.6,
                                    '[70-80)': 0.7,
                                    '[80-90)': 0.8,
                                    '[90-100)': 0.9,
                                    '[100-110)': 1.0})
        df['n_lab_procedures_grouped'] = (df['n_lab_procedures'] // 10).astype(int)
        df['n_medications_grouped'] = (df['n_medications'] // 5).astype(int)
        df['n_outpatient'] = df['n_outpatient'].map({0: 0, 1: 1}).fillna(2).astype(int)
        df['n_inpatient'] = df['n_inpatient'].map({0: 0, 1: 1}).fillna(2).astype(int)
        df['n_emergency'] = df['n_emergency'].map({0: 0, 1: 1}).fillna(2).astype(int)

        df = df.drop(columns=['n_lab_procedures',
                            # 'medical_specialty',
                            # 'glucose_test',
                            'n_medications'],
                    )
        return df

    def preprocessor() -> Pipeline:
        data_cleaner = FunctionTransformer(make_clean_data)

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
