import numpy as np
import pandas as pd

from ml_logic.preprocessor import clean_data, preprocess_features
from ml_logic.registry import load_model

def preprocess():
    pass

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    if X_pred is None:
        X_pred = pd.DataFrame({
            'age': ['[70-80)'],
            'time_in_hospital': [5],
            'n_lab_procedures': [43],
            'n_procedures': [1],
            'n_medications': [16],
            'n_outpatient': [0],
            'n_inpatient': [1],
            'n_emergency':[0],
            'diag_1': ['Circulatory'],
            'diag_2': ['Respiratory'],
            'diag_3': ['Other'],
            'A1Ctest': ['no'],
            'change': ['no'],
            'diabetes_med': ['yes'],
        })

    model = load_model()

    X_preproc = preprocess_features(X_pred)
    y_pred = model.predict(X_preproc)

    print()
    print('Prediction done.', y_pred, type(y_pred), float(y_pred))
    print()

    return y_pred

if __name__ == '__main__':
    preprocess()
    pred()
