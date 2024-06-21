import pandas as pd
from fastapi import FastAPI
import pickle
#from fastapi.middleware.cors import CORSMiddleware

from ml_logic.preprocessor import preprocess_features
from ml_logic.registry import load_model

app = FastAPI()
app.state.model = load_model()

@app.get("/")
def root():
    return {'greeting': "hello, World!"}

@app.get('/predict')
def predict(
    age='[70-80)',
    diag_1='Circulatory',
    diag_2='Respiratory',
    diag_3='Other',
    A1Ctest='no',
    change='no',
    diabetes_med='yes',
    time_in_hospital=5,
    n_lab_procedures=43,
    n_procedures=1,
    n_medications=16,
    n_outpatient=0,
    n_inpatient=1,
    n_emergency=0
):

    X_pred = pd.DataFrame({
        'age': age,
        'time_in_hospital': time_in_hospital,
        'n_lab_procedures': n_lab_procedures,
        'n_procedures': n_procedures,
        'n_medications': n_medications,
        'n_outpatient': n_outpatient,
        'n_inpatient': n_inpatient,
        'n_emergency':n_emergency,
        'diag_1': diag_1,
        'diag_2': diag_2,
        'diag_3': diag_3,
        'A1Ctest': A1Ctest,
        'change': change,
        'diabetes_med': diabetes_med,
    }, index=[0])

    X_preproc = preprocess_features(X_pred)
    print(X_preproc)
    # prediction = app.state.model.predict(X_preproc)[0]
    # print()
    # print('Prediction done.', prediction, type(prediction), float(prediction))
    # print()

    # return {'Hospital readmission:': prediction}
    return 'NO MODEL YET'
