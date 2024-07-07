import pandas as pd
from fastapi import FastAPI
from ml_logic.registry import load_model
from ml_logic.preprocessor import preprocess_features
import warnings


app = FastAPI()
app.state.model = load_model('best_model.pkl')

# Define the endpoint for root
@app.get("/")
def root():
    return {'greeting': "hello, World!"}

# Define the endpoint for prediction
@app.get('/predict')
def predict(
    age: int = 15,
    gender: str = 'Female',
    race: str = 'AfricanAmerican',
    level1_diag_1: float = 0,
    level1_diag_2: float = 4,
    level1_diag_3: float = 0,
    num_medications: int = 18,
    num_lab_procedures: int = 59,
    num_procedures: int = 0,
    numchange: int = 1,
    nummed: int = 1,
    A1Cresult: str = "-99.0",
    metformin: int = 0,
    pioglitazone: int = 0,
    insulin: int = 1,
    glipizide: int = 0,
    glimepiride: int = 0,
    diabetesMed: str = 'Yes',
    comorbidity_count: int = 2,
    number_diagnoses: int = 9,
    admission_type_id: str = "1",
    discharge_disposition_id: str = "1",
    admission_source_id: str = "1",
    total_visits: int = 0,
    time_in_hospital: int = 3,
    change: int = 1,
    age_comorbidity_count: int = 30,
    age_number_diagnoses: int = 135,
    num_medications_num_lab_procedures: int = 1062,
    num_medications_time_in_hospital: int = 56,
    num_medications_num_procedures: int = 0,
    num_medications_number_diagnoses: int = 162,
    number_diagnoses_time_in_hospital: int = 27,
    time_in_hospital_num_lab_procedures: int = 177,
):


    # Prepare input data as a DataFrame
    X_pred = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'race': [race],
        'level1_diag_1': [level1_diag_1],
        'level1_diag_2': [level1_diag_2],
        'level1_diag_3': [level1_diag_3],
        'num_medications': [num_medications],
        'num_lab_procedures': [num_lab_procedures],
        'num_procedures': [num_procedures],
        'numchange': [numchange],
        'nummed': [nummed],
        'A1Cresult': [A1Cresult],
        'metformin': [metformin],
        'pioglitazone': [pioglitazone],
        'insulin': [insulin],
        'glipizide': [glipizide],
        'glimepiride': [glimepiride],
        'diabetesMed': [diabetesMed],
        'comorbidity_count': [comorbidity_count],
        'number_diagnoses': [number_diagnoses],
        'admission_type_id': [admission_type_id],
        'discharge_disposition_id': [discharge_disposition_id],
        'admission_source_id': [admission_source_id],
        'total_visits': [total_visits],
        'time_in_hospital': [time_in_hospital],
        'change': [change],
        'age|comorbidity_count': [age_comorbidity_count],
        'age|number_diagnoses': [age_number_diagnoses],
        'num_medications|num_lab_procedures': [num_medications_num_lab_procedures],
        'num_medications|time_in_hospital': [num_medications_time_in_hospital],
        'num_medications|num_procedures': [num_medications_num_procedures],
        'num_medications|number_diagnoses': [num_medications_number_diagnoses],
        'number_diagnoses|time_in_hospital': [number_diagnoses_time_in_hospital],
        'time_in_hospital|num_lab_procedures': [time_in_hospital_num_lab_procedures]

    }, index=[0])


    # Make prediction
    prediction = app.state.model.predict(X_pred)[0]
    prediction = float(prediction)

    probability = app.state.model.predict_proba(X_pred)[0][1]
    probability = float(probability)


    print()
    print('Prediction done.')
    print('Readmission: ', prediction)
    print('Readmission Probability: ', probability)
    print()


    # Return prediction
    return {'Hospital readmission': float(prediction),
            'Probability': float(probability)}

# Define the endpoint for CSV prediction
@app.post('/predict_csv')
def predict_csv(file: UploadFile = File(...)):
    X_pred = pd.read_csv(file.file)

    prediction = app.state.model.predict(X_pred)[0]
    prediction = float(prediction)

    probability = app.state.model.predict_proba(X_pred)[0][1]
    probability = float(probability)

    # Return prediction
    return {'Hospital readmission': float(prediction),
            'Probability': float(probability)}
