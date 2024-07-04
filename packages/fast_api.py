import pandas as pd
from fastapi import FastAPI
from ml_logic.registry import load_model
from ml_logic.preprocessor import preprocess_features
import warnings
import pickle
import shap

app = FastAPI()
app.state.model = load_model('best_model.pkl')
explainer = shap.Explainer('best_model.pkl')

with open("models/explainer.pkl", "wb") as explainer_file:
    pickle.dump(explainer, explainer_file)
# Define the endpoint for root
@app.get("/")
def root():
    return {'greeting': "hello, World!"}

# Define the endpoint for prediction
@app.get('/predict')
def predict(
    race: str = 'Caucasian',
    gender: str = 'Male',
    age: int = 25,
    admission_type_id: str = "1",
    discharge_disposition_id: str = "1",
    admission_source_id: str = "7",
    time_in_hospital: int = 14,
    num_lab_procedures: int = 50,
    num_procedures: int = 1,
    num_medications: int = 11,
    number_diagnoses: int = 5,
    max_glu_serum: str = "-99.0",
    A1Cresult: str = "-99.0",
    metformin: int = 0,
    repaglinide: int = 0,
    nateglinide: int = 0,
    chlorpropamide: int = 0,
    glimepiride: int = 0,
    acetohexamide: int = 0,
    glipizide: int = 0,
    glyburide: int = 0,
    tolbutamide: int = 0,
    pioglitazone: int = 0,
    rosiglitazone: int = 0,
    acarbose: int = 0,
    miglitol: int = 0,
    troglitazone: int = 0,
    tolazamide: int = 0,
    insulin: int = 0,
    glyburide_metformin: int = 0,
    glipizide_metformin: int = 0,
    glimepiride_pioglitazone: int = 0,
    metformin_pioglitazone: int = 0,
    change: int = 0,
    diabetesMed: str = 'Yes',
    comorbidity_count: int = 3,
    total_visits: int = 0,
    long_stay: int = 0,
    numchange: int = 1,
    nummed: int = 1,
    num_medications_time_in_hospital: int = 54,
    num_medications_num_procedures: int = 0,
    time_in_hospital_num_lab_procedures: int = 177,
    num_medications_num_lab_procedures: int = 1062,
    num_medications_number_diagnoses: int = 162,
    age_number_diagnoses: int = 135,
    age_comorbidity_count: int = 30,
    change_num_medications: int = 18,
    number_diagnoses_time_in_hospital: int = 27,
    num_medications_numchange: int = 18,
    level1_diag_1: float = 1,
    level1_diag_2: float = 1,
    level1_diag_3: float = 1
):


    # Prepare input data as a DataFrame
    X_pred = pd.DataFrame({
        'race': [race],
        'gender': [gender],
        'age': [age],
        'admission_type_id': [admission_type_id],
        'discharge_disposition_id': [discharge_disposition_id],
        'admission_source_id': [admission_source_id],
        'time_in_hospital': [time_in_hospital],
        'num_lab_procedures': [num_lab_procedures],
        'num_procedures': [num_procedures],
        'num_medications': [num_medications],
        'number_diagnoses': [number_diagnoses],
        'max_glu_serum': [max_glu_serum],
        'A1Cresult': [A1Cresult],
        'metformin': [metformin],
        'repaglinide': [repaglinide],
        'nateglinide': [nateglinide],
        'chlorpropamide': [chlorpropamide],
        'glimepiride': [glimepiride],
        'acetohexamide': [acetohexamide],
        'glipizide': [glipizide],
        'glyburide': [glyburide],
        'tolbutamide': [tolbutamide],
        'pioglitazone': [pioglitazone],
        'rosiglitazone': [rosiglitazone],
        'acarbose': [acarbose],
        'miglitol': [miglitol],
        'troglitazone': [troglitazone],
        'tolazamide': [tolazamide],
        'insulin': [insulin],
        'glyburide-metformin': [glyburide_metformin],
        'glipizide-metformin': [glipizide_metformin],
        'glimepiride-pioglitazone': [glimepiride_pioglitazone],
        'metformin-pioglitazone': [metformin_pioglitazone],
        'change': [change],
        'diabetesMed': [diabetesMed],
        'comorbidity_count': [comorbidity_count],
        'total_visits': [total_visits],
        'long_stay': [long_stay],
        'numchange': [numchange],
        'nummed': [nummed],
        'num_medications|time_in_hospital': [num_medications_time_in_hospital],
        'num_medications|num_procedures': [num_medications_num_procedures],
        'time_in_hospital|num_lab_procedures': [time_in_hospital_num_lab_procedures],
        'num_medications|num_lab_procedures': [num_medications_num_lab_procedures],
        'num_medications|number_diagnoses': [num_medications_number_diagnoses],
        'age|number_diagnoses': [age_number_diagnoses],
        'age|comorbidity_count': [age_comorbidity_count],
        'change|num_medications': [change_num_medications],
        'number_diagnoses|time_in_hospital': [number_diagnoses_time_in_hospital],
        'num_medications|numchange': [num_medications_numchange],
        'level1_diag_1': [level1_diag_1],
        'level1_diag_2': [level1_diag_2],
        'level1_diag_3': [level1_diag_3]
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
    return {'Hospital readmission': prediction,
            'Probability': probability}

# Define the endpoint for CSV prediction
@app.post('/predict_csv')
def predict_csv(file: UploadFile = File(...)):
    X_pred = pd.read_csv(file.file)

    prediction = app.state.model.predict(X_pred)[0]
    prediction = float(prediction)

    probability = app.state.model.predict_proba(X_pred)[0][1]
    probability = float(probability)

    # Return prediction
    return {'Hospital readmission': prediction,
            'Probability': probability}
