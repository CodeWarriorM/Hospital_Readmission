from fastapi import FastAPI
import pickle
#from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.get("/")
def root():
    return {'greeting': "hello, World!"}

@app.get('/predict')
def predict( #add non-numeric predicting variables
    time_in_hospital=5,
    n_lab_procedures=43,
    n_procedures=1,
    n_medications=16,
    n_outpatient=0,
    n_inpatient=1,
    n_emergency=0
):
    with open('../models/test_model.pkl', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict([[time_in_hospital, n_lab_procedures, n_procedures, n_medications,n_outpatient,n_inpatient,n_emergency]])[0]
    return {'Hospital readmission:' : prediction}
