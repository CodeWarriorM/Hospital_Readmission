import streamlit as st
import requests

st.title('Hospital Readmission app')

st.write('Select your features below:')

time_in_hospital = st.slider('Select a value for number of days in hospital', min_value=1, max_value=14, value=5, step=1)
n_lab_procedures = st.slider('Select a value for number of procedures performed', min_value=1, max_value=113, value=43, step=1)
n_procedures = st.slider('Select a value for number of laboratory procedures', min_value=0, max_value=6, value=1, step=1)
n_medications = st.slider('Select a value for number of medications administered', min_value=1, max_value=79, value=16, step=1)
n_outpatient = st.slider('Select a value for number of outpatient visits', min_value=0, max_value=33, value=0, step=1)
n_inpatient = st.slider('Select a value for number of inpatient visits', min_value=0, max_value=15, value=0, step=1)
n_emergency = st.slider('Select a value for number of visits to the emergency room', min_value=0, max_value=64, value=0, step=1)

# Method 1
url = 'https://hospitalreadmission1575-jgoxnpqt5a-ew.a.run.app/predict'
params = {'time_in_hospital': time_in_hospital,
          'n_lab_procedures': n_lab_procedures,
          'n_procedures': n_procedures,
          'n_medications': n_medications,
          'n_outpatient': n_outpatient,
          'n_inpatient': n_inpatient,
          'n_emergency': n_emergency,
          }


response = requests.get(url=url,
                        params=params).json()



st.write('Hospital readmission is expected', str(response['Hospital readmission:']))
