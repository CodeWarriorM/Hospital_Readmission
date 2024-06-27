import streamlit as st
import requests
import pandas as pd
from PIL import Image
import base64
import matplotlib.pyplot as plt
import shap
import numpy as np
from ml_logic.data import map_age, map_n_diverse

#st.image('raw_data/hospital.jpg', caption='Hospital', use_column_width=True)
def convert_jpg_to_png(jpg_file, png_file):
    with Image.open(jpg_file) as img:
        img.save(png_file, 'PNG')

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data=f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    body {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

jpg_file = 'raw_data/hospital.jpg'
png_file = 'raw_data/hospital.png'
convert_jpg_to_png(jpg_file, png_file)

set_background(png_file)

st.title('Hospital Readmission App')
st.header('Will my patient be readmitted to the hospital within 30 days after discharge?')

st.write('Select your features below:')

age = st.selectbox('Age', ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
diag_1 = st.selectbox('Primary Diagnosis', ['Circulatory',  'Injury', 'Digestive', 'Respiratory','Diabetes', 'Musculoskeletal', 'Other','Missing'])
diag_2 = st.selectbox('Secondary Diagnosis 1', ['Circulatory',  'Injury', 'Digestive', 'Respiratory','Diabetes', 'Musculoskeletal', 'Other','Missing'])
diag_3 = st.selectbox('Secondary Diagnosis 2', ['Circulatory',  'Injury', 'Digestive', 'Respiratory', 'Diabetes', 'Musculoskeletal','Other', 'Missing'])
A1Ctest = st.selectbox('A1C Test Result', ['no', 'yes'])
change = st.selectbox('Change in Medication', ['no', 'yes'])
diabetes_med = st.selectbox('On Diabetes Medication', ['no', 'yes'])
time_in_hospital = st.slider('Select a value for number of days in hospital', min_value=1, max_value=14, value=5, step=1)
n_lab_procedures = st.slider('Select a value for number of procedures performed', min_value=1, max_value=113, value=43, step=1)
n_procedures = st.slider('Select a value for number of laboratory procedures', min_value=0, max_value=6, value=1, step=1)
n_medications = st.slider('Select a value for number of medications administered', min_value=1, max_value=79, value=16, step=1)
n_outpatient = st.slider('Select a value for number of outpatient visits', min_value=0, max_value=33, value=0, step=1)
n_inpatient = st.slider('Select a value for number of inpatient visits', min_value=0, max_value=15, value=0, step=1)
n_emergency = st.slider('Select a value for number of visits to the emergency room', min_value=0, max_value=64, value=0, step=1)

# preprocess some values
age = map_age(pd.Series(age))[0]
n_outpatient = map_n_diverse(pd.Series(n_outpatient))[0]
n_inpatient = map_n_diverse(pd.Series(n_inpatient))[0]
n_emergency = map_n_diverse(pd.Series(n_emergency))[0]

if st.button('Predict'):
    # Create a DataFrame from the input
    X_pred = pd.DataFrame({
        'age': [age],
        'time_in_hospital': [time_in_hospital],
        'n_lab_procedures': [n_lab_procedures],
        'n_procedures': [n_procedures],
        'n_medications': [n_medications],
        'n_outpatient': [n_outpatient],
        'n_inpatient': [n_inpatient],
        'n_emergency': [n_emergency],
        'diag_1': [diag_1],
        'diag_2': [diag_2],
        'diag_3': [diag_3],
        'A1Ctest': [A1Ctest],
        'change': [change],
        'diabetes_med': [diabetes_med],
    })

# Method 1
url = 'https://hospitalreadmission1575-jgoxnpqt5a-ew.a.run.app/predict'
params = {'time_in_hospital': time_in_hospital,
          'n_lab_procedures': n_lab_procedures,
          'n_procedures': n_procedures,
          'n_medications': n_medications,
          'n_outpatient': n_outpatient,
          'n_inpatient': n_inpatient,
          'n_emergency': n_emergency,
          'age' : age,
          'diag_1' : diag_1,
          'diag_2' : diag_2,
          'diag_3' : diag_3,
          'A1Ctest' : A1Ctest,
          'change' : change,
          'diabetes_med' : diabetes_med
          }


response = requests.get(url=url,
                        params=params).json()



st.write('Will my patient be readmitted?')
st.write(str(response['Hospital readmission:'].capitalize()))

def display_shap_summary_plot(shap_values, X):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.title('SHAP Summary Plot')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
    plt.clf()
st.title("SHAP Summary Plot")

if st.button('Generate SHAP Summary Plot'):
    display_shap_summary_plot(shap.shap_values, X_pred)
