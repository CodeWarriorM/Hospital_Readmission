import streamlit as st
import requests
import matplotlib.pyplot as plt
from packages.utils import classify_diag_level1

# Title and description
st.title('Hospital Readmission Prediction App')
st.write('Select your features below:')

# Sidebar sliders for user input
age = st.slider('Age', min_value=1, max_value=100, value=25, step=1)
gender = st.selectbox('Gender', ['Male','Female'])
race = st.selectbox('Race', ['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'])
diag_1 = st.slider('Diagnosis 1 (ICD-9 Code)', min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
insulin = st.selectbox('Insulin', [0, 1])
change = st.selectbox('Change', [0, 1])
num_lab_procedures = st.slider('Number of Lab Procedures', min_value=1, max_value=150, value=50, step=1)
num_medications = st.slider('Number of Medications', min_value=1, max_value=100, value=11, step=1)
number_diagnoses = st.slider('Number of Diagnoses', min_value=1, max_value=20, value=5, step=1)
admission_type_id = st.selectbox('Admission Type ID', ['1', '2', '3', '4', '5'])
time_in_hospital = st.slider('Time in Hospital', min_value=1, max_value=14, value=14, step=1)

#Params compute
long_stay = 1 if time_in_hospital > 7 else 0

# Button to trigger prediction
if st.button('Predict'):
    # Convert inputs to appropriate types
    params = {
        'age': int(age),
        'gender': gender,
        'race': race,
        'level1_diag_1': float(classify_diag_level1(diag_1)),
        'insulin': int(insulin),
        'change': int(change),
        'num_lab_procedures': int(num_lab_procedures),
        'num_medications': int(num_medications),
        'number_diagnoses': int(number_diagnoses),
        'admission_type_id': str(admission_type_id),
        'time_in_hospital': int(time_in_hospital),
        'long_stay': int(long_stay),

        # Fixed
        'discharge_disposition_id': "1",
        'admission_source_id': "7",
        'max_glu_serum': "-99.0",
        'A1Cresult': "-99.0",
        'metformin': 0,
        'repaglinide': 0,
        'nateglinide': 0,
        'chlorpropamide': 0,
        'glimepiride': 0,
        'acetohexamide': 0,
        'glipizide': 0,
        'glyburide': 0,
        'tolbutamide': 0,
        'pioglitazone': 0,
        'rosiglitazone': 0,
        'acarbose': 0,
        'miglitol': 0,
        'troglitazone': 0,
        'tolazamide': 0,
        'glyburide_metformin': 0,
        'glipizide_metformin': 0,
        'glimepiride_pioglitazone': 0,
        'metformin_pioglitazone': 0,
        'diabetesMed': 'Yes',
        'comorbidity_count': 3,
        'total_visits': 0,
        'numchange': 0,
        'nummed': 1,

        # Interaction
        'num_medications_time_in_hospital': 54,
        'num_medications_num_procedures': 0,
        'time_in_hospital_num_lab_procedures': 177,
        'num_medications_num_lab_procedures': 1062,
        'num_medications_number_diagnoses': 162,
        'age_number_diagnoses': 135,
        'age_comorbidity_count': 30,
        'change_num_medications': 18,
        'number_diagnoses_time_in_hospital': 27,
        'num_medications_numchange': 18,

    }

    # Endpoint URL of your FastAPI application
    url = 'https://hospitalreadmission1575-jgoxnpqt5a-ew.a.run.app/predict'

    try:
        # Make a GET request to the FastAPI endpoint
        response = requests.get(url, params=params)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            result = response.json()

            # Access the correct keys from the response
            prediction = result['Hospital readmission']
            probability = result['Probability']

            # Display prediction in a user-friendly format
            readmission_text = 'No' if prediction == 0.0 else 'Yes'
            color = 'red' if prediction == 0.0 else 'green'

            st.markdown(f"<h2 style='color:{color};'>Predicted Hospital Readmission: <b>{readmission_text}</b></h2>", unsafe_allow_html=True)

            # Create a bar plot for the probability
            # Create a donut plot for the probability
            fig, ax = plt.subplots()
            size = 0.3

            # Create data for the donut chart
            values = [probability, 1 - probability]
            colors = ['#66b3ff', '#e6e6e6']

            ax.pie(values, colors=colors, radius=1, wedgeprops=dict(width=size, edgecolor='w'))

            # Add a circle in the center to create the donut shape
            centre_circle = plt.Circle((0,0), 1-size, color='white', fc='white', linewidth=0)
            fig.gca().add_artist(centre_circle)

            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')

            # Add the percentage text in the center of the donut chart
            plt.text(0, 0, f'{probability * 100:.2f}%', ha='center', va='center', fontsize=20, color='black')


            # Display the plot in Streamlit
            st.pyplot(fig)

        else:
            st.write('Failed to receive prediction. Please try again.')

    except requests.exceptions.RequestException as e:
        st.write('Error making prediction request:', e)
