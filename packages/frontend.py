import streamlit as st
import requests
import matplotlib.pyplot as plt

# Title and description
st.title('Hospital Readmission Prediction App')
st.write('Select your features below:')

# Mapping
diag_1_mapping = {
    'Circulatory': 1,
    'Respiratory': 2,
    'Digestive': 3,
    'Diabetes': 4,
    'Injury': 5,
    'Musculoskeletal': 6,
    'Genitourinary': 7,
    'Neoplasms': 8,
    'Others': 9
}

insulin_mapping = {
    'Not prescribed': 0,
    'Not Adjusted': 0,
    'Increased': 1,
    'Decreased': 1,
}

change_mapping = {
    'No': 0,
    'Yes': 1,
}

admission_type_mapping = {
    'Emergency': 1,
    'Urgent': 1,
    'Elective': 3,
    'Newborn': 4,
    'Not Available': 5,
    'Trauma Center': 5,
    'Not Mapped': 1,
    'Other': 5
}

# Sidebar sliders for user input
age = st.slider('Age', min_value=1, max_value=100, value=25, step=1)
gender = st.selectbox('Gender', ['Male','Female'])
race = st.selectbox('Race', ['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'])
diag_1 = st.selectbox('Primary diagnosis', list(diag_1_mapping.keys()))
insulin = st.selectbox('Insulin', list(insulin_mapping.keys()))
change = st.selectbox('Change of diabetic medications', list(change_mapping.keys()))
num_lab_procedures = st.slider('Number of Lab Procedures', min_value=1, max_value=150, value=50, step=1)
num_medications = st.slider('Number of Medications', min_value=1, max_value=100, value=11, step=1)
number_diagnoses = st.slider('Number of Diagnoses', min_value=1, max_value=20, value=5, step=1)
admission_type = st.selectbox('Admission Type', list(admission_type_mapping.keys()))
time_in_hospital = st.slider('Time in Hospital', min_value=1, max_value=14, value=14, step=1)

#Params compute
long_stay = 1 if time_in_hospital > 7 else 0
diag_1_number = diag_1_mapping[diag_1]
insulin_number = insulin_mapping[insulin]
change_number = change_mapping[change]
admission_type_id = admission_type_mapping[admission_type]

num_medications_time_in_hospital = num_medications*time_in_hospital
num_medications_num_procedures = num_medications*1
time_in_hospital_num_lab_procedures = time_in_hospital*num_lab_procedures
num_medications_num_lab_procedures = num_medications*num_lab_procedures
num_medications_number_diagnoses = num_medications*number_diagnoses
age_number_diagnoses = age*number_diagnoses
age_comorbidity_count = age*3
change_num_medications = change_number*num_medications
number_diagnoses_time_in_hospital = number_diagnoses*time_in_hospital
num_medications_numchange = num_medications*0

# Button to trigger prediction
if st.button('Predict'):
    # Convert inputs to appropriate types
    params = {
        'age': int(age),
        'gender': gender,
        'race': race,
        'level1_diag_1': float(diag_1_number),
        'insulin': int(insulin_number),
        'change': int(change_number),
        'num_lab_procedures': int(num_lab_procedures),
        'num_medications': int(num_medications),
        'number_diagnoses': int(number_diagnoses),
        'admission_type_id': str(admission_type_id),
        'time_in_hospital': int(time_in_hospital),
        'long_stay': int(long_stay),

        # Fixed
        'num_procedures': 1,
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
        'num_medications_time_in_hospital': num_medications_time_in_hospital,
        'num_medications_num_procedures': num_medications_num_procedures,
        'time_in_hospital_num_lab_procedures': time_in_hospital_num_lab_procedures,
        'num_medications_num_lab_procedures': num_medications_num_lab_procedures,
        'num_medications_number_diagnoses': num_medications_number_diagnoses,
        'age_number_diagnoses': age_number_diagnoses,
        'age_comorbidity_count': age_comorbidity_count,
        'change_num_medications': change_num_medications,
        'number_diagnoses_time_in_hospital': number_diagnoses_time_in_hospital,
        'num_medications_numchange': num_medications_numchange,

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
            fig, ax = plt.subplots()
            size = 0.3

            # Create data for the donut chart
            values = [probability, 1 - probability]
            color = 'red' if prediction == 0.0 else 'green'
            colors = [color, '#e6e6e6']

            ax.pie(values, colors=colors, radius=1, wedgeprops=dict(width=size, edgecolor='#94ECBE'))

            # Add a circle in the center to create the donut shape
            centre_circle = plt.Circle((0,0), 1-size, color='#94ECBE', fc='#94ECBE', linewidth=0)
            fig.gca().add_artist(centre_circle)

            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')

            # Add the percentage text in the center of the donut chart
            plt.text(0, 0, f'Probability:\n{probability * 100:.0f}%', ha='center', va='center', fontsize=20, color=color)


            # Display the plot in Streamlit
            st.pyplot(fig)

        else:
            st.write('Failed to receive prediction. Please try again.')

    except requests.exceptions.RequestException as e:
        st.write('Error making prediction request:', e)
