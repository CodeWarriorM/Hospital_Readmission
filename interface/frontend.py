
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import StringIO
import shap
import joblib
#from ml_logic.registry import load_shap_explainer
#from ml_logic.preprocessor import preprocess_features

# Set up page navigation in the sidebar
page = st.sidebar.selectbox("Select a Page", ["User Input", "Description"])

# Mapping
discharge_disposition_mapping = {
    'Home': 1,
    'Transferred': 2,
    'Expired': 3,
    'Left AMA': 4,
    'Inpatient': 5,
    'Other/Unknown': 6
}

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


# Page One: User Input
if page == "User Input":
    # Title and description
    st.title('Hospital Readmission Prediction App')
    st.write("""
    <div class='description'>
    Please enter the patient details below to receive a prediction about hospital readmission risk.
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio("Select input method", ("Manual Input", "CSV Upload"))

    if input_method == "Manual Input":
    # Sidebar sliders for user input
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age (in years)', min_value=1, max_value=100, value=25, step=1, help="Enter the patient's age.")
            gender = st.selectbox('Gender', ['Male', 'Female'], help="Select the patient's gender.")
            race = st.selectbox('Race', ['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'], help="Select the patient's race.")
            discharge_disposition_id = st.selectbox('Discharge Disposition', list(discharge_disposition_mapping.keys()), help="Select where the patient was discharged to.")
            diag_1 = st.selectbox('Primary Diagnosis', list(diag_1_mapping.keys()), help="Select the primary diagnosis category.")

        with col2:
            total_visits = st.number_input('Total Number of Visits', min_value=0, max_value=80, value=0, step=1, help="Enter the total number of visits by the patient.")
            number_diagnoses = st.number_input('Number of Diagnoses', min_value=1, max_value=20, value=5, step=1, help="Enter the total number of diagnoses recorded.")
            num_procedures = st.number_input('Number of Procedures', min_value=0, max_value=10, value=0, step=1, help="Enter the number of procedures performed.")
            num_lab_procedures = st.number_input('Number of Lab Procedures', min_value=1, max_value=150, value=50, step=1, help="Enter the number of lab procedures conducted.")
            num_medications = st.number_input('Number of Medications', min_value=1, max_value=100, value=11, step=1, help="Enter the number of medications prescribed.")


        data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'race': [race],
            'discharge_disposition_id': [discharge_disposition_id],
            'diag_1': [diag_1],
            'total_visits': [total_visits],
            'number_diagnoses': [number_diagnoses],
            'num_procedures': [num_procedures],
            'num_lab_procedures': [num_lab_procedures],
            'num_medications': [num_medications],
        })

    else:
        # CSV Template download
        # Generate a CSV template with example data
        csv_template = StringIO()
        example_data = pd.DataFrame({
            'age': [70],
            'gender': ['Female'],
            'race': ['Caucasian'],
            'discharge_disposition_id': ['Home'],
            'diag_1': ['Diabetes'],
            'total_visits': [0],
            'number_diagnoses': [9],
            'num_procedures': [0],
            'num_lab_procedures': [59],
            'num_medications': [18]
        })
        example_data.to_csv(csv_template, index=False)

        # Add the CSV template download button
        st.download_button(
            label="Download CSV template",
            data=csv_template.getvalue(),
            file_name='example.csv',
            mime='text/csv'
        )


        # CSV Upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

    if 'data' in locals():

         # Add the "Predict" button
        if st.button('Predict', key='predict_button'):
            with st.spinner('Processing your request...'):

                # Apply mappings
                data['diag_1'] = data['diag_1'].map(diag_1_mapping)
                data['discharge_disposition_id'] = data['discharge_disposition_id'].map(discharge_disposition_mapping)

                # Compute derived parameters
                data['age_comorbidity_count'] = data['age'] * 2
                data['age_number_diagnoses'] = data['age'] * data['number_diagnoses']
                data['num_medications_num_lab_procedures'] = data['num_medications'] * data['num_lab_procedures']
                data['num_medications_time_in_hospital'] = data['num_medications'] * 3
                data['num_medications_num_procedures'] = data['num_medications'] * 1
                data['num_medications_number_diagnoses'] = data['num_medications'] * data['number_diagnoses']
                data['number_diagnoses_time_in_hospital'] = data['number_diagnoses'] * 3
                data['time_in_hospital_num_lab_procedures'] = 3 * data['num_lab_procedures']

                # Create a prediction for each row
                for idx, row in data.iterrows():
                    params = {
                        'age': int(row['age']),
                        'gender': row['gender'],
                        'race': row['race'],
                        'level1_diag_1': float(row['diag_1']),
                        'level1_diag_2': float(4),
                        'level1_diag_3': float(0),
                        'num_medications': int(row['num_medications']),
                        'num_lab_procedures': int(row['num_lab_procedures']),
                        'num_procedures': int(row['num_procedures']),
                        'numchange': 0,
                        'nummed': 1,
                        'A1Cresult': "-99.0",
                        'metformin': 0,
                        'pioglitazone': 0,
                        'insulin': 1,
                        'glipizide': 0,
                        'glimepiride': 0,
                        'diabetesMed': 'Yes',
                        'comorbidity_count': 3,
                        'number_diagnoses': int(row['number_diagnoses']),
                        'admission_type_id': "1",
                        'discharge_disposition_id': str(row['discharge_disposition_id']),
                        'admission_source_id': "7",
                        'total_visits': int(row['total_visits']),
                        'time_in_hospital': 3,
                        'change': 1,
                        'age_comorbidity_count': row['age_comorbidity_count'],
                        'age_number_diagnoses': row['age_number_diagnoses'],
                        'num_medications_num_lab_procedures': row['num_medications_num_lab_procedures'],
                        'num_medications_time_in_hospital': row['num_medications_time_in_hospital'],
                        'num_medications_num_procedures': row['num_medications_num_procedures'],
                        'num_medications_number_diagnoses': row['num_medications_number_diagnoses'],
                        'number_diagnoses_time_in_hospital': row['number_diagnoses_time_in_hospital'],
                        'time_in_hospital_num_lab_procedures': row['time_in_hospital_num_lab_procedures']
                    }
                    # Get SHAP values
                    #explainer = load_shap_explainer('shap_explainer.pkl')

                    # Create DataFrame from params
                    input_df = pd.DataFrame([params])

                    # Rename columns in input_df
                    input_df = input_df.rename(columns={
                        'age_comorbidity_count': 'age|comorbidity_count',
                        'age_number_diagnoses': 'age|number_diagnoses',
                        'num_medications_num_lab_procedures': 'num_medications|num_lab_procedures',
                        'num_medications_time_in_hospital': 'num_medications|time_in_hospital',
                        'num_medications_num_procedures': 'num_medications|num_procedures',
                        'num_medications_number_diagnoses': 'num_medications|number_diagnoses',
                        'number_diagnoses_time_in_hospital': 'number_diagnoses|time_in_hospital',
                        'time_in_hospital_num_lab_procedures': 'time_in_hospital|num_lab_procedures'
                    })

                    # Important: Preprocess Input!
                    processed_data = preprocess_features(input_df)
                    shap_values = explainer.shap_values(processed_data, check_additivity=False)

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
                            color = 'red' if probability > 0.8 else 'orange' if 0.4 < probability <= 0.8 else 'green'

                            st.markdown(f"<h2 style='color:{color};'>Predicted Hospital Readmission: <b>{readmission_text}</b></h2>", unsafe_allow_html=True)

                            # Create a bar plot for the probability
                            sns.set(style="whitegrid")

                            # Create a bar plot for the probability
                            fig, ax = plt.subplots()
                            size = 0.3

                            # Set the background color to yellow
                            fig.patch.set_facecolor('#1EBE9B')
                            ax.set_facecolor('#1EBE9B')

                            # Create data for the donut chart
                            values = [probability, 1 - probability]
                            color = 'red' if probability > 0.8 else 'orange' if 0.4 < probability <= 0.8 else 'green'
                            colors = [color, '#e6e6e6']

                            ax.pie(values, colors=colors, radius=1, wedgeprops=dict(width=size, edgecolor='#1EBE9B'))

                            # Add a circle in the center to create the donut shape
                            centre_circle = plt.Circle((0,0), 1-size, color='#1EBE9B', fc='#1EBE9B', linewidth=0)
                            fig.gca().add_artist(centre_circle)

                            # Equal aspect ratio ensures that pie is drawn as a circle
                            ax.axis('equal')

                            # Add the percentage text in the center of the donut chart
                            plt.text(0, 0, f'Probability:\n{probability * 100:.0f}%', ha='center', va='center', fontsize=20, color=color)

                            # Display the plot in Streamlit
                            st.pyplot(fig)

                            # Display SHAP summary plot
                            #st.subheader("SHAP Summary Plot")
                            #st.write('Provides a global explanation of the model, showing the feature importance for all samples.')

                            #fig2, ax = plt.subplots()
                            #size = 0.3
                            #shap.summary_plot(shap_values, processed_data)
                            #st.pyplot(fig2)

                        else:
                            st.write(f"Failed to receive prediction for row {idx+1}. Please try again.")

                    except Exception as e:
                        st.write(f"An error occurred while making the prediction for row {idx+1}: {e}")

# Page Two: Description
else:
    st.title('Description')

    # Descriptive Text
    st.markdown("""
    ### How to Use This App
    Follow these simple steps to use the app effectively:

    1. **Navigate** to the 'User Input' page using the sidebar.
    2. **Input Data**:
       - **Manually:** Use the sliders, dropdowns, and text inputs to enter feature values.
       - **Upload CSV:** Alternatively, upload a CSV file that matches the provided template. Ensure the CSV adheres to the expected structure to avoid errors.
    3. **Get Prediction:** Click the 'Predict' button to receive the hospital readmission prediction based on your input data.
    """)

    st.write("""
    ### About the Prediction
    The prediction is based on a machine learning model trained to predict hospital readmissions. The probability value indicates the likelihood of readmission.
    """)

    st.write("""
    ### Features Description
    The following table provides a brief description of each feature used in the prediction model.
    """)

    # Data for the table
    data = {
        'Feature': [
            'Age',
            'Gender',
            'Race',
            'Discharge Disposition',
            'Primary Diagnosis',
            'Total Number of Visits',
            'Number of Diagnoses',
            'Number of Procedures',
            'Number of Lab Procedures',
            'Number of Medications'
        ],
        'Description': [
            'The age of the patient.',
            'The gender of the patient.',
            'The race of the patient.',
            'The discharge disposition indicates where the patient was discharged to.',
            'The primary diagnosis category of the patient.',
            'The total number of visits made by the patient.',
            'The total number of diagnoses recorded for the patient.',
            'The number of procedures performed during the encounter.',
            'The number of lab tests conducted during the encounter.',
            'The number of distinct medications prescribed during the encounter.'
        ],
        'Possible Values': [
            'Any integer from 1 to 100',
            "'Male', 'Female'",
            "'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'",
            "'Home', 'Transferred', 'Expired', 'Left AMA', 'Inpatient', 'Other/Unknown'",
            "'Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Others'",
            'Any number from 0 to 80',
            'Any number from 1 to 20',
            'Any number from 0 to 10',
            'Any number from 1 to 150',
            'Any number from 1 to 100'
        ]
    }

    df = pd.DataFrame(data)

    # Display the table
        # Display the table with styled DataFrame
    st.write(
        df.style
        .set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#E3ECEC'), ('font-weight', 'bold')]}])
        .set_properties(**{'text-align': 'left'})
        .hide(axis='index')
        .to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    # Add space between table and expander
    st.markdown("---")  # Blank line for spacing
    st.write("### Learn more about the project")

    # Project Information
    st.write("""
        This project aims to predict hospital readmission of patients, particularly focusing on diabetic patients. Hospital readmission rates are indicators of hospital quality and affect the cost of care. This prediction model can help hospitals save millions of dollars while improving the quality of care by identifying the factors that lead to higher readmission rates.

        #### Data Information
        The model is trained on a medical claims dataset provided by the Center for Clinical and Translational Research at Virginia Commonwealth University, which is an abstract of the Health Facts database (Cerner Corporation). The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008).

        #### Model Information
        We utilized a Random Forest model to predict hospital readmissions. The model achieved an recall of 84.27% in our evaluations.

        ### GitHub Repository
        For more details about the project, including code implementation and further information, you can visit the [GitHub repository](https://github.com/CodeWarriorM/hospital_readmission).

        ### Contact Information
        For any inquiries or feedback, please contact the project contributors:
        - Michael Augustynik (augustynik@me.com)
        - Manuela Brunner (Manuela@brunner-sr.de)
        - Olaf Hilgenfeld (olaf.hilgenfeld@gmail.com)
        - Gaelle Massart (gaelle@id-transition.eu)
        - Virginia Wenger (wenger_virginia@gmx.ch)
        """)
