import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import StringIO
import pickle
import shap



# Load the SHAP explainer
def load_explainer():
    with open("models/explainer.pkl", "rb") as explainer_file:
        explainer = pickle.load(explainer_file)
    return explainer

explainer = load_explainer()
# Set up page navigation in the sidebar
page = st.sidebar.selectbox("Select a Page", ["User Input", "Description"])

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

# Page One: User Input
if page == "User Input":
    # Title and description
    st.title('Hospital Readmission Prediction App')
    st.write('Select your features below:')

    input_method = st.radio("Select input method", ("Manual Input", "CSV Upload"))

    if input_method == "Manual Input":
    # Sidebar sliders for user input
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age', min_value=1, max_value=100, value=25, step=1)
            gender = st.selectbox('Gender', ['Male', 'Female'])
            race = st.selectbox('Race', ['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'])
            num_lab_procedures = st.number_input('Number of Lab Procedures', min_value=1, max_value=150, value=50, step=1)
            num_medications = st.number_input('Number of Medications', min_value=1, max_value=100, value=11, step=1)
            number_diagnoses = st.number_input('Number of Diagnoses', min_value=1, max_value=20, value=5, step=1)

        with col2:

            diag_1 = st.selectbox('Primary diagnosis', list(diag_1_mapping.keys()))
            insulin = st.selectbox('Insulin', list(insulin_mapping.keys()))
            change = st.selectbox('Change of diabetic medications', list(change_mapping.keys()))
            admission_type = st.selectbox('Admission Type', list(admission_type_mapping.keys()))
            time_in_hospital = st.number_input('Time in Hospital', min_value=1, max_value=14, value=14, step=1)

        data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'race': [race],
            'diag_1': [diag_1],
            'insulin': [insulin],
            'change': [change],
            'num_lab_procedures': [num_lab_procedures],
            'num_medications': [num_medications],
            'number_diagnoses': [number_diagnoses],
            'admission_type': [admission_type],
            'time_in_hospital': [time_in_hospital]
        })

    else:
        # CSV Template download
        # Generate a CSV template with example data
        csv_template = StringIO()
        example_data = pd.DataFrame({
            'age': [70],
            'gender': ['Female'],
            'race': ['Caucasian'],
            'diag_1': ['Diabetes'],
            'insulin': ['Increased'],
            'change': ['Yes'],
            'num_lab_procedures': [15],
            'num_medications': [25],
            'number_diagnoses': [10],
            'admission_type': ['Emergency'],
            'time_in_hospital': [5]
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
        if st.button('Predict'):

            # Apply mappings
            data['diag_1'] = data['diag_1'].map(diag_1_mapping)
            data['insulin'] = data['insulin'].map(insulin_mapping)
            data['change'] = data['change'].map(change_mapping)
            data['admission_type'] = data['admission_type'].map(admission_type_mapping)
            data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
            data['race'] = data['race'].map({'AfricanAmerican': 1, 'Asian': 2, 'Caucasian': 3, 'Hispanic': 4, 'Other': 5})

            # Compute derived parameters
            data['long_stay'] = data['time_in_hospital'].apply(lambda x: 1 if x > 7 else 0)
            data['num_medications_time_in_hospital'] = data['num_medications'] * data['time_in_hospital']
            data['num_medications_num_procedures'] = data['num_medications'] * 1
            data['time_in_hospital_num_lab_procedures'] = data['time_in_hospital'] * data['num_lab_procedures']
            data['num_medications_num_lab_procedures'] = data['num_medications'] * data['num_lab_procedures']
            data['num_medications_number_diagnoses'] = data['num_medications'] * data['number_diagnoses']
            data['age_number_diagnoses'] = data['age'] * data['number_diagnoses']
            data['age_comorbidity_count'] = data['age'] * 3
            data['change_num_medications'] = data['change'] * data['num_medications']
            data['number_diagnoses_time_in_hospital'] = data['number_diagnoses'] * data['time_in_hospital']
            data['num_medications_numchange'] = data['num_medications'] * 0

            # Create a prediction for each row
            for idx, row in data.iterrows():
                params = {
                    'age': int(row['age']),
                    'gender': int(row['gender']),
                    'race': int(row['race']),
                    'level1_diag_1': float(row['diag_1']),
                    'insulin': int(row['insulin']),
                    'change': int(row['change']),
                    'num_lab_procedures': int(row['num_lab_procedures']),
                    'num_medications': int(row['num_medications']),
                    'number_diagnoses': int(row['number_diagnoses']),
                    'admission_type_id': str(row['admission_type']),
                    'time_in_hospital': int(row['time_in_hospital']),
                    'long_stay': int(row['long_stay']),
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
                    'num_medications_time_in_hospital': row['num_medications_time_in_hospital'],
                    'num_medications_num_procedures': row['num_medications_num_procedures'],
                    'time_in_hospital_num_lab_procedures': row['time_in_hospital_num_lab_procedures'],
                    'num_medications_num_lab_procedures': row['num_medications_num_lab_procedures'],
                    'num_medications_number_diagnoses': row['num_medications_number_diagnoses'],
                    'age_number_diagnoses': row['age_number_diagnoses'],
                    'age_comorbidity_count': row['age_comorbidity_count'],
                    'change_num_medications': row['change_num_medications'],
                    'number_diagnoses_time_in_hospital': row['number_diagnoses_time_in_hospital'],
                    'num_medications_numchange': row['num_medications_numchange'],
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

                        #Display Shap Plots
                        st.subheader("SHAP Summary Plot")
                        st.write('Provides a global explanation of the model, showing the feature importance for all samples.')
                        row = data.iloc[idx:idx+1]
                        fig2, ax = plt.subplots()
                        size = 0.3
                        row = data.iloc[[idx]]
                        shap_values = explainer.shap_values(row, check_additivity=False)
                        shap.summary_plot(shap_values, row)
                        st.pyplot(fig2)

                        st.subheader("SHAP Waterfall Plot")
                        st.write('Provides a local explanation for a specific prediction, showing how each feature contributes to the final prediction for the specific instance.')
                        fig3, ax =plt.subplots()
                        size = 0.3
                        shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data =row.iloc[0]))
                        st.pyplot(fig3)

                        #st.subheader("SHAP Dependence Plot")
                        #st.write('This plot shows the relationship between the SHAP values of a feature and the feature itself. It helps in understanding how changes in the feature values impact the prediction. In this example, it is plotted for the feature "age".')
                        #fig4, ax = plt.subplots()
                        #shap.dependence_plot("age", shap_values, row, interaction_index=None, ax = ax)
                        #st.pyplot(fig4)

                        #st.subheader("SHAP Force Plot")
                        #st.write('This plot provides a visual explanation of a single prediction. It shows how the feature values push the prediction from the base value to the final output.')
                        #fig5, ax = plt.subplots()
                        #shap.initjs()
                        #shap.force_plot(explainer.expected_value, shap_values[0], row.iloc[0], matplotlib=True)
                        #st.pyplot(fig5)
                    else:
                        st.write(f"Failed to receive prediction for row {idx+1}. Please try again.")

                except Exception as e:
                    st.write(f"An error occurred while making the prediction for row {idx+1}: {e}")

# Page Two: Description
# Page Two: Description
else:
    st.title('Description')

    # Descriptive Text
    st.write("""
    ### How to Use This App
    1. Navigate to the 'User Input' page using the sidebar.
    2. You have two options for inputting feature values:
       - **Manually:** Select the values for each input feature using the provided sliders, dropdowns, and text inputs.
       - **Upload CSV:** Alternatively, you can upload a CSV file that adheres to the provided template. Ensure the CSV file follows the structure outlined in the template to avoid errors.
    3. Click the 'Predict' button to get the hospital readmission prediction.
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
            'Primary Diagnosis',
            'Insulin',
            'Change of Diabetic Medications',
            'Number of Lab Procedures',
            'Number of Medications',
            'Number of Diagnoses',
            'Admission Type',
            'Time in Hospital'
        ],
        'Description': [
            'The age of the patient.',
            'The gender of the patient.',
            'The race of the patient.',
            'The primary diagnosis category.',
            'The insulin level status.',
            'Whether there was a change in diabetic medications.',
            'The number of lab procedures conducted.',
            'The number of medications prescribed.',
            'The number of diagnoses made.',
            'The type of hospital admission.',
            'The duration of the hospital stay.'
        ],
        'Possible Values': [
            'Any integer from 1 to 100',
            "'Male', 'Female'",
            "'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'",
            "'Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Others'",
            "'Not prescribed', 'Not Adjusted', 'Increased', 'Decreased'",
            "'No', 'Yes'",
            'Any integer from 1 to 150',
            'Any integer from 1 to 100',
            'Any integer from 1 to 20',
            "'Emergency', 'Urgent', 'Elective', 'Newborn', 'Not Available', 'Trauma Center', 'Not Mapped', 'Other'",
            'Any integer from 1 to 14'
        ]
    }

    df = pd.DataFrame(data)

    # Display the table
    st.write(pd.DataFrame(df).to_html(index=False), unsafe_allow_html=True)

    # Add space between table and expander
    st.markdown("---")  # Blank line for spacing
    st.write("### Learn more about the project")

    # Dropdown box for Project Information
    st.write("""
        This project aims to predict hospital readmission of patients, particularly focusing on diabetic patients. Hospital readmission rates are indicators of hospital quality and affect the cost of care. This prediction model can help hospitals save millions of dollars while improving the quality of care by identifying the factors that lead to higher readmission rates.

        #### Data Information
        The model is trained on a medical claims dataset provided by the Center for Clinical and Translational Research at Virginia Commonwealth University, which is an abstract of the Health Facts database (Cerner Corporation). The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008).

        #### Model Information
        We utilized a Random Forest model to predict hospital readmissions. The model achieved an accuracy of 81.48% in our evaluations.

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
