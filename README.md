# Hospital Readmission Prediction

## Project Overview

This project is the final assignment of the Le Wagon Data Science & AI course. The main objective is to demonstrate the skills and knowledge acquired during the course by building a predictive model for hospital readmission. The project features a user-friendly application where users can either input data via a user interface or upload a CSV file to receive predictions and probabilities of hospital readmission.

## Project Aim

This project aims to predict hospital readmission of patients, particularly focusing on diabetic patients. Hospital readmission rates are indicators of hospital quality and affect the cost of care. This prediction model can help hospitals save millions of dollars while improving the quality of care by identifying the factors that lead to higher readmission rates.

## Installation Instructions

To set up the project, follow these steps:

1. **Clone the Repository**:
```
git clone https://github.com/CodeWarriorM/hospital_readmission.git
cd hospital_readmission
```
2. **Set Up Environment**:
It is recommended to use a Python virtual environment. You can create and activate one using the following commands:
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install Dependencies**:
Install the required Python packages using the requirements.txt file:
```
pip install -r requirements.txt
```

4. **Build and Run Docker Container (if using Docker)**:
See Makefile

## Usage Instructions
The project includes a Streamlit application that allows you to interact with the model through a web interface.
1. **Open the App**:
LINK

2. **Input Data**:
- Via User Interface: Enter the patient data into the form on the Streamlit app.
- Via CSV Upload: Upload a CSV file containing the patient data. The CSV file should be placed in the raw_data folder. Ensure the CSV file format matches the expected format.

3. **Receive Predictions**:
The application will display the prediction and probability of hospital readmission based on the provided data.

## Data
The dataset used for training and testing the model should be located in the raw_data folder. The dataset is a medical claims dataset abstracted from the Health Facts database.

## Configuration
The configuration files and scripts are organized as follows:

- Dockerfile: Contains instructions to build the Docker image.
- Makefile: Defines tasks for building and managing the project.
- requirements.txt: Lists the Python packages required for the project.
- setup.py: Defines the package settings for Python.

The preprocessing pipeline and trained model are stored in the following files:

- Preprocessing Pipeline: preprocessor/preprocessing_pipeline.pkl
- Trained Model: models/best_model.pkl
- SHAP Explainer: shap/shap_explainer.pkl

## Model Information
The project utilizes a Random Forest model for predicting hospital readmissions. The model's performance on the test set is as follows:

- Test Accuracy: 0.2969
- Test Precision: 0.1014
- Test Recall: 0.8427

### Model Details
To address class imbalance in the dataset, the following preprocessing steps were applied:
- Undersampling: The majority class was undersampled to match the minority class count.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please submit a pull request or open an issue.

## Acknowledgements
- Data Source: Center for Clinical and Translational Research at Virginia Commonwealth University.
- Original Dataset: UCI Machine Learning Repository.

## Contact
For any inquiries or feedback, please contact the project contributors:

- Michael Augustynik: augustynik@me.com
- Manuela Brunner: Manuela@brunner-sr.de
- Olaf Hilgenfeld: olaf.hilgenfeld@gmail.com
- Gaelle Massart: gaelle@id-transition.eu
- Virginia Wenger: wenger_virginia@gmx.ch
