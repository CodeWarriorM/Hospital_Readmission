import os
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)

    def clean_data(self, df):
        # Drop columns with a large number of missing values
        df = df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1)

        # Drop rows with specific missing values
        df = df[df['diag_1'] != '?']
        df = df[df['diag_2'] != '?']
        df = df[df['diag_3'] != '?']
        df = df[df['race'] != '?']
        df = df[df['gender'] != 'Unknown/Invalid']
        df = df[df['discharge_disposition_id'] != 11]

        # Replace values in A1Cresult and max_glu_serum
        a1c_replacements = {'>7': 1, '>8': 1, 'Norm': 0}
        df['A1Cresult'] = df['A1Cresult'].replace(a1c_replacements).fillna(-99)

        max_glu_serum_replacements = {'>200': 1, '>300': 1, 'Norm': 0}
        df['max_glu_serum'] = df['max_glu_serum'].replace(max_glu_serum_replacements).fillna(-99)

        # Drop columns with identical values in all rows
        columns_to_drop = ['examide', 'citoglipton', 'metformin-rosiglitazone']
        df = df.drop(columns_to_drop, axis=1)

        return df

    def feature_engineering(self, df):
        # Comorbidity count
        comorbidity_ranges = {
            'infectious_parasitic_diseases': (1, 139),
            'neoplasms': (140, 239),
            'endocrine_nutritional_metabolic_immunity_disorders': (240, 279),
            'blood_diseases': (280, 289),
            'mental_disorders': (290, 319),
            'nervous_system_diseases': (320, 389),
            'circulatory_system_diseases': (390, 459),
            'respiratory_system_diseases': (460, 519),
            'digestive_system_diseases': (520, 579),
            'genitourinary_system_diseases': (580, 629),
            'pregnancy_childbirth_complications': (630, 679),
            'skin_diseases': (680, 709),
            'musculoskeletal_system_diseases': (710, 739),
            'congenital_anomalies': (740, 759),
            'perinatal_conditions': (760, 779),
            'symptoms_signs_ill_defined_conditions': (780, 799),
            'injury_poisoning': (800, 999),
            'external_causes_supplemental': ('E', 'V')
        }

        def is_comorbidity(code, comorbidity_ranges):
            try:
                code_int = int(code)
                for comorbidity, (start, end) in comorbidity_ranges.items():
                    if start != 'E' and start != 'V' and start <= code_int <= end:
                        return True
            except ValueError:
                if any(code.startswith(prefix) for prefix in ['E', 'V']):
                    return True
            return False

        def count_comorbidities(row, comorbidity_ranges):
            count = 0
            if is_comorbidity(row['diag_1'], comorbidity_ranges):
                count += 1
            if is_comorbidity(row['diag_2'], comorbidity_ranges):
                count += 1
            if is_comorbidity(row['diag_3'], comorbidity_ranges):
                count += 1
            return count

        df['comorbidity_count'] = df.apply(lambda row: count_comorbidities(row, comorbidity_ranges), axis=1)

        # Visit history summary
        df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

        # Recode columns
        admission_type_mapping = {2: 1, 7: 1, 6: 5, 8: 5}
        discharge_disposition_mapping = {
            6: 1, 8: 1, 9: 1, 13: 1, 3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2, 12: 10,
            15: 10, 16: 10, 17: 10, 25: 18, 26: 18
        }
        admission_source_mapping = {
            2: 1, 3: 1, 5: 4, 6: 4, 10: 4, 22: 4, 25: 4, 15: 9, 17: 9, 20: 9, 21: 9, 13: 11, 14: 11
        }

        df['admission_type_id'] = df['admission_type_id'].replace(admission_type_mapping)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(discharge_disposition_mapping)
        df['admission_source_id'] = df['admission_source_id'].replace(admission_source_mapping)

        # Long stay feature
        df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)

        # Medication columns
        medication_cols = df.loc[:, 'metformin':'metformin-pioglitazone'].columns
        for col in medication_cols:
            colname = str(col) + 'temp'
            df[colname] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)
        df['numchange'] = df[[str(col) + 'temp' for col in medication_cols]].sum(axis=1)
        df.drop(columns=[str(col) + 'temp' for col in medication_cols], inplace=True)

        for col in medication_cols:
            df[col] = df[col].replace('No', 0)
            df[col] = df[col].replace('Steady', 1)
            df[col] = df[col].replace('Up', 1)
            df[col] = df[col].replace('Down', 1)

        # Number of medications used
        df['nummed'] = 0
        for col in medication_cols:
            df['nummed'] = df['nummed'] + df[col]


        # Calculate age_midpoint from age column
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
            '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
        }
        df['age'] = df['age'].replace(age_mapping)

        # Convert change to numeric
        df['change'] = df['change'].replace('Ch', 1)
        df['change'] = df['change'].replace('No', 0)

        # Interaction terms
        interaction_terms = [
            ('num_medications', 'time_in_hospital'),
            ('num_medications', 'num_procedures'),
            ('time_in_hospital', 'num_lab_procedures'),
            ('num_medications', 'num_lab_procedures'),
            ('num_medications', 'number_diagnoses'),
            ('age', 'number_diagnoses'),
            ('age', 'comorbidity_count'),
            ('change', 'num_medications'),
            ('number_diagnoses', 'time_in_hospital'),
            ('num_medications', 'numchange')
        ]

        for inter in interaction_terms:
            name = inter[0] + '|' + inter[1]
            df[name] = df[inter[0]] * df[inter[1]]

        # Categorizing diagnosis
        for col in ['diag_1', 'diag_2', 'diag_3']:
            df[f'level1_{col}'] = df[col]

        for col in ['level1_diag_1', 'level1_diag_2', 'level1_diag_3']:
            df[col] = df[col].replace({r'^V.*': 0, r'^E.*': 0}, regex=True)

        df.replace('?', -1, inplace=True)

        for col in ['level1_diag_1', 'level1_diag_2', 'level1_diag_3']:
            df[col] = df[col].astype(float)

        def classify_diag_level1(value):
            if value >= 390 and value < 460 or np.floor(value) == 785:
                return 1
            elif value >= 460 and value < 520 or np.floor(value) == 786:
                return 2
            elif value >= 520 and value < 580 or np.floor(value) == 787:
                return 3
            elif np.floor(value) == 250:
                return 4
            elif value >= 800 and value < 1000:
                return 5
            elif value >= 710 and value < 740:
                return 6
            elif value >= 580 and value < 630 or np.floor(value) == 788:
                return 7
            elif value >= 140 and value < 240:
                return 8
            else:
                return 0

        df['level1_diag1'] = df['level1_diag_1'].apply(classify_diag_level1)
        df['level1_diag2'] = df['level1_diag_2'].apply(classify_diag_level1)
        df['level1_diag3'] = df['level1_diag_3'].apply(classify_diag_level1)

        # Drop original columns that have been encoded or aggregated
        df.drop(['number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3'],
          axis=1, inplace=True)

        # Change column data types
        cols = ['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'A1Cresult', 'max_glu_serum']
        df[cols] = df[cols].astype('object')

        # Encoding the target variable
        readmitted_mapping = {'>30': 0, '<30': 1, 'NO': 0}
        df['readmitted'] = df['readmitted'].replace(readmitted_mapping)

        # Dropping Duplicates
        df = df.drop_duplicates(subset= ['patient_nbr'], keep = 'first')
        df = df.drop(['encounter_id', 'patient_nbr'], axis=1)

        return df
