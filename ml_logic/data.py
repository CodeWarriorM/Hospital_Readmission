import os
import numpy as np
import pandas as pd

def map_age(series: pd.Series) -> pd.Series:
    series = series.map({   '[0-10]': 0.0,
                            '[10-20)': 0.1,
                            '[20-30)': 0.2,
                            '[30-40)': 0.3,
                            '[40-50)': 0.4,
                            '[50-60)': 0.5,
                            '[60-70)': 0.6,
                            '[70-80)': 0.7,
                            '[80-90)': 0.8,
                            '[90-100)': 0.9,
                            '[100-110)': 1.0})
    return series

def map_n_diverse(series: pd.Series) -> pd.Series:
    series = series.map({0: 0, 1: 1}).fillna(2).astype(int)
    return series

def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['diag_1'] != 'Missing']
    df = df[df['diag_2'] != 'Missing']
    df = df[df['diag_3'] != 'Missing']

    df['age'] = map_age(df['age'])
    df['n_outpatient'] = map_n_diverse(df['n_outpatient'])
    df['n_inpatient'] = map_n_diverse(df['n_inpatient'])
    df['n_emergency'] = map_n_diverse(df['n_emergency'])

    df = df.drop(columns=[
                            'medical_specialty',
                            'glucose_test',
                        ])

    print('Dataset cleaned. New shape ', df.shape)

    return df

def load_data() -> pd.DataFrame:
    project_path = os.path.dirname(os.path.dirname(__file__))
    data = pd.read_csv(project_path + '/raw_data/hospital_readmissions.csv')

    print('loaded dataset with shape ', data.shape)

    return data
