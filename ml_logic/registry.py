import os
import glob
import pickle

def load_model(preferred=None):
    project_path = os.path.dirname(os.path.dirname(__file__))
    list_of_models = glob.glob(project_path + '/models/*.pkl')
    latest_model = max(list_of_models, key=os.path.getctime)

    if preferred is None:
        model_to_open = latest_model
    elif (project_path + f'/models/{preferred}') not in list_of_models:
        model_to_open = latest_model
        print('Preferred model not found. Will load latest model instead...')
    else:
        model_to_open = project_path + f'/models/{preferred}'

    with open(model_to_open, 'rb') as file:
        model = pickle.load(file)

    print('loaded model: ', {model_to_open.split('/')[-1]})

    return model
