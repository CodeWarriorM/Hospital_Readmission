import os
import glob
import pickle

def load_model():
    project_path = os.path.dirname(os.path.dirname(__file__))
    list_of_models = glob.glob(project_path + '/models/*.pkl')
    latest_model = max(list_of_models, key=os.path.getctime)

    with open(latest_model, 'rb') as file:
        model = pickle.load(file)

    print('loaded model: ', {latest_model.split('/')[-1]})

    return model
