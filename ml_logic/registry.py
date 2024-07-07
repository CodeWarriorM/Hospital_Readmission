import os
import glob
import pickle

def load_model(preferred=None):
    """
    Load a model explainer from the models folder.
    """
    project_path = os.path.dirname(os.path.dirname(__file__))
    list_of_models = glob.glob(project_path + '/models/*.pkl')
    latest_model = max(list_of_models, key=os.path.getctime)
    print('available models: ', list_of_models)
    print('latest model: ', latest_model)
    print('preferred model: ', preferred)

    if preferred is None:
        model_to_open = latest_model
        print('No model preferred. Will load latest model...')
    elif (project_path + f'/models/{preferred}') not in list_of_models:
        print('Preferred model not found. Will load latest model instead...')
        model_to_open = latest_model
    else:
        model_to_open = project_path + f'/models/{preferred}'

    with open(model_to_open, 'rb') as file:
        model = pickle.load(file)

    print('loaded model: ', {model_to_open.split('/')[-1]})

    return model

import os
import glob
import pickle

def load_shap_explainer(preferred=None):
    """
    Load a SHAP explainer from the shap folder.
    """
    project_path = os.path.dirname(os.path.dirname(__file__))
    shap_path = os.path.join(project_path, 'shap')
    list_of_explainers = glob.glob(os.path.join(shap_path, '*.pkl'))
    latest_explainer = max(list_of_explainers, key=os.path.getctime)
    print('available explainers: ', list_of_explainers)
    print('latest explainer: ', latest_explainer)
    print('preferred explainer: ', preferred)

    if preferred is None:
        explainer_to_open = latest_explainer
        print('No explainer preferred. Will load latest explainer...')
    elif os.path.join(shap_path, preferred) not in list_of_explainers:
        print('Preferred explainer not found. Will load latest explainer instead...')
        explainer_to_open = latest_explainer
    else:
        explainer_to_open = os.path.join(shap_path, preferred)

    with open(explainer_to_open, 'rb') as file:
        explainer = pickle.load(file)

    print('loaded explainer: ', explainer_to_open.split('/')[-1])

    return explainer
