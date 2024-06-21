import pickle

def load_model():
    with open('models/test_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
