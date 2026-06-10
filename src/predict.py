from src import forward_propagation
import numpy as np


def load_model():
    model = np.load("models/model.npz")
    parameters = {
        "W1": model["W1"],
        "b1": model["b1"],
        "W2": model["W2"],  
        "b2": model["b2"],
        "W3": model["W3"],
        "b3": model["b3"]  
    }
    return parameters

def predict_one(image, parameters):
    probabilities, _ = forward_propagation(image,parameters)
    prediction = np.argmax(probabilities,axis=1)[0]
    confidence = probabilities[0]
    return prediction, confidence