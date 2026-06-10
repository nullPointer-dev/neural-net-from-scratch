import numpy as np
from config import *
from src import relu, relu_derivative, softmax

def initialize_parameters():
    parameters = {
        "W1": np.random.randn(INPUT_SIZE, HIDDEN1_SIZE) * np.sqrt(2. / INPUT_SIZE),
        "b1": np.zeros((1, HIDDEN1_SIZE)),
        "W2": np.random.randn(HIDDEN1_SIZE, HIDDEN2_SIZE) * np.sqrt(2. / HIDDEN1_SIZE),
        "b2": np.zeros((1, HIDDEN2_SIZE)),
        "W3": np.random.randn(HIDDEN2_SIZE, OUTPUT_SIZE) * np.sqrt(2. / HIDDEN2_SIZE),
        "b3": np.zeros((1, OUTPUT_SIZE))    
    }
    return parameters

def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    cache = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3
    }

    return A3, cache

def backward_propagation(X, Y, parameters, cache):
    m = X.shape[0]
    W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]

    dZ3 = A3 - Y
    dW3 = A2.T @ dZ3 / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(cache["Z2"])
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(cache["Z1"])
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3
    }

def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    parameters["W3"] -= learning_rate * gradients["dW3"]
    parameters["b3"] -= learning_rate * gradients["db3"]
    return parameters