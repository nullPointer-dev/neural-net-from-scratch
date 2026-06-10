import os
import numpy as np
import matplotlib.pyplot as plt

from src import *
from config import *
from data import load_mnist_data
np.random.seed(42)

(x_train, y_train), (x_test, y_test) = load_mnist_data()

x_train = normalize(flatten(x_train))
x_test = normalize(flatten(x_test))

y_train_labels = y_train.copy()
y_test_labels = y_test.copy()

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)


parameters = initialize_parameters()

loss_history = []
accuracy_history = []
test_epochs = []


for epoch in range(EPOCHS):

    predictions, cache = forward_propagation(x_train, parameters)
    loss = cross_entropy(y_train, predictions)
    loss_history.append(loss)

    gradients = backward_propagation(x_train, y_train, parameters, cache)

    parameters = update_parameters(parameters, gradients, LEARNING_RATE)

    if epoch % 10 == 0:

        test_probs, _ = forward_propagation(x_test, parameters)
        preds = np.argmax(test_probs, axis=1)
        test_acc = np.mean(preds == y_test_labels)
        accuracy_history.append(test_acc * 100)
        test_epochs.append(epoch)

        print(f"Epoch {epoch} Loss: {loss:.4f} Test Accuracy: {test_acc * 100:.2f}%")

test_probs, _ = forward_propagation(x_test,parameters)

predictions = np.argmax(test_probs,axis=1)
accuracy = np.mean(predictions == y_test_labels)

print(f"Final Loss: {loss_history[-1]:.4f}")
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

if accuracy >= 0.90:
    np.savez(
        "models/model.npz",
        W1=parameters["W1"],
        b1=parameters["b1"],
        W2=parameters["W2"],
        b2=parameters["b2"],
        W3=parameters["W3"],
        b3=parameters["b3"],
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
    print("Model saved.")

plt.figure(figsize=(10, 5))

# loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, linewidth=2)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# test accuracy
plt.subplot(1, 2, 2)
plt.plot(test_epochs, accuracy_history, linewidth=2)
plt.title("Test Accuracy During Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()

plt.savefig("models/training_metrics.png")

plt.show()