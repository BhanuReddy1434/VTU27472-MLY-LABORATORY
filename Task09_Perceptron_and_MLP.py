# Task 9: Perceptron Logic Gates + MLP from Scratch

import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron(inputs, weights, bias):
    return step_function(np.dot(inputs, weights) + bias)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

gates = {
    "AND": (np.array([1, 1]), -1.5),
    "OR": (np.array([1, 1]), -0.5),
    "NAND": (np.array([-1, -1]), 1.5)
}

for gate, (weights, bias) in gates.items():
    print(f"\n{gate} Gate:")
    for x in X:
        print(x, "->", perceptron(x, weights, bias))

# ---------------- MLP FROM SCRATCH ----------------
np.random.seed(42)

X = np.random.rand(100, 2)
y = (X.sum(axis=1) > 1).astype(int).reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

input_size = 2
hidden_size = 4
output_size = 1
lr = 0.5
epochs = 1000

W1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros((1, output_size))

for epoch in range(epochs):
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)

    loss = np.mean((output - y) ** 2)

    output_error = output - y
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = output_delta @ W2.T
    hidden_delta = hidden_error * sigmoid_derivative(hidden)

    W2 -= hidden.T @ output_delta * lr
    b2 -= np.sum(output_delta, axis=0, keepdims=True) * lr
    W1 -= X.T @ hidden_delta * lr
    b1 -= np.sum(hidden_delta, axis=0, keepdims=True) * lr

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

predictions = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
predictions = (predictions > 0.5).astype(int)

accuracy = np.mean(predictions == y) * 100
print(f"\nFinal Training Accuracy: {accuracy:.2f}%")
