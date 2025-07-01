import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ocr import NeuralNetwork

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=0)

def one_hot(y, size=10):
    oh = np.zeros((size, len(y)))
    oh[y, np.arange(len(y))] = 1
    return oh

# Load MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X, y = X / 255.0, y.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model setup
nn = NeuralNetwork()
lr = 0.01
epochs = 5

# Training (very basic SGD, no batching)
for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i].reshape(-1, 1)
        y_true = one_hot([y_train[i]])

        # Forward
        z1 = nn.w1 @ x + nn.b1
        a1 = nn.sigmoid(z1)
        z2 = nn.w2 @ a1 + nn.b2
        a2 = softmax(z2)

        # Backward
        dz2 = a2 - y_true
        dw2 = dz2 @ a1.T
        db2 = dz2
        dz1 = (nn.w2.T @ dz2) * a1 * (1 - a1)
        dw1 = dz1 @ x.T
        db1 = dz1

        # Update
        nn.w1 -= lr * dw1
        nn.b1 -= lr * db1
        nn.w2 -= lr * dw2
        nn.b2 -= lr * db2

    print(f"Epoch {epoch+1} done")

# Save weights
np.save("model_weights.npy", {
    'w1': nn.w1,
    'b1': nn.b1,
    'w2': nn.w2,
    'b2': nn.b2
})
