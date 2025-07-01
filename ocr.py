import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        self.w1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        x = x.reshape(-1, 1)
        z1 = self.w1 @ x + self.b1
        a1 = self.sigmoid(z1)
        z2 = self.w2 @ a1 + self.b2
        return np.argmax(z2)

    def load_weights(self, path):
        data = np.load(path, allow_pickle=True).item()
        self.w1, self.b1, self.w2, self.b2 = data['w1'], data['b1'], data['w2'], data['b2']
