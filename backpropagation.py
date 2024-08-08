import numpy as np

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly with mean 0
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def forward(self, X):
        # Forward pass
        self.hidden = sigmoid(np.dot(X, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def backward(self, X, y, output):
        # Calculate the error
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Update the weights
        self.weights2 += self.hidden.T.dot(output_delta)
        self.weights1 += X.T.dot(hidden_delta)

    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)

# Example usage
if __name__ == "__main__":
    # Input dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Output dataset
    y = np.array([[0], [1], [1], [0]])

    # Initialize the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

    # Train the network
    nn.train(X, y, epochs=10000)

    # Test the network
    print("Predicted outputs:")
    print(nn.predict(X))
