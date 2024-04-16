import numpy as np
class NeuralNetwork():

    def __init__(self):

        self.weights = np.array([0.1, 0, -0.1]).T
        self.bias = 0
        self.step = float(0.0001)
        self.convergence = float(10e-3)
        self.magnitude_combined = 10

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, xi, zi):
        a = np.dot(self.weights.T, xi) + self.bias
        q = self.sigmoid(a)
        l = zi*np.log(q) + (1-zi)*np.log(1-q)
        return q, l

    def backward(self, q, xi, zi):
        dL_dq = zi/q - ((1-zi)/(1-q))
        dq_da = q*(1 - q)
        da_dw = xi
        da_db = 1
        dL_dw = dL_dq * dq_da * da_dw
        dL_db = dL_dq * dq_da * da_db

        return dL_dw, dL_db
    def calculate(self, dL_dw, dL_db):
        self.bias = self.bias + self.step*dL_db
        self.weights = self.weights + self.step*dL_dw
        self.magnitude_combined = np.sqrt(dL_db ** 2 + dL_dw[0] ** 2 + dL_dw[1] ** 2 + dL_dw[2] ** 2)
        return self.weights, self.bias


    def train(self, x, z):
        j = 0
        while self.magnitude_combined >= self.convergence:
            i = 0
            while i < int(len(x)):
                q, l = self.forward(x[i], z[i])
                dL_dw, dL_db = self.backward(q, x[i], z[i])
                self.calculate(dL_dw, dL_db)
                i = i+1
            j = j+1

            # print(f'\rValue magnitude_combined: {self.magnitude_combined:.2f}\n', end='')

            print(f'value magnitude_combined {self.magnitude_combined}, value magnitude_combined {self.convergence}')
        print(j)
        return self.weights, self.bias

    def testing(self, x_test, y_test):
        i = 0
        accur = 0
        while i < int(len(x_test)):
            a_test = np.dot(self.weights.T, x_test[i]) + self.bias
            y_pred = self.sigmoid(a_test)
            if y_pred <=0.5:
                y = 0

            else:
                y = 1

            accur = self.accuracy(y, y_test[i], accur, x_test, i)
            i = i + 1
        print(f'The accuricy is {accur}%')
        return accur

    def accuracy(self, y_pred, y_test, accur, x_test, i):
        if y_pred == y_test:
            accur = accur + 1
            if i == int(len(x_test) - 1):
                accur = accur/int(len(x_test))*100

        return accur










if __name__ == "__main__":
    # Initialize the single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.weights)
    print("Random starting synaptic bias: ")
    print(neural_network.bias)


    # The training set, with 4 examples consisting of 3
    # input values and 1 output value
    training_inputs = np.array([[1, 2, 3],
                                [2, 3, 1],
                                [3, 1, 2],
                                [1, 0, 0]])

    training_outputs = np.array([0, 0, 1, 1]).T

    neural_network.train(training_inputs, training_outputs)
    y_pred = neural_network.testing(training_inputs, training_outputs)
    # neural_network.accuracy(y_pred, training_outputs)
    # Train the neural network
    # neural_network.train(training_inputs, training_outputs, 10000)
    #
    # print("Synaptic weights after training: ")
    # print(neural_network.synaptic_weights)
    #
    # A = str(input("Input 1: "))
    # B = str(input("Input 2: "))
    # C = str(input("Input 3: "))
    #
    # print("New situation: input data = ", A, B, C)
    # print("Output data: ")
    # print(neural_network.think(np.array([A, B, C])))