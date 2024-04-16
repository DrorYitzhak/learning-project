import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import numpy as np


class NeuralNetwork:

    def __init__(self, z, number_neurons=11,  convergence=float(10e-3)):
        self.y_size = z.shape[1]
        self.number_neurons = number_neurons
        self.weights_1 = np.array([np.squeeze(2 * np.random.rand(2, 1) - 1) for _ in range(number_neurons)])
        self.weights_2 = np.array([np.squeeze(2 * np.random.rand(number_neurons, 1) - 1) for _ in range(self.y_size)])
        self.bias_1 = np.outer(np.zeros(number_neurons), 1)
        self.bias_2 = np.outer(np.zeros(self.y_size), 1)
        self.step = float(0.02)
        self.convergence = convergence
        self.magnitude_combined_el = True
        self.magnitude_combined_el_for_graph = []
        self.iter = []


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, xi):
        h_array = []
        c_array = []
        for n_n in range(self.number_neurons):
            a = np.dot(self.weights_1[n_n].T, xi) + self.bias_1[n_n]
            h = self.sigmoid(a)
            h_array = np.append(h_array, h)
        for e in range(self.y_size):
            c = np.dot(self.weights_2[e].T, np.outer(h_array, 1)) + self.bias_2[e]
            c_array = np.append(c_array, c)
            q_numerator = np.exp(c_array)
            q_denominator = np.sum(np.exp(c_array))
            q = q_numerator / q_denominator


        return q, h_array

    def backward(self, q, xi, zi, h_array):
        de_dq = (q - zi)
        dq_dc = q * (1 - q)
        dc_db2 = 1
        dc_dw2 = h_array
        dc_dh = self.weights_2
        dh_da = h_array * (1-h_array)
        da_db1 = 1
        da_dw1 = np.outer(xi, 1)

        # de_dw1 = de_dq * dq_dc * dc_dh * dh_da * da_dw1
        de_dw1 = np.dot(np.outer((np.dot(np.array(dc_dh).T, de_dq * dq_dc) * dh_da), 1), da_dw1.T)

        # de_db1 = de_dq * dq_dc * dc_dh * dh_da * da_db1
        de_db1 = np.dot(np.outer((np.dot(np.array(dc_dh).T, de_dq * dq_dc) * dh_da), 1), da_db1)

        # de_dw2 = de_dq * dq_dc * dc_dw2
        de_dw2 = np.dot(np.outer((de_dq * dq_dc), 1), np.outer(dc_dw2, 1).T)

        # de_dw2 = de_dq * dq_dc * de_db2
        de_db2 = np.outer(de_dq * dq_dc * dc_db2, 1)

        return de_dw2, de_db2, de_dw1, de_db1

    def calculate(self, de_dw2, de_db2, de_dw1, de_db1, qi, zi):
        self.weights_1 = self.weights_1 - self.step * de_dw1
        self.weights_2 = self.weights_2 - self.step * de_dw2
        self.bias_1 = self.bias_1 - self.step*de_db1
        self.bias_2 = self.bias_2 - self.step*de_db2

        self.magnitude_combined_el = (1 / 2) * sum((qi - zi)**2)

        # return self.weights_1, self.weights_2, self.bias_1, self.bias_2


    def train(self, x, z):

        j = 0
        while self.magnitude_combined_el >= self.convergence:
            i = 0
            while i < int(len(x)):
                qi, h_array = self.forward(x[i])
                de_dw2, de_db2, de_dw1, de_db1 = self.backward(qi, x[i], z[i], h_array)
                self.calculate(de_dw2, de_db2, de_dw1, de_db1, qi, z[i])
                if self.magnitude_combined_el <= self.convergence and j > 1:
                    break

                i = i+1

            j = j+1
            self.iter = np.append(self.iter, j)
            self.magnitude_combined_el_for_graph = np.append(self.magnitude_combined_el_for_graph, self.magnitude_combined_el)
            print(f'value magnitude_combined {self.magnitude_combined_el}, value magnitude_combined {self.convergence} ,iteration {j}')

            # print(f'\rValue magnitude_combined: {self.magnitude_combined_el_el:.2f}\n', end='')

        print(j)
        return self.weights_1, self.weights_2, self.bias_1, self.bias_2

    def loss_graph(self):

        x_values = self.iter
        y_values = self.magnitude_combined_el_for_graph
        plt.plot(x_values, y_values)
        plt.xlabel('X - iter')
        plt.ylabel('Y - magnitude')
        plt.title('Graph Title')
        plt.show()




if __name__ == "__main__":

    file_path = r'C:\Users\sagiv.a\OneDrive - SolarEdge\Desktop\Sagiv\תואר שני\למידה עמוקה\תרגיל בית\NewMatrix.mat'

    mat_data = loadmat(file_path)
    df = pd.DataFrame(mat_data['NewMatrix'])

    x1_x2 = mat_data['NewMatrix'][:, :2]
    y = mat_data['NewMatrix'][:, 2:]
    print("x1_x2:")
    print(x1_x2)

    print("y:")
    print(y)


    def shuffle_dataframe(df):
        indices = list(df.index)
        np.random.shuffle(indices)
        df_shuffled = df.iloc[indices].reset_index(drop=True)

        train = np.array(df_shuffled[:]).T
        x_train = np.array([train[0], train[1]]).T
        min_val = np.min(x_train)
        max_val = np.max(x_train)
        x_train = (x_train - min_val) / (max_val - min_val)
        y_train = np.array([train[2], train[3], train[4]]).T

        return x_train, y_train


    x_train, y_train = shuffle_dataframe(df)

    NN = NeuralNetwork(y_train)
    NN.train(x_train, y_train)
    NN.loss_graph()
