import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class FeedForward_NN:

    def __init__(self, z, number_neurons=11,  convergence=float(50e-3)):
        self.y_size = z.shape[1]
        self.number_neurons = number_neurons
        # self.weights_1 = np.array([np.squeeze(2 * np.random.rand(2, 1) - 1) for _ in range(number_neurons)])
        self.weights_1 = np.array([[1.94980627, 1.44785474], [2.68661933, 2.77856598], [0.10575249, - 11.76072016], [2.05644441, 3.61867669], [-10.42349242, 5.74684178], [2.03606377, 1.24290638], [9.94681543, 5.70844653], [0.4915529, 1.01300102], [4.53216953, 3.38880207], [0.88677085, 1.35359425], [3.0816388, 1.66166305]])
        # self.weights_2 = np.array([np.squeeze(2 * np.random.rand(number_neurons, 1) - 1) for _ in range(self.y_size)])
        self.weights_2 = np.array([[41.32453492, 39.28377298, - 5.39783009, 38.80520143, - 6.38005418, 42.07165626, - 0.3114017,  42.76662866, 42.98369444, 42.08013645, 42.3256203], [40.24453471, 41.24472629,  7.08157394, 38.52376019,  5.78728366, 41.62759977, 11.72004209, 41.0634616,  39.5003162,  40.66641404, 41.75026727],[39.25040761,38.58195944,27.21166151,38.79607758,25.51031503,38.36369947,31.17514096,38.22988556,38.21603047,38.05556597,39.20307506]])
        # self.bias_1 = np.outer(np.zeros(number_neurons), 1)
        self.bias_1 = np.array([[6.56750687], [5.77719766], [2.83907154], [5.68911487], [-0.63042709], [6.6408539], [-10.68735794], [7.46841625], [5.05647139], [7.08529946], [6.12844616]])
        # self.bias_2 = np.outer(np.zeros(self.y_size), 1)
        self.bias_2 = np.array([[51.70452543], [51.56863749], [49.2323948]])
        self.step = float(0.03)
        self.convergence = convergence
        self.magnitude_combined_el = True
        self.magnitude_combined_el_for_graph = []
        self.iterations = []
        self.y_pred = []

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, xi):
        h_array = []
        c_array = []
        try:
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
        except Exception as m:
            print(m)
            print(c_array)

        return q, h_array

    def backward(self, q, xi, zi, h_array):
        de_dq = (q - zi)
        dq_dc = q * (1 - q)
        dc_db2 = 1
        dc_dw2 = h_array
        dc_dh = self.weights_2
        dh_da = h_array * (1 - h_array)
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

        # de_dw1_total_sum = np.sum(np.power(de_dw2, 2), axis=None)
        # de_dw2_total_sum = np.sum(np.power(de_dw1, 2), axis=None)
        # de_db1_total_sum = np.sum(np.power(de_db1, 2), axis=None)
        # de_db2_total_sum = np.sum(np.power(de_db2, 2), axis=None)
        # self.magnitude_combined_el = np.sqrt(de_dw1_total_sum + de_dw2_total_sum + de_db1_total_sum + de_db2_total_sum)

        self.magnitude_combined_el_for_graph = np.append(self.magnitude_combined_el_for_graph, self.magnitude_combined_el)

    def train(self, x, z):

        j = 0
        while self.magnitude_combined_el >= self.convergence:
            i = 0
            while i < int(len(x)):
                qi, h_array = self.forward(x[i])
                de_dw2, de_db2, de_dw1, de_db1 = self.backward(qi, x[i], z[i], h_array)
                self.calculate(de_dw2, de_db2, de_dw1, de_db1, qi, z[i])
                # if self.magnitude_combined_el <= self.convergence and j > 1:
                #     break

                i = i+1

            j = j+1
            self.iterations = np.append(self.iterations, j)
            print(f'value magnitude_combined {self.magnitude_combined_el}, value magnitude_combined {self.convergence} ,iteration {j}')

            # print(f'\rValue magnitude_combined: {self.magnitude_combined_el_el:.2f}\n', end='')

        print(j)
        return self.weights_1, self.weights_2, self.bias_1, self.bias_2

    # def train_criterion_accuracy(self, x, z):
    #
    #     j = 0
    #     while self.magnitude_combined_el >= self.convergence:
    #         i = 0
    #         while i < int(len(x)):
    #             qi, h_array = self.forward(x[i])
    #             de_dw2, de_db2, de_dw1, de_db1 = self.backward(qi, x[i], z[i], h_array)
    #             self.calculate(de_dw2, de_db2, de_dw1, de_db1, qi, z[i])
    #             # if self.magnitude_combined_el <= self.convergence and j > 1:
    #             #     break
    #
    #             i = i+1
    #
    #         j = j+1
    #         self.iterations = np.append(self.iterations, j)
    #         print(f'value magnitude_combined {self.magnitude_combined_el}, value magnitude_combined {self.convergence} ,iteration {j}')
    #
    #         # print(f'\rValue magnitude_combined: {self.magnitude_combined_el_el:.2f}\n', end='')
    #
    #     print(j)
    #     return self.weights_1, self.weights_2, self.bias_1, self.bias_2


    # def testing(self, x_test, y_test):
    #     i = 0
    #     accur = 0
    #     while i < int(len(x_test)):
    #         a_test = np.dot(self.weights.T, x_test[i]) + self.bias
    #         y_pred = self.sigmoid(a_test)
    #         if y_pred <=0.5:
    #             y = 0
    #
    #         else:
    #             y = 1
    #
    #         accur = self.accuracy(y, y_test[i], accur, x_test, i)
    #         i = i + 1
    #     print(f'The accuricy is {accur}%')
    #     return accur

    # def accuracy(self, y_pred, y_test, accur, x_test, i):
    #     if y_pred == y_test:
    #         accur = accur + 1
    #         if i == int(len(x_test) - 1):
    #             accur = accur/int(len(x_test))*100
    #     return accur


    def loss_graph(self):
        """
        Plots the loss graph.
        """
        x_values = self.iterations
        y_values = self.magnitude_combined_el_for_graph
        plt.plot(x_values, y_values)
        plt.xlabel('X - iterations')
        plt.ylabel('Y - loss')
        plt.title('Loss over Iterations')
        plt.show()

    def predict(self, x):
        y_pred = []
        qj = [None, None, None]
        i = 0
        while i < int(len(x)):
            qi, h_array = self.forward(x[i])
            for j in range(len(qi)):
                if qi[j] > 0.5:
                    qj[j] = int(1)
                else:
                    qj[j] = int(0)
            y_pred_j = [qj[0], qj[1], qj[2]]
            y_pred.append(y_pred_j)
            i = i + 1
        return y_pred

    def accuracy(self, y_pred, y_test):
        q = 0
        for i in range(len(y_pred)):
            print(f"Row {i}--------------")
            print(y_pred[i])
            print(y_test[i])
            print("----------------------")
            if y_pred[i][0] == y_test[i][0] and y_pred[i][1] == y_test[i][1] and y_pred[i][2] == y_test[i][2]:
                q = q + 1


        accuracy = q / len(y_test) * 100
        print(accuracy)
        print(self.iterations)
        # print(f' weights_1 {self.weights_1}, weights_2 {self.weights_2}, bias_1 {self.bias_1}, bias_2 {self.bias_2}')
        return accuracy


if __name__ == "__main__":
    # Initialize the single neuron neural network
    neural_network = FeedForward_NN(convergence=float(10e-4))

