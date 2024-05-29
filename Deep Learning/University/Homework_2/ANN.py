import time
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.io import loadmat
import pandas as pd



class FeedForward_NN:

    def __init__(self, z, number_neurons=800,  convergence=float(10e-3)):
        self.y_size = z.shape[0]
        self.number_neurons = number_neurons
        self.weights_1 = np.array([np.squeeze(2 * np.random.rand(self.y_size, 1) - 1) for _ in range(number_neurons)])
        self.weights_2 = np.array([np.squeeze(2 * np.random.rand(number_neurons, 1) - 1) for _ in range(self.y_size)])
        self.bias_1 = np.outer(np.zeros(number_neurons), 1)
        self.bias_2 = np.outer(np.zeros(self.y_size), 1)
        self.step = float(0.003)
        self.convergence = convergence
        self.magnitude_combined_el = True
        self.magnitude_combined_el_for_graph = []
        self.iterations = []
        self.observations_number = []
        self.y_pred = []
        self.accuracy_train = []

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        if x<0:
            y = 0
        else:
            y = x
        return y

    def forward(self, xi):
        h_array = []
        r_array = []
        try:
            for n_n in range(self.number_neurons):
                a = np.dot(self.weights_1[n_n].T, xi) + self.bias_1[n_n]
                h = self.relu(a)
                h_array = np.append(h_array, h)
            for e in range(self.y_size):
                c = np.dot(self.weights_2[e].T, h_array) + self.bias_2[e]
                r = self.sigmoid(c)
                r_array = np.append(r_array, r)

        except Exception as m:
            print(m)
            print(r_array)

        return r_array, h_array

    def test_relu_a(self, h_array):
        dh_da = []
        for h in h_array:
            if h == 0:
                dh_da.append(int(h))
            else:
                h = 1
                dh_da.append(int(h))

        return dh_da

    def backward(self, x, r_array, h_array):

        dl_dr = -(x - r_array)
        dr_dc = r_array * (1 - r_array)
        dc_dh = self.weights_2
        dc_db2 = 1
        dc_dw2 = h_array
        dh_da = np.array(self.test_relu_a(h_array))
        da_db1 = 1
        da_dw1 = np.array(x)

        # q = np.outer(da_dw1.T, 1)
        # p = np.outer((np.dot(np.array(dc_dh).T, dl_dr * dr_dc) * dh_da), 1)

        # de_dw1 = dl_dr * dr_dc * dc_dh * dh_da * da_dw1
        dl_dw1 = np.dot(np.outer((np.dot(np.array(dc_dh).T, dl_dr * dr_dc) * dh_da), 1), np.outer(da_dw1, 1).T)

        # de_db1 = de_dq * dq_dc * dc_dh * dh_da * da_db1
        de_db1 = np.dot(np.outer((np.dot(np.array(dc_dh).T, dl_dr * dr_dc) * dh_da), 1), da_db1)

        # de_dw2 = de_dq * dq_dc * dc_dw2
        de_dw2 = np.dot(np.outer((dl_dr * dr_dc), 1), np.outer(dc_dw2, 1).T)

        # de_dw2 = de_dq * dq_dc * de_db2
        de_db2 = np.outer(dl_dr * dr_dc * dc_db2, 1)

        return de_dw2, de_db2, dl_dw1, de_db1

    def update_parameters(self, de_dw2, de_db2, de_dw1, de_db1):

        self.weights_1 = self.weights_1 - self.step * de_dw1
        self.weights_2 = self.weights_2 - self.step * de_dw2
        self.bias_1 = self.bias_1 - self.step*de_db1
        self.bias_2 = self.bias_2 - self.step*de_db2
    def calculate(self, r, x):

        self.magnitude_combined_el = (1 / len(x)) * sum((x - r)**2)
        self.magnitude_combined_el_for_graph = np.append(self.magnitude_combined_el_for_graph, self.magnitude_combined_el)
        return self.magnitude_combined_el

    def train(self, x, z):
        observations = x.shape[1]

        j = 0
        while self.magnitude_combined_el >= self.convergence:
            magnitude_combined_el_cum = 0
            i = 0
            while i < observations:

                r_array, h_array = self.forward(x.T[i])
                de_dw2, de_db2, de_dw1, de_db1 = self.backward(z.T[i], r_array, h_array)
                self.update_parameters(de_dw2, de_db2, de_dw1, de_db1)
                magnitude_combined_el = self.calculate(r_array, z.T[i])
                i = i+1
                self.observations_number = np.append(self.observations_number, i)

                if self.magnitude_combined_el <= self.convergence and j > 1:
                # if self.magnitude_combined_el <= self.convergence:
                    self.compare_images(z.T[i], r_array, j, i)
                    self.loss_graph()
                    break
                # if i in [4000, 4001]:
                #     self.compare_images(x.T[i], r_array, j, i)

                magnitude_combined_el_cum = magnitude_combined_el * 100 + magnitude_combined_el_cum
                print(f'value magnitude_combined {self.magnitude_combined_el}, value magnitude_combined {self.convergence} ,observations {i},iteration {j}')

            accuracy = 100 - (magnitude_combined_el_cum / observations)
            self.accuracy_train = np.append(self.accuracy_train, accuracy)

            print(f'train accuracy value = {self.accuracy_train}%')

            j = j+1
            self.iterations = np.append(self.iterations, j)
            print(f'value magnitude_combined {self.magnitude_combined_el}, value magnitude_combined {self.convergence} ,iteration {j}')


            # print(f'\rValue magnitude_combined: {self.magnitude_combined_el_el:.2f}\n', end='')

        print(j)
        return self.weights_1, self.weights_2, self.bias_1, self.bias_2

    def loss_graph(self):
        """
        Plots the loss graph.
        """
        # Setting up the figure with adjusted height
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # כאן אני משנה את גובה התמונה ל-8 אינץ'
        fig.subplots_adjust(hspace=0.5)  # כאן אני משנה את המרווח בין התת-גרפים לערך חיובי

        # Plotting the loss graph with plt.scatter and connecting lines
        ax1.plot(range(len(self.observations_number)), self.magnitude_combined_el_for_graph, linestyle='-')
        ax1.set_xlabel('observations number')
        ax1.set_ylabel('loss')
        ax1.set_title('Loss over Observations Number')

        # Plotting the accuracy graph with plt.scatter and connecting lines
        ax2.plot(range(len(self.iterations)), self.accuracy_train, marker='o', linestyle='-')
        ax2.set_xlabel('iterations number')
        ax2.set_ylabel('average accuracy %')
        ax2.set_title('Average Accuracy over Iterations Number')

        # Adding a title to the entire figure
        plt.suptitle(f'Model Accuracy Results\n convergence value {self.convergence}, number neurons {self.number_neurons}',
                     y=0.99)  # Model accuracy results convergence value, number of neurons

        plt.tight_layout()
        plt.show()

    def predict_and_accuracy(self, x):
            observations = x.shape[1]
            magnitude_combined_el_cum = 0
            i = 0
            while i < observations:
                r_array, h_array = self.forward(x.T[i])
                magnitude_combined_el = self.calculate(r_array, x.T[i])
                magnitude_combined_el_cum = magnitude_combined_el*100 + magnitude_combined_el_cum
                i = i + 1
            accuracy = 100 - (magnitude_combined_el_cum / observations)

            print(f'accuracy value = {accuracy}%')

            return accuracy

    def compare_images(self, new_image, old_image, iterations, observations):
        plt.ion()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        image_data1 = new_image.reshape((28, 28))
        axes[0].imshow(image_data1, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f'X_Image_observation {observations}, iterations {iterations}')

        image_data2 = old_image.reshape((28, 28))
        axes[1].imshow(image_data2, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(f'R_Image_observation {observations}, iterations {iterations}')

        plt.draw()
        plt.pause(0.1)
        plt.show()
        time.sleep(2)


if __name__ == "__main__":
    # Define the path to the MAT file

    file_path = r'C:\Users\drory\OneDrive - Mobileye\Desktop\תואר שני\למידה עמוקה\עבודות להגשה\תרגיל בית 3\images.mat'
    file_path_noise = r'C:\Users\drory\OneDrive - Mobileye\Desktop\תואר שני\למידה עמוקה\עבודות להגשה\תרגיל בית 3\images_noise.mat'

    # Load data from the MAT file
    mat_data = loadmat(file_path)
    df = pd.DataFrame(mat_data['observations'])

    mat_data_noise = loadmat(file_path_noise)
    df_noise = pd.DataFrame(mat_data_noise['observations_noise'])


    # region Function to shuffle the dataframe
    def shuffle_dataframe(dataframe, observations_to_train):
        indices = list((dataframe.T).index)
        # np.random.shuffle(indices)
        shuffled_dataframe = (dataframe.T).iloc[indices].reset_index(drop=True)

        train_data = np.array(shuffled_dataframe[0:int(len(shuffled_dataframe) * observations_to_train) - 1]).T
        test_data = np.array(
            shuffled_dataframe[int(len(shuffled_dataframe) * observations_to_train):len(shuffled_dataframe) - 1]).T

        # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        return train_data, test_data


    # endregion

    # region Shuffle and preprocess the data
    observations_to_train = 1
    train_data, test_data = shuffle_dataframe(df, observations_to_train)
    train_data_noise, test_data_noise = shuffle_dataframe(df_noise, observations_to_train)
    # endregion

    neural_network = FeedForward_NN(train_data)

    # region Train the model to
    neural_network.train(train_data, train_data)
    # endregion

    # region Train the model filte noise
    # neural_network.train(train_data_noise, train_data)
    # endregion

    # Plot the loss graph
    # neural_network.loss_graph()
