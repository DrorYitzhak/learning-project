from scipy.io import loadmat
import pandas as pd
import numpy as np
import System.FeedForward_NN as NN
from sklearn.model_selection import train_test_split

# Define the path to the MAT file
file_path = r'C:\Users\drory\OneDrive - Mobileye\Desktop\תואר שני\למידה עמוקה\עבודות להגשה\תרגיל בית 1\NewMatrix.mat'

# Load data from the MAT file
mat_data = loadmat(file_path)
df = pd.DataFrame(mat_data['NewMatrix'])

# Define input and output data
input_data = mat_data['NewMatrix'][:, :2]
output_data = mat_data['NewMatrix'][:, 2:]

# print("Input data (x1_x2):")
# print(input_data)
#
# print("Output data (y):")
# print(output_data)

# Function to shuffle the dataframe
def shuffle_dataframe(dataframe):
    indices = list(dataframe.index)
    np.random.shuffle(indices)
    shuffled_dataframe = dataframe.iloc[indices].reset_index(drop=True)

    train_data = np.array(shuffled_dataframe[:]).T
    x_train = np.array([train_data[0], train_data[1]]).T
    min_val = np.min(x_train)
    max_val = np.max(x_train)
    x_train = (x_train - min_val) / (max_val - min_val)
    y_train = np.array([train_data[2], train_data[3], train_data[4]]).T

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return x_train, x_test, y_test, y_train

# Shuffle and preprocess the data
x_train, x_test, y_test, y_train = shuffle_dataframe(df)

# Create the neural network model
neural_network = NN.FeedForward_NN(y_train)

# Train the model
neural_network.train(x_train, y_train)

y_pred = neural_network.predict(x_test)

# Plot the loss graph
# neural_network.loss_graph()

accuracy = neural_network.accuracy(y_pred, y_test)

pass
