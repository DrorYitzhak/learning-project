from scipy.io import loadmat
import pandas as pd
import numpy as np
import System.LogLikelihood_Sagiv as LK

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

NN = LK.NeuralNetwork(y_train)
NN.train(x_train, y_train)
NN.loss_graph()


