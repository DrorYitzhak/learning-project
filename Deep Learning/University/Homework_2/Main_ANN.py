from scipy.io import loadmat
import pandas as pd
import numpy as np
import ANN as ann

# Define the path to the MAT file

file_path = r'C:\Users\drory\OneDrive - Mobileye\Desktop\תואר שני\למידה עמוקה\עבודות להגשה\תרגיל בית 3\images.mat'
file_path_noise = r'C:\Users\drory\OneDrive - Mobileye\Desktop\תואר שני\למידה עמוקה\עבודות להגשה\תרגיל בית 3\images_noise.mat'



# Load data from the MAT file
mat_data = loadmat(file_path)
df = pd.DataFrame(mat_data['observations'])

mat_data_noise = loadmat(file_path_noise)
df_noise = pd.DataFrame(mat_data_noise['observations_noise'])


# region Function to shuffle the dataframe
def shuffle_dataframe(dataframe,observations_to_train):
    indices = list((dataframe.T).index)
    # np.random.shuffle(indices)
    shuffled_dataframe = (dataframe.T).iloc[indices].reset_index(drop=True)

    train_data = np.array(shuffled_dataframe[0:int(len(shuffled_dataframe)*observations_to_train) - 1]).T
    test_data = np.array(shuffled_dataframe[int(len(shuffled_dataframe)*observations_to_train):len(shuffled_dataframe) - 1]).T


    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return train_data, test_data
# endregion

# region Shuffle and preprocess the data
observations_to_train = 1
train_data, test_data = shuffle_dataframe(df, observations_to_train)
train_data_noise, test_data_noise = shuffle_dataframe(df_noise, observations_to_train)
# endregion

neural_network = ann.FeedForward_NN(train_data)

# region Train the model to
neural_network.train(train_data, train_data)
# endregion

# region Train the model filte noise
# neural_network.train(train_data_noise, train_data)
# endregion


# Plot the loss graph
# neural_network.loss_graph()




