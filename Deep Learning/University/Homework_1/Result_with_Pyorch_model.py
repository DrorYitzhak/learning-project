import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
# matplotlib inline

from sklearn.model_selection import train_test_split

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
  # Input layer (4 features of the flower) -->
  # Hidden Layer1 (number of neurons) -->
  # H2 (n) -->
  # output (3 classes of iris flowers)
  def __init__(self, in_features=2, h1=8, h2=9, out_features=3):
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x


# Pick a manual seed for randomization
torch.manual_seed(41)
# Create an instance of model
model = Model()

# Define the path to the MAT file
file_path = r'C:\Users\drory\OneDrive - Mobileye\Desktop\תואר שני\למידה עמוקה\עבודות להגשה\תרגיל בית 1\NewMatrix.mat'

# Load data from the MAT file
mat_data = loadmat(file_path)
df = pd.DataFrame(mat_data['NewMatrix'])

# Define input and output data
input_data = mat_data['NewMatrix'][:, :2]
output_data = mat_data['NewMatrix'][:, 2:]

indices = list(df.index)
np.random.shuffle(indices)
shuffled_dataframe = df.iloc[indices].reset_index(drop=True)

train_data = np.array(shuffled_dataframe[:]).T
x_train = np.array([train_data[0], train_data[1]]).T
min_val = np.min(x_train)
max_val = np.max(x_train)
x_train = (x_train - min_val) / (max_val - min_val)
y_train_ = np.array([train_data[2], train_data[3], train_data[4]]).T
i = 0
y_train = []
for i in range(len(y_train_)):
    if np.array_equal(y_train_[i], [1.0, 0.0, 0.0]):
        y_train.append(0.0)
    elif np.array_equal(y_train_[i],  [0.0, 1.0, 0.0]):
        y_train.append(1.0)
    else:
        y_train.append(2.0)

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)




# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 600
losses = []
for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X_train) # Get predicted results

  # Measure the loss/error, gonna be high at first
  loss = criterion(y_pred, y_train) # predicted values vs the y_train

  # Keep Track of our losses
  losses.append(loss.detach().numpy())

  # print every 10 epoch
  if i % 10 == 0:
    print(f'Epoch: {i} and loss: {loss}')

  # Do some back propagation: take the error rate of forward propagation and feed it back
  # thru the network to fine tune the weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


# Graph it out!
time.sleep(1)
plt.plot(range(epochs), losses)
time.sleep(1)
plt.ylabel("loss/error")
plt.xlabel('Epoch')


#Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
  y_eval = model.forward(X_test) # X_test are features from our test set, y_eval will be predictions
  loss = criterion(y_eval, y_test) # Find the loss or error

print(f'loss: {loss}')

correct = 0
with torch.no_grad():
  for i, data in enumerate(X_test):
    y_val = model.forward(data)
 
    if y_test[i] == 0:
      x = "Setosa"
    elif y_test[i] == 1:
      x = 'Versicolor'
    else:
      x = 'Virginica'


    # Will tell us what type of flower class our network thinks it is
    print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

    # Correct or not
    if y_val.argmax().item() == y_test[i]:
      correct +=1

print(f'We got {correct} correct!')