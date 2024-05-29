import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
# matplotlib inline


# Convert MNIST Image Files into a Tensor of 4-Dimensions (# of images, Height, Width, Color Channels)
transform = transforms.ToTensor()
# transform = transforms.Compose[transforms.ToTensor(), transforms.Normalize(0.5, 0.24)]

# Train Data
train_data = datasets.MNIST(root='/cnn_data', train=True, download=True, transform=transform)

# Test Data
test_data = datasets.MNIST(root='/cnn_data', train=False, download=True, transform=transform)

# Create a small batch size for images...let's say 10
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# dataset = datasets.ImageFolder(root=root_dir, transform=transform)
# region example==================================================================================================
# # Define Our CNN Model
# # Describe convolutional layer and what it's doing (2 convolutional layers)
# # This is just an example in the next video we'll build out the actual model
# conv1 = nn.Conv2d(1, 6, 3, 1)
# conv2 = nn.Conv2d(6, 16, 3, 1)
#
# # Grab 1 MNIST record/image
# for i, (X_Train, y_train) in enumerate(train_data):
#   break
#
# X_Train.shape
#
# x = X_Train.view(1,1,28,28)
#
# # Perform our first convolution
# x = F.relu(conv1(x)) # Rectified Linear Unit for our activation function
#
# # 1 single image, 6 is the filters we asked for, 26x26
# print(x.shape)
#
# # pass thru the pooling layer
# x = F.max_pool2d(x,2,2) # kernal of 2 and stride of 2
#
# print(x.shape) # 26 / 2 = 13
#
# # Do our second convolutional layer
# x = F.relu(conv2(x))
#
# print(x.shape) # Again, we didn't set padding so we lose 2 pixles around the outside of the image
#
# # Pooling layer
# x = F.max_pool2d(x,2,2)
#
# print(x.shape) # 11 / 2 = 5.5 but we have to round down, because you can't invent data to round up
# endregion example==================================================================================================#

# Model Class
class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,3,1)
    self.bn1 = nn.BatchNorm2d(6)
    self.conv2 = nn.Conv2d(6,16,3,1)
    self.bn1 = nn.BatchNorm2d(16)
    # Fully Connected Layer
    self.fc1 = nn.Linear(5*5 *16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2
    # Second Pass
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2

    # Re-View to flatten it out
    X = X.view(-1, 16*5*5) # negative one so that we can vary the batch size

    # Fully Connected Layers
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return F.log_softmax(X, dim=1)


# Create an Instance of our Model
torch.manual_seed(41)
model = ConvolutionalNetwork()
print(model)

# Loss Function Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Smaller the Learning Rate, longer its gonna take to train.


start_time = time.time()

# Create Variables To Tracks Things
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# For Loop of Epochs
for i in range(epochs):
  trn_corr = 0
  tst_corr = 0


  # Train
  for b,(X_train, y_train) in enumerate(train_loader):
    b+=1 # start our batches at 1
    y_pred = model(X_train) # get predicted values from the training set. Not flattened 2D
    loss = criterion(y_pred, y_train) # how off are we? Compare the predictions to correct answers in y_train

    predicted = torch.max(y_pred.data, 1)[1] # add up the number of correct predictions. Indexed off the first point
    batch_corr = (predicted == y_train).sum() # how many we got correct from this batch. True = 1, False=0, sum those up
    trn_corr += batch_corr # keep track as we go along in training.

    # Update our parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # Print out some results
    if b%600 == 0:
      print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')

  train_losses.append(loss)
  train_correct.append(trn_corr)


  # Test
  with torch.no_grad(): #No gradient so we don't update our weights and biases with test data
    for b,(X_test, y_test) in enumerate(test_loader):
      y_val = model(X_test)
      predicted = torch.max(y_val.data, 1)[1] # Adding up correct predictions
      tst_corr += (predicted == y_test).sum() # T=1 F=0 and sum away


  loss = criterion(y_val, y_test)
  test_losses.append(loss)
  test_correct.append(tst_corr)



current_time = time.time()
total = current_time - start_time
print(f'Training Took: {total/60} minutes!')

# Graph the loss at epoch
train_losses = [tl.item() for tl in train_losses]
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Validation Loss")
plt.title("Loss at Epoch")
plt.legend()



# graph the accuracy at the end of each epoch
plt.plot([t/600 for t in train_correct], label="Training Accuracy")
plt.plot([t/100 for t in test_correct], label="Validation Accuracy")
plt.title("Accuracy at the end of each Epoch")
plt.legend()


test_load_everything = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
  correct = 0
  for X_test, y_test in test_load_everything:
    y_val = model(X_test)
    predicted = torch.max(y_val, 1)[1]
    correct += (predicted == y_test).sum()

# Did for correct
correct.item()/len(test_data)*100

test_data[1978][0].reshape(28,28)
plt.imshow(test_data[1978][0].reshape(28,28))


start_time = time.time()