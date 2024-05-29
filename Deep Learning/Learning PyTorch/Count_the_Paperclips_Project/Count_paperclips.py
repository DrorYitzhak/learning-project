import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from Data_organization import PaperclipsDataset


# matplotlib inline


data_dir = r'C:\Users\drory\learning-project\Deep Learning\Learning PyTorch\Count_the_Paperclips_Project\data'
clips_data_dir = os.path.join(data_dir, 'clips_data_2020', 'clips')
train_csv_path = os.path.join(data_dir, 'train.csv')
test_csv_path = os.path.join(data_dir, 'test.csv')

# הגדר טרנספורמציות לתמונות
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# צור את ה-Datasets
train_data = PaperclipsDataset(csv_file=train_csv_path, root_dir=clips_data_dir, transform=transform)
test_data = PaperclipsDataset(csv_file=test_csv_path, root_dir=clips_data_dir, transform=transform)

# צור את ה-DataLoaders
train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False, num_workers=0)

class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super(ConvolutionalNetwork, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, 3, 1)  # 4 input channels, 8 output channels, kernel size 3, stride 1
    self.bn1 = nn.BatchNorm2d(8)  # Batch Normalization after conv1
    self.conv2 = nn.Conv2d(8, 16, 3, 1)  # 8 input channels, 16 output channels, kernel size 3, stride 1
    self.bn2 = nn.BatchNorm2d(16)  # Batch Normalization after conv2
    self.conv3 = nn.Conv2d(16, 16, 3, 1)  # 16 input channels, 16 output channels, kernel size 3, stride 1
    self.bn3 = nn.BatchNorm2d(16)  # Batch Normalization after conv3
    self.fc1 = nn.Linear(16 * 30 * 30, 500)  # 16 channels * 30x30 image size after 3 max poolings
    self.bn4 = nn.BatchNorm1d(500)  # Batch Normalization after fc1
    self.fc2 = nn.Linear(500, 200)
    self.bn5 = nn.BatchNorm1d(200)  # Batch Normalization after fc2
    self.fc3 = nn.Linear(200, 76)


  def forward(self, X):
    X = F.relu(self.bn1(self.conv1(X)))
    X = F.max_pool2d(X, 2, 2)  # Max pooling with 2x2 kernel and stride 2
    X = F.relu(self.bn2(self.conv2(X)))
    X = F.max_pool2d(X, 2, 2)  # Max pooling with 2x2 kernel and stride 2
    X = F.relu(self.bn3(self.conv3(X)))
    X = F.max_pool2d(X, 2, 2)  # Max pooling with 2x2 kernel and stride 2
    X = X.view(-1, 16 * 30 * 30)  # Flatten the output for the fully connected layers
    X = F.relu(self.bn4(self.fc1(X)))
    X = F.relu(self.bn5(self.fc2(X)))
    X = self.fc3(X)
    return F.log_softmax(X, dim=1)

# Create an Instance of our Model
torch.manual_seed(41)
model = ConvolutionalNetwork()
print(model)

# Loss Function Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Smaller the Learning Rate, longer its gonna take to train.


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
    y_pred = model(X_train)  # get predicted values from the training set. Not flattened 2D
    loss = criterion(y_pred, y_train) # how off are we? Compare the predictions to correct answers in y_train

    predicted = torch.max(y_pred.data, 1)[1]  # add up the number of correct predictions. Indexed off the first point
    batch_corr = (predicted == y_train).sum()  # how many we got correct from this batch. True = 1, False=0, sum those up
    trn_corr += batch_corr  # keep track as we go along in training.

    # Update our parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # Print out some results
    if b%600 == 0:
      print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')
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

test_data[1978][0].reshape(256, 256)
plt.imshow(test_data[1978][0].reshape(256, 256))


start_time = time.time()