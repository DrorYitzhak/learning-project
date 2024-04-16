import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from LogisticRegression import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('C:\PycharmProjects\pythonProject5\Database for training\diabetes_risk_prediction_dataset.csv')
df.head()
df['Polyuria'] = df['Polyuria'].replace({'No': 0, 'Yes': 1})
df['Polydipsia'] = df['Polydipsia'].replace({'No': 0, 'Yes': 1})
df['sudden weight loss'] = df['sudden weight loss'].replace({'No': 0, 'Yes': 1})
df['weakness'] = df['weakness'].replace({'No': 0, 'Yes': 1})
df['Polyphagia'] = df['Polyphagia'].replace({'No': 0, 'Yes': 1})
df['Genital thrush'] = df['Genital thrush'].replace({'No': 0, 'Yes': 1})
df['visual blurring'] = df['visual blurring'].replace({'No': 0, 'Yes': 1})
df['Itching'] = df['Itching'].replace({'No': 0, 'Yes': 1})
df['Irritability'] = df['Irritability'].replace({'No': 0, 'Yes': 1})
df['delayed healing'] = df['delayed healing'].replace({'No': 0, 'Yes': 1})
df['partial paresis'] = df['partial paresis'].replace({'No': 0, 'Yes': 1})
df['muscle stiffness'] = df['muscle stiffness'].replace({'No': 0, 'Yes': 1})
df['Alopecia'] = df['Alopecia'].replace({'No': 0, 'Yes': 1})
df['Obesity'] = df['Obesity'].replace({'No': 0, 'Yes': 1})
df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})
df['class'] = df['class'].replace({'Negative': 0, 'Positive': 1})


x1 = df.drop(['class'], axis=1).to_numpy()
y1 = df['class'].to_numpy()
print("start: ", x1.shape, y1.shape)

X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=1234)


print("train: ", X_train.shape, y_train.shape)
print("test: ", X_test.shape, y_test.shape)

# normaliz of the data
scaler = MinMaxScaler()
print("Max: ", X_train.max())
print("Min: ", X_train.min())

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

print("Max: ", X_train.max())
print("Min: ", X_train.min())

clf = LogisticRegression(lr=0.01)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)