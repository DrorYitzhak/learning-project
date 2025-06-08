import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("/content/drive/MyDrive/Data/Iris.csv")
df

df.value_counts('Species')

X=df.drop(['Id','Species'],axis=1)
X

y = df['Species']
y

X1 = X.to_numpy()
y1 = y.to_numpy()
print("start: ",X1.shape, y1.shape)

# Euclidean distance version #1
def euclidean_distance(a,b):
    return np.linalg.norm(a - b)

# Euclidean distance version #2, same as version #1
def euclidean_distance2(a,b):
    sum=0
    for a1,b1 in zip(a,b):
        sum += (a1-b1)**2
    return np.sqrt(sum)


import sys

# demo KNN with k=1
def OneNN(x):
  minDistance = sys.maxsize
  minDistance_index = 0
  for i in range(len(X1)):
    dis = euclidean_distance(x,X1[i])
    if dis < minDistance:
      minDistance = dis
      minDistance_index = i
  print(minDistance_index, minDistance,y1[minDistance_index])

# demo KNN with k=1
OneNN([1, 2, 3, 4])



class KNN:

    def __init__(self,n_neighbors=3):
        self._k = n_neighbors

    def fit(self, X1, y1):
        self._X1 = X1
        self._y1 = y1

    def predict_one(self, X12):
        distances = []
        for X11,y11 in zip(self._X1,self._y1):
            dist = euclidean_distance(X11, X12)
            distances.append([dist,len(distances),y11])
        distances.sort(key=lambda elem: elem[0])
        kdistances=distances[:self._k]
        klabels = [row[2] for row in kdistances]
        return max(klabels, key=klabels.count)

    def predict(self, XPred):
        return np.array([self.predict_one(x) for x in XPred])

    def score(self,X_test, y_test):
        pred = self.predict(X_test)
        return np.sum(pred==y_test) / y_test.shape[0]



X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.2)
print("train: ", X_train.shape, y_train.shape)
print("test:  " , X_test.shape, y_test.shape)
print("train: ", sum(y_train=='Iris-setosa'),sum(y_train=='Iris-versicolor'), sum(y_train=='Iris-virginica'))
print("test: ", sum(y_test=='Iris-setosa'),sum(y_test=='Iris-versicolor'), sum(y_test=='Iris-virginica'))


k = 3
knn1 = KNN(k)
knn1.fit(X_train,y_train)
print(knn1.predict([[2.0,2.0,1.0,5.0],[2.0,2.0,1.0,5.5]]))
print(knn1.predict([[2.0,2.0,1.0,5.0]]))


knn2 = KNeighborsClassifier(n_neighbors = k)
knn2.fit(X_train,y_train)
print(knn2.predict([[2.0,2.0,1.0,5.0],[2.0,2.0,1.0,5.5]]))
print(knn2.predict([[2.0,2.0,1.0,5.0]]))



pred1 = knn1.predict(X_test)
pred1

pred2 = knn2.predict(X_test)
pred2


print("********************\nk : ",k)
print("total: {} same: {}".format(y_test.shape[0], np.sum(pred1==pred2)))
print("score train: {} {}".format(knn1.score(X_train, y_train), knn2.score(X_train, y_train)))
print("score test: {} {}".format(knn1.score(X_test, y_test),knn2.score(X_test, y_test)))
print("cross val score :", np.mean(cross_val_score(knn2, X1, y1)))
print("confusion matrix test:\n", confusion_matrix(pred2, y_test))

confusion_matrix(knn2.predict(X_train),y_train)