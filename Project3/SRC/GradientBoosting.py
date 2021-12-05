#Remove before handing inn:
#https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/gradient_boosting.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#from NN import Loss_CC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate



class GradientBoosting():

    def __init__(self, n_epoch, learningrate):
        self.n_epoch = n_epoch
        self.learningrate = learningrate


        self.trees = []
        for i in range(n_epoch):
            self.trees.append(DecisionTreeClassifier())


    def fit(self, X, y):
        y_predictict = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_epoch):
            grad = self.grad(y, y_predictict)
            self.trees[i].fit(X, grad)
            update = self.trees[i].predict(X)

            y_predict -= np.matmul(self.learningrate, u)

    def predict(self, X):
        y_predictict = np.array([])

        for tree in self.trees:
            u = tree.predict(X)
            u = np.matmul(self.learningrate, u)
            y_predictict = -u if not y_predictict.any() else y_predictict - u
            y_predictict = np.exp(y_predict) / np.expand_dims(np.sum(np.exp(y_predict), axis=1), axis=1)

            y_predict = np.argmax(y_predict, axis=1)
        return y_predict

    def grad(self, y, predict):
        predict = np.clip(predict, 1e-14, 1-1e-14)
        r = y/predict + (1-y)/(1-predict)
        return r



'''Test Wisconsin Cancer Data'''
data = load_breast_cancer()
X = data['data']
y = data['target']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



clf = GradientBoosting(n_epoch = 100, learningrate=0.1)

clf.fit(X_train, y_train)
#Cross validation
accuracy = cross_validate(clf, X_test, y_test,cv=10)['test_score']
print(accuracy)
