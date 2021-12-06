#Remove before handing inn:
#https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/gradient_boosting.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#from NN import Loss_CC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn import svm



class GradientBoosting():

    def __init__(self, n_epoch, learningrate, max_depth):
        self.n_epoch = n_epoch
        self.learningrate = learningrate
        self.max_depth = max_depth


        self.trees = []
        for i in range(n_epoch):
            #self.trees.append(DecisionTreeClassifier())
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth))


    def fit(self, X, y):
        y_predict = np.full(np.shape(y), np.mean(y, axis=0))

        for i in range(self.n_epoch):
            grad = self.grad(y, y_predict)
            self.trees[i].fit(X, grad)
            up = self.trees[i].predict(X)
            #fit_tree = DecisionTreeRegressor().fit(X, grad)
            #up = DecisionTreeRegressor().predict(X)
            #up = self.predict(DecisionTreeRegressor(), X, y)

            y_predict -= np.multiply(self.learningrate, up)


    def predict(self, X):
        y_predictict = np.array([])

        for tree in self.trees:
            u = tree.predict(X)
            u = np.multiply(self.learningrate, u)
            y_predict = -u if not y_predictict.any() else y_predict - u
            y_predict = np.exp(y_predict) / np.expand_dims(np.sum(np.exp(y_predict), axis=1), axis=1)

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
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)



clf = GradientBoosting(n_epoch = 100, learningrate=1.0, max_depth = 3)

print("---------------------------------------------")
print(clf.fit(X_train, y_train))
print("---------------------------------------------")

clf.fit(X_train, y_train)
#clf.predict(X_train)

#Cross validation
#cg = svm.SVC(clf).fit(X_train, y_train)

score = cross_validate(clf, X_test, y_test ,cv=10)['test_score'] #scoring="accuracy"
print(score)

#print(cg.score(X_train, y_train))
#print(cg.score(X_test, y_test))







"""

import scikitplot as skplt
from sklearn.ensemble import GradientBoostingClassifier



#now scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

gd_clf = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=1.0)
gd_clf.fit(X_train_scaled, y_train)
#Cross validation
accuracy = cross_validate(gd_clf,X_test_scaled,y_test,cv=10)['test_score']
print(accuracy)
print("Test set accuracy with Random Forests and scaled data: {:.2f}".format(gd_clf.score(X_test_scaled,y_test)))



y_pred = gd_clf.predict(X_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

plt.show()
y_probas = gd_clf.predict_proba(X_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas)

plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas)

plt.show()
"""
