#https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/gradient_boosting.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from NN import Loss_CC


class GradientBoosting():

    def __init__(self, n_epoch, learningrate, X, y):
        self.n_epoch = n_epoch
        self.learningrate = learningrate
        self.X = X
        self.y = y


        self.trees = []
        for i in range(n_epoch):
            self.tress.append(DecisionTreeClassifier(X, y))


    def fit(self):
        for i in range(self.n_epoch):
            grad = self.loss.gradient(y, y_predictict) #change to our model
            self.trees[i].fit(X, grad)
            update = self.trees[i].predict(X)

            y_predict -= np.matmul(self.learningrate, up)

    def predict(self, X):
        y_predictict = np.array([])

        for tree in self.trees:
            u = tree.predict(X)
            u = np.matmul(self.learningrate, u)

            y_predictict = np.exp(y_predict) / np.expand_dims(np.sum(np.exp(y_predict), axis=1), axis=1)

            y_predict = np.argmax(y_predict, axis=1)
        return y_predict
