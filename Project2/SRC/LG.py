import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Logistic_Regression:

    def __init__(self, learning_rate = 1, n_iter = 2000):
        self.lr = learning_rate

        self.n_iter = n_iter
        self.weights = []
        self.bias = 0

    def wb_init(self, dim):
        weights = np.zeros((dim,1))
        bias = 0
        return weights, bias

    def sigmoid(self, z):
        s = 1/(1 + np.exp(-z))
        return s

    def assumption(self, weights, X, bias):
        assum = self.sigmoid(np.dot(weights.T, X) + bias)
        return assum

    def cost(self, assum, y, num_sample):
        #print(assum.shape)
        #print(y.shape)
        cost = -np.sum(y * np.log(assum) + (1-y) * np.log(1-assum)) / num_sample
        #cost = np.squeeze(cost)
        return cost

    def grad_cal(self, weights, assum, X, y):
        num_sample = X.shape[1]
        grad_weights = np.dot(X,(assum-y).T) / num_sample
        grad_bias = np.sum(assum-y) / num_sample
        grads = {
            "grad_weights": grad_weights,
            "grad_bias": grad_bias
        }
        return grads

    def model_status(self, weights, bias, X, y):
        num_sample = X.shape[1]
        assum = self.assumption(weights, X, bias)
        cost = self.cost(assum, y, num_sample)
        grads = self.grad_cal(weights, assum, X, y)
        return grads, cost

    #weights and bias was optimized by stochastic gradient descent algorithm with momentum
    def sgd(self, weights, bias, X, y, print_cost = False):
        costs = []

        for i in range(self.n_iter+1):

            grads, cost = self.model_status(weights, bias, X, y)

            grad_weights = grads["grad_weights"]
            grad_bias = grads["grad_bias"]

            #updates
            weights = weights - self.lr * grad_weights
            bias = bias - self.lr * grad_bias


            #record cost
            if i % 100 == 0:
                costs.append(cost)

            #print cost every 100 training epoch
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        parameters = {
            "weights": weights,
            "bias": bias
        }

        gradients = {
            "grad_weights": grad_weights,
            "grad_bias": grad_bias
        }

        return parameters, gradients, costs

    def predict(self, X):
        X = np.array(X)
        num_sample = X.shape[1]

        y_pred = np.zeros((1,num_sample))

        weights = self.weights.reshape(X.shape[0], 1)
        bias = self.bias

        assum = self.assumption(weights, X, bias)

        for i in range(assum.shape[1]):
            if assum[0,i] >= 0.5:
                y_pred[0,i] = 1
            else:
                y_pred[0,i] = 0

        return y_pred

    def accuracy(self, y_pred, y):
        accuracy = 100 - np.mean(np.abs(y_pred - y)) * 100
        return accuracy

    def train(self, X_train, y_train, X_test, y_test, print_cost = True):
        #initilize parameters with normal distribution
        weights, bias = self.wb_init(X_train.shape[0])

        #update parameters with sgd algorithm
        parameters, gradients, costs = self.sgd(weights, bias, X_train, y_train, print_cost)
        self.weights = parameters["weights"]
        self.bias = parameters["bias"]

        #make predictions
        y_pred_test = self.predict(X_test)
        y_pred_train = self.predict(X_train)
        #print(y_pred_train, y_train)

        train_acc = self.accuracy(y_pred_train, y_train)
        test_acc = self.accuracy(y_pred_test, y_test)

        print ("train accuracy: %f" %(train_acc))
        print ("test accuracy: %f" %(test_acc))

        output = {
            "costs": costs,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "weights": self.weights,
            "bias": self.bias,
            "learning rate": self.lr,
            "train accuracy": train_acc,
            "test accuracy": test_acc
        }

        return output

data = load_breast_cancer()
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train).T
X_test = sc.transform(X_test).T
y_train = y_train.reshape((1,y_train.shape[0]))
y_test = y_test.reshape((1,y_test.shape[0]))

lg = Logistic_Regression(learning_rate = 0.2, n_iter = 1000)
output = lg.train(X_train, y_train, X_test, y_test)
