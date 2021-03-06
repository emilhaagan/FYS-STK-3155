import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from autograd import grad
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
from sklearn import linear_model
from SGD import StochasticGradientDecent


"""Feed Forward Neural Network"""
class NeuralNetwork(object):
    def __init__(self, x, y, hidden_neurons=50, categories= 10, n_epochs=10 ,batch_sz = 100, lmbda = 0.001, activation_func_hidden = "sigmoid", activation_func_out = "tanh"):

        #Setting up initial conditions for class
        self.X_full = x
        self.Y_full = y

        self.input = x.shape[0]
        self.feauters = x.shape[1]
        self.hidden_neurons = hidden_neurons
        self.categories = categories
        self.n_epochs = n_epochs
        self.lmbda = lmbda
        self.batch_sz=  batch_sz
        self.iter = self.input // self.batch_sz
        self.eta = (self.n_epochs/2)/(self.n_epochs/2+self.n_epochs) #Learning scheduel

        #Initilize theta from SGD
        self.theta = StochasticGradientDecent(x, y, n_epoc = self.n_epochs, M = self.categories, n=np.size(self.input), gamma=0.3).SGD()

        #Allows other activation function for hidden layer
        if activation_func_hidden == "sigmoid":
            self.activation_func_hidden = self.sigmoid
        #Allows other activation function for output layer
        elif activation_func_out == "tanh":
            self.activation_func_out = self.tanh
        elif activation_func_out == "relu":
            self.activation_func_out = self.re

        #Creating bias and weight by running function
        self.crt_b_w()

    def __call__(self):

        #Checks matrix size of rows
        size_matrix = self.x_full.shape[0]
        if size_matrix != self.y_full.shape[0]:
            raise ValueError("'x' and 'y' must have same rows")

        #Small tests to check input
        size_matrix = X.shape[0]
        self.n_epochs = int(self.n_epoch)
        if not 0 < self.n_epoch <= size_matrix:
            raise ValueError("Must have a 'epochs' size less or equal to observations and greater than zero")

        self.batch_sz = int(self.batch_sz)
        if  self.batch_sz <= 0:
            raise ValueError("'Batch size' must be greater than 0.")



    def crt_b_w(self):
        # weights and bias in our hidden
        #Note addind +0.01 so that if we have zero its changed to a low value
        self.h_weights = 0.01 + np.random.normal(self.features, self.hidden_neurons) #with normal distribution
        self.h_bias = np.zeros(self.hidden_neurons)

        # weights and bias in our output
        self.out_weights = 0.01 + np.random.normal(self.hidden_neurons, self.categories)
        self.out_bias = np.zeros(self.categories)

    #Sigmoid activation function
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    #Tanh activation function
    def tanh(self, x):
	    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    #ReLU activation function
    def relu(self, x):
        return np.maximum(0,x)

    def cost_MSE(self y_h):
        n = np.size(y)
        C = 0
        for i in range(n):
            C += (self.Y_full[i] - y_h[i])**2
        return 1/n * C

    def ff(self):
        #Feed forward for network saved globaly in class
        self.z_hidden = np.matmul(self.X_full, self.h_weights) + self.h_bias

        activation_hidden = self.activation_func_hidden(self.z_hidden)

        self.z_out = np.matmul(activation_hidden, self.out_weights) + self.out_bias
        self.a_expect = self.activation_func_out(self.z_out)
        self.probability = self.a_expect/np.sum(self.a_expect, axis=1, keepdim=True)


    def ff_out(self):
        #feed forward output saved localy in function
        z_hidden = np.matmul(X, self.h_weights) + self.h_bias

        activation_hidden = self.activation_func_hidden(z_hidden)

        z_out = np.matmul(activation_hidden, self.out_weights) + self.out_bias
        a_expect = self.activation_func_out(z_out)
        probability = a_expect/np.sum(a_expect, axis=1, keepdim=True)
        return probability

    def backprop(self):
        self.error_in_out = self.probability - self.Y_full

        self.error_in_hidden = np.matmul(self.error_in_out, self.out_weights.T) * self.activation_hidden*(1-self.activation_hidden)

        self.grad_weight_out = np.matmul(self.activation_hidden.T, self.error_in_hidden)
        self.grad_bias_out = np.sum(self.error_in_out, axis=0)


        self.grad_weight_hidden = np.matmul(self.X_full.T, self.error_in_hidden)
        self.grad_bias_hidden = np.sum(self.error_in_hidden, axis=0)

        if self.lmbda > 0:
            self.grad_weight_out += self.lmbda * self.out_weights
            self.grad_weight_hidden += self.lmbda * self.h_weights

        self.out_weights -= self.eta*self.grad_weight_out
        self.out_bias -= self.eta*self.grad_bias_out

        self.h_weights -= self.eta*self.grad_weight_hidden
        self.h_bias -= self.eta*self.grad_bias_hidden

    def predict(self, X):
        return np.argmax(self.ff_out(X), axis=1)

    def pred_prob(self, X):
        return self.ff_out(X)

    def train_function(self):
        indec = np.arange(self.inputs)

        for i in range(self.epochs):
            for l in range(self.iter):
                data_points=np.random.choice(indec, size=self.batch_sz, replace=False)

                self.X_full = self.X_full[data_points]
                self.Y_full = self.Y_full[data_points]

                self.ff()
                self.backprop()
