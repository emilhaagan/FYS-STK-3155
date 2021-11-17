import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ensure the same random numbers appear every time
np.random.seed(100)

class Logistic_Regression:

    def __init__(self, learning_rate = 1, lamda = 0, n_iter = 200):
        self.lr = learning_rate
        self.regularizer_l2 = lamda
        self.n_iter = n_iter
        self.weights = []
        self.bias = 0
        self.data_cost = []
        self.reg_cost = []

    def wb_init(self, dim):
        weights = np.zeros((dim,1))
        bias = 0
        return weights, bias

    def sigmoid(self, z):
        s = 1/(1 + np.exp(-z))
        return s

    def probability(self, weights, X, bias):
        prob = self.sigmoid(np.dot(weights.T, X) + bias)
        return prob

    def cost(self, prob, y, num_sample, weights):
        data_cost = np.mean(-np.sum(y * np.log(prob) + (1-y) * np.log(1-prob)))
        self.data_cost.append(data_cost)

        reg_cost = 0
        if self.regularizer_l2 > 0:
            reg_cost = self.regularizer_l2 * np.sum(weights**2)
            self.reg_cost.append(reg_cost)

        #print("data_cost", data_cost)
        #print("reg_cost", reg_cost)


        cost = data_cost + reg_cost
        #print("cost", cost)

        return cost

    def grad_cal(self, weights, prob, X, y):
        num_sample = X.shape[1]
        grad_weights = np.dot(X,(prob-y).T) / num_sample
        grad_bias = np.sum(prob-y) / num_sample
        grads = {
            "grad_weights": grad_weights,
            "grad_bias": grad_bias
        }
        return grads

    def model_status(self, weights, bias, X, y):
        num_sample = X.shape[1]
        prob = self.probability(weights, X, bias)
        prob_clip = np.clip(prob, 1e-8, 1 - 1e-8)
        cost = self.cost(prob_clip, y, num_sample, weights)
        grads = self.grad_cal(weights, prob, X, y)
        return grads, cost

    #weights and bias was optimized by stochastic gradient descent algorithm with momentum
    def sgd(self, weights, bias, X, y, print_cost = False):
        costs = []

        for i in range(self.n_iter):

            grads, cost = self.model_status(weights, bias, X, y)
            grad_weights = grads["grad_weights"]
            grad_bias = grads["grad_bias"]

            #updates
            weights = weights - self.lr * grad_weights
            bias = bias - self.lr * grad_bias

            #record cost
            costs.append(cost)

            #print cost every 100 training epoch
            if print_cost:
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

        prob = self.probability(weights, X, bias)

        for i in range(prob.shape[1]):
            if prob[0,i] >= 0.5:
                y_pred[0,i] = 1
            else:
                y_pred[0,i] = 0

        return y_pred

    def accuracy(self, y_pred, y):
        accuracy = 100 - np.mean(np.abs(y_pred - y)) * 100
        return accuracy

    def train(self, X_train, y_train, X_test, y_test, print_cost = False, print_score = False):
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
        if print_score:
            print ("train accuracy: %f" %(train_acc))
            print ("test accuracy: %f" %(test_acc))

        output = {
            "costs": costs,
            "data_cost": self.data_cost,
            "reg_cost": self.reg_cost,
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

#set up learning rate eta and regularization lambda for grid search
lr = np.logspace(-5, 1, 7)
lmbd = np.logspace(-5, 1, 7)

train_accuracy = np.zeros((len(lr), len(lmbd)))
test_accuracy = np.zeros((len(lr), len(lmbd)))

# grid search
for i, eta in enumerate(lr):
    for j, lm in enumerate(lmbd):
        # Instantiate the model
        lg = Logistic_Regression(learning_rate = eta, lamda = lm)
        out = lg.train(X_train, y_train, X_test, y_test)

        train_accuracy[i][j] = out['train accuracy']
        test_accuracy[i][j] = out['test accuracy']

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy of our own LG" )
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig('../Results/eta_lmd_train_acc_LG.png')

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy of our own LG")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig('../Results/eta_lmd_test_acc_LG.png')

#plot the cost based on the incresing epochs
lg = Logistic_Regression(n_iter = 2000, lamda = 1)
d = lg.train(X_train, y_train, X_test, y_test)

costs = d["costs"]
data_cost = d["data_cost"]
reg_cost = d["reg_cost"]
x = np.linspace(0, len(data_cost), len(data_cost))
fig = plt.figure()
plt.plot(x, data_cost, label = "data_cost")
plt.plot(x, reg_cost, label = "reg_cost")
plt.plot(x, costs, label = "costs")
fig.suptitle('Training cost')
plt.xlabel('Epoch')
plt.ylabel('cost')
plt.legend()
plt.savefig('../Results/costs_epoch_LG.png')
