import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#define Layers
"""Initializing for weights an biases for flexible number of layers with number of imputs and amount of neurons or hidden layers."""
class Layer():
    def __init__(self, n_features, neurons, lmbd = 0):
        # Initialize weights and biases
        self.n_features = n_features
        self.neurons = neurons
        self.weights = 0.01 * np.random.randn(self.n_features, self.neurons) #with normal distribution
        self.bias = np.zeros((1,self.neurons))

        # Set strength of regularization, the regularizer should be greater than or equal to 0
        self.weight_regularizer_l2 = lmbd

    def forward(self, inputs, train):
        self.input = inputs

        #Calculate output value from previous layer's inputs, weights and biases
        self.out = np.dot(inputs, self.weights) + self.bias

    def back(self, d_val):
        # gradients on weights and biases
        self.grad_weights = np.dot(self.input.T, )
        self.grad_bias = np.sum(, axis=0, keepdims=True)

        #L2 regularization on weights
        if self.weight_regularizer_l2 > 0:
            self.grad_weights += 2 * self.weight_regularizer_l2 * self.weights

        #Gradient on inputs
        self.grad_input = np.dot(d_val, self.weights.T)

class Layer_Input:
    def forward(self, inputs, train):
        self.out = inputs


'''Define activation function'''
class Activ_Sigmoid():
    def forward(self, inputs, train):
        self.input = inputs

        self.out =  1/(1 + np.exp(-inputs))
    def back(self, d_val):
        self.grad_input = d_val * (1 - self.out) * self.out

    def predict(self, out):
        return (out > 0.5) * 1

class Activ_Leaky_ReLU():
    def forward(self, inputs, train):
        self.input = inputs

        self.out = np.maximum(0.01 * inputs, inputs)

    def back(self, d_val):

        self.grad_input = d_val.copy()

        #gradient will be 0.01 when inputs are negative
        self.grad_input[self.input <= 0] = 0.01

    def predict(self, out):
        return out

class Activ_ReLU():
    def forward(self, inputs, train):
        self.input = inputs

        self.out = np.maximum(0, inputs)

    def back(self, d_val):

        self.grad_input = d_val.copy()

        #gradient will be 0 when inputs are negative
        self.grad_input[self.input <= 0] = 0

    def predict(self, out):
        return out

class Activ_Linear():
    def forward(self, inputs, train):
        self.input = inputs
        self.out = inputs

    def back(self, d_val):
        self.grad_input = d_val.copy()

    def predict(self, out):
        return out

class Activ_tanh():
    def forward(self, inputs, train):
        self.input = inputs

        self.out  = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

    def back(self, d_val):
        self.grad_input = d_val.copy()

        self.grad_input = 1-self.out**2

    def predict(self, out):
        return out

class Activ_Softmax():

    def forward(self, inputs, train):
        self.input = inputs

        # Get unnormalized probabilities
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)

        self.out = probabilities

    def back(self, d_val):

        # Create uninitialized array
        self.grad_input = np.empty_like(d_val)

        # Enumerate outputs and gradients
        for index, (single_out, single_d_val) in enumerate(zip(self.out, d_val)):
            # Flatten output array
            single_out = single_out.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_out) - np.dot(single_out, single_out.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.grad_input[index] = np.dot(jacobian_matrix, single_d_val)

    def predict(self, out):
        return np.argmax(out, axis=1)

'''Define Optimizer'''
class Optim_SGD():

    def __init__(self, momentum = 0, decay = 0, lr = 1):
        self.momentum = momentum
        self.lr = lr
        self.decay = decay

        self.cur_lr = lr
        self.iter = 0

    def pre_up_par(self):
        if self.decay:
            self.cur_lr = self.lr * (1 / (1 + self.decay * self.iter))

    def up_par(self, layer):
        if self.momentum:
            #If layer dont consist of momentum we create them
            if not hasattr(layer, "weight_momentum"):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.bias)

            #Creating weight updates
            weight_up = self.momentum * layer.weight_momentum - self.cur_lr * layer.grad_weights
            layer.weight_momentum = weight_up

            #Creating bias updates
            bias_up = self.momentum * layer.bias_momentum - self.cur_lr * layer.grad_bias
            layer.bias_momentum = bias_up

        else:
            weight_up = -self.cur_lr * layer.grad_weights

            bias_up = -self.cur_lr * layer.grad_bias


        layer.weights += weight_up
        layer.bias += bias_up

    def post_up_par(self):
        self.iter += 1

'''Define Loss'''
class Loss:

    def reg_loss(self):
        #initialize regularization loss
        reg_loss = 0

        #calculate regularization loss for each the training layer
        for layer in self.training_layer:
            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                reg_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        return reg_loss

    def remember_training_layer(self, training_layer):
        self.training_layer = training_layer

    #calculate losses from data and regularization with model output and true values
    def cal(self, out, y, *, regularization=False):
        #loss from sample
        samp_loss = self.forward(out, y)
        #mean loss
        loss_dat = np.mean(samp_loss)

        if not regularization:
            return loss_dat

        return loss_dat, self.reg_loss()


# Categorical Cross-entropy loss
class Loss_CC(Loss):

    def forward(self, predict, y_real):
        #size of each batch
        sample = len(predict)

        # Clip data to avoid denominator of 0
        predict_clip = np.clip(predict, 1e-8, 1 - 1e-8)

        #Probabilities for target values of categorical labels
        if len(y_real.shape) == 1:
            cc = predict_clip[range(sample), y_real]

        elif len(y_real.shape) == 2:
            cc = np.sum(predict_clip * y_real, axis = 1)

        nll = -np.log(cc)
        return nll

    def back(self, d_val, y_real):

        sample = len(d_val)
        lab = len(d_val[0])

        if len(y_real.shape) == 1:
            y_real = np.eye(lab)[y_real]

        #Claculate and normalize
        self.grad_input = (-y_real/d_val)/sample

class Loss_MSE(Loss):

    def forwar(self, predict, y_real):
        samp_loss = np.mean((y_real - predict)**2, axis=-1)

        return samp_loss

    def backward(self, d_val, y_real):
        sample = len(d_val)

        out = len(d_val[0])

        #Claculate and normalize
        self.grad_input = (-2 * (y_real - d_val) / out) / sample

'''Define Accuracy'''
class Accuracy:

    def cal(self, predict, y):
        # Get comparison results
        comparisons = self.compare(predict, y)
        acc = np.mean(comparisons)

        return acc

    def cal_accum(self):

        acc = self.accum_sum / self.accum_count

        return acc

    # Reset variables for accumulated accuracy
    def reset_var(self):

        self.accum_sum = 0
        self.accum_count = 0


class Accuracy_Classification(Accuracy):

    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predict , y):

        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis = 1)

        return predict == y


class Accuracy_Regression(Accuracy):

    def __init__(self, y):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # Compares predictions to the ground truth values
    def compare(self, predict, y):
        return np.absolute(predict - y) < self.precision

'''Define Model'''
class Method():

    def __init__(self):
        self.layer = []
        self.train_acc = None
        self.test_acc = None

    def add_to_list(self, layer):
        self.layer.append(layer)

    def set_param(self, *, loss, accuracy, optimiz):
        self.optimiz = optimiz
        self.accuracy = accuracy
        self.loss = loss

    def finall(self):

        self.layer_inp = Layer_Input()
        layer_iter = len(self.layer)
        self.tlayer = []

        for i in range(layer_iter):

            if i == 0:
                self.layer[i].prev = self.layer_inp
                self.layer[i].next = self.layer[i+1]

            elif i < layer_iter - 1:
                self.layer[i].prev = self.layer[i-1]
                self.layer[i].next = self.layer[i+1]

            else:
                self.layer[i].prev = self.layer[i-1]
                self.layer[i].next = self.loss
                self.activ_out = self.layer[i]

            if hasattr(self.layer[i], "weights"):
                self.tlayer.append(self.layer[i])

        # Update loss object with trainable layers
        self.loss.remember_training_layer(self.tlayer)


    def train(self, X, y, *, n_epoc = 1, validation_data = None, print_epoch = False):

        self.accuracy.init(y)

        for epochs in range(1, n_epoc+1):

            out = self.forward(X, train=True)

            loss_dat, reg_loss = self.loss.cal(out, y, regularization = True)
            loss = loss_dat + reg_loss

            predict = self.activ_out.predict(out) #??
            accuracy = self.accuracy.cal(predict, y)

            self.back(out, y)

            self.optimiz.pre_up_par()
            for layer in self.tlayer:
                self.optimiz.up_par(layer)
            self.optimiz.post_up_par()

            """
            if print_epoch:
                print(f'n_epoch: {epochs}, ' +
                      f'accuracy: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'loss in data: {loss_dat:.3f}, ' +
                      f'loss in reglarization: {reg_loss:.3f}), ' +
                      f'learningrate: {self.optimiz.cur_lr}')
            """

            if epochs == n_epoc:
                self.train_acc = accuracy


        if validation_data is not None:

            X_val, y_val = validation_data

            out = self.forward(X_val, train=False)

            loss = self.loss.cal(out, y_val)

            predict = self.activ_out.predict(out)

            accuracy = self.accuracy.cal(predict, y_val)

            self.test_acc = accuracy

            if print_epoch:
                print(f'validation, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f}')

        return self.train_acc, self.test_acc


    def forward(self, X, train):

        self.layer_inp.forward(X, train)

        for layer in self.layer:
            layer.forward(layer.prev.out, train)

        return layer.out

    def back(self, out, y):

        self.loss.back(out, y)

        for layer in reversed(self.layer):
            layer.back(layer.next.grad_input)

'''Test Wisconsin Cancer Data'''
data = load_breast_cancer()
X = data['data']
y = data['target']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

n_neurons = 100
train_accuracy = []
test_accurary = []
for n in range(n_neurons):

    # Instantiate the model
    model = Method()
    # Add layers
    model.add_to_list(Layer(30, n, lmbd=5e-4))
    model.add_to_list(Activ_ReLU())
    model.add_to_list(Layer(n, 2))
    model.add_to_list(Activ_Softmax())

    # Set loss, optimizer and accuracy objects
    model.set_param(
        loss=Loss_CC(),
        optimiz=Optim_SGD(lr=0.05, decay=5e-5, momentum=0.9),
        accuracy=Accuracy_Classification()
    )

    # Finalize the model
    model.finall()

    # Train the model
    train_acc, test_acc = model.train(X_train, y_train, validation_data=(X_test, y_test), n_epoc=200, print_epoch = False)
    train_accuracy.append(train_acc)
    test_accurary.append(test_acc)

x = range(n_neurons)
fig = plt.figure()
plt.plot(x, train_accuracy, label = "train_accuracy")
plt.plot(x, test_accurary, label = "test_accurary")
fig.suptitle('Accuracy of train and test dataset with different number of neurons in the hidden layer')
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../Results/acc_n_nruons_NN.png')
