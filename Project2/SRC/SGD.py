
#import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from autograd import grad
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
from sklearn import linear_model

class StochasticGradientDecent(object):
    """docstring for StochasticGradientDecent."""

    """ x and y are the input data where y must have the same number of rows as x
        e_epoc is the number of times we run the minibatches, default 50
        M is the size of each minibatch, default 10
        n is the amount of rows or data enteries, default 1000
        gamma is a set value 0 <= gamma <= 1, default 0.3"""

    def __init__(self, x, y, n_epoc = 50, M = 10, n=1000, gamma=0.3):

        self.x_full = x
        self.y_full = y
        self.n_epoc = n_epoc

        #size of each minibatch
        self.M = M
        self.n = n
        self.gamma = 0.3

        #Some initial conditions
        #number of minibatch
        self.m = int(self.n/self.M)
        self.t0 = self.M
        self.t1 = self.n_epoc

        #theta dimension is based on the number of columns in design matrix
        self.v_ = 0

    def __call__(self):

        #Checks matrix size of rows
        size_matrix = self.x_full.shape[0]
        if size_matrix != self.y_full.shape[0]:
            raise ValueError("'x' and 'y' must have same rows.")

        #Check to see if batches are right size
        self.n_epoc = int(self.n_epoc)
        if not 0 < self.n_epoc <= size_matrix:
            raise ValueError("Must have a batch size less or equal to observations and greater than zero.")

        #Checks gamma is in range
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be equal or greater than zero and equal or less than 1.")

    #gradient
    def gradient(self, x, y, theta):
        return (2.0/self.M)*x.T @ ((x @ theta) - y)

    #This the learning scheduel for eta
    def ls(self, t):
        return self.t0/(t+self.t1)

    #The eta values function
    #def eta(self, t):
        #return self.t0**2/(t+self.t1)


    def SGD(self):
        X = np.c_[np.ones((n,1)), self.x_full]
        #size_matrix = x.shape[0]
        self.theta = np.random.randn(X.shape[1],1) #Initilize theta for matrix shape.

        #xy = np.c_[x.reshape(size_matrix, -1), y.reshape(size_matrix, 1)]

        #Main SGD loop for epochs of minibatches
        for epoc in range(self.n_epoc):
            #Second SGD loop with random choice of k
            for k in range(self.m):
                random_index = np.random.randint(self.m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]

                eta = self.ls(epoc*self.m+k) #Calling function to cal. eta


                #self.v_ = gamma*self.v_ + eta*gradient(x_iter, y_iter, self.theta - gamma*self.v_) #Cal. v where gradient is from autograd
                place_hold = self.theta + self.gamma*self.v_
                x_grad = egrad(self.gradient, 2) #Gradient with respect to theta
                self.v_ = self.gamma*self.v_ + eta * self.gradient(xi, yi, self.theta) * x_grad(xi, yi, place_hold) #Cal. v where gradient is from autograd
                self.theta = self.theta - self.v_ #Theta +1 from this itteration of theta and v

        return self.theta

n = 1000
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x]

#print(X)


if __name__ == "__main__":
    theta = StochasticGradientDecent(x, y).SGD()
    print(theta)
"""
#Standard gradient decent
def gradient(x, y, theta):
    return 1/2*x.T @ ((x * theta) - y)

#This is out gamma and the learning scheduel
def learning_schedule(t0, t1, dt):
    return t0/(t0+t1*dt)

#The eta values function
def eta(t0, t1, dt):
    return dt**2/(t0+t1*dt)
#SGD which takes the arrays x and y, with n_epoc batches and M minibatches for n itterations
def SGD_02(learning_schedule, eta, x, y, n_epoc = 50, M = 5, n=1000, dtype = "float64"):

    d_type = np.dtype(dtype)

    #Tests for collable functions
    if not callable(learning_schedule):
        raise TypeError("Can not call 'learning_schedule' must be a callable function")

    if not callable(eta):
        raise TypeError("Can not call 'eta' must be a callable function")

    #Checks matrix size
    size_matrix = x.shape[0]
    if size_matrix != y.shape[0]:
        raise ValueError("'x' and 'y' must have same dimentions")

    #Check to see if batches are right size
    n_epoc = int(n_epoc)
    if not 0 < n_epoc <= size_matrix:
        raise ValueError("Must have a batch size less or equal to observations and greater than zero")


    #Check n is greater than zero
    n = int(n)
    if  n <= 0:
        raise ValueError("'n' must be greater than 0 ")

    #Some initial conditions
    m = int(n/M)
    t0 = M
    t1 = n_epoc

    theta = 0
    v_ = 0

    #Setting up arrays
    x = np.array(x, dtype=d_type)
    y = np.array(y, dtype=d_type)

    xy = np.c_[x.reshape(size_matrix, -1), y.reshape(size_matrix, 1)]

    #Main SGD loop
    for epoc in range(n_epoc):
        #Second SGD loop
        for i in range(m):
            end = i + n_epoc
            #Defining x and y for each itteration
            x_iter = xy[i:end, :-1]
            y_iter = xy[i:end, -1:]

            gamma = learning_schedule(t0, t1, epoc/(m+i)) #Calling function to cal. gamma
            eta_ = eta(t0, t1, epoc/(m+i)) #Calling function to cal. eta

            v_ = gamma*v_ + eta_*grad(gradient)(x_iter, y_iter, theta - gamma*v_) #Cal. v where gradient is from autograd
            theta = theta - v_ #Theta +1 from this itteration of theta and v

    return theta

#Test run
n = 1000
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)

func = SGD_02(learning_schedule, eta, x=x, y=y)
print("This is the SGD:")
print(func)

"""



#SGDReg =linear_model.SGDRegressor(max_iter = 1000,penalty = "elasticnet",loss = 'huber',tol = 1e-3, average = True)
#SGDReg.fit(x, y)
#print(SGDReg.coef_.fit(x, y))

"""
#Old SGD



def SGD(gradient, x, y, first_iter=1, lr = 0.1, batch_sz = 5, n=100, n_tol = 1e-7, dtype = "float64"):

    #See if we can call gradient
    if not callable(gradient):
        raise TypeError("Can not call 'gradient' must be callable")

    d_type = np.dtype(dtype)

    #Setting x and y to arrays
    x = np.array(x, dtype=d_type)
    y = np.array(y, dtype=d_type)
    size_matrix = x.shape[0]

    #initilizing the vector
    v = np.array(first_iter, dtype = d_type)

    #Checking matrix size, learningrate and batch size
    if size_matrix != y.shape[0]:
        raise ValueError("x and y must have same dimentions")

    lr = np.array(lr, dtype = d_type)
    if np.any(lr <= 0):
        raise ValueError("The learning rate must be > 0")

    batch_sz = int(batch_sz)
    if not 0 < batch_sz <= size_matrix:
        raise ValueError("Must have a batch size less or equal to observations and greater than zero")

    #Checking n itterations and the tolerance
    n = int(n)
    if  n <= 0:
        raise ValueError("n must be > 0 ")

    n_tol = np.array(n_tol, dtype=d_type)
    if np.any(n_tol <= 0):
        raise ValueError("The tolerance must be > 0")

    #Might need to change
    theta = 0
    eta_ = 0
    v_ = 0
    xy = np.c_[x.reshape(size_matrix, -1), y.reshape(size_matrix, 1)]

    #gradient descent
    for j in range(n):
        for i in range(0, size_matrix, batch_sz):
            end = i + batch_sz
            x_iter = xy[i:end, :-1]
            y_iter = xy[i:end, -1:]

            gamma = learning_schedule(j*size_matrix+i, batch_sz, n)
            #theta = theta - gamma*gradient(x_iter, y_iter, theta)
            eta_ = eta(j*size_matrix+i, batch_sz, n)
            a = theta+gamma*v_
            v_ = gamma*v_ + eta_*gradient(x_iter, y_iter, a)
            theta = theta - v_
    return theta
"""



"""
#From lecture just test
def SGD(x, y, M = 5, n_epoch = 50, ):
    #M size of each minibatch

    n = len(x)
    X = np.c_[np.ones((n,1)), x]
    theta_linregres = np.linalg.inv(X.T @ X) @ (X.T @ y)

    theta = np.random.randn(2,1)
    eta = 10^(-5)



    m = int(n/M) #number of minibatches

    for e in range(n_epoch):
        for i in range(m):
            rand_indx = np.random.randint(m)
            x_i = X[rand_indx:rand_indx+1]
            y_i = y[rand_indx:rand_indx+1]
            grad = 2*x_i.T @ ((x_i @ theta) - y_i)
            eta = learning_schedule(e*m+i, M, n_epoch)
            theta = theta - eta*grad

    x_new = np.array([[0],[2]])
    X_new = np.c_[np.ones((2,1)), x_new]
    y_predict = X_new.dot(theta)

    return theta, x_new, y_predict
"""


"""Test code above"""
"""
n = 1000
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)

theta, x_new, y_predict = SGD(x, y)

print(theta)

plt.plot(x_new, y_predict, "b-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.show()
"""

"""Simple Gradient descent"""
"""
def step_length(t,t0,t1):
    return t0/(t+t1)

n = 100 #100 datapoints
M = 5   #size of each minibatch
m = int(n/M) #number of minibatches
n_epochs = 500 #number of epochs
t0 = 1.0
t1 = 10

gamma_j = t0/t1
j = 0
for epoch in range(1,n_epochs+1):
    for i in range(m):
        k = np.random.randint(m) #Pick the k-th minibatch at random
        #Compute the gradient using the data in minibatch Bk
        #Compute new suggestion for beta
        t = epoch*m+i
        gamma_j = step_length(t,t0,t1)
        j += 1

print("gamma_j after %d epochs: %g" % (n_epochs,gamma_j))
"""
