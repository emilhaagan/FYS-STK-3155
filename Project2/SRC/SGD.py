
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from autograd import grad
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs



def gradient(x, y, theta):
    return 2*x.T @ ((x * theta) - y)

def learning_schedule(t, t_0, t_1):
    return t_0/(t+t_1)

def SGD(gradient, x, y, first_iter=1, lr = 0.1, batch_sz = 1, n=100, n_tol = 1e-7, dtype = "float64"):

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
    xy = np.c_[x.reshape(size_matrix, -1), y.reshape(size_matrix, 1)]

    #gradient descent
    for j in range(n):
        for i in range(0, size_matrix, batch_sz):
            end = i + batch_sz
            x_iter = xy[i:end, :-1]
            y_iter = xy[i:end, -1:]

            gamma = learning_schedule(j*size_matrix+i, batch_sz, n)
            theta = theta - gamma*gradient(x_iter, y_iter, theta)

            v = theta

    return v

n = 1000
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)
v = SGD(gradient,x=x, y=y, n = n)
print(v)


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
