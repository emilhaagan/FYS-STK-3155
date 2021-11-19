
#import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from autograd import grad
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
from autograd import holomorphic_grad as hgrad
from sklearn import linear_model
#from AnalysisFunctions import *
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import cm

"""
Analasys functions from project 1
"""

def FrankeFunction(x,y, noise = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return (term1 + term2 + term3 + term4 + noise*np.random.randn(len(x)))


def R2(zReal, zPredicted):
    """
    :param zReal: actual z-values, size (n, 1)
    :param zPredicted: predicted z-values, size (n, 1)
    :return: R2-score
    """
    R2 = 1 - (np.sum((zReal - zPredicted)**2)/np.sum((zReal - np.mean(zReal))**2))
    return R2

def MeanSquaredError(zReal, zPredicted):
    """
    :param zReal: actual z-values, size (n, 1)
    :param zPredicted: predicted z-values, size (n, 1)
    :return: Mean squared error
    """
    MSE = np.sum((zReal - zPredicted)**2)/len(z)
    return MSE

def betaCI_OLS(zReal, beta_mean, X):
    """
    :param zReal: actual z-values, size (n, 1)
    :param beta_mean: mean of beta
    :param X: dataset
    Compute a 90% confidence interval for the beta coefficients
    """

    # Calculate variance squared in the error
    z_hat = X.dot(beta)
    N, P = np.shape(X)
    sigma2 = (np.sum(np.power((zReal-z_hat), 2)))/N

    # Calculate the variance squared of the beta coefficients
    var_beta = np.diag(sigma2*np.linalg.inv((X.T.dot(X))))

    # The square root of var_beta is the standard error. Confidence intervals are calculated as mean +/- Z*SE
    ci_minus = beta_mean - 1.645*var_beta
    ci_plus = beta_mean + 1.645*var_beta

    return ci_minus, ci_plus


def betaCI_Ridge(zReal, beta_mean, X, l):
    """
    :param zReal: actual z-values, size (n, 1)
    :param beta_mean: mean of beta
    :param X: dataset
    Compute a 90% confidence interval for the beta coefficients - Ridge
    """

    # Calculate variance squared in the error
    z_hat = X.dot(beta)
    N, P = np.shape(X)
    sigma_2 = (np.sum(np.power((zReal-z_hat), 2)))/N

    # Calculate the variance squared of the beta coefficients
    XTX= X.T.dot(X)
    R, R = np.shape(XTX)
    var_beta = np.diag(sigma_2*np.linalg.inv((XTX + l*np.identity(R))))

    # The square root of var_beta is the standard error. Confidence intervals are calculated as mean +/- Z*SE
    ci_minus = beta_mean - 1.645*var_beta
    ci_plus = beta_mean + 1.645*var_beta

    return ci_minus, ci_plus

def plotFrankes(x_, y_, z_):
    """
    Plot Franke's function
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z - Franke')

    # Add a color bar which maps values to colors.
    clb = fig.colorbar(surf, shrink=0.5, aspect=5)
    clb.ax.set_title('Level')

    plt.show()

#Ordinary Least Squared function
def ols(x, y, z, degree = 5):
    #x: vector of size(n, 1)
    #y: vector of size(n,1)
    #z: vector of size(n,1)
    xyb_ = np.c_[x, y]
    poly = PolynomialFeatures(degree)
    xyb = poly.fit_transform(xyb_)
    #Change from inverse to SGD
    beta = StochasticGradientDecent(xyb.T.dot(xyb), z).SGD().dot(xyb.T).dot(z)
    #beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)

    return beta

def RidgeRegression(x, y, z, degree=5, l=0.0001):
    """
    :param x: numpy vector of size (n, 1)
    :param y: numpy vector of size (n, 1)
    :param degree: degree of polynomial fit
    :param l: Ridge penalty coefficient
    :return: numpy array with the beta coefficients
    """
    # Calculate matrix with x, y - polynomials
    M_ = np.c_[x, y]
    poly = PolynomialFeatures(degree)
    M = poly.fit_transform(M_)

    # Calculate beta
    A = np.arange(1, degree + 2)
    rows = np.sum(A)
    #Change from inverse to SGD
    beta = (StochasticGradientDecent(M.T.dot(M) + l * np.identity(rows), z)).SGD().dot(M.T).dot(z)

    return beta

def Lasso(x, y, z, degree=5, a=1e-06):

    X = np.c_[x, y]
    poly = PolynomialFeatures(degree=degree)
    X_ = poly.fit_transform(X)

    clf = linear_model.Lasso(alpha=a, max_iter=5000, fit_intercept=False)
    clf.fit(X_, z)
    beta = clf.coef_

    return beta

def bootstrap(x, y, z, p_degree, method, n_bootstrap=100):
    # Randomly shuffle data
    data_set = np.c_[x, y, z]
    np.random.shuffle(data_set)
    set_size = round(len(x)/5)

    # Extract test-set, never used in training. About 1/5 of total data
    x_test = data_set[0:set_size, 0]
    y_test = data_set[0:set_size, 1]
    z_test = data_set[0:set_size, 2]
    test_indices = np.linspace(0, set_size-1, set_size)

    # And define the training set as the rest of the data
    x_train = np.delete(data_set[:, 0], test_indices)
    y_train = np.delete(data_set[:, 1], test_indices)
    z_train = np.delete(data_set[:, 2], test_indices)

    Z_predict = []

    MSE = []
    R2s = []
    for i in range(n_bootstrap):
        x_, y_, z_ = resample(x_train, y_train, z_train)

        if method == 'Ridge':
            # Ridge regression, save beta values
            beta = RidgeRegression(x_, y_, z_, degree=p_degree)
        elif method == 'Lasso':
            beta = Lasso(x_, y_, z_, degree=p_degree)
        elif method == 'OLS':
            beta = ols(x_, y_, z_, degree=p_degree)
        else:
            print('ERROR: Cannot recognize method')
            return 0

        M_ = np.c_[x_test, y_test]
        poly = PolynomialFeatures(p_degree)
        M = poly.fit_transform(M_)
        z_hat = M.dot(beta)

        Z_predict.append(z_hat)

        # Calculate MSE
        MSE.append(np.mean((z_test - z_hat)**2))
        R2s.append(R2(z_test, z_hat))

    # Calculate MSE, Bias and Variance
    MSE_M = np.mean(MSE)
    R2_M = np.mean(R2s)
    bias = np.mean((z_test - np.mean(Z_predict, axis=0, keepdims=True))**2)
    variance = np.mean(np.var(Z_predict, axis=0, keepdims=True))
    return MSE_M, R2_M, bias, variance



class StochasticGradientDecent(object):
    """docstring for StochasticGradientDecent."""

    def __init__(self, x, y, n_epoc = 50, M = 10, n=1000, gamma=0.9, dtype = "float64"):

        self.x_full = x
        self.y_full = y
        self.n_epoc = n_epoc
        #size of each minibatch
        self.M = M
        self.n = n
        self.gamma = gamma
        #Some initial conditions
        #nunber of minibatch
        self.m = int(self.n/self.M)
        self.t0 = self.M
        self.t1 = self.n_epoc
        #theta dimension is based on the number of columns in design matrix
        self.v_ = 0

    def __call__(self):

        #Checks matrix size of rows
        size_matrix = self.x_full.shape[0]
        if size_matrix != self.y_full.shape[0]:
            raise ValueError("'x' and 'y' must have same rows")

        #Check to see if batches are right size
        self.n_epoc = int(self.n_epoc)
        if not 0 < self.n_epoc <= size_matrix:
            raise ValueError("Must have a batch size less or equal to observations and greater than zero.")

        #Checks gamma is in range
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be equal or greater than zero and equal or less than 1.")

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
                random_index = self.M*np.random.randint(self.m)
                xi = X[random_index:random_index+self.M]
                yi = y[random_index:random_index+self.M]

                eta = self.ls(epoc*self.m+k) #Calling function to cal. eta


                #self.v_ = gamma*self.v_ + eta*gradient(x_iter, y_iter, self.theta - gamma*self.v_) #Cal. v where gradient is from autograd
                place_hold = self.theta + self.gamma*self.v_
                x_grad = egrad(self.gradient, 2) #Gradient with respect to theta
                self.v_ = self.gamma*self.v_ + eta * self.gradient(xi, yi, self.theta) * x_grad(xi, yi, place_hold) #Cal. v where gradient is from autograd
                self.theta = self.theta - self.v_ #Theta +1 from this itteration of theta and v

        return self.theta


"""
    Analysis of a Lasso Regression model of Franke's function
"""

# Load data
X = np.load('data.npy')
x = X[:, 0]
y = X[:, 1]
z = FrankeFunction(x, y)

alphas = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3]
alpha_logs = [-10, -9, -8, -7, -6, -5, -4, -3]

Bs = []
for al in alphas:
    Bs.append(Lasso(x, y, z, degree=5, a=al))

# Generate new test data
x_test = np.random.rand(200)
y_test = np.random.rand(200)
z_test = FrankeFunction(x_test, y_test)

M_ = np.c_[x_test, y_test]
poly = PolynomialFeatures(5)
M = poly.fit_transform(M_)
MSEs = []
R2s = []
text_file = open("../Results/SGD/Bootstrap_lasso_SGD.txt", "w")
for i in range(len(alphas)):
    z_predict = M.dot(Bs[i])
    MSE = MeanSquaredError(z_test, z_predict)
    MSEs.append(MSE)
    R2_score = R2(z_test, z_predict)
    R2s.append(R2_score)
    text_file.write('--- Alpha value: {0} ---\n Mean Squared error: {1:.7f} \n R2 Score: {2:.7f}\n'.format(alphas[i], MSE, R2_score))

# make plot
fig, ax1 = plt.subplots()
ax1.plot(alpha_logs, MSEs, 'bo-')
ax1.set_xlabel('Logarithmic alpha')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('MSE', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(alpha_logs, R2s, 'r*-')
ax2.set_ylabel('R2 score', color='r')
ax2.tick_params('y', colors='r')

plt.title('Influence of alpha on MSE and R2 Score')
fig.tight_layout()
plt.savefig('../Results/SGD/MSE_R2_alpha_SGD.png')

# Investigate how the alpha values are influenced by noise
noise = np.arange(0, 0.4, 0.01)
alphas = [10**-7, 10**-3, 1]
Bs = []

# Generate more data to test
x_test = np.random.rand(200)
y_test = np.random.rand(200)
M_ = np.c_[x_test, y_test]
poly5 = PolynomialFeatures(5)
M = poly5.fit_transform(M_)

for al in alphas:
    B = []
    #print(al)
    for n in noise:
        z = FrankeFunction(x, y, noise=n)
        B.append(Lasso(x, y, z, degree=5, a=al))
    Bs.append(B)

lines = []
plt.figure()
for i in range(len(alphas)):
    text_file.write('--- alpha value: {} --- \n'.format(alphas[i]))
    line = []
    for j in range(len(noise)):
        z_test = FrankeFunction(x_test, y_test, noise=noise[j])
        z_predict = M.dot(Bs[i][j])
        MSE = MeanSquaredError(z_test, z_predict)
        line.append(MSE)
        R2_score = R2(z_test, z_predict)
        text_file.write(' Noise: {0} \n Mean Squared error: {1:.7f} \n R2 Score: {2:.7f}\n'.format(noise[j], MSE, R2_score))
    plt.plot(noise, line, label='Alpha = {0}'.format(alphas[i]))

plt.legend()
plt.xlabel('Degree of noise')
plt.ylabel('MSE')
plt.title('Influence of alpha and noise on MSE')
plt.savefig('../Results/SGD/alpha_noise_MSE_SGD.png')

MSE_l, R2_l, bias_l, variance_l = bootstrap(x, y, z, method='Lasso', p_degree=5)
text_file.write('--- BOOTSTRAP --- \n')
text_file.write('MSE: {} \n'.format(MSE_l))
text_file.write('R2: {} \n'.format(R2_l))
text_file.write('Bias: {} \n'.format(bias_l))
text_file.write('Variance: {} \n'.format(variance_l))

text_file.close()


"""
n = 1000
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x]

#print(X)


if __name__ == "__main__":
    theta = StochasticGradientDecent(x, y).SGD()
    print(theta)
"""
