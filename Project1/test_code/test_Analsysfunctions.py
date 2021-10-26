import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn import linear_model
import unittest

"""
To run test with python unittest run line:
python -m unittest test_Analsysfunctions
"""


#Define Franke function for all test
def FrankeFunction(x,y, noise = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return (term1 + term2 + term3 + term4 + noise*np.random.randn(len(x)))



#Class defined to run unittest
class Test_AnalasysFunctions(unittest.TestCase):
    def test_MSE(self):
        y = np.eye(3)
        y_ = np.ones((3,3))
        MSE = np.sum((y - y_)**2)/len(y)
        self.assertAlmostEqual(MSE, 2.0)

    def test_R2(self):
        zReal = np.eye(3)
        zPredicted = np.ones((3,3))
        R2 = 1 - (np.sum((zReal - zPredicted)**2)/np.sum((zReal - np.mean(zReal))**2))
        self.assertAlmostEqual(R2, -2.0)

    def test_betaCI_OLS(self):
        X = np.array([[1, 1, 0],[1, 0, 0], [0, 0, 1]])
        beta = np.array([[1, 1, 0],[1, 0, 0], [0, 0, 1]])

        z_hat = X.dot(beta)
        N, P = np.shape(X)

        z_real = np.array([[1, 1, 0],[1, 0, 0], [0, 0, 1]])
        z_hat = np.array([[1, 1, 0],[1, 0, 0], [0, 0, 1]])

        sigma2 = (np.sum(np.power((z_real-z_hat), 2)))/N
        var_beta = np.diag(sigma2*np.linalg.inv((X.T.dot(X))))

        ci_minus = beta - 1.645*np.sqrt(var_beta/N)
        ci_plus = beta + 1.645*np.sqrt(var_beta/N)
        #ci_minus_exp = np.array([[1., 1., 0.],[1., 0., 0.],[0., 0., 1.]])
        #ci_plus_exp = np.array([[1., 1., 0.],[1., 0., 0.],[0., 0., 1.]])

        ci_minus_exp = 0.4444444
        ci_plus_exp = 0.4444444

        #Change to mean value which is eaysier to test
        self.assertAlmostEqual(np.mean(ci_minus), ci_minus_exp)
        self.assertAlmostEqual(np.mean(ci_plus), ci_plus_exp)

    def test_betaCI_Ridge(self):

        X = np.array([[1, 1, 0],[1, 0, 0], [0, 0, 1]])
        beta = np.array([[1, 1, 0],[1, 0, 0], [0, 0, 1]])
        l=10**-4

        z_real = np.array([[1, 1, 0],[1, 0, 0], [0, 0, 1]])
        z_hat = np.array([[1, 1, 0],[1, 0, 0], [0, 0, 1]])
        # Calculate variance squared in the error
        z_hat = X.dot(beta)
        N, P = np.shape(X)
        sigma_2 = (np.sum(np.power((z_real-z_hat), 2)))/N

        # Calculate the variance squared of the beta coefficients
        XTX= X.T.dot(X)
        R, R = np.shape(XTX)
        var_beta = np.diag(sigma_2*np.linalg.inv((XTX + l*np.identity(R))))

        # The square root of var_beta is the standard error. Use it to calculate confidence intervals
        ci_minus = beta - 1.645*np.sqrt(var_beta/N)
        ci_plus = beta + 1.645*np.sqrt(var_beta/N)


        ci_minus_exp = -0.4380003
        ci_plus_exp = 1.3268892

        #Change to mean value which is eaysier to test
        self.assertAlmostEqual(np.mean(ci_minus), ci_minus_exp)
        self.assertAlmostEqual(np.mean(ci_plus), ci_plus_exp)
    #Ordinary Least Squared function


    def test_ols(self):
        X = np.load('data.npy')
        x = X[:, 0]
        y = X[:, 1]
        z = FrankeFunction(x, y)
        degree = 3
        xyb_ = np.c_[x, y]
        poly = PolynomialFeatures(degree)
        xyb = poly.fit_transform(xyb_)
        beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)
        self.assertAlmostEqual(np.mean(beta), 0.0202193)

    def test_RidgeRegression(self):
        X = np.load('data.npy')
        x = X[:, 0]
        y = X[:, 1]
        z = FrankeFunction(x, y)
        degree = 3
        l=0.0001
        M_ = np.c_[x, y]
        poly = PolynomialFeatures(degree)
        M = poly.fit_transform(M_)

        # Calculate beta
        A = np.arange(1, degree + 2)
        rows = np.sum(A)
        beta = (np.linalg.inv(M.T.dot(M) + l * np.identity(rows))).dot(M.T).dot(z)
        self.assertAlmostEqual(np.mean(beta), 0.0201915)

    def test_Lasso(self):
        X = np.load('data.npy')
        x = X[:, 0]
        y = X[:, 1]
        z = FrankeFunction(x, y)

        degree=5
        a=1e-06
        X = np.c_[x, y]
        poly = PolynomialFeatures(degree=degree)
        X_ = poly.fit_transform(X)

        clf = linear_model.Lasso(alpha=a, max_iter=5000, fit_intercept=False)
        clf.fit(X_, z)
        beta = clf.coef_
        self.assertAlmostEqual(np.mean(beta), 0.00549964)
