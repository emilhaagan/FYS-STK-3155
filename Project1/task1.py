import numpy as np
import matplotlib.pyplot as plt
from frankefunction import FrankeFunction
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


z = FrankeFunction(x, y)

"""First we define our X from SCD decomposition"""
def singular_value_decom(x):
    P, D, Q = np.linalg.svd(x, full_matrices=False)
    X = P @ diag(D) @ Q
    return X

"""Defining the Standard least square regression """
def OLS(x, y):
    X = singular_value_decom(x)
    # Numpy implimentation:
    # fit = np.linalg.lstsq(X, y, rcond = None)[0]
    # y_til = dp.dot(fit, X.T)

    # Sklearn implimentation:
    model = Pipeline([('poly', PolynomialFeatures(degree=5)), ('linear'), LinearReression(fit_intercept=False)])
    model = model.fit(X[:,np.newaxis], y)

    return model


""" Combining the obve functions to obtain Mean Squared Error """
def MSE(x, y, data):
    y_model = OLS(x, y)
    X = singular_value_decom(x)
    n = np.size(y_model)
    MSE = np.sum((y - y_model)**2/n)
    y_tildenp = np.dot(y_model, X.T)
    #data = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
    #y_tilde = X@beta
    return MSE



    """
    X_sequ = np.linspace(X.min(), X.max(), n).reshape(-1, 1)
    reg = skl.linear_model.LinearRegression()
    reg.fit(X, y)
    for i in range(np.linspace(1, 5, 5)):
        koef = np.polyfit(X.values.flatten(), y.values.flatten(), i)
    """


"""Defining R2"""

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model)**2)/np.sum((y_data- np.mean(y_data))**2)
