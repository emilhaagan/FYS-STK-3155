import numpy as np
import matplotlib.pyplot as plt
from frankefunction import FrankeFunction
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

""" First we define our X from SCD decomposition """
def singular_value_decom(x):
    P, D, Q = np.linalg.svd(x, full_matrices=False)
    X = P @ diag(D) @ Q
    return X


def OLS(x, y):
    X = singular_value_decom(x, n)
    # Numpy implimentation:
    # fit = np.linalg.lstsq(X, y, rcond = None)[0]
    # y_til = dp.dot(fit, X.T)

    # Sklearn implimentation:
    model = Pipeline([('poly', PolynomialFeatures(degree=5)), ('linear'), LinearReression(fit_intercept=False)])


    """
    X_sequ = np.linspace(X.min(), X.max(), n).reshape(-1, 1)
    reg = skl.linear_model.LinearRegression()
    reg.fit(X, y)
    for i in range(np.linspace(1, 5, 5)):
        koef = np.polyfit(X.values.flatten(), y.values.flatten(), i)
    """
