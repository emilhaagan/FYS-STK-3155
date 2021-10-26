from AnalysisFunctions import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.utils import resample
"""
    Analysis of bootstrap resampling technique
"""
# load data
X = np.load('data.npy')
x = X[:, 0]
y = X[:, 1]
z = FrankeFunction(x, y)

MSE, R2_b, bias, variance = bootstrap(x, y, z, method='OLS', p_degree=5)
text_file = open("../Results/ex2/Bootstrap_ols.txt", "w")
text_file.write('--- BOOTSTRAP for OLS --- \n')
text_file.write('MSE: {} \n'.format(MSE))
text_file.write('R2: {} \n'.format(R2_b))
text_file.write('Bias: {} \n'.format(bias))
text_file.write('Variance: {} \n'.format(variance))
text_file.close()
