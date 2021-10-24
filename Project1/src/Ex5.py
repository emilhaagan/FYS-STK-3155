from AnalysisFunctions import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import resample

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
text_file = open("../Results/ex5/Bootstrap_lasso.txt", "w")
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
plt.savefig('../Results/ex5/MSE_R2_alpha.png')

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
plt.savefig('../Results/ex5/alpha_noise_MSE.png')

MSE_l, R2_l, bias_l, variance_l = bootstrap(x, y, z, method='Lasso', p_degree=5)
text_file.write('--- BOOTSTRAP --- \n')
text_file.write('MSE: {} \n'.format(MSE_l))
text_file.write('R2: {} \n'.format(R2_l))
text_file.write('Bias: {} \n'.format(bias_l))
text_file.write('Variance: {} \n'.format(variance_l))

text_file.close()
