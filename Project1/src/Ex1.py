from AnalysisFunctions import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
    Analysis of Ordinary Least Square (OLS) on the Franke function
"""

# Load random data
X = np.load('data.npy')
x = X[:, 0]
y = X[:, 1]

#Noise
noises = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
#output the info about MSE and R2
text_file = open("../Results/ex1/MSE_R2_detail.txt", "w")
#explore Polynomial degree up to 5th
for d in range(1, 6):
    MSEs = []
    R2Scores = []
    for n in noises:
        z = FrankeFunction(x, y, noise=n)
        beta = []
        beta = ols(x, y, z, degree=d)

        # Generate new test data
        x_test = np.random.rand(200)
        y_test = np.random.rand(200)
        z_test = FrankeFunction(x_test, y_test, noise=n)

        #fit model
        M_ = np.c_[x_test, y_test]
        poly = PolynomialFeatures(d)
        M = poly.fit_transform(M_)

        z_predict= M.dot(beta)
        MSE = MeanSquaredError(z_test, z_predict)
        R2score = R2(z_test, z_predict)
        MSEs.append(MSE)
        R2Scores.append(R2score)

        text_file.write("Model of degree {} with noise: {}, MSE: {}, R2 score {} \n".format(d, n, MSE, R2score))

    #plot MSE and R2 score
    fig2, ax1 = plt.subplots()
    plt.setp(ax1, xticks=[0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4], xticklabels=['0', '0.001', '0.01', '0.1', '0.2', '0.3', '0.4'])
    ax1.plot(noises, MSEs, 'bo-')
    ax1.set_xlabel('Data with different degree of noise')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('MSE', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(noises, R2Scores, 'r*-')
    ax2.set_ylabel('R2 score', color='r')
    ax2.tick_params('y', colors='r')

    plt.title('Influence of noise on MSE and R2 score with polynomial degree {}'.format(d))
    fig2.tight_layout()
    plt.savefig('../Results/ex1/Degree {}.png'.format(d))

text_file.close()

########################################################################################################
#confidence interval
########################################################################################################
# Generate test data
x_test = np.random.rand(1000)
y_test = np.random.rand(1000)
z_test = FrankeFunction(x_test, y_test, noise=0.1)

# Calculate beta values and polynomial matrix
beta = ols(x, y, z, degree=5)
M_ = np.c_[x_test, y_test]
poly5 = PolynomialFeatures(5)
M = poly5.fit_transform(M_)

# Calculate beta confidence intervals
conf1, conf2 = betaCI_OLS(z_test, beta, M)
text_file = open("../Results/ex1/BetaCI_ols.txt", "w")
for i in range(len(conf1)):
    text_file.write('Beta {0}: {1:5f} & [{2:5f}, {3:5f}] \n'.format(i, beta[i], conf1[i], conf2[i]))
text_file.close()
