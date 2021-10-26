from AnalysisFunctions import *
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

# Load the terrain
terrain1 = imread('SRTM_data_Norway_2.tif')
# Show the terrain
plt.figure()
plt.title('Terrain area')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Choose a smaller part of the data set
terrain = terrain1[500:750, 0:250]
# Show the terrain
plt.figure()
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('../Results/ex6/terrain_original.png')

# Make zero matrix to later fit data
num_rows, num_cols = np.shape(terrain)
num_observations = num_rows * num_cols
X = np.zeros((num_observations, 3))

# make a matrix with all the values from the data on the form [x y z]
index = 0
#X = X-np.mean(X)
for i in range(0, num_rows):
    for j in range(0, num_cols):
        X[index, 0] = i  # x
        X[index, 1] = j  # y
        X[index, 2] = terrain[i, j]  # z
        index += 1

# OLS example
# extract x, y, z
xt = X[:,0, np.newaxis]
yt = X[:,1, np.newaxis]
zt = X[:,2, np.newaxis]

degree = [2, 4, 6, 8]
text_file = open("../Results/ex6/terrain_CI_ols.txt", "w")
for d in degree:
    beta = ols(xt, yt, zt, degree=d)

    M_ = np.c_[xt, yt]
    poly = PolynomialFeatures(d)
    M = poly.fit_transform(M_)
    z_predict = M.dot(beta)


    T = np.zeros([num_rows, num_cols])
    index = 0
    # create matrix for imshow
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            T[i, j] = (z_predict[index])
            index += 1
    plt.figure()
    plt.imshow(T, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('../Results/ex6/terrain_ols_d{}.png'.format(d))


    z_test = np.zeros(zt.shape[0])
    for i in range(zt.shape[0]):
        z_test[i] = zt[i][0]

    beta_test = np.zeros(beta.shape[0])
    for i in range(beta.shape[0]):
        beta_test[i] = zt[i][0]


    conf1, conf2 = betaCI_OLS(z_test, beta_test, M)
    #print(conf2.shape)
    for i in range(len(conf1)):
        text_file.write('Beta {0}: {1:5f} & [{2:5f}, {3:5f}] \n'.format(i, beta_test[i], conf1[i], conf2[i]))

text_file.close()

# Evaluate model with bootstrap algorithm
text_file = open("../Results/ex6/terrain_mse_ols.txt", "w")
MSE, R2, bias, variance = bootstrap(xt, yt, zt, p_degree=8, method='OLS', n_bootstrap=100)
text_file.write('MSE: {0:5f} & R2: {1:5f} & bias: {2:5f} & var: {3:5f}'.format(MSE, R2, bias, variance))
text_file.close()

########################################################################################
##Ridge Regression
########################################################################################
text_file = open("../Results/ex6/terrain_CI_ridge.txt", "w")
for d in degree:
    beta = RidgeRegression(xt, yt, zt, degree=d)

    M_ = np.c_[xt, yt]
    poly = PolynomialFeatures(d)
    M = poly.fit_transform(M_)
    z_predict = M.dot(beta)


    T = np.zeros([num_rows, num_cols])
    index = 0
    # create matrix for imshow
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            T[i, j] = (z_predict[index])
            index += 1
    plt.figure()
    plt.imshow(T, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('../Results/ex6/terrain_ridge_d{}.png'.format(d))


    z_test = np.zeros(zt.shape[0])
    for i in range(zt.shape[0]):
        z_test[i] = zt[i][0]

    beta_test = np.zeros(beta.shape[0])
    for i in range(beta.shape[0]):
        beta_test[i] = zt[i][0]


    conf1, conf2 = betaCI_OLS(z_test, beta_test, M)
    #print(conf2.shape)
    for i in range(len(conf1)):
        text_file.write('Beta {0}: {1:5f} & [{2:5f}, {3:5f}] \n'.format(i, beta_test[i], conf1[i], conf2[i]))

text_file.close()

# Evaluate model with bootstrap algorithm
text_file = open("../Results/ex6/terrain_mse_ridge.txt", "w")
MSE, R2, bias, variance = bootstrap(xt, yt, zt, p_degree=8, method='Ridge', n_bootstrap=100)
text_file.write('MSE: {0:5f} & R2: {1:5f} & bias: {2:5f} & var: {3:5f}'.format(MSE, R2, bias, variance))
text_file.close()

########################################################################################
##Lasso Regression
########################################################################################
text_file = open("../Results/ex6/terrain_CI_lasso.txt", "w")
for d in degree:
    beta = Lasso(xt, yt, zt, degree=d)

    M_ = np.c_[xt, yt]
    poly = PolynomialFeatures(d)
    M = poly.fit_transform(M_)
    z_predict = M.dot(beta)


    T = np.zeros([num_rows, num_cols])
    index = 0
    # create matrix for imshow
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            T[i, j] = (z_predict[index])
            index += 1
    plt.figure()
    plt.imshow(T, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('../Results/ex6/terrain_lasso_d{}.png'.format(d))


    z_test = np.zeros(zt.shape[0])
    for i in range(zt.shape[0]):
        z_test[i] = zt[i][0]

    beta_test = np.zeros(beta.shape[0])
    for i in range(beta.shape[0]):
        beta_test[i] = zt[i][0]


    conf1, conf2 = betaCI_OLS(z_test, beta_test, M)
    #print(conf2.shape)
    for i in range(len(conf1)):
        text_file.write('Beta {0}: {1:5f} & [{2:5f}, {3:5f}] \n'.format(i, beta_test[i], conf1[i], conf2[i]))
text_file.close

# Evaluate model with bootstrap algorithm
text_file = open("../Results/ex6/terrain_mse_lasso.txt", "w")
MSE_l, R2_l, bias_l, variance_l = bootstrap(xt, yt, zt, method='Lasso', p_degree=8)
text_file.write('MSE: {0:5f} & R2: {1:5f} & bias: {2:5f} & var: {3:5f}'.format(MSE_l, R2_l, bias_l, variance_l))
text_file.close()
