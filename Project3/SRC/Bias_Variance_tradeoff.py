

from mlxtend.evaluate import bias_variance_decomp
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import linear_model

"""Collecting data with pandas"""
df = pd.read_csv("heart.csv")
df.info()
X = df.drop(["HeartDisease"], axis=1) #Everything but "HeartDisease"
y = df["HeartDisease"] #All data

df.head() #data structure


"""Deviding data into numerical and categorical to see what is what"""
numerical= df.drop(['HeartDisease'], axis=1).select_dtypes('number').columns

categorical = df.select_dtypes('object').columns

print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')


X_oh = pd.get_dummies(X) #Getting dummy matrix of X
X_oh

X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(X_oh, y, test_size=0.2)



#change to numpy values
X_train_oh = X_train_oh.values
X_test_oh = X_test_oh.values
y_train_oh = y_train_oh.values
y_test_oh = y_test_oh.values




""" SVM with different degrees as complexity """
degree = np.linspace(0, 10, 11, dtype=int)
mse_ = np.zeros(len(degree))
bias_ = np.zeros(len(degree))
var_ = np.zeros(len(degree))

for i in degree:
    model = make_pipeline(StandardScaler(), SVC(degree = i, kernel="poly"))
    mse_[i], bias_[i], var_[i] = bias_variance_decomp(model, X_train_oh, y_train_oh, X_test_oh, y_test_oh , num_rounds=200, random_seed=1)

plt.plot(degree, bias_**2 + var_, label="$Loss(x)$")
plt.plot(degree, bias_**2, label="$bias(x)^2$")
plt.plot(degree, var_, label="$variance(x)$")
plt.legend(loc=1)
plt.xlabel("Degree of polynomial")
plt.ylabel("Score")
plt.title("Bias-variance tradeoff SVM")
plt.show()


""" Linear Regression with Ridge and alpha as complexity """

alpha = np.linspace(0, 10e-10, 10)
mse_ = np.zeros(len(alpha))
bias_ = np.zeros(len(alpha))
var_ = np.zeros(len(alpha))
k = 0

for i in alpha:
    model  = linear_model.Ridge(alpha=i)
    mse_[k], bias_[k], var_[k] = bias_variance_decomp(model, X_train_oh, y_train_oh, X_test_oh, y_test_oh, num_rounds=200, random_seed=1)
    k += 1



plt.plot(alpha, bias_**2 + var_, label="$Loss(x)$")
plt.plot(alpha, bias_**2, label="$bias(x)^2$")
plt.plot(alpha, var_, label="$variance(x)$")
plt.legend(loc=7)
plt.xlabel("Alpha")
plt.ylabel("Score")
plt.title("Bias-variance tradeoff Ridge")
plt.show()


""" Linear Regression with Lasso and alpha as complexity """

alpha = np.linspace(0, 10e-10, 10)
mse_ = np.zeros(len(alpha))
bias_ = np.zeros(len(alpha))
var_ = np.zeros(len(alpha))
k = 0

for i in alpha:
    model  = linear_model.Lasso(alpha=i)
    mse_[k], bias_[k], var_[k] = bias_variance_decomp(model, X_train_oh, y_train_oh, X_test_oh, y_test_oh, num_rounds=200, random_seed=1)
    k += 1

plt.plot(alpha, bias_**2 + var_, label="$Loss(x)$")
plt.plot(alpha, bias_**2, label="$bias(x)^2$")
plt.plot(alpha, var_, label="$variance(x)$")
plt.legend(loc=7)
plt.xlabel("Alpha")
plt.ylabel("Score")
plt.title("Bias-variance tradeoff Lasso")
plt.show()





""" GradientBoosting with ccp_alpha as complexity"""

n = np.linspace(0, 0.1, 11)
mse_ = np.zeros(len(n))
bias_ = np.zeros(len(n))
var_ = np.zeros(len(n))
k=0



for i in n:
    model = make_pipeline(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0, ccp_alpha = i))
    mse_[k], bias_[k], var_[k] = bias_variance_decomp(model, X_train_oh, y_train_oh, X_test_oh, y_test_oh, num_rounds=200, random_seed=1)
    k += 1


plt.plot(n, bias_**2 + var_, label="$Loss(x)$")
plt.plot(n, bias_**2, label="$bias(x)^2$")
plt.plot(n, var_, label="$variance(x)$")
plt.legend(loc=1)
plt.xlabel("$a_{eff}$")
plt.ylabel("Score")
plt.title("Bias-variance tradeoff Gradient Boosting Classifier")
plt.show()


plt.plot(n, mse_, label="$MSE(x)$")
plt.plot(n, bias_**2, label="$bias(x)^2$")
plt.plot(n, var_, label="$variance(x)$")
plt.legend(loc=1)
plt.xlabel("$a_{eff}$")
plt.ylabel("Score")
plt.title("Bias-variance tradeoff Gradient Boosting Classifier")
plt.show()














