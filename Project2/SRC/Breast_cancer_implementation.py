from NN import *
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numba as nb
import timeit

# ensure the same random numbers appear every time
np.random.seed(100)

# load breast cancer data
data = load_breast_cancer()
X = data['data']
y = data['target']

#@nb.jit(nopython=True)
def run_NN(X, y, active = "ReLU"):

    sns.set()
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #set up learning rate eta and regularization lambda for grid search
    lr = np.logspace(-5, 1, 7)
    lmbd = np.logspace(-5, 1, 7)

    train_accuracy = np.zeros((len(lr), len(lmbd)))
    test_accuracy = np.zeros((len(lr), len(lmbd)))

    # grid search
    for i, eta in enumerate(lr):
        for j, lm in enumerate(lmbd):
            # Instantiate the model
            model = Method()

            if active == "ReLU":
                # Add layers ReLU
                model.add_to_list(Layer(30, 64, lmbd=lm))
                model.add_to_list(Activ_ReLU())
                model.add_to_list(Layer(64, 64, lmbd=lm))
                model.add_to_list(Activ_ReLU())
                model.add_to_list(Layer(64, 2))
                model.add_to_list(Activ_Softmax())

            elif active == "Sigmoid":
                # Add layers Sigmoid
                model.add_to_list(Layer(30, 64, lmbd=lm))
                model.add_to_list(Activ_Sigmoid())
                model.add_to_list(Layer(64, 64, lmbd=lm))
                model.add_to_list(Activ_Sigmoid())
                model.add_to_list(Layer(64, 2))
                model.add_to_list(Activ_Softmax())

            elif active == "Leaky_ReLU":
                # Add layers Leaky_ReLU
                model.add_to_list(Layer(30, 64, lmbd=lm))
                model.add_to_list(Activ_Leaky_ReLU())
                model.add_to_list(Layer(64, 64, lmbd=lm))
                model.add_to_list(Activ_Leaky_ReLU())
                model.add_to_list(Layer(64, 2))
                model.add_to_list(Activ_Softmax())

            elif active == "Linear":
                # Add layers Linear
                model.add_to_list(Layer(30, 64, lmbd=lm))
                model.add_to_list(Activ_Linear())
                model.add_to_list(Layer(64, 64, lmbd=lm))
                model.add_to_list(Activ_Linear())
                model.add_to_list(Layer(64, 2))
                model.add_to_list(Activ_Softmax())

            elif active == "Tanh":
                # Add layers tanh
                model.add_to_list(Layer(30, 64, lmbd=lm))
                model.add_to_list(Activ_tanh())
                model.add_to_list(Layer(64, 64, lmbd=lm))
                model.add_to_list(Activ_tanh())
                model.add_to_list(Layer(64, 2))
                model.add_to_list(Activ_Softmax())

            elif active == "sigmoid_tanh":
                # Add layers sigmoid and tanh
                model.add_to_list(Layer(30, 64, lmbd=lm))
                model.add_to_list(Activ_Sigmoid())
                model.add_to_list(Layer(64, 64, lmbd=lm))
                model.add_to_list(Activ_tanh())
                model.add_to_list(Layer(64, 2))
                model.add_to_list(Activ_Softmax())

            elif active == "sigmoid_ReLU":
                # Add layers sigmoid and ReLU
                model.add_to_list(Layer(30, 64, lmbd=lm))
                model.add_to_list(Activ_Sigmoid())
                model.add_to_list(Layer(64, 64, lmbd=lm))
                model.add_to_list(Activ_ReLU())
                model.add_to_list(Layer(64, 2))
                model.add_to_list(Activ_Softmax())


            # Set loss, optimizer and accuracy objects
            model.set_param(
                loss=Loss_CC(),
                optimiz=Optim_SGD(lr=eta, decay=5e-5, momentum=0.9),
                accuracy=Accuracy_Classification()
            )

            # Finalize the model
            model.finall()

            # Train the model
            train_acc, test_acc = model.train(X_train, y_train, validation_data=(X_test, y_test), n_epoc=200,print_epoch = False)

            train_accuracy[i][j] = train_acc
            test_accuracy[i][j] = test_acc


    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy of our own NN with activ func " + active )
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('../Results/eta_lmd_train_acc_' + active + ".png")

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy of our own NN with activ func " + active)
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('../Results/eta_lmd_test_acc_' + active + ".png")

    return train_accuracy, test_accuracy


start = timeit.default_timer()
run_NN(X, y, active = "Tanh")
stop = timeit.default_timer()
print('Time: ', stop - start)




"""
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy of our own NN")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig('../Results/eta_lmd_train_acc.png')

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy of our own NN")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig('../Results/eta_lmd_test_acc.png')
"""
