import numpy as np
from sklearn.model_selection import train_test_split



def train_test_split(inputs, labels, train_size = 0.8):

    labels = to_categorical(labels)

    # split into train and test data
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                        test_size=test_size)

    return X_train, X_test, Y_train, Y_test
