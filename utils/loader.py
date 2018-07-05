import pickle

import numpy as np


def load_cifar_10(dataset_path, number_of_batches=5):
    Xs = []
    ys = []
    for i in range(1, number_of_batches + 1):
        X, y = load_batch(f'{dataset_path}/data_batch_{i}')
        Xs.append(X)
        ys.append(y)

    print('Loaded trainset.')

    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)

    del Xs, ys

    X_test, y_test = load_batch(f'{dataset_path}/test_batch')
    print('Loaded testset.')

    return X_train, y_train, X_test, y_test


def load_batch(filename):
    batch_dict = unpickle_file(filename)
    X = batch_dict['data']
    y = batch_dict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(y)
    return X, y


def unpickle_file(filename):
    with open(filename, 'rb') as file_object:
        unpickled_dict = pickle.load(file_object, encoding='latin1')
    return unpickled_dict
