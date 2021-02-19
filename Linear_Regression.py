import numpy as np
import random


def normalize(x):
    return np.array([value / np.amax(x, axis=0) for value in x]), np.amax(x, axis=0)


def predict(X, w):
    return X @ w


def fit_batch(X, y, alpha, epochs=500, epsilon=1.0e-4):

    """
    Batch gradient descent
    :param X: Our X Matrix
    :param y: Our y Array
    :param alpha: Learning rate
    :param epochs: Number of iterations.
    :param epsilon: Tolerance.
    :return: w - Weight vector
    """

    w = np.zeros(X.shape[1]).reshape((-1, 1))

    alpha /= len(X)

    for epoch in range(epochs):
        w0 = w
        loss = y - predict(X, w)
        gradient = X.T @ loss
        w = w + alpha * gradient
        if np.linalg.norm(w - w0) / np.linalg.norm(w) < epsilon:
            break

    return w


def fit_stoch(X, y, alpha, epochs=500, epsilon=1.0e-3):

    """
    Stochastic gradient descent
    :param X: Our X Matrix
    :param y: Our y Array
    :param alpha: Learning rate
    :param epochs: Number of iterations.
    :param epsilon: Tolerance.
    :return: w - Weight vector
    """

    w = np.zeros(X.shape[1]).reshape((-1, 1))

    index_list = list(range(len(X)))

    for epoch in range(epochs):

        random.shuffle(index_list)
        w0 = w

        for i in index_list:
            loss = y[i] - predict(X[i], w)[0]
            gradient = loss * np.array([X[i]]).T
            w = w + alpha * gradient

        # The normal of the difference between two consecutive weight vectors.
        if np.linalg.norm(w - w0) / np.linalg.norm(w) < epsilon:
            break

    return w


def main():
    return


if __name__ == "__main__":
    main()