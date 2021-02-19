import numpy as np
import random


def predict(X, w):
    return 1.0 if (1 / (1 + np.exp(-w @ X)) >= 0.5) else 0


def fit_stoch(X, y, epochs=1000, epsilon=1.0e-5):

    """
    Stochastic gradient descent
    :param X: Our X Matrix
    :param y: Our y Array
    :param epochs: Number of iterations.
    :param epsilon: Tolerance.
    :return: w - Weight vector, e - epoch
    """

    w = np.zeros(X.shape[1]).reshape((-1, 1))
    random.seed(0)
    index_list = list(range(len(X)))

    for epoch in range(epochs):

        alpha = 1000 / (1000 + epoch)
        random.shuffle(index_list)

        w0 = w

        for i in index_list:
            loss = y[i] - predict(X[i], w.T)
            gradient = loss * np.array([X[i]]).T
            w = w + alpha * gradient

        # Break Condition : The normal of the difference between two consecutive weight vectors.
        if np.linalg.norm(w - w0) / np.linalg.norm(w) < epsilon:
            break

    return w, epoch


def leave_one_out_cross_val(X, y, fitting_function, length=30):

    amount = 0

    for i in range(0, length):

        correct = False

        train_x = np.delete(X, i, axis=0)
        train_y = np.delete(y, i, axis=0)

        w, e = fitting_function(train_x, train_y)

        y_hat = predict(X[i], w.T)

        if y_hat == y[i]:
            correct = True
            amount += 1

        print("Fold", i + 1, "on 30:", "epochs:", e, "weights:", [value[0] for value in w],
              "Correct." if correct else "Incorrect.")

    return amount / length


def main():
    return


if __name__ == "__main__":
    main()