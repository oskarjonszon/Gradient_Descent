import numpy as np
import random


def normalize(X):
    return np.array([value / np.amax(X, axis=0) for value in X]), np.amax(X, axis=0)


def predict(X, w):
    y = np.matmul(X, w)
    y = np.vectorize(lambda val: 1 if val > 0 else 0)(y)
    return y


def fit_stoch(X, y, epochs=1000, max_misclassified=0):

    """
    Stochastic gradient descent
    :param X: Our X Matrix
    :param y: Our y Array
    :param epochs: Number of iterations.
    :param max_misclassified: Break condition.
    :return: w - Weight vector, e - epoch
    """

    index_list = list(range(len(X)))
    w = np.zeros(X.shape[1]).reshape((-1, 1))

    for epoch in range(epochs):

        alpha = 1000 / (1000 + epoch)
        random.shuffle(index_list)
        misclassified = 0

        for i in index_list:

            x = X[i]
            h_w = predict(x, w)

            # Mistake on positive
            if y[i] == 0.0 and h_w == 1.0:

                misclassified += 1

                for i in range(len(x)):
                    w[i][0] = w[i][0] + x[i] * -alpha

            # Mistake on negative
            if y[i] == 1.0 and h_w == 0.0:

                misclassified += 1

                for i in range(len(x)):
                    w[i][0] = w[i][0] + x[i] * alpha

        # Break condition
        if misclassified == max_misclassified:
            break

    return w, epoch


def leave_one_out_cross_val(X, y, fitting_function, length=30):

    amount = 0

    for i in range(0, length):

        correct = False

        train_x = np.delete(X, i, axis=0)
        train_y = np.delete(y, i, axis=0)

        w, e = fitting_function(train_x, train_y)

        y_hat = predict(X[i], w)

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