import numpy as np
import util
import logistic
import matplotlib.pyplot as plt
from decimal import Decimal

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = logistic.LogisticRegression()
    clf.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    y_pre = clf.predict(x_valid)
    np.savetxt(save_path, y_pre)
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path, valid_path, save_path)