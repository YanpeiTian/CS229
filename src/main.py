import numpy as np
import util
import logistic
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    x_train = x_train[:, 1:]
    x_valid = x_valid[:, 1:]

    # normalize the data: (skip binary features)
    x_train[:, :-1] = (x_train[:, :-1] - np.mean(x_train[:, :-1], axis=0)) / np.std(x_train[:, :-1], axis=0)
    x_valid[:, :-1] = (x_valid[:, :-1] - np.mean(x_valid[:, :-1], axis=0)) / np.std(x_valid[:, :-1], axis=0)

    # add intercept for logistic regression:
    x_train = util.add_intercept(x_train)
    x_valid = util.add_intercept(x_valid)


    clf = logistic.LogisticRegression(step_size=1, max_iter=100000000)
    clf.fit(x_train, y_train)


    y_pred_prob = clf.predict(x_valid)
    y_pred = y_pred_prob.round()

    print(classification_report(y_valid, y_pred))
    print(confusion_matrix(y_valid, y_pred))
    print(np.sum(y_valid))

    np.savetxt(save_path, y_pred)


if __name__ == '__main__':
    main("../Example Data/one_day_2018-03-01_2018-03-02_merged.csv", "../Example Data/one_day_2018-06-01_2018-06-02_merged.csv", "prediction_output.txt")
    # main("../Example Data/one_month_2018-04-01_2018-05-01_merged.csv", "../Example Data/one_day_2018-03-01_2018-03-02_merged.csv", "prediction_output.txt")