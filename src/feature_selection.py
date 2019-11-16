import numpy as np
import util
import logistic
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def backward_selection(x_train, y_train, x_valid, y_valid):
    n = x_train.shape[0] # number of examples
    d = x_train.shape[1] # number of features
    # Wrapper feature selection: forward search
    remove_list = []
    F_list = np.arange(d).tolist()
    score_all = []
    index = np.arange(d).tolist()
    i = 0  # iteration times
    while len(F_list) > 0:
        i += 1
        remove_f = []
        score_f = []
        for k in range(d):
            if k in F_list:
                remove_f.append(k)
                f = F_list[:]
                f.remove(k)
                x_train_f = x_train[:, f]
                x_valid_f = x_valid[:, f]
                # add intercept for logistic regression:
                x_train_f = util.add_intercept(x_train_f)
                x_valid_f = util.add_intercept(x_valid_f)
                clf = logistic.LogisticRegression(step_size=1, max_iter=100000000, verbose=False)
                clf.fit(x_train_f, y_train)
                y_pred_f_prob = clf.predict(x_valid_f)
                y_pred_f = y_pred_f_prob.round()
                f_accuracy = np.mean(y_pred_f == y_valid)
                score_f.append(f_accuracy)
                print('Acc = %.6f' %(f_accuracy), f)
        best_score = np.amax(score_f)
        best_f_index = np.argwhere(score_f == best_score)
        best_f_index = best_f_index.flatten().tolist()

        remove_all = True
        if remove_all:
            for f_index in best_f_index:
                best_f = remove_f[f_index]
                remove_list.append(best_f)
                F_list.remove(best_f)
                score_all.append(best_score)
                index[len(remove_list)-1] = i
                print('')
                print('Acc_best = %.6f' % (best_score), F_list)
                print('')
        else:
            if len(best_f_index) == 1:
                f_index = best_f_index[0]
            else:  # more than one best choice
                f_index = random.choice(best_f_index)
            best_f = remove_f[f_index]
            remove_list.append(best_f)
            F_list.remove(best_f)
            score_all.append(best_score)
            print('Acc_best = %.6f' %(best_score), F_list)

    return remove_list, score_all, index


def forward_selection(x_train, y_train, x_valid, y_valid):
    n = x_train.shape[0] # number of examples
    d = x_train.shape[1] # number of features
    # Wrapper feature selection: forward search
    F_list = []
    score_all = []
    for i in range(d):
        add_f = []
        score_f = []
        for k in range(d):
            if k not in F_list:
                add_f.append(k)
                f = F_list + [k]
                x_train_f = x_train[:, f]
                x_valid_f = x_valid[:, f]
                # add intercept for logistic regression:
                x_train_f = util.add_intercept(x_train_f)
                x_valid_f = util.add_intercept(x_valid_f)
                clf = logistic.LogisticRegression(step_size=1, max_iter=100000000, verbose=False)
                clf.fit(x_train_f, y_train)
                y_pred_f_prob = clf.predict(x_valid_f)
                y_pred_f = y_pred_f_prob.round()
                f_accuracy = np.mean(y_pred_f == y_valid)
                score_f.append(f_accuracy)
                print(f, f_accuracy)
        print(score_f)
        best_score = np.max(score_f)
        best_f_index = np.argwhere(score_f == best_score)
        best_f_index = best_f_index.flatten().tolist()
        if len(best_f_index) == 1:
            best_f_index = best_f_index[0]
        else:  # more than one best choice
            best_f_index = random.choice(best_f_index)
        best_f = add_f[best_f_index]
        F_list.append(best_f)
        score_all.append(best_score)
        print('%.8f' %(best_score), F_list)

        remove_list.append(best_f)

    best_score_all = np.max(score_all)
    best_score_index = np.argmax(score_all)
    F_best = F_list[:int(best_score_index)+1]

    return F_best, F_list, score_all


def correlation(train_data, fig_path):
    for key in train_data.keys():
        if 'parsed_' in key:
            new_key = key.replace('parsed_', '')
            train_data = train_data.rename(columns={key: new_key})
    corr = train_data.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(train_data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(train_data.columns)
    ax.set_yticklabels(train_data.columns)
    plt.savefig(fig_path)

    corr_y = corr[['Label']][:-1]
    return corr_y


def main(train_path, valid_path, save_path, csv_path, fig_path):
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

    # normalize the data:
    x_train[:, :-1] = (x_train[:, :-1] - np.mean(x_train[:, :-1], axis=0)) / np.std(x_train[:, :-1], axis=0)
    x_valid[:, :-1] = (x_valid[:, :-1] - np.mean(x_valid[:, :-1], axis=0)) / np.std(x_valid[:, :-1], axis=0)

    # correlation
    train_data = pd.read_csv(train_path, index_col=0)
    corr_y = correlation(train_data, fig_path)
    corr_y.to_csv(csv_path.replace('X', 'correlation'), header=True, index=True)

    # forward select features:
    # F_best, F_list, score_all = forward_selection(x_train, y_train, x_valid, y_valid)
    # output = pd.DataFrame(list(zip(F_list, score_all)), columns=['feature_index', 'accuracy'])
    # output.to_csv(csv_path.replace('X', 'forward'))

    # backward select features:
    remove_list, score_all, index = backward_selection(x_train, y_train, x_valid, y_valid)
    remove_feature = train_data.columns[[remove_list]]
    remove_feature = remove_feature.to_list()
    output = pd.DataFrame(list(zip(index, remove_list, remove_feature, score_all)), columns=['iteration', 'remove_feature_index', 'remove_feature', 'accuracy'])
    output.to_csv(csv_path.replace('X', 'backward'), index=False)


if __name__ == '__main__':
    main("../Example Data/one_day_2018-03-01_2018-03-02_merged.csv", "../Example Data/one_day_2018-06-01_2018-06-02_merged.csv", "feature_selection/prediction_output.txt", "feature_selection/feature_selection_X.csv", "feature_selection/feature_correlation.png")
    # main("../Example Data/one_month_2018-04-01_2018-05-01_merged.csv",
    #     "../Example Data/one_day_2018-03-01_2018-03-02_merged.csv", "prediction_output.txt")