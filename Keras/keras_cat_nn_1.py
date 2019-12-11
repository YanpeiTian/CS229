
# first neural network with keras tutorial
from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import util
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad

def simple_nn_all(x_train, y_train, x_test, y_test, h1, h2, h3, h4, epoch):
    input = Input(shape=(x_train.shape[1],))
    layer1 = Dense(h1, activation='relu')(input)  # 300
    layer2 = Dense(h2, activation='relu')(layer1)  # 100
    layer3 = Dense(h3, activation='sigmoid')(layer2)
    layer4 = Dense(h4, activation='sigmoid')(layer3)
    output = Dense(1, activation='sigmoid')(layer4)
    model = Model(inputs=input, outputs=output)

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(x_train, y_train, epochs=epoch, batch_size=16)

    # evaluate the keras model on test set
    predictions = model.predict(x_test)
    predictions = classify(predictions)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    _, train_accuracy = model.evaluate(x_train, y_train)
    _, test_accuracy = model.evaluate(x_test, y_test)
    print('Train Accuracy: %.2f%%; Test Accuracy: %.2f%%' % (train_accuracy * 100, test_accuracy * 100))

    return train_accuracy, test_accuracy

def simple_nn_orig(x_train, y_train, x_test, y_test, h1, h2, h3, epoch):
    # Because we are using a neural network, standardize the data.
    # scaler_x=StandardScaler()
    # scaler_x.fit(x_train)
    #
    # x_train=scaler_x.transform(x_train)
    # x_test=scaler_x.transform(x_test)

    x_train_original = x_train[:, -13:]  # for all original 13 features
    x_test_original = x_test[:, -13:]

    # Densed bert layer as a tensor
    input_orig = Input(shape=(x_train_original.shape[1],))
    orig_1 = Dense(h1, activation='relu')(input_orig)  # 300
    orig_2 = Dense(h2, activation='relu')(orig_1)  # 100
    orig_3 = Dense(h3, activation='sigmoid')(orig_2)
    output = Dense(1, activation='sigmoid')(orig_3)
    model = Model(inputs=input_orig, outputs=output)

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(x_train_original, y_train, epochs=epoch, batch_size=16)

    # evaluate the keras model on test set
    predictions = model.predict(x_test_original)
    predictions = classify(predictions)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    _, train_accuracy = model.evaluate(x_train_original, y_train)
    _, test_accuracy = model.evaluate(x_test_original, y_test)
    print('Train Accuracy: %.2f%%; Test Accuracy: %.2f%%' % (train_accuracy * 100, test_accuracy * 100))

    return train_accuracy, test_accuracy


def keras_cat_nn(x_train, y_train, x_test, y_test, h1, h2, h3, h4, epoch, feature):
    # Because we are using a neural network, standardize the data.
    # scaler_x=StandardScaler()
    # scaler_x.fit(x_train)
    #
    # x_train=scaler_x.transform(x_train)
    # x_test=scaler_x.transform(x_test)

    num_orig = 13
    print(x_train.shape, x_test.shape)
    if feature:
        x_train, x_test = feature_selection(x_train, x_test)
        num_orig -= 5
    print(x_train.shape, x_test.shape)
    x_train_bert = x_train[:, 1:-num_orig]  # for all bert features
    x_test_bert = x_test[:, 1:-num_orig]

    x_train_original = x_train[:, -num_orig:]  # for all original 13 features
    x_test_original = x_test[:, -num_orig:]

    print(x_train_bert.shape, x_test_bert.shape)
    print(x_train_original.shape, x_test_original.shape)

    # define the keras model
    # model = Sequential()
    # model.add(Dense(500, input_dim=x_train.shape[1], activation='tanh'))  # relu/ tanh
    # model.add(Dense(100, activation='tanh'))
    # model.add(Dense(1, activation='sigmoid'))

    # Densed bert layer as a tensor
    input_bert = Input(shape=(x_train_bert.shape[1],))
    bert_1 = Dense(h1, activation='relu')(input_bert)  # 300
    bert_2 = Dense(h2, activation='relu')(bert_1)  # 100

    # Original features merged with densed bert layer
    input_orig = Input(shape=(x_train_original.shape[1],))
    merge_1 = concatenate([bert_2, input_orig])

    # Combined model
    combine_1 = Dense(h3, activation='sigmoid')(merge_1)
    combine_2 = Dense(h4, activation='sigmoid')(combine_1)
    combine_3 = Dense(1, activation='sigmoid')(combine_2)
    model = Model(inputs=[input_bert, input_orig], outputs=combine_3)

    # 12 8 1， 65.84

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit([x_train_bert, x_train_original], y_train, epochs=epoch, batch_size=16)

    # evaluate the keras model on test set
    predictions = model.predict([x_test_bert, x_test_original])
    predictions = classify(predictions)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    _, train_accuracy = model.evaluate([x_train_bert, x_train_original], y_train)
    _, test_accuracy = model.evaluate([x_test_bert, x_test_original], y_test)
    print('Train Accuracy: %.2f%%; Test Accuracy: %.2f%%' % (train_accuracy * 100, test_accuracy * 100))

    return train_accuracy, test_accuracy


def classify(x):
    x_class = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] >= 0.5:
            x_class[i] = 1
        else:
            x_class[i] = 0
    return x_class


def feature_selection(x_train, x_test):
    delete_list = [-13, -8, -4, -3, -1]  # unigramCost, user view, block code, block code line, edit
    for column in delete_list:
        x_train = np.delete(x_train, column, 1)
        x_test = np.delete(x_test, column, 1)
    return x_train, x_test


def main(train_path, test_path):
    # load the dataset
    # Load the data.
    # Load headers
    x_train, y_train = util.load_dataset(train_path[0])

    if len(train_path) == 2:
        x_train2, y_train2 = util.load_dataset(train_path[1])
        x_train = np.concatenate((x_train, x_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)

    x_test, y_test = util.load_dataset(test_path)

    h1 = 600
    h2 = 200
    h3 = 12
    h4 = 8
    epoch = 150

    # acc_train_all, acc_test_all = simple_nn_all(x_train, y_train, x_test, y_test, h1, h2, h3, h4, epoch)
    # acc_train_orig, acc_test_orig = simple_nn_orig(x_train, y_train, x_test, y_test, 10, 10, 10, epoch)
    acc_train, acc_test = keras_cat_nn(x_train, y_train, x_test, y_test, h1, h2, h3, h4, epoch, feature=False)

    # print('All:          Train Accuracy: %.2f%%; Test Accuracy: %.2f%%' % (acc_train_all * 100, acc_test_all * 100))
    # print('Original:     Train Accuracy: %.2f%%; Test Accuracy: %.2f%%' % (acc_train_orig * 100, acc_test_orig * 100))
    print('Concatenated: Train Accuracy: %.2f%%; Test Accuracy: %.2f%%' % (acc_train * 100, acc_test * 100))

    # acc = np.zeros([2, 5, 2, 2])
    # hidden_layer1 = [300, 500]
    # hidden_layer2 = [25, 50, 75, 100, 150]
    # epochs = [100, 150]
    # for idx1, h1 in enumerate(hidden_layer1):
    #     for idx2, h2 in enumerate(hidden_layer2):
    #         for idx3, epoch in enumerate(epochs):
    #             acc_trains = np.zeros(5)
    #             acc_tests = np.zeros(5)
    #             for i in range(5):
    #               acc_train, acc_test = keras_cat_nn(x_train, y_train, x_test, y_test, h1, h2, epoch)
    #                acc_trains[i] = acc_train
    #                acc_tests[i] = acc_test
    #            acc[idx1, idx2, idx3, 0] = np.mean(acc_trains)
    #            acc[idx1, idx2, idx3, 1] = np.mean(acc_tests)

    #    print(acc)



if __name__ == '__main__':
    #one_month_2018-04-01_2018-05-01_merged.csv
    #keras_cat_nn('/Users/yanhaojiang/Desktop/DOC/cs229/CS229/src/keras_train/one_day_2018-03-01_2018-03-02_merged.csv','/Users/yanhaojiang/Desktop/DOC/cs229/CS229/src/keras_train/one_day_2018-06-01_2018-06-02_merged.csv')

    #keras_cat_nn(['../Example Data/one_month_2018-04-01_2018-05-01_merged.csv','../Example Data/three_month_2018-01-01_2018-04-01_merged.csv'],
    #             '../Example Data/one_day_2018-06-01_2018-06-02_merged.csv')

    main(['../Example Data/one_month_2018-04-01_2018-05-01_merged.csv'],
                 '../Example Data/one_day_2018-06-01_2018-06-02_merged.csv')   #  '../Example Data/one_day_2018-03-01_2018-03-02_merged.csv'
