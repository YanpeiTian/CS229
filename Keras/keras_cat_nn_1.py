
# first neural network with keras tutorial
from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import util
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.utils import model_to_dot, plot_model

from PIL import Image


from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad


def keras_cat_nn(x_train, y_train, x_test, y_test, h1, h2, h3, h4, epoch, feature):
    # Because we are using a neural network, standardize the data.
    scaler_x=StandardScaler()
    scaler_x.fit(x_train)

    x_train=scaler_x.transform(x_train)
    x_test=scaler_x.transform(x_test)

    num_orig = 13

    x_train_bert = x_train[:, 1:-num_orig]  # for all bert features
    x_test_bert = x_test[:, 1:-num_orig]

    x_train_original = x_train[:, -num_orig:]  # for all original 13 features
    x_test_original = x_test[:, -num_orig:]



    # Densed bert layer as a tensor
    input_bert = Input(shape=(x_train_bert.shape[1],))
    bert_1 = Dense(h1, activation='sigmoid')(input_bert)  # 300
    bert_2 = Dense(h2, activation='sigmoid')(bert_1)  # 100

    # Original features merged with densed bert layer
    input_orig = Input(shape=(x_train_original.shape[1],))
    merge_1 = concatenate([bert_2, input_orig])

    # Combined model
    combine_1 = Dense(h3, activation='sigmoid')(merge_1)
    combine_2 = Dense(h4, activation='sigmoid')(combine_1)
    combine_3 = Dense(1, activation='sigmoid')(combine_2)
    model = Model(inputs=[input_bert, input_orig], outputs=combine_3)

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    history=model.fit([x_train_bert, x_train_original], y_train, epochs=epoch, batch_size=16)
    # history=model.fit([x_train_bert, x_train_original], y_train, epochs=epoch, batch_size=16)
    # evaluate the keras model on test set

    predictions = model.predict([x_test_bert, x_test_original])
    predictions = classify(predictions)
    print(classification_report(y_test, predictions))

    plot_accuracy(history)
    plot_loss(history)


def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('hierarchical_nn_accuracy_bert.png')
    plt.close()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('hierarchical_nn_loss_bert.png')
    plt.close()


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

    h1 = 300
    h2 = 70
    h3 = 12
    h4 = 8
    epoch = 100

    # acc_train_all, acc_test_all = simple_nn_all(x_train, y_train, x_test, y_test, h1, h2, h3, h4, epoch)
    # acc_train_orig, acc_test_orig = simple_nn_orig(x_train, y_train, x_test, y_test, 10, 10, 10, epoch)
    acc_train, acc_test = keras_cat_nn(x_train, y_train, x_test, y_test, h1, h2, h3, h4, epoch, feature=False)

    # print('All:          Train Accuracy: %.2f%%; Test Accuracy: %.2f%%' % (acc_train_all * 100, acc_test_all * 100))
    # print('Original:     Train Accuracy: %.2f%%; Test Accuracy: %.2f%%' % (acc_train_orig * 100, acc_test_orig * 100))



if __name__ == '__main__':
    #one_month_2018-04-01_2018-05-01_merged.csv
    #keras_cat_nn('/Users/yanhaojiang/Desktop/DOC/cs229/CS229/src/keras_train/one_day_2018-03-01_2018-03-02_merged.csv','/Users/yanhaojiang/Desktop/DOC/cs229/CS229/src/keras_train/one_day_2018-06-01_2018-06-02_merged.csv')

    #keras_cat_nn(['../Example Data/one_month_2018-04-01_2018-05-01_merged.csv','../Example Data/three_month_2018-01-01_2018-04-01_merged.csv'],
    #             '../Example Data/one_day_2018-06-01_2018-06-02_merged.csv')

    main(['../Final_Data/one_month_2018-04-01_2018-05-01_merged.csv','../Final_Data/three_month_2018-01-01_2018-04-01_merged.csv'],
                 '../Final_Data/one_day_2018-06-01_2018-06-02_merged.csv')
    # main(['../Final_Data/one_day_2018-03-01_2018-03-02_merged.csv'],
    #              '../Final_Data/one_day_2018-06-01_2018-06-02_merged.csv')
