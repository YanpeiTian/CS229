
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


from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad

EPOCHS=300


def keras_cat_nn(train_path, test_path):
    # load the dataset
    # Load the data.
    x_train, y_train = util.load_dataset(train_path[0])

    if len(train_path)==2:
        x_train2, y_train2 = util.load_dataset(train_path[1])
        x_train=np.concatenate((x_train,x_train2),axis=0)
        y_train=np.concatenate((y_train,y_train2),axis=0)


    x_test, y_test = util.load_dataset(test_path)

    # delete bert features
    x_train=x_train[:,-13:]
    x_test=x_test[:,-13:]

    # Because we are using a neural network, standardize the data.
    scaler_x=StandardScaler()
    scaler_x.fit(x_train)

    x_train=scaler_x.transform(x_train)
    x_test=scaler_x.transform(x_test)


    # define the keras model
    model = Sequential()
    model.add(Dense(500, input_dim=x_train.shape[1], activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))


    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    history=model.fit(x_train, y_train, epochs=EPOCHS, batch_size=16)

    # evaluate the keras model on test set
    predictions = model.predict_classes(x_test)
    print(classification_report(y_test, predictions))

    plot_accuracy(history)
    plot_loss(history)

    return

def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('simple_nn_accuracy.png')
    plt.close()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('simple_nn_loss.png')
    plt.close()

if __name__ == '__main__':
    #one_month_2018-04-01_2018-05-01_merged.csv
    #keras_cat_nn('/Users/yanhaojiang/Desktop/DOC/cs229/CS229/src/keras_train/one_day_2018-03-01_2018-03-02_merged.csv','/Users/yanhaojiang/Desktop/DOC/cs229/CS229/src/keras_train/one_day_2018-06-01_2018-06-02_merged.csv')

    keras_cat_nn(['../Final_Data/one_month_2018-04-01_2018-05-01_merged.csv','../Final_Data/three_month_2018-01-01_2018-04-01_merged.csv'],
                 '../Final_Data/one_day_2018-06-01_2018-06-02_merged.csv')
    # keras_cat_nn(['../Final_Data/one_day_2018-03-01_2018-03-02_merged.csv'],
    #              '../Final_Data/one_day_2018-06-01_2018-06-02_merged.csv')
