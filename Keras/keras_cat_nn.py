
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



def keras_cat_nn0(train_path, test_path):
    # load the dataset
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:, 0:8]
    y = dataset[:, 8]

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)

    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))

    # make probability predictions with the model
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    return


def keras_original_nn(train_path, test_path):
    # load the dataset
    # Load the data.
    x_train, y_train = util.load_dataset(train_path)
    x_test, y_test = util.load_dataset(test_path)

    # Because we are using a neural network, standardize the data.
    # scaler_x=StandardScaler()
    # scaler_x.fit(x_train)
    #
    # x_train=scaler_x.transform(x_train)
    # x_test=scaler_x.transform(x_test)


    x_train_bert= x_train[:,-13:] # for all bert features
    x_test_bert=x_test[:,-13:]

    x_train_original= x_train[:,:-13] # for all original 13 features
    x_test_original=x_test[:,:-13]


    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 12 8 1， 65.84

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(x_train, y_train, epochs=150, batch_size=16)


    # evaluate the keras model on test set
    predictions = model.predict_classes(x_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


    _, accuracy = model.evaluate(x_test, y_test)
    print('tttAccuracy: %.2f' % (accuracy * 100))

    return


def keras_cat_nn(train_path, test_path):
    # load the dataset
    # Load the data.
    x_train, y_train = util.load_dataset(train_path[0])

    if len(train_path)==2:
        x_train2, y_train2 = util.load_dataset(train_path[1])
        x_train=np.concatenate((x_train,x_train2),axis=0)
        y_train=np.concatenate((y_train,y_train2),axis=0)

        
    x_test, y_test = util.load_dataset(test_path)

    # Because we are using a neural network, standardize the data.
    # scaler_x=StandardScaler()
    # scaler_x.fit(x_train)
    #
    # x_train=scaler_x.transform(x_train)
    # x_test=scaler_x.transform(x_test)


    x_train_bert= x_train[:,-13:] # for all bert features
    x_test_bert=x_test[:,-13:]

    x_train_original= x_train[:,:-13] # for all original 13 features
    x_test_original=x_test[:,:-13]


    # define the keras model
    model = Sequential()
    model.add(Dense(500, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 12 8 1， 65.84

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(x_train, y_train, epochs=150, batch_size=16)


    # evaluate the keras model on test set
    predictions = model.predict_classes(x_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


    _, accuracy = model.evaluate(x_test, y_test)
    print('tttAccuracy: %.2f' % (accuracy * 100))

    return

if __name__ == '__main__':
    #one_month_2018-04-01_2018-05-01_merged.csv
    #keras_cat_nn('/Users/yanhaojiang/Desktop/DOC/cs229/CS229/src/keras_train/one_day_2018-03-01_2018-03-02_merged.csv','/Users/yanhaojiang/Desktop/DOC/cs229/CS229/src/keras_train/one_day_2018-06-01_2018-06-02_merged.csv')

    keras_cat_nn(['one_month_2018-04-01_2018-05-01_merged.csv','three_month_2018-01-01_2018-04-01_merged.csv'],
                 'one_day_2018-06-01_2018-06-02_merged.csv')
