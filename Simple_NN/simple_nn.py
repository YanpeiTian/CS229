from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import util
import numpy as np

# Parameters used to define the neural network.
HIDDEN_LAYER_SIZES=(500,500,)
# ACTIVATION='logistic','tanh','relu'
ACTIVATION='logistic'
MAX_ITER=10000
VERBOSE=True
REGULARIZATION=1e-04
SOLVER='adam'
# SOLVER='lbfgs','sgd','adam'
TOL=1e-06

def main(train_path,test_path):
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


    # Define the neural network.
    clf=MLPClassifier(hidden_layer_sizes=(HIDDEN_LAYER_SIZES),activation=ACTIVATION,\
    solver=SOLVER,max_iter=MAX_ITER,verbose=VERBOSE,alpha=REGULARIZATION,tol=TOL)
    # clf=MLPRegressor(hidden_layer_sizes=(HIDDEN_LAYER_SIZES),activation=ACTIVATION,\
    # max_iter=MAX_ITER,verbose=VERBOSE,alpha=REGULARIZATION)

    # Training.
    clf.fit(x_train,y_train)

    # Predicting
    prediction=clf.predict(x_test)

    print(classification_report(y_test, prediction))

    prediction=clf.predict(x_train)

    print(classification_report(y_train, prediction))

if __name__ == '__main__':
    # main(['../Final_Data/one_month_2018-04-01_2018-05-01_merged.csv','../Final_Data/three_month_2018-01-01_2018-04-01_merged.csv'],\
    # '../Final_Data/one_day_2018-06-01_2018-06-02_merged.csv')
    main(['../Final_Data/one_month_2018-04-01_2018-05-01_merged.csv'],\
    '../Final_Data/one_day_2018-06-01_2018-06-02_merged.csv')
