from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import util
import numpy as np

# Parameters used to define the neural network.
HIDDEN_LAYER_SIZES=(10,10,10)
# ACTIVATION='logistic','tanh','relu'
ACTIVATION='logistic'
MAX_ITER=10000
VERBOSE=True
REGULARIZATION=1e-04
SOLVER='adam'
# SOLVER='lbfgs','sgd','adam'
TOL=1e-06

def main():
    # Path of the train set, test set and save path.
    # train_path='cal_train.csv'
    # test_path='cal_test.csv'
    # save_path='cal_pred.txt'
    train_path='Data/one_month_1.csv'
    test_path='Data/one_day_2.csv'
    save_path='Data/pred.txt'

    # Load the data.
    x_train, y_train = util.load_dataset(train_path)
    x_test, y_test = util.load_dataset(test_path)

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

    # Save the result
    np.savetxt(save_path,prediction)

    # Performance measurement
    score=np.sum([1 for i in range(len(y_test)) if prediction[i]==y_test[i]])
    # score=np.sum([1 for i in range(len(y_test)) if np.isclose(prediction[i],y_test[i])])
    print("Correct prediction rate is: "+str(score)+"/"+str(len(y_test))+"="+str(score/len(y_test)))
    print(score)
    print(np.sum(y_test))
    print(np.sum(prediction))
    print(np.sum([1 for i in range(len(y_test)) if (prediction[i]==1 and y_test[i]==1)]))
    return score/len(y_test)

if __name__ == '__main__':
    main()
