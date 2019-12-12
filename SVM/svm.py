from sklearn import svm
from sklearn.preprocessing import StandardScaler
import util
import numpy as np
from sklearn.metrics import classification_report

KERNEL='rbf'

def main(train_path,test_path):

    x_train, y_train = util.load_dataset(train_path[0])

    if len(train_path) == 2:
        x_train2, y_train2 = util.load_dataset(train_path[1])
        x_train = np.concatenate((x_train, x_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)

    # Load the data.
    x_test, y_test = util.load_dataset(test_path)

    # # delete the bert entries
    # x_train=x_train[:,-13:]
    # x_test=x_test[:,-13:]


    scaler_x=StandardScaler()
    scaler_x.fit(x_train)
    x_train=scaler_x.transform(x_train)
    x_test=scaler_x.transform(x_test)

    # Define the SVM
    clf=svm.SVC(kernel=KERNEL,verbose=True)

    # Training.
    clf.fit(x_train,y_train)
    # Predicting
    prediction=clf.predict(x_test)

    print(classification_report(y_test, prediction))

    prediction=clf.predict(x_train)

    print(classification_report(y_train, prediction))

if __name__ == '__main__':
    main(['../Final_Data/one_month_2018-04-01_2018-05-01_merged.csv','../Final_Data/three_month_2018-01-01_2018-04-01_merged.csv'],\
    '../Final_Data/one_day_2018-06-01_2018-06-02_merged.csv')
