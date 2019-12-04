from sklearn import svm
from sklearn.preprocessing import StandardScaler
import util
import numpy as np

KERNEL='rbf'

def main():
    # Path of the train set, test set and save path.
    train_path='Data/one_month_2018-04-01_2018-05-01_merged.csv'
    # train_path='Data/one_day_2018-03-01_2018-03-02_merged.csv'
    test_path='Data/one_day_2018-06-01_2018-06-02_merged.csv'
    save_path='Data/pred.txt'

    # Load the data.
    x_train, y_train = util.load_dataset(train_path)
    x_test, y_test = util.load_dataset(test_path)

    # scaler_x=StandardScaler()
    # scaler_x.fit(x_train)
    # x_train=scaler_x.transform(x_train)
    # x_test=scaler_x.transform(x_test)

    # Define the SVM
    clf=svm.SVC(kernel=KERNEL)

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
