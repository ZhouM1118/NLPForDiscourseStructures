from sklearn import svm

X = [[0, 0], [1, 1], [4, 4], [6, 6], [10, 10]]
Y = [0, 0, 0, 1, 1]
clf = svm.SVC()
clf.fit(X, Y)
# print(clf.predict([[4.5, 5]]))
# print(clf.support_vectors_)
# print(clf.support_)
# print(clf.n_support_)

import numpy as np
import sklearn
from sklearn import cross_validation

x = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[9, 10]])
y = np.array([1, 2, 1, 2, 3])
def show_cross_val(method):
    if method == "lolo":
        labels = np.array(["summer", "winter", "summer", "winter", "spring"])
        cv = cross_validation.LeaveOneLabelOut(labels)
    elif method == 'lplo':
        labels = np.array(["summer", "winter", "summer", "winter", "spring"])
        cv = cross_validation.LeavePLabelOut(labels, p=2)
    elif method == 'loo':
        cv = cross_validation.LeaveOneOut(n=len(y))
    elif method == 'lpo':
        cv = cross_validation.LeavePOut(n=len(y), p=3)
    for train_index, test_index in cv:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print ("X_train: ", X_train)
        print ("y_train: ", y_train)
        print ("X_test: ", X_test)
        print ("y_test: ", y_test)

show_cross_val("lpo")