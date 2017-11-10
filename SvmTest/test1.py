from sklearn import svm

X = [[0, 0], [1, 1], [4, 4], [6, 6], [10, 10]]
Y = [0, 0, 0, 1, 1]
clf = svm.SVC()
clf.fit(X, Y)
print(clf.predict([[5.5, 5]]))
# print(clf.support_vectors_)
# print(clf.support_)
# print(clf.n_support_)