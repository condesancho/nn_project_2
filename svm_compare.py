from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import metrics
from preprocessing import preprocessing, reduce_set_size
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time

xtrain, ytrain, xtest, ytest = preprocessing("../samples")

# Reduce the training and testing data to 6000 and 1000 respectively
xtrain, ytrain = reduce_set_size(xtrain, ytrain)
xtest, ytest = reduce_set_size(xtest, ytest)

gammavalues = [0.001, 0.01, 0.1, 1, 10]

trainingError = []
testingError = []
for gamma in gammavalues:
    clf = SVC(kernel="rbf", gamma=gamma)
    clf = clf.fit(xtrain, ytrain.values.ravel())
    pred = clf.predict(xtrain)
    trainingError.append(1 - metrics.accuracy_score(ytrain, pred))
    pred = clf.predict(xtest)
    testingError.append(1 - metrics.accuracy_score(ytest, pred))

plt.plot(trainingError, c="blue")
plt.plot(testingError, c="red")
# plt.ylim(0, 0.5)
plt.xticks(range(len(gammavalues)), gammavalues)
plt.legend(["Training Error", "Testing Error"])
plt.xlabel("Gamma")
plt.ylabel("Error")
plt.show()

# accuracies = []
# for gamma in gammavalues:
#     clf = svm.SVC(kernel="rbf", gamma=gamma)
#     scores = cross_val_score(clf, xtrain, ytrain, cv=10)
#     accuracies.append(scores.mean())

# print("Best gamma: ", gammavalues[np.argmax(accuracies)])
