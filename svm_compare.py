from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import metrics
from preprocessing import preprocessing, reduce_set_size
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time

xtrain, ytrain, xtest, ytest = preprocessing("./samples")

# Reduce the training and testing data to 6000 and 1000 respectively
xtrain, ytrain = reduce_set_size(xtrain, ytrain)
xtest, ytest = reduce_set_size(xtest, ytest)

gammavalues = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1]
exec_times = []
total_test_acc = []
total_train_acc = []
legend = []
for gamma in gammavalues:
    clf = SVC(kernel="rbf", gamma=gamma)
    # Start timer
    start_time = time.time()

    clf = clf.fit(xtrain, ytrain.values.ravel())

    exec_times.append(time.time() - start_time)

    pred = clf.predict(xtrain)
    total_train_acc.append(metrics.accuracy_score(ytrain, pred))

    pred = clf.predict(xtest)
    total_test_acc.append(metrics.accuracy_score(ytest, pred))

plot1 = plt.figure(1)
plt.plot(total_train_acc, c="blue")
plt.plot(total_test_acc, c="red")
plt.xticks(range(len(gammavalues)), gammavalues)
plt.legend(["Training Accuracy", "Testing Accuracy"])
plt.xlabel("gamma")
plt.ylabel("accuracy")
plt.title("Train and accuracy for different gamma values")

plot2 = plt.figure(2)
plt.plot(range(len(gammavalues)), exec_times, "-o")
plt.xticks(range(len(gammavalues)), gammavalues)
plt.xlabel("gamma values")
plt.ylabel("execution times")

plt.show()

accuracies = []
for gamma in gammavalues:
    clf = SVC(kernel="rbf", gamma=gamma)
    scores = cross_val_score(clf, xtrain, ytrain.values.ravel(), cv=10)
    accuracies.append(scores.mean())

print(
    f"Best gamma: {gammavalues[np.argmax(accuracies)]} with best accuracy: {100*max(accuracies)}%"
)
