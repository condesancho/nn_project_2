from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import metrics
from preprocessing import preprocessing
import time

xtrain, ytrain, xtest, ytest = preprocessing("../samples")

clf = LinearSVC(max_iter=2000)

start_time = time.time()
clf.fit(xtrain, ytrain.values.ravel())
print(
    "The time that took the model to train is %s seconds." % (time.time() - start_time)
)

start_time = time.time()
ypred = clf.predict(xtest)
print("The time passed for the prediction is %s seconds." % (time.time() - start_time))


ytrain_pred = clf.predict(xtrain)
train_accur = metrics.accuracy_score(ytrain, ytrain_pred)
print(f"The accuracy of the model for the train set is: {100*train_accur}%")

test_accur = metrics.accuracy_score(ytest, ypred)
print(f"The accuracy of the model for the test set is: {100*test_accur}%")
