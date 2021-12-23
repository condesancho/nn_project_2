from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import metrics
from preprocessing import preprocessing, reduce_set_size
import time

xtrain, ytrain, xtest, ytest = preprocessing("./samples")

# Reduce the training and testing data to 6000 and 1000 respectively
xtrain, ytrain = reduce_set_size(xtrain, ytrain)
xtest, ytest = reduce_set_size(xtest, ytest)

# Create the model
clf = SVC()

# Fit and time the model
start_time = time.time()
clf.fit(xtrain, ytrain.values.ravel())
print(
    "The time that took the model to train is %s seconds." % (time.time() - start_time)
)

# Make prediction on the test set
start_time = time.time()
ypred = clf.predict(xtest)
print("The time passed for the prediction is %s seconds." % (time.time() - start_time))

# Find accuracies
ytrain_pred = clf.predict(xtrain)
train_accur = metrics.accuracy_score(ytrain, ytrain_pred)
print(f"The accuracy of the model for the train set is: {100*train_accur}%")

test_accur = metrics.accuracy_score(ytest, ypred)
print(f"The accuracy of the model for the test set is: {100*test_accur}%")
