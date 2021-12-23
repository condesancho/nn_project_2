from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from mnist import MNIST

import numpy as np

import time

# Import data from the folder
mndata = MNIST("../samples")

xtrain, ytrain = mndata.load_training()
xtest, ytest = mndata.load_testing()

# Change labels of data to 0 if the number is even or to 1 if it is odd
ytrain = np.array(ytrain) % 2
ytest = np.array(ytest) % 2

# # Check the first 10 images
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(xtrain, ytrain)

# pred = model.predict(xtest)
# print('The first 10 predicted values are:', pred[:10].tolist())
# print('The first 10 actual values are:', ytest[:10])

for k in [1, 3, 5, 7, 9]:
    # Start timer
    start_time = time.time()
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)

    pred = model.predict(xtest)

    score = accuracy_score(ytest, pred)

    print("For k =", k, "we have accuracy:", 100 * score, "%")
    print("Time passed: %s seconds." % (time.time() - start_time))
