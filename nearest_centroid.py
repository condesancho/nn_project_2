from sklearn.neighbors import NearestCentroid
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

# Start timer
start_time = time.time()

model = NearestCentroid()
model.fit(xtrain, ytrain)

pred = model.predict(xtest)

# # Check the first 10 images
print("The first 10 predicted values are:", pred[:10].tolist())
print("The first 10 actual values are:", ytest[:10])

acc = accuracy_score(ytest, pred)

print("For the nearest centroid algorithm the accuracy is:", acc * 100, "%")
print("Time passed: %s seconds." % (time.time() - start_time))
