import pandas as pd
import numpy as np
from mnist import MNIST
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocessing(path, scale=True, pca=False, num_components=0.95):
    mndata = MNIST(path)

    xtrain, ytrain = mndata.load_training()
    xtest, ytest = mndata.load_testing()

    # Turn list to array for faster tranformation to DataFrame
    xtrain = np.array(xtrain)
    xtest = np.array(xtest)

    # If we want the samples to be scaled
    if scale:
        xtrain = StandardScaler().fit_transform(xtrain)
        xtest = StandardScaler().fit_transform(xtest)

    # If you want to use PCA
    if pca:
        clf = PCA(n_components=num_components)
        xtrain = clf.fit_transform(xtrain)

        xtest = clf.transform(xtest)

    xtrain = pd.DataFrame(xtrain)
    xtest = pd.DataFrame(xtest)

    ytrain = pd.DataFrame({"label": ytrain})
    ytest = pd.DataFrame({"label": ytest})

    # Change labels of data to 0 if the number is even or to 1 if it is odd
    ytrain = ytrain % 2
    ytest = ytest % 2

    return xtrain, ytrain, xtest, ytest


# Reduce the size of the dataset according to the reduce_factor
# E.g. if originally we had 60000 samples and the reduce_factor is 10 a new set of 6000 will be returned
def reduce_set_size(samples, labels, reduce_factor=10):

    # Find the new size of the set to be returned
    reduced_size = labels.shape[0] // reduce_factor

    # Value that checks if the new dataset is balanced
    even_labels_precentage = 1.0

    while even_labels_precentage < 0.4 or even_labels_precentage > 0.6:
        # Generate new indices
        indices = np.random.randint(0, labels.shape[0], reduced_size)
        new_labels = labels.loc[indices]
        # Sum the values with label 1 and divide by the total number of labels
        even_labels_precentage = new_labels.value_counts()[1] / reduced_size

    new_samples = samples.loc[indices, :]

    return new_samples, new_labels
