import copy
import numpy as np
import sys
sys.path.append("/home/ubuntu/scikit-learn")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler

class Preprocess():

    """ Accepts a list of lists and outputs a matrix """
    @classmethod
    def to_matrix(self, data):
        arrays = [np.array(d) for d in data]
        return np.matrix(arrays)

    @classmethod
    def standard_deviation(self, matrix):
        return np.std(matrix, axis=0)

    """ accepts a numpy matrix and returns another numpy matrix with 
    all of the columns removed that have standard deviations of zero """
    @classmethod
    def remove_constants(self, matrix):
        std = np.std(matrix, axis=0)
        std = std.tolist()[0]
        nonzero_std_indices = [i for i,d in enumerate(std) if d != 0]
        return matrix[:,(nonzero_std_indices)]

    @classmethod
    def polynomial(self, matrix, polynomial):
        out = copy.copy(matrix)
        for i in range(polynomial+1)[2:]:
            print "...raising to power of ", i
            raised = np.power(matrix, i)
            out = np.concatenate((out, raised), axis=1)
        return out

    @classmethod
    def scale(self, matrix):
        scaler = StandardScaler()
        scaler.fit(matrix)
        matrix = scaler.transform(matrix)
        return matrix

    @classmethod
    def pca(self, matrix, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(matrix)
        return pca.transform(matrix)

    @classmethod
    def rbf_kernel(self, matrix, n_components):
        rbf = RBFSampler(n_components = n_components)
        print rbf
        matrix_features = rbf.fit_transform(matrix)
        return matrix_features 
        
    @classmethod
    def norm(self, matrix):
        min_max_scaler = MinMaxScaler()
        matrix = min_max_scaler.fit_transform(matrix)
        return matrix
