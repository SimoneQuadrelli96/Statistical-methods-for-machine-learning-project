import numpy as np
import scipy
from scipy import sparse
import itertools



class Perceptron():
    def __init__(self):
        self.fitted_predictor = sparse.csr_matrix((95281,1), dtype=np.float32)

    def fit(self,X, Y, n_iter, seed=42):
        #Y csr matrix (20000 ,1 )
        #X csr matrix (20000, 95281)
        np.random.seed(seed)
        max_index = X.shape[0]
        w = sparse.csr_matrix((95281,1), dtype=np.float32)
        sum_predictors = sparse.csr_matrix((95281,1), dtype=np.float32)
        for example_index in range(n_iter):
           example_index = np.random.randint(0, max_index)
           x = X.getrow(example_index)
           y = Y[example_index]
           score = y*((x.dot(w)).toarray())
           if score <= 0:
              w +=  y*x.transpose()
           sum_predictors += w
        self.fitted_predictor = sum_predictors/n_iter


        #x vector csr (1, 95281)
    def predict(self,x):
        if (x.dot(self.fitted_predictor)[0,0] >= 0):
            return 1
        else:
            return -1

        # X is a  csr matrix
    def predict_all(self,X):
        res = (X.dot(self.fitted_predictor)).toarray()
        prediction = np.where(res >= 0,1,-1)
        prediction = list(itertools.chain(*prediction))
        return np.array(prediction)
