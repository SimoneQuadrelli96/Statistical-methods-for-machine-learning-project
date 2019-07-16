import numpy as np
import scipy
from scipy import sparse
import itertools



class Pegasos():
    def __init__(self):
        self.fitted_predictor = sparse.csr_matrix((95281,1), dtype=np.float32)

    def fit(self,X, Y,n_iter,lambda1,seed=42):
        #Y csr matrix (20000 ,1 )
        #X csr matrix (20000, 95281)
        np.random.seed(seed)
        max_index = X.shape[0]
        sum_predictors = sparse.csr_matrix((95281,1), dtype=np.float32)
        w = sparse.csr_matrix((95281,1), dtype=np.float32)
        for i in range(n_iter):
           eta = 1. / (lambda1*(i+1))
           example_index = np.random.randint(0, max_index)
           x = X.getrow(example_index)
           y = Y[example_index]
           score = x.dot(w)
           score = score.toarray()
           if 1-y*score > 0:  # hinge loss > 0
              w = w - (eta*(-y*(x.transpose()) + lambda1*w))
           else:
              w = w -(eta*(lambda1*w))
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
