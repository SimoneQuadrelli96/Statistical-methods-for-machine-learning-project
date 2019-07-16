import itertools
import random
import numpy as np

'''
takes the length of a vector or of a matrix and returns a tuple contatining the indices of
n_test_examples should be a multiple of 20000

'''
def train_test_split_index(length,n_test_examples):
     indices =  random.sample(range(length), length)
     return (indices[0:n_test_examples],indices[n_test_examples:])

'''
measures train or test error
'''
def error_performance(ground_truths,prediction):
    correct_preditction = np.sum((prediction == ground_truths).astype(int))
    return 1-(correct_preditction/len(ground_truths))

'''
returns a list of lists containing the indices of the examples in the folds
'''
def get_folds(length,n_folds=4, seed=42):
     random.seed(seed)
     indices =  random.sample(range(length), length)
     dimension_fold = length/n_folds
     list_indices = []
     for i in range(n_folds):
         list_indices.append(indices[int(dimension_fold*i):int(dimension_fold*(i+1))])
     return list_indices

def holdout(indices, index_test,n_fold=4):
    test_indices = []
    train_indices =[]
    test_error = []
    train_error =[]
    test_indices = indices[index_test]
    train_indices = []
    for i in range(n_fold):
        if(index_test != i):
            train_indices.append(indices[i])
    train_indices = list(itertools.chain(*train_indices))
    return [test_indices,train_indices]
