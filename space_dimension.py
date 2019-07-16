import pandas as pd
import numpy as np
import scipy
from scipy import sparse
import pickle
'''
search the max key value in order to create a sparse matrix with dimension
number_of_examples x max_key_value
'''
with open("../github/SingerSplit_sorted_20k.data","r") as f:
       features_name = []
       number_of_examples = 0
       for line in f.readlines():
           number_of_examples +=1
           data = line.split(' ')
           data[len(data)-1] = data[len(data)-1].strip()
           data = data[data.index("=")+1:len(data)]
           features = []
           for i in range(len(data)):
               features.append(int(data[i].split(':')[0]))
           features_name.append(np.max(np.array(features)))


max_key_value = np.max(features_name)
#found that 95280 =  max key value


'''
creation of matrix_X of shape number_of_examples x max_key_value
where values are stored in position [example, key]
'''
with open("../github/SingerSplit_sorted_20k.data","r") as f:
       # max_key_value + 1 since the counting of labels starts from 0
       matrix_X = sparse.dok_matrix((number_of_examples,  max_key_value+1), dtype=np.float32)
       row = 0
       for line in f.readlines():
           data = line.split(' ')
           data[len(data)-1] = data[len(data)-1].strip()
           data = data[data.index("=")+1:len(data)]
           for i in range(len(data)):
               column = int(data[i].split(':')[0])
               value = np.float32(data[i].split(':')[1])
               matrix_X[row, column] = value
           row += 1


'''
creation of matrix.pickle in csr format
'''
with open('matrix.pickle', 'wb') as handle:
    pickle.dump(matrix_X.tocsr(), handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('matrix.pickle', 'rb') as handle:
    b = pickle.load(handle)
