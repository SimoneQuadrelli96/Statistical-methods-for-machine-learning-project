import pandas as pd
import numpy as np
'''
row is a list of elements
construct a list containing the labels
'''

def label_count(row):
  return [i for i in data[1:row.index("=")]]
'''
for each row of the dataset extract its id and its labels
returns a list of lists
'''
with open("SingerSplit_sorted_20k.data","r") as f:
       labels = []
       ids = []
       for line in f.readlines():
           data = line.split(' ')
           ids.append(data[0])
           data[len(data)-1] = data[len(data)-1].strip()
           labels.append(label_count(data))

'''
flatten the list of lists
'''
import itertools
labels_flat = list(itertools.chain(*labels))
labels_flat = np.array(labels_flat)

'''
for each label extract its frequency
'''
unique_elements, counts_elements = np.unique(labels_flat, return_counts=True)

'''
select the four most frequent label
'''
lab = []
for i in range(0,4):
    pos = np.argmax(counts_elements)
    lab.append(unique_elements[pos])
    counts_elements[pos] = 0
'''
create a matrix of zeros
'''
label_encoding = -np.ones(shape=(20000,4), dtype=int)
label_encoding

'''
encode the label in a boolean matrix
'''
for i in range(0,len(labels)):
    for elem in range(0,len(lab)):
        if  lab[elem] in labels[i]:
            label_encoding[i][elem] = 1

'''
create a dataframe with objects ids and the label encoding
'''
df = pd.DataFrame(ids, columns = ['id'])
for i in range(0,4):
    str = 'l_' + lab[i]
    df.insert(i+1, str, label_encoding[:,i])

'''
save the dataset
'''
df.to_csv('extracted_labels.csv',index=False)
