import sys
import random
import pandas as pd
import numpy as np
import io
import multiprocessing as mp
import time

import torch

#Load Set from a pickled file
import pickle
validation =int(sys.argv[1])

print("Loading Set from pickle file.")

file_to_read = open("/blue/tkahveci/aysegul.bumin/Data/Set.pkl", "rb")

Set = pickle.load(file_to_read)

#print(Loaded_Set.keys())

file_to_read.close()
print("Loading run1...")

file_name = "/blue/tkahveci/aysegul.bumin/Data/run1.pkl"

open_file = open(file_name, "rb")
run1 = pickle.load(open_file)
open_file.close()
print("Loading run2...")

file_name = "/blue/tkahveci/aysegul.bumin/Data/run2.pkl"

open_file = open(file_name, "rb")
run2 = pickle.load(open_file)
open_file.close()
print("Loading run3...")

file_name = "/blue/tkahveci/aysegul.bumin/Data/run3.pkl"

open_file = open(file_name, "rb")
run3 = pickle.load(open_file)
open_file.close()
print("Loading run4...")

file_name = "/blue/tkahveci/aysegul.bumin/Data/run4.pkl"

open_file = open(file_name, "rb")
run4 = pickle.load(open_file)
open_file.close()
print("Loading run5...")

file_name = "/blue/tkahveci/aysegul.bumin/Data/run5.pkl"

open_file = open(file_name, "rb")
run5 = pickle.load(open_file)
open_file.close()

for key in Set.keys():
    Set[key]= torch.FloatTensor(Set[key])

# O is the missing values
# 1 is the test values
# 2 is the validation 
# 3,4,5 are the training samples

val_list = []
test_list = []
train_list = []

val = validation

test_list = [(i,j) for i,j  in list(Set.keys()) if run1[i,j]== 1.0]

val_list = [(i,j) for i,j  in list(Set.keys()) if run1[i,j]== val]


train_list = [(i,j) for i,j in list(Set.keys()) if (run1[i,j]!=val and  run1[i,j] != 1.0 and run1[i,j] != 0) ] #if it is not missing value (zero) and if it is not test (one) and if it is not validation (val) then it is training 

print("Training Samples: ", len(train_list))
print("Validation Samples: ", len(val_list))
print("Test Samples: ", len(test_list))

def PredictwithMedian(i1,j1,k,Set):
    rows = []
    cols = []
    #for i1, j1 in test_samples:
    # fixed i- row 
    for (i_, j_) in list(Set.keys()):
      if (i_, j_) not in test_list:
      	if j_ !=j1 and i_==i1:
       	    rows.append( float(Set[(i1,j_)][k]))
    for (i_,j_) in list(Set.keys()):
      if (i_, j_) not in test_list:
	if i_!=i1 and j_==j1:
            cols.append(float(Set[(i_,j1)][k]))
    rows= np.array(rows)
    rows_median = np.median(rows)
    cols= np.array(cols)
    cols_median = np.median(cols)

    predicted= (rows_median+ cols_median)/2.0
    
    return predicted

import time
#0.950 with test set size 1605 training set 24K
normalized_loss_list = []
mse_loss_list = []
rmse_list=[]

summed_error = 0
summed_norm = 0 
for i,j in val_list:
    for k in range(978):
      predicted = PredictwithMedian(i,j,k,Set)
      actual = Set[(i,j)][k]
      
      summed_norm += (actual)**2
      error = (predicted-actual)**2
      summed_error += error

rmse = np.sqrt(summed_error)/ np.sqrt(summed_norm)
print("Test error:", rmse)
