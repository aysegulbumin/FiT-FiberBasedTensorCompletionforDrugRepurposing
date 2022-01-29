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

#validation = int(sys.argv[3])
test_id = int(sys.argv[1])
iterate = int(sys.argv[2])
multi = int(sys.argv[3])
print("Loading Set from pickle file.")

file_to_read = open("/blue/tkahveci/aysegul.bumin/Data/Set.pkl", "rb")

Set = pickle.load(file_to_read)

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

def combine_tensors(first_tensor, second_tensor):
    third_tensor = torch.cat((first_tensor, second_tensor), 0)
    return third_tensor

# Adjusted such that we can append more genes , this part is just for time analysis purposes.
for key in Set.keys():

    multi_tensor = torch.FloatTensor(Set[key])
    one_tensor = torch.FloatTensor(Set[key])

    #multi has to be >=1 
    for i in range(multi-1):
        multi_tensor =  combine_tensors(multi_tensor,one_tensor)

    Set[key]= multi_tensor

print("Multitensor shape:", multi_tensor.shape)

class TF_SGD(): #stands for tensor factorization
    
    def __init__(self,Dict, K, alpha, iterations, mode,training_sample, test_sample):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.K = K
        self.alpha = alpha
        self.iterations = iterations
        self.mode=mode
        self.mood=0 #By default it will do SGD
        self.Dict=Dict
        self.training_samples=list(training_sample)
        self.validation_samples=[]#list(validation_sample)
        self.test_samples=list(test_sample)
        self.Error={}
        self.Genes_predicted = {}
        self.Genes_actual = {}
      

    def train(self,mood, seed):
        self.mood=mood
        self.height=80
        self.length=1330
        key = list(self.Dict.keys())[0]
        
        self.depth=len(self.Dict[key])
        print("Latent factors are being initialized")
        # Initialize user and item latent feature matrice
        np.random.seed(seed+0)
        self.A=np.random.normal(scale=1./self.K, size=(self.height, self.K))
        np.random.seed(seed+1)
        self.B= np.random.normal(scale=1./self.K, size=(self.length, self.K))
        np.random.seed(seed+2)
        self.C= np.random.normal(scale=1./self.K, size=(self.depth, self.K))

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        test_error_list = []
        test_norm_square_list =[]
        time_array=[]
        inter=0
        time_array.append(inter)
        start_=time.process_time()
        initial_mse, e1, n1 = self.mse_test()
        training_process.append((0, initial_mse))
        print("Iteration: %d ; error = %.4f" % (0,initial_mse))
        
        for i in range(1,self.iterations):
            
            np.random.shuffle(self.training_samples)
            
            if(self.mood==0): #Traditional SGD
                self.sgd(i)
                print (i, "SGD done")
            elif(self.mood==1):#AdaGrad
                print("Calling Adagrad")
                self.AdaGrad(i)
                print (i, "AdaGrad done")
            else:
                raise Exception("No such mood is implemented.Please select within range [0,1].")
            if i%1== 0:
                mse, test_error , test_norm_square = self.mse_test()
                inter = time.process_time()-start_
                time_array.append(inter)
                test_error_list.append(test_error)
                test_norm_square_list.append(test_norm_square)
                training_process.append((i, mse))
                print("Iteration: %d ; error = %.4f" % (i, mse))
                print("Time:", inter)
        return training_process, time_array , test_error_list, test_norm_square_list
    
    def calculate_imputed_tensor(self):
        
        print("Calculating the imputed tensor.")
        self.ImputedTensor = {}
        for x,y in self.Dict.keys():
            Hadamart_ai_bj = torch.mul(torch.tensor(self.A[x, :][:]), torch.tensor(self.B[y, :][:])) 
            self.ImputedTensor[(x,y)] = torch.tensor(self.C)@Hadamart_ai_bj
        return self.ImputedTensor
        
    def mse_validation(self):
        """
        A function to compute the total mean square error
        """
        print("Calculating mse...")
        summed_error = 0
        count = 0
        summed_norm_square = 0
        #global Error
        for x, y in self.validation_samples :
            norm_square = 0
            error = 0
            count = count+1
            
            Hadamart_ai_bj = torch.mul(torch.tensor(self.A[x, :][:]), torch.tensor(self.B[y, :][:])) 
            Fiber_actual = self.Dict[(x,y)]
            
            sub=torch.sub(Fiber_actual, torch.tensor(self.C)@Hadamart_ai_bj, alpha=1)
            error += torch.norm(sub)*torch.norm(sub)
            norm_square  += torch.norm(Fiber_actual)*torch.norm(Fiber_actual)
            
            summed_error = summed_error+error
            summed_norm_square = summed_norm_square+norm_square
            
            self.Error[(x,y)]= np.sqrt(error.numpy())/ np.sqrt(norm_square.numpy())
        return np.sqrt(summed_error.numpy())/ np.sqrt(summed_norm_square.numpy()) , error.numpy() , norm_square.numpy()

    def mse_test(self):
        """
        A function to compute the total mean square error
        """
        self.Genes_predicted = {}
        self.Genes_actual = {}
        summed_error = 0
        count = 0
        summed_norm_square = 0
        #global Error
        index=0
        for x, y in self.test_samples :
            norm_square = 0
            error = 0
            count=count+1

            Hadamart_ai_bj = torch.mul(torch.tensor(self.A[x, :][:]), torch.tensor(self.B[y, :][:])) 
            Fiber_actual = self.Dict[(x,y)]
            
            sub=torch.sub(Fiber_actual, torch.tensor(self.C)@Hadamart_ai_bj, alpha=1)
            error = torch.norm(sub)*torch.norm(sub)
            norm_square  = torch.norm(Fiber_actual)*torch.norm(Fiber_actual)

            summed_error = summed_error+error
            summed_norm_square = summed_norm_square+norm_square

            self.Error[(x,y)]= np.sqrt(error.numpy())/ np.sqrt(norm_square.numpy())
        return np.sqrt(summed_error.numpy())/ np.sqrt(summed_norm_square.numpy()) , summed_error.numpy() , summed_norm_square.numpy()



    def sgd(self,iterations):
        """
        Perform stochastic gradient descent
        """
        if(self.mode==0):
            m=0
            #print('Alpha:', self.alpha)
        if(self.mode==1):
            self.alpha=float(1/(iterations))
            #print('Alpha:', self.alpha)
        np.random.seed(iterations+0)
        np.random.shuffle(self.training_samples)
        mini_samples=self.training_samples[:1000]
        for i, j in mini_samples:
            for each in range(1):
          
                # Computer prediction and error
                # Create copy of row of P since we need to update it but use older values for update on Q
                A_i = self.A[i, :][:]
                B_j = self.B[j, :][:]               
                Ce = self.C

                
                Hadamart_ai_bj = torch.mul(torch.tensor(self.A[i, :][:]), torch.tensor(self.B[j, :][:])) 
                Fiber_actual = self.Dict[(i,j)]
                predicted = torch.tensor(self.C)@Hadamart_ai_bj
                sub = torch.sub(Fiber_actual, torch.tensor(self.C)@Hadamart_ai_bj, alpha=1)
                
                Grad_A = - torch.mul(torch.tensor(self.C).T@sub,torch.tensor(self.B[j, :][:]))
                
                Grad_B = - torch.mul(torch.tensor(self.C).T@sub,torch.tensor(self.A[i, :][:]))
                
                transpose= Hadamart_ai_bj.T
                Grad_C = - sub.unsqueeze(1)@transpose.unsqueeze(0)

                self.A[i, :] -= self.alpha * Grad_A.numpy()
                self.B[j, :] -= self.alpha * Grad_B.numpy()
                self.C -= self.alpha * Grad_C.numpy()
                
    def AdaGrad(self,iterations):
        """
        Perform stochastic gradient descent with AdaGrad update
        """
        print('Alpha:', self.alpha)
        
        gti_a=np.zeros(self.K)
        gti_b=np.zeros(self.K)
        gti_c=np.zeros(self.K)
        
        np.random.shuffle(self.training_samples)
        mini_samples=self.training_samples
        for i, j in mini_samples:
            for k, r  in enumerate(self.Dict[(i,j)]):
                # Create copy of row of P since we need to update it but use older values for update on Q
                A_i = self.A[i, :][:]
                B_j = self.B[j, :][:]
                C_k = self.C[k, :][:]

                ac=np.multiply(A_i,C_k)
                bc=np.multiply(B_j,C_k)
                ab=np.multiply(A_i,B_j)

                innerproduct=np.matmul(np.multiply(self.A[i, :][:], self.B[j, :][:]), self.C[k,:][:])
                n= innerproduct-r

                fudge_factor=1e-6

                grad_a=( bc*n.numpy())
                gti_a+=grad_a**2
                adjusted_grad_a = grad_a / np.sqrt(fudge_factor +(gti_a)) 

                grad_b=(  ac*n.numpy())
                gti_b+=grad_b**2
                adjusted_grad_b = grad_b / np.sqrt(fudge_factor + (gti_b))

                grad_c=(  ab*n.numpy())
                gti_c+=grad_c**2
                adjusted_grad_c = grad_c / np.sqrt(fudge_factor + (gti_c))


                self.A[i, :] -= self.alpha * adjusted_grad_a
                self.B[j, :] -= self.alpha * adjusted_grad_b
                self.C[k, :] -= self.alpha * adjusted_grad_c

    def get_rating(self, i, j,k ):
        """
        Get the predicted rating of user i and item j
        """
        predicted=np.matmul(np.multiply(self.A[i, :][:], self.B[j, :][:]), self.C[k,:][:])
        return predicted


# O is the missing values
# 1 is the test values
# 2 is the validation 
# 3,4,5 are the training samples

#val_list = []
test_list = []
train_list = []

#val = validation

test_list = [(i,j) for i,j  in list(Set.keys()) if run1[i,j]== test_id]

#val_list = [(i,j) for i,j  in list(Set.keys()) if run1[i,j]== val]

train_list =[(i,j) for i,j in list(Set.keys()) if (run1[i,j] != test_id and run1[i,j] != 0) ]
#train_list = [(i,j) for i,j in list(Set.keys()) if (run1[i,j]!=val and  run1[i,j] != 1.0 and run1[i,j] != 0) ] #if it is not missing value (zero) and if it is not test (one) and if it is not validation (val) then it is training 

print("Training Samples: ", len(train_list))
#print("Validation Samples: ", len(val_list))
print("Test Samples: ", len(test_list))

#TRAINING 
print("Training is starting...")
SetList=[Set]
General= []
GenesPredicted = []
GenesActual = []
Time = []
General_test_error = []
General_test_norm_square = []
count=0
ImputedTensors = []
for each_set in SetList:
    #STOCHASTIC GRADIENT DESCENT Diminishing step size (1/t) actually (1/100t)
    count = count+1
    print("Cluster ",count)
    start = time.time()
    start_clock = time.process_time()

    # storing the arguments
    alpha_= 0.01
    rank = 500

    tf_sgd_constant = TF_SGD(each_set,rank, alpha_, iterate,2, train_list, test_list)
    general_sgd_constant, time_sgd_constant, test_error, test_norm = tf_sgd_constant.train(1,0) #, [(0, seed) for seed in range(2)]) #0 mood SGD, 1 mood AdaGrad, 2 mood ADAM
    
    print("No Fiber Time:")
    print(time_sgd_constant)

    mse_test = tf_sgd_constant.mse_test()

    
    ImputedTensor = tf_sgd_constant.calculate_imputed_tensor()
    ImputedTensors.append(ImputedTensor)
    
    GenesPredicted.append(tf_sgd_constant.Genes_predicted)
    GenesActual.append(tf_sgd_constant.Genes_actual)
    
    #GenePr
    General_test_error.append(test_error)
    General_test_norm_square.append(test_norm)
    
    General.append(general_sgd_constant)
    Time.append(time_sgd_constant)
    
    end=time.time()
    end_clock=time.process_time()

    print(len(tf_sgd_constant.Error))
    print(len(general_sgd_constant))
    print(len(time_sgd_constant))

    print("Loss Array: " , general_sgd_constant)

    print("Time Array: " , time_sgd_constant)

    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Adagrad Total Run Time \n {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
    print("Time clock : ", end_clock-start_clock)

    print("Adagrad Alpha "+ str(alpha_))

print("Test Error in Every Iteration : ")

import math
error = 0
norm = 0
for i in range(1):
    error += General_test_error[i][-1]
    norm +=General_test_norm_square[i][-1]
print(math.sqrt(error)/math.sqrt(norm))

print("Test Error in Every Iteration Min :")

error = 0
norm = 0
c = 0
for i in range(1):
    min_value = min(General_test_error[i])
    min_index = General_test_error[i]. index(min_value)
    error += General_test_error[i][min_index]
    norm += General_test_norm_square[i][min_index]
print(math.sqrt(error)/math.sqrt(norm))


print("Calculating the Imputed Tensor : ")


ImputedTensorSeed = {}

for key, value in Set.items():
    
    ImputedTensorSeed[key] = torch.zeros(978)
    
    ImputedTensorSeed[key] = ImputedTensors[0][key]

# define dictionary
# Set

# create a binary pickle file 
f = open("ImputedTensorNoClusterTest"+str(test_id)+".pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(ImputedTensorSeed,f)

# close file
f.close()

print("Calculating the Test Error :")

rmse = 0
norm_square=0
error=0
for key in test_list:
    for k in range(978):
        predicted = ImputedTensorSeed[key][k]
        actual = Set[key][k]
        error += (predicted-actual)**2
        norm_square += (actual)**2
    
rmse = np.sqrt(error)/np.sqrt(norm_square)
print(rmse)

print("END")
