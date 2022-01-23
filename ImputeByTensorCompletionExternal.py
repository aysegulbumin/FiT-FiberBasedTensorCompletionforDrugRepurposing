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

cluster_seed = int(sys.argv[1])
cluster_level = int(sys.argv[2])
#validation = int(sys.argv[3])
test_id = int(sys.argv[3])

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

class TF_SGD_External(): #stands for tensor factorization
    
    def __init__(self,Dict, K, alpha, iterations, mode,training_sample, test_sample, GeneInteract):
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
        self.test_samples=list(test_sample)
        self.Error={}
        #self.BinaryMatrix=BinaryMatrix
        self.GeneInteract = GeneInteract
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
                #prediction = self.get_rating(i, j ,k)
                #e = (r - prediction)
            
                # Create copy of row of P since we need to update it but use older values for update on Q
                A_i = self.A[i, :][:]
                B_j = self.B[j, :][:]
                #C_k = self.C[k, :][:]
                
                Ce = self.C

                A_Tilde = torch.zeros(978, 978, dtype=torch.float)
                for key in self.GeneInteract.keys():
                  x = int(key[0])
                  y = int(key[1])
                  I = torch.eye(self.K)
                  one = torch.ones(self.K)
                  summation =  one @ one.T
                  mean = (1.0/self/K)*summation
                  ci = Ce[i]
                  cj = Ce[j]
                  A_Tilde[i][j] = 1.0 / (torch.norm((I- mean)@ci) * torch.norm((I- mean)@cj)* torch.norm((I- mean)@ci) * torch.norm((I- mean)@cj))

                
                Hadamart_ai_bj = torch.mul(torch.tensor(self.A[i, :][:]), torch.tensor(self.B[j, :][:])) 
                Fiber_actual = self.Dict[(i,j)]
                predicted = torch.tensor(self.C)@Hadamart_ai_bj
                sub = torch.sub(Fiber_actual, torch.tensor(self.C)@Hadamart_ai_bj, alpha=1)
                
                Grad_A = - torch.mul(torch.tensor(self.C).T@sub,torch.tensor(self.B[j, :][:])) + 2*0.0001*torch.tensor(self.A[i, :][:])
                #Grad_A2 = - torch.tensor(self.C).T@torch.mul(sub,torch.tensor(self.B[j, :][:]))
                Grad_B = - torch.mul(torch.tensor(self.C).T@sub,torch.tensor(self.A[i, :][:])) + 2*0.0001*torch.tensor(self.B[j, :][:])
                #Grad_B2 = - torch.tensor(self.C).T@torch.mul(sub,torch.tensor(self.A[i, :][:]))
                I = torch.eye(self.K)
                one = torch.ones(self.K)
                summation =  one @ one.T
                mean = (1.0/self.K)*summation
                
                transpose= Hadamart_ai_bj.T
                #print(transpose.unsqueeze(0).shape)
                #print(sub.unsqueeze(1).shape)
                #print(torch.tensor(self.C).shape)
                Grad_C = - sub.unsqueeze(1)@transpose.unsqueeze(0) + 2*0.0001*(A_Tilde@self.C@(I-mean))
                
                self.A[i, :] -= self.alpha * Grad_A.numpy()
                self.B[j, :] -= self.alpha * Grad_B.numpy()
                self.C -= self.alpha * Grad_C.numpy()
                #self.C[k, :] -= self.alpha * ( ab*n)
    def AdaGrad(self,iterations):
        """
        Perform stochastic gradient descent with AdaGrad update
        """
        print('Alpha:', self.alpha)
        
        gti_a=np.zeros(self.K)
        gti_b=np.zeros(self.K)
        gti_c=np.zeros((self.depth,self.K))
        
        np.random.shuffle(self.training_samples)
        mini_samples=self.training_samples
        cou=0
        A_Tilde = torch.zeros(self.depth, self.depth, dtype=torch.float)
        for i, j in mini_samples:
            cou=cou+1
            # Create copy of row of P since we need to update it but use older values for update on Q
            A_i = self.A[i, :][:]
            B_j = self.B[j, :][:]
            # C_k = self.C[k, :][:]
            Ce = self.C
            # Correlation, demeaned version of the External Information.
            if cou%100 == 1:
                A_Tilde = torch.zeros(self.depth, self.depth, dtype=torch.float)
                for key in self.GeneInteract.keys():
                    x = int(key[0])
                    y = int(key[1])
                    I = torch.eye(self.K)
                    one = torch.ones(self.K)
                    summation =  one @ one.T
                    mean = (1.0/self.K)*summation
                    ci = Ce[x]
                    cj = Ce[y]
                    A_Tilde[x][y] = 1.0 / (torch.norm((I- mean)@ci) * torch.norm((I- mean)@cj))
                A_Tilde=A_Tilde.to_sparse()
                
            Hadamart_ai_bj = torch.mul(torch.tensor(self.A[i, :][:]), torch.tensor(self.B[j, :][:])) 
            Fiber_actual = self.Dict[(i,j)]
            predicted = torch.tensor(self.C)@Hadamart_ai_bj
            sub = torch.sub(Fiber_actual, torch.tensor(self.C)@Hadamart_ai_bj, alpha=1)
            
    
            Grad_A = - torch.mul(torch.tensor(self.C).T@sub,torch.tensor(self.B[j, :][:])) + 0.0001*torch.tensor(self.A[i, :][:])
            Grad_B = - torch.mul(torch.tensor(self.C).T@sub,torch.tensor(self.A[i, :][:])) + 0.0001*torch.tensor(self.B[j, :][:])
            
            I = torch.eye(self.K)
            one = torch.ones(self.K)
            summation =  one @ one.T
            mean = (1.0/self.K)*summation
                
            transpose= Hadamart_ai_bj.T
            multiplied = torch.sparse.mm(A_Tilde ,torch.tensor(self.C).float()@(I-mean))
            Grad_C = - sub.unsqueeze(1)@transpose.unsqueeze(0)+0.0001*multiplied#- 0.01*multiplied
            
            fudge_factor=1e-6
            
            #grad_a=( bc*n)
            #print(Grad_A.shape)
            #print(torch.tensor(Grad_A**2).shape)
            gti_a+=(Grad_A.numpy()**2)
            adjusted_grad_a = Grad_A.numpy() / np.sqrt(fudge_factor +(gti_a)) 
            
            #grad_b=(  ac*n)
            gti_b+=(Grad_B.numpy()**2)
            adjusted_grad_b = Grad_B.numpy() / np.sqrt(fudge_factor + (gti_b))
     
            #grad_c=(  ab*n)
            gti_c+=(Grad_C.numpy()**2)
            adjusted_grad_c = Grad_C .numpy()/ np.sqrt(fudge_factor + (gti_c))
        
       
            self.A[i, :] -= self.alpha * adjusted_grad_a
            self.B[j, :] -= self.alpha * adjusted_grad_b
            self.C -= self.alpha * adjusted_grad_c
   

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
                #prediction = self.get_rating(i, j ,k)
                #e = (r - prediction)
            
                # Create copy of row of P since we need to update it but use older values for update on Q
                A_i = self.A[i, :][:]
                B_j = self.B[j, :][:]
                #C_k = self.C[k, :][:]
                
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
        gti_c=np.zeros((self.depth,self.K))
        
        np.random.shuffle(self.training_samples)
        mini_samples=self.training_samples
        for i, j in mini_samples:

            # Create copy of row of P since we need to update it but use older values for update on Q
            A_i = self.A[i, :][:]
            B_j = self.B[j, :][:]
            # C_k = self.C[k, :][:]
            Ce = self.C

                
            Hadamart_ai_bj = torch.mul(torch.tensor(self.A[i, :][:]), torch.tensor(self.B[j, :][:])) 
            Fiber_actual =self.Dict[(i,j)]
            predicted = torch.tensor(self.C)@Hadamart_ai_bj
            sub = torch.sub(Fiber_actual, torch.tensor(self.C)@Hadamart_ai_bj, alpha=1)
                
            Grad_A = - torch.mul(torch.tensor(self.C).T@sub,torch.tensor(self.B[j, :][:]))
            Grad_B = - torch.mul(torch.tensor(self.C).T@sub,torch.tensor(self.A[i, :][:]))
            
                
            transpose= Hadamart_ai_bj.T
            Grad_C = - sub.unsqueeze(1)@transpose.unsqueeze(0)
            
            
            fudge_factor=1e-6
            
            #grad_a=( bc*n)
            gti_a+=(Grad_A.numpy()**2)
            adjusted_grad_a = Grad_A.numpy() / np.sqrt(fudge_factor +(gti_a)) 
            
            #grad_b=(  ac*n)
            gti_b+=(Grad_B.numpy()**2)
            adjusted_grad_b = Grad_B.numpy() / np.sqrt(fudge_factor + (gti_b))
     
            #grad_c=(  ab*n)
            gti_c+=(Grad_C.numpy()**2)
            adjusted_grad_c = Grad_C .numpy()/ np.sqrt(fudge_factor + (gti_c))
        
       
            self.A[i, :] -= self.alpha * adjusted_grad_a
            self.B[j, :] -= self.alpha * adjusted_grad_b
            self.C -= self.alpha * adjusted_grad_c
 
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

GeneNames=["DDR1" , "PAX8 " , "RPS5 " , "ABCF1 " , "SPAG7 " , "RHOA " , "RNPS1 " , "SMNDC1 " , "ATP6V0B " , "RPS6 " , "USP22 " , "APP " , "CLTC " , "MLEC " , "CSRP1 " , "CALM3 " , "PTPRF " , "DNAJB1 " , "XBP1 " , "GRN " , "HK1 " , "KDELR2 " , "SKP1 " , "CAPN1 " , "CALU " , "CTSD " , "MAT2A " , "STMN1 " , "ECH1 " , "IQGAP1 " , "HSPA1A " , "HSPD1 " , "CIRBP " , "PSME1 " , "PAFAH1B1 " , "HYOU1 " , "PSMD2 " , "EPRS " , "PSMD4 " , "PGAM1 " , "STAT1 " , "FKBP4 " , "KTN1 " , "TMED10 " , "ALDOA " , "TSPAN3 " , "GNAS " , "HIF1A " , "AARS " , "RPN1 " , "PAICS " , "BLCAP " , "HADH " , "GNAI2 " , "PSMF1 " , "MYL9 " , "MMP2 " , "SMARCC1 " , "TM9SF2 " , "PIP4K2B " , "PXN " , "COPB2 " , "PFKL " , "PGRMC1 " , "ITGB5 " , "PLP2 " , "NFE2L2 " , "MBNL1 " , "NMT1 " , "BHLHE40 " , "TERF2IP " , "FBXO7 " , "HTRA1 " , "LRPAP1 " , "CREG1 " , "PCNA " , "LGMN " , "ILK " , "ATP1B1 " , "SYPL1 " , "TXNRD1 " , "NUDCD3 " , "TOP2A " , "STK25 " , "EBNA1BP2 " , "ARHGEF12 " , "SCP2 " , "YME1L1 " , "TMEM109 " , "TPD52L2 " , "CRTAP " , "TRAP1 " , "IGF2R " , "PHGDH " , "LRP10 " , "SOX4 " , "CAT " , "RHEB " , "MAPKAPK2 " , "SCRN1 " , "JUN " , "SHC1 " , "SQSTM1 " , "PRCP " , "USP7 " , "NFKBIA " , "G3BP1 " , "TOMM70A " , "RPA1 " , "ZFP36 " , "DUSP3 " , "KDM5B " , "MCM3 " , "CLIC4 " , "CLSTN1 " , "ID2 " , "DCTD " , "FAT1 " , "SMC1A " , "NISCH " , "PWP1 " , "ICMT " , "RUVBL1 " , "MBTPS1 " , "INSIG1 " , "RRAGA " , "IER3 " , "UBE2L6 " , "SMC4 " , "USP14 " , "EGR1 " , "PNP " , "DNMT1 " , "CCND3 " , "NIPSNAP1 " , "MYBL2 " , "EPB41L2 " , "ELAVL1 " , "KIAA0100 " , "TP53 " , "RPA2 " , "MTHFD2 " , "PSME2 " , "DAXX " , "ELAC2 " , "NCAPD2 " , "EFCAB14 " , "DDX42 " , "LBR " , "SH3BP5 " , "UBE3C " , "SCARB1 " , "SCCPDH " , "SMARCD2 " , "NET1 " , "HDAC2 " , "HSPB1 " , "LIPA " , "BNIP3 " , "CDC25B " , "ATMIN " , "TOMM34 " , "MPZL1 " , "IL13RA1 " , "PSRC1 " , "UBE2A " , "COASY " , "LRRC41 " , "MYCBP2 " , "RBM6 " , "PGM1 " , "MYO10 " , "RSU1 " , "EGFR " , "KIAA0196 " , "EXT1 " , "SPEN " , "PTPN12 " , "TWF2 " , "TJP1 " , "MEST " , "ALDOC " , "KDM5A " , "RAI14 " , "BIRC2 " , "CTSL " , "PAF1 " , "BIRC5 " , "RALB " , "ARFIP2 " , "ARHGAP1 " , "CPNE3 " , "ABL1 " , "TRAK2 " , "PYCR1 " , "NUP62 " , "PCM1 " , "BLMH " , "MVP " , "NUP133 " , "PLOD3 " , "PPP2R5A " , "NUP93 " , "ARL4C " , "KIAA0907 " , "CRK " , "CHERP " , "PLK1 " , "TRIB1 " , "CDK4 " , "STXBP1 " , "VPS72 " , "HSD17B10 " , "CDKN1A " , "COL1A1 " , "SENP6 " , "ACBD3 " , "CSK " , "CSNK1E " , "TOR1A " , "TRAM2 " , "TCEAL4 " , "GNPDA1 " , "RGS2 " , "ABCF3 " , "TCERG1 " , "USP1 " , "KEAP1 " , "KAT6A " , "MPC2 " , "MYC " , "SLC35B1 " , "PLSCR1 " , "DECR1 " , "ERBB3 " , "PRSS23 " , "PAPD7 " , "CTNNAL1 " , "IKBKAP " , "PPIE " , "DNAJB2 " , "SNAP25 " , "BCL7B " , "GALE " , "HMGCR " , "PRKCD " , "VAPB " , "MYLK " , "S100A13 " , "NRIP1 " , "HTATSF1 " , "ADAM10 " , "EAPP " , "SERPINE1 " , "APPBP2 " , "TOPBP1 " , "POLR2K " , "ICAM1 " , "NRAS " , "LPGAT1 " , "PSMB10 " , "SDHB " , "RASA1 " , "GTF2A2 " , "NPC1 " , "GTF2E2 " , "RNMT " , "RBM15B " , "OXSR1 " , "DUSP11 " , "CCNB2 " , "HIST2H2BE " , "PTPN1 " , "TES " , "GFPT1 " , "LIG1 " , "PKIG " , "P4HA2 " , "EBP " , "PHKB " , "PIK3R3 " , "WRB " , "GPC1 " , "SYNE2 " , "CASP3 " , "DNTTIP2 " , "ZMYM2 " , "OXCT1 " , "NNT " , "MAPKAPK3 " , "INPP1 " , "SACM1L " , "PRKACA " , "INTS3 " , "STAMBP " , "GAA " , "TARBP1 " , "SLC25A4 " , "SLC37A4 " , "PCK2 " , "HPRT1 " , "FAH " , "POP4 " , "CDC20 " , "CYTH1 " , "DDIT4 " , "MAP7 " , "NIT1 " , "NUP88 " , "WFS1 " , "ADGRE5 " , "MSH6 " , "FAM20B " , "PIN1 " , "ETFB " , "FPGS " , "FHL2 " , "CRYZ " , "UBE2C " , "RFX5 " , "ARNT2 " , "PYGL " , "POLD4 " , "LYPLA1 " , "ECD " , "PTPRK " , "TIMELESS " , "STK10 " , "TP53BP1 " , "TCTA " , "PDHX " , "KLHL21 " , "COG2 " , "DNM1L " , "PTK2B " , "PAN2 " , "CCDC86 " , "TP53BP2 " , "SLC11A2 " , "SPTLC2 " , "KIF5C " , "RB1 " , "TBP " , "HAT1 " , "PAK4 " , "TIMP2 " , "RRP8 " , "S100A4 " , "B4GAT1 " , "ABCB6 " , "PMM2 " , "MTFR1 " , "RFC5 " , "CDK1 " , "ST3GAL5 " , "MAPK9 " , "TLE1 " , "PAFAH1B3 " , "IL4R " , "NPRL2 " , "CDH3 " , "DRAP1 " , "DFFA " , "EDEM1 " , "HS2ST1 " , "KIAA0355 " , "CNOT4 " , "DMTF1 " , "DCK " , "DYNLT3 " , "BAMBI " , "SLC35A1 " , "NCK2 " , "ITGB1BP1 " , "PPP2R5E " , "CEBPZ " , "TIMM17B " , "UGDH " , "MTF2 " , "EZH2 " , "MYCBP " , "DUSP14 " , "SOCS2 " , "RPS6KA1 " , "APOE " , "HES1 " , "PSMG1 " , "SATB1 " , "DDB2 " , "CCNA2 " , "EML3 " , "PRAF2 " , "SPR " , "EPN2 " , "MRPL19 " , "CEP57 " , "TRAPPC3 " , "ZNF318 " , "STX4 " , "IPO13 " , "PCBD1 " , "MNAT1 " , "AGL " , "LOXL1 " , "NFIL3 " , "CSNK2A2 " , "RAB4A " , "POLB " , "IGF1R " , "FGFR2 " , "MBNL2 " , "TATDN2 " , "TRIM13 " , "HMOX1 " , "NUCB2 " , "BCL2 " , "RFC2 " , "FZD7 " , "PHKG2 " , "GADD45A " , "LAMA3 " , "SKIV2L " , "BUB1B " , "SLC25A13 " , "SSBP2 " , "AKAP8 " , "PDIA5 " , "RAB11FIP2 " , "RAB21 " , "LYRM1 " , "RAP1GAP " , "TCEA2 " , "NFKBIE " , "MRPL12 " , "ATF6 " , "CEBPD " , "GNB5 " , "DUSP4 " , "CEBPA " , "WASF3 " , "PRKX " , "SLC5A6 " , "MAP3K4 " , "AURKA " , "CCNH " , "TESK1 " , "CDC45 " , "FOXO3 " , "ENOSF1 " , "IFNAR1 " , "CDK2 " , "ELOVL6 " , "PMAIP1 " , "PIK3C3 " , "CREB1 " , "PIK3CA " , "GSTM2 " , "FOSL1 " , "FZD1 " , "PLA2G15 " , "SNCA " , "MMP1 " , "PIK3C2B " , "CD44 " , "DPH2 " , "PPIC " , "BRCA1 " , "ST6GALNAC2 " , "IKBKE " , "FGFR4 " , "SLC25A14 " , "CGRRF1 " , "CCDC85B " , "ACD " , "TFAP2A " , "SHB " , "CCP110 " , "KCNK1 " , "CDC25A " , "KIAA0753 " , "NCK1 " , "STX1A " , "PTGS2 " , "MAP2K5 " , "C2CD2L " , "USP6NL " , "FAS " , "PPOX " , "TMEM5 " , "CLPX " , "ZW10 " , "MELK " , "CCNF " , "RAD9A " , "TCFL5 " , "ZNF274 " , "ICAM3 " , "DDX10 " , "TRAPPC6A " , "CDK5R1 " , "ATF5 " , "STAT5B " , "CCNE2 " , "LSM 6.00 " , "IKZF1 " , "CENPE " , "KIT " , "ITGAE " , "IL1B " , "ORC1 " , "MAMLD1 " , "SGCB " , "MOK " , "CD40 " , "PEX11A " , "CLTB " , "CD58 " , "PLS1 " , "PCMT1 " , "RELB " , "GNA15 " , "INPP4B " , "CBR3 " , "CHEK1 " , "SMAD3 " , "DAG1 " , "PHKA1 " , "FOXO4 " , "PIGB " , "PDGFA " , "CASP10 " , "GHR " , "C5 " , "BTK " , "RPP38 " , "SNX7 " , "NOS3 " , "SCYL3 " , "ALAS1 " , "SYNGR3 " , "BPHL " , "POLG2 " , "NOLC1 " , "NFATC4 " , "CCNA1 " , "POLE2 " , "DNAJA3 " , "FOXJ3 " , "RNH1 " , "RAD51C " , "EPHA3 " , "PRKCH " , "FUT1 " , "ADRB2 " , "GABPB1 " , "EGF " , "KIF14 " , "ETV1 " , "CSNK1A1 " , "MAP4K4 " , "GLRX " , "PTPN6 " , "CPSF4 " , "LPAR2 " , "DFFB " , "SLC35A3 " , "HDAC6 " , "GLI2 " , "CDKN2A " , "E2F2 " , "CDK6 " , "AKT1 " , "CASP7 " , "TNIP1 " , "TERT " , "PTPRC " , "TGFBR2 " , "NFATC3 " , "RAC2 " , "POLR1C " , "NFKB2 " , "SYK " , "CASK " , "NCOA3 " , "PSMD9 " , "PROS1 " , "CASC3 " , "MUC1 " , "ST7 " , "NVL " , "MEF2C " , "CHP1 " , "HMGA2 " , "CASP2 " , "LSR " , "MALT1 " , "TBPL1 " , "CTNND1 " , "CIAPIN1 " , "BAX " , "PPARG " , "SPTAN1 " , "EIF4G1 " , "VAT1 " , "MACF1 " , "PARP1 " , "FDFT1 " , "HSPA8 " , "PDLIM1 " , "EIF5 " , "CCND1 " , "TMCO1 " , "OXA1L " , "CDC42 " , "TSC22D3 " , "PTK2 " , "ADH5 " , "REEP5 " , "DUSP6 " , "HLA.DRA " , "ATP6V1D " , "CYCS " , "CAST " , "LGALS8 " , "BECN1 " , "ALDH7A1 " , "STAT3 " , "DNAJB6 " , "COPS7A " , "PSMB8 " , "XPNPEP1 " , "RALGDS " , "CORO1A " , "GLOD4 " , "DLD " , "IFRD2 " , "TSPAN6 " , "CDKN1B " , "PRPF4 " , "CYB561 " , "MAN2B1 " , "MBOAT7 " , "FOS " , "TUBB6 " , "CBR1 " , "MFSD10 " , "SORBS3 " , "SMC3 " , "SFN " , "NR2F6 " , "TSPAN4 " , "GADD45B " , "PSIP1 " , "IKBKB " , "BAD " , "STXBP2 " , "ABCC5 " , "KIF2C " , "GRB10 " , "ARHGEF2 " , "AURKB " , "MKNK1 " , "RPA3 " , "RAB27A " , "HDGFRP3 " , "GSTZ1 " , "RRS1 " , "EED " , "GNAI1 " , "PRUNE " , "EPHB2 " , "GATA3 " , "ACAT2 " , "PAK1 " , "CETN3 " , "CBLB " , "GATA2 " , "TGFB3 " , "CXCL2 " , "HIST1H2BK " , "ANXA7 " , "SPP1 " , "PUF60 " , "CFLAR " , "PRKCQ " , "MAPK13 " , "FYN " , "RPL39L " , "PLA2G4A " , "DYRK3 " , "ME2 " , "ACLY " , "CHEK2 " , "GPER1 " , "HMG20B " , "LYN " , "GRB7 " , "DHRS7 " , "TPM1 " , "HSPA4 " , "MLLT11 " , "CDK7 " , "RAE1 " , "BMP4 " , "BDH1 " , "BID " , "BLVRA " , "LSM 5.00 " , "TXNDC9 " , "MTA1 " , "CXCR4 " , "COL4A1 " , "RNF167 " , "WIPF2 " , "TBC1D9B " , "ADGRG1 " , "HN1L " , "ZMIZ1 " , "PDS5A " , "IGFBP3 " , "XPO7 " , "CRKL " , "COG4 " , "H2AFV " , "FBXO21 " , "ATP2C1 " , "TMEM97 " , "SUZ12 " , "TXLNA " , "TFDP1 " , "VGLL4 " , "UBE3B " , "KIF1BP " , "SPRED2 " , "KAT6B " , "GPATCH8 " , "ADO " , "ATP11B " , "ZNF451 " , "RBM34 " , "ARID5B " , "CHN1 " , "MAPK1IP1L " , "DHX29 " , "JADE2 " , "TIPARP " , "KDM3A " , "PCCB " , "PLEKHM1 " , "JMJD6 " , "PIK3R4 " , "CAMSAP2 " , "KIAA1033 " , "SLC1A4 " , "ASCC3 " , "SLC25A46 " , "RRP1B " , "AXIN1 " , "DCUN1D4 " , "MAPKAPK5 " , "C2CD2 " , "WDR7 " , "SUPV3L1 " , "CDK19 " , "THAP11 " , "C2CD5 " , "POLR2I " , "RFNG " , "RPIA " , "TLK2 " , "TIAM1 " , "HOXA10 " , "COG7 " , "TICAM1 " , "KLHL9 " , "SNX13 " , "SRC " , "GDPD5 " , "PLCB3 " , "TBX2 " , "APBB2 " , "FCHO1 " , "FAM69A " , "ASAH1 " , "SMARCA4 " , "SOX2 " , "HOXA5 " , "TMEM110 " , "ATP5S " , "TBC1D31 " , "NFKBIB " , "CTTN " , "PARP2 " , "AKR7A2 " , "ACAA1 " , "SPDEF " , "RALA " , "ETS1 " , "CCNB1 " , "ZNF131 " , "FEZ2 " , "NSDHL " , "DNM1 " , "UBQLN2 " , "MAST2 " , "TRIM2 " , "IGHMBP2 " , "NR3C1 " , "PPP1R13B " , "CCL2 " , "ERBB2 " , "RRP12 " , "HOMER2 " , "VDAC1 " , "HERPUD1 " , "GAPDH " , "HLA.DMA " , "IDE " , "NGRN " , "CNDP2 " , "TM9SF3 " , "ADI1 " , "RAB31 " , "TMEM50A " , "HACD3 " , "YKT6 " , "SNX6 " , "BZW2 " , "UBE2J1 " , "EVL " , "BACE2 " , "MIF " , "PIH1D1 " , "CAB39 " , "IARS2 " , "DSG2 " , "KLHDC2 " , "BAG3 " , "CNPY3 " , "LAP3 " , "STUB1 " , "NOSIP " , "ENOPH1 " , "HSD17B11 " , "SQRDL " , "MRPS2 " , "NUP85 " , "FIS1 " , "NUSAP1 " , "MRPS16 " , "UFM1 " , "NT5DC2 " , "AKAP8L " , "NPDC1 " , "ANKRD10 " , "DERA " , "TEX10 " , "UBR7 " , "TMEM2 " , "TRIB3 " , "ZNF395 " , "ADCK3 " , "ISOC1 " , "CCDC92 " , "GOLT1B " , "SCAND1 " , "NR1H2 " , "TSKU " , "ZDHHC6 " , "PLEKHJ1 " , "PRKAG2 " , "TIMM9 " , "SESN1 " , "GMNN " , "CRELD2 " , "NUDT9 " , "CDCA4 " , "NENF " , "CERK " , "DNAJC15 " , "HEBP1 " , "DNMT3A " , "KCTD5 " , "ERO1A " , "CD320 " , "DHDDS " , "CHMP4A " , "ABHD4 " , "TCTN1 " , "HEATR1 " , "CISD1 " , "SUV39H1 " , "VPS28 " , "EXOSC4 " , "NARFL " , "CHMP6 " , "PACSIN3 " , "KIF20A " , "HOOK2 " , "TXNL4B " , "VAV3 " , "SLC35F2" , "DUSP22 " , "IGF2BP2 " , "PPP2R3C " , "TNFRSF21 " , "FAM57A " , "NOTCH1 " , "ANO10 " , "PNKP " , "EDN1 " , "FASTKD5 " , "METRN " , "LAGE3 " , "PXMP2 " , "AMDHD2 " , "PRR15L " , "FSD1 " , "TIMM22 " , "FBXO11 " , "RBKS " , "CHAC1 " , "MSRA " , "HERC6 " , "MTERF3 " , "ADAT1 " , "FKBP14 " , "PAK6 " , "PSMD10 " , "CHIC2 " , "LRRC16A " , "TSEN2 " , "ZNF586 " , "PRR7 " , "GFOD1 " , "SPAG4 " , "MCOLN1 " , "ZNF589 " , "SLC2A6 " , "MCUR1 " , "FBXL12 " , "SNX11 " , "FAIM " , "GTPBP8 " , "TLR4 " , "DENND2D " , "PECR " , "ARID4B " , "FRS2 " , "ITFG1 " , "BNIP3L " , "ARPP19 " , "ATG3 " , "UTP14A " , "WDR61 " , "EIF4EBP1 " , "GRWD1 " , "ABHD6 " , "SIRT3 " , "NOL3 " , "STAP2 " , "ACOT9 " , "CANT1 " , "YTHDF1 " , "HMGCS1 " , "MICALL1 " , "FAM63A " , "ATF1 " , "P4HTM " , "SLC27A3 " , "TBXA2R" , "RTN2 " , "TSTA3 " , "PPARD " , "GNA11 " , "WDTC1 " , "PLSCR3 " , "NPEPL1"]

print("Creating Gene Vector...")
#For each gene find the transcription values vector
GeneVector={}
gene_index=0
for gene_index in range(978):
    #print(gene_index)
    GeneVector[gene_index] = []
    for key, values in Set.items():
        if key not in test_list:
            GeneVector[gene_index].append(float(values[gene_index]))

#According to the Experimental Evidence the following genes are found to have score above the threshold(900)
AverageOver = [{2, 9},
 {5, 20, 181},
 {8, 620},
 {31, 91, 682},
 {38, 129},
 {57, 784},
 {75, 108, 209, 275, 416, 450},
 {112, 446, 545},
 {118, 647},
 {128, 145},
 {132, 341},
 {195, 202},
 {210, 241, 481},
 {255, 328, 523,692},
 {256, 766},
 {265, 401, 444},
 {379, 667},
 {394, 845,849},
 {421, 954},
 {453, 750},
 {590, 615},
 {606, 801}]

#Gene Vector 2 is the adjusted version of the Gene Vector using the external information (Experimental Evidence)
GeneVector2 = GeneVector.copy()
new_ind = -1
for each in AverageOver:
    average = [0.0 for j in range(len(GeneVector[0]))]
    for i in each:
        average = [a + b for a, b in zip(average, GeneVector[i])]
        del GeneVector2[i]
    #print(average)
    average = [a*1.0/len(each) for a in average] 
    GeneVector2[new_ind] =average
    new_ind=new_ind-1

#Calculating the Gene List based on the external information
from sklearn.cluster import KMeans
import numpy as np
level = cluster_level
GeneList2 = [GeneVector2]
newGeneList2 = []
for i in range(level):
    for j, gene_vector in enumerate(GeneList2):
        data = list(gene_vector.values())
        an_array = np.array(data)
        if(len(an_array)>1):
            #print("Kmeans, level " + str(i)+ " Gene List index "+str(j))
            kmeans = KMeans(n_clusters=2, random_state=cluster_seed).fit(an_array)
            #Genes
            Genes0 = {}
            Genes1 = {}
            for index, label in enumerate(kmeans.labels_):
                if label==1:
                    Genes1[list(gene_vector.keys())[index]] = gene_vector[list(gene_vector.keys())[index]]
                else:
                    Genes0[list(gene_vector.keys())[index]] = gene_vector[list(gene_vector.keys())[index]]
            newGeneList2.append(Genes0)
            newGeneList2.append(Genes1)
        else:
            newGeneList2.append(gene_vector)
    print("New Gene List has " +str(len(newGeneList2))+" clusters.")
    GeneList2 = newGeneList2
    newGeneList2 = []

len_=0
for i, each in enumerate(GeneList2):
    len_ += len(each)
    print("Cluster "+str(i+1)+" length: ", len(list(each.keys())), list(each.keys()))

## Replace the indices back
for set_of_genes in GeneList2:
    copy = set_of_genes.copy()
    for key, value in copy.items():
        if key < 0 : 
            #print(key)
            del set_of_genes[key]
            index = (-1)*key-1
            for k in AverageOver[index]:
                set_of_genes[k] = GeneVector[k]
#len_=0
SetList2 = []
BinaryMatrixList= []
GeneInteractionList = []
for i, each in enumerate(GeneList2):
    #len_ += len(each)
    SetCluster = {}
    GeneInter ={}
    BinaryMatrix900 = torch.zeros(len(list(each.keys())), len(list(each.keys())), dtype=torch.float)
    for every in AverageOver:
        if list(every)[0] in list(each.keys()):
            for e in every:
                for v in every:
                    if e!=v:
                        BinaryMatrix900[list(each.keys()).index(e)][list(each.keys()).index(v)]= -1
                        GeneInter[(list(each.keys()).index(e),list(each.keys()).index(v))] = 1 
    for j in list(each.keys()):
        for key, values in Set.items():
            SetCluster.setdefault( key, [] ).append( values[j] )
    for key in SetCluster.keys():
        SetCluster[key]= torch.FloatTensor(SetCluster[key])   
    print("Cluster "+str(i+1)+" length: ", list(each.keys()))
    GeneInteractionList.append(GeneInter)
    BinaryMatrixList.append(BinaryMatrix900)
    SetList2.append(SetCluster)

#Print the number of genes that are interacting within that cluster
print("Number of Interacting Genes per Cluster")
for i,each in enumerate(GeneInteractionList):
    print("Cluster " +str(i)+" : ",len(each.keys()))

#Create the Graph Laplacian.
           
for BM in BinaryMatrixList:
    #Create the Graph Laplacian
    i_range= BM.shape[0]
    j_range= BM.shape[1]
    for i in range(i_range):
        for j in range(j_range):
            if (i==j):
                BM[i][j]=-1*torch.sum(BM[i])
               
#TRAINING 
print("Training is starting...")

General= []
GenesPredicted = []
GenesActual = []
Time = []
General_test_error = []
General_test_norm_square = []
count=0
ImputedTensors = []
for each_set in SetList2:
    #STOCHASTIC GRADIENT DESCENT Diminishing step size (1/t) actually (1/100t)
    count = count+1
    print("Cluster ",count)
    start = time.time()
    start_clock = time.process_time()

    # storing the arguments
    alpha_= 0.01
    rank = 500

    tf_sgd_constant = TF_SGD_External(each_set,rank, alpha_, 120,2, train_list, test_list,GeneInteractionList[count-1])
    general_sgd_constant, time_sgd_constant, test_error, test_norm = tf_sgd_constant.train(1,0) #, [(0, seed) for seed in range(2)]) #0 mood SGD, 1 mood AdaGrad, 2 mood ADAM
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

total_clusters = 2**(cluster_level)
import math
error = 0
norm = 0
for i in range(total_clusters):
    error += General_test_error[i][-1]
    norm +=General_test_norm_square[i][-1]
print(math.sqrt(error)/math.sqrt(norm))

print("Test Error in Every Iteration Min :")

error = 0
norm = 0
c = 0
for i in range(total_clusters):
    min_value = min(General_test_error[i])
    min_index = General_test_error[i]. index(min_value)
    error += General_test_error[i][min_index]
    norm += General_test_norm_square[i][min_index]
print(math.sqrt(error)/math.sqrt(norm))


print("Calculating the Imputed Tensor : ")


ImputedTensorSeed = {}

for key, value in Set.items():
    
    ImputedTensorSeed[key] = torch.zeros(978)
    
    for i, each in enumerate(GeneList2):
        #i is the cluster each is the cluster gene indices
        gene_indices = list(each.keys())
        for j, index in enumerate(gene_indices):
            #j iterative on the list of gene indices
            ImputedTensorSeed[key][index] = ImputedTensors[i][key][j]

# define dictionary
# Set

# create a binary pickle file 
f = open("ImputedTensorWithExternalSeed"+str(cluster_seed)+"Level"+str(cluster_level)+".pkl","wb")

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
