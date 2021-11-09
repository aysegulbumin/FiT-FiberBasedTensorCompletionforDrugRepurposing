# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import random
import pandas as pd
import numpy as np
import io
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import torch
import multiprocessing as mp
import time
from sklearn.cluster import KMeans
import numpy as np

pandas2ri.activate()


class TF_SPPA():  # stands for tensor factorization

    def __init__(self, Random, K, l, iterations, mood, training_sample, test_sample):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        self.Random = Random
        self.training_samples = training_sample
        self.test_samples = test_sample
        self.K = K
        self.l = l
        self.iterations = iterations
        self.mood = mood
        self.height = 80
        self.length = 1330
        self.depth = 978

    def train(self, seed):

        # Initialize user and item latent feature matrice
        np.random.seed(seed + 0)
        self.A = np.random.normal(scale=1. / self.K, size=(self.height, self.K))
        np.random.seed(seed + 1)
        self.B = np.random.normal(scale=1. / self.K, size=(self.length, self.K))
        np.random.seed(seed + 2)
        self.C = np.random.normal(scale=1. / self.K, size=(self.depth, self.K))

        np.random.seed(seed + 40)
        self.Ar = np.random.normal(scale=1. / self.K, size=(self.height, self.K))
        np.random.seed(seed + 41)
        self.Br = np.random.normal(scale=1. / self.K, size=(self.length, self.K))
        np.random.seed(seed + 42)
        self.Cr = np.random.normal(scale=1. / self.K, size=(self.depth, self.K))

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        initial_mse = self.mse()
        training_process.append((0, initial_mse))
        start = time.time()
        time_array = []
        time_array.append(0)
        # print("Iteration: %d ; error = %.4f" % (0,initial_mse))
        for i in range(1, self.iterations):
            np.random.shuffle(self.training_samples)
            self.sppa(i)
            print(i, "SPPA done")
            mse = self.mse()
            time_array.append(time.time() - start)
            training_process.append((i, mse))
            # print("Iteration: %d ; error = %.4f" % (i, mse))

        return training_process, time_array

    def mse(self):
        """
        A function to compute the total mean square error
        """
        error = 0
        count = 0
        norm = 0
        for x, y, z, r in self.test_samples:
            count = count + 1
            predicted = np.matmul(np.multiply(self.A[x, :][:], self.B[y, :][:]), self.C[z, :][:])
            error += pow(self.Random[(x, y, z)] - predicted, 2)
            norm += pow(self.Random[(x, y, z)], 2)
        return np.sqrt(error) / np.sqrt(norm)

    def sppa(self, iterations):
        """
        Perform stochastic proximal point algorithm
        """
        count = 0
        if (self.mood == 0):
            print('Lambda:', self.l)
        if (self.mood == 1):
            self.l = float(1 / (iterations + 1))
            print('Lambda:', self.l)
        if (self.mood == 2):
            self.l = float(1 / math.sqrt(iterations))
            print('Lambda:', self.l)
        np.random.shuffle(self.training_samples)
        mini_samples = self.training_samples[:20000]
        for i, j, k, r in mini_samples:

            for m in range(1):
                innerproduct = np.matmul(np.multiply(self.Ar[i, :][:], self.Br[j, :][:]), self.Cr[k, :][:])
                t1 = np.matmul(np.multiply(self.A[i, :][:], self.Br[j, :][:]), self.Cr[k, :][:])
                t2 = np.matmul(np.multiply(self.Ar[i, :][:], self.B[j, :][:]), self.Cr[k, :][:])
                t3 = np.matmul(np.multiply(self.Ar[i, :][:], self.Br[j, :][:]), self.C[k, :][:])

                Ar_i = self.Ar[i, :][:]
                Br_j = self.Br[j, :][:]
                Cr_k = self.Cr[k, :][:]

                ac = np.multiply(Ar_i, Cr_k)
                bc = np.multiply(Br_j, Cr_k)
                ab = np.multiply(Ar_i, Br_j)

                numerator = (2 * innerproduct + self.Random[i, j, k].item()) - (t1 + t2 + t3)
                numerator = self.l * numerator
                denominator = 1 + self.l * (bc @ bc.T + ac @ ac.T + ab @ ab.T)

                [self.Ar[i, :], self.Br[j, :], self.Cr[k, :]] = np.vstack(
                    (self.A[i, :], self.B[j, :], self.C[k, :])) + (float(numerator / denominator)) * np.vstack(
                    (bc, ac, ab))

            [self.A[i, :], self.B[j, :], self.C[k, :]] = [self.Ar[i, :], self.Br[j, :], self.Cr[k, :]]

    def get_rating(self, i, j, k):
        """
        Get the predicted rating of user i and item j
        """
        predicted = np.matmul(np.multiply(self.A[i, :][:], self.B[j, :][:]), self.C[k, :][:])
        return predicted


class TF_SGD():  # stands for tensor factorization

    def __init__(self, Dict, K, alpha, iterations, mode, training_sample, test_sample):
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
        self.mode = mode
        self.mood = 0  # By default it will do SGD
        self.Dict = Dict
        self.training_samples = list(training_sample)
        self.test_samples = list(test_sample)
        self.Error = {}
        self.Genes_predicted = {}
        self.Genes_actual = {}

    def train(self, mood, seed):
        self.mood = mood
        self.height = 80
        self.length = 1330
        key = list(self.Dict.keys())[0]

        self.depth = len(self.Dict[key])

        # Initialize user and item latent feature matrice
        np.random.seed(seed + 0)
        self.A = np.random.normal(scale=1. / self.K, size=(self.height, self.K))
        np.random.seed(seed + 1)
        self.B = np.random.normal(scale=1. / self.K, size=(self.length, self.K))
        np.random.seed(seed + 2)
        self.C = np.random.normal(scale=1. / self.K, size=(self.depth, self.K))

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        time_array = []
        start_ = time.time()
        initial_mse = self.mse()
        inter = 0
        time_array.append(inter)
        training_process.append((0, initial_mse))
        # print("Iteration: %d ; error = %.4f" % (0,initial_mse))
        # for i in range(1,self.iterations):
        i = 0
        while i < 1000:
            i = i + 1
            np.random.shuffle(self.training_samples)
            if (self.mood == 0):  # Traditional SGD
                self.sgd(i)
                print(i, "SGD done")
            elif (self.mood == 1):  # AdaGrad
                self.AdaGrad(i)
                print(i, "AdaGrad done")
            elif (self.mood == 2):  # ADAM
                self.Adam(i)
                print(i, "ADAM done")
            else:
                raise Exception("No such mood is implemented.Please select within range [0,2].")
            if i % 1 == 0:
                mse = self.mse()
                inter = time.time() - start_
                time_array.append(inter)
                training_process.append((i, mse))
                print("Iteration: %d ; error = %.4f" % (i, mse))
                print("Time:", inter)

            if (len(training_process) % 100 == 0):
                with open("./fiberupdateNoExternalwithClustering.txt", 'w') as file_handler:
                    file_handler.write("Iteration " + str(i * 500) + "\n")
                    for item in training_process:
                        file_handler.write("{},".format(item))
                    file_handler.write("\n")

        return training_process, time_array

    def mse(self):
        """
        A function to compute the total mean square error
        """
        summed_error = 0
        count = 0
        summed_norm = 0
        # global Error
        for x, y in self.test_samples:
            norm = 0
            error = 0
            count = count + 1
            Hadamart_ai_bj = torch.mul(torch.tensor(self.A[x, :][:]), torch.tensor(self.B[y, :][:]))
            Fiber_actual = self.Dict[(x, y)]
            # print("Matrix C ", torch.tensor(self.C))
            # print("Hadamart Shape", Hadamart_ai_bj.shape)
            # print("Fiber Actual", Fiber_actual.shape)
            sub = torch.sub(Fiber_actual, torch.tensor(self.C) @ Hadamart_ai_bj, alpha=1)
            error += torch.norm(sub)
            norm += torch.norm(Fiber_actual)
            # print("Fiber diff:",sub.shape)
            # print("Norm", torch.norm(sub))
            # predicted=np.matmul(np.multiply(self.A[x, :][:], self.B[y, :][:]), self.C[z,:][:])
            # error += pow(self.Dict[(x,y,z)] - predicted, 2)
            # norm +=pow(self.Dict[(x,y,z)], 2)
            summed_error = summed_error + error
            summed_norm = summed_norm + norm
            self.Error[(x, y)] = np.sqrt(error.numpy()) / np.sqrt(norm.numpy())
        return np.sqrt(summed_error.numpy()) / np.sqrt(summed_norm.numpy())

    def mse2(self):
        """
        A function to compute the total mean square error
        """
        self.Genes_predicted = {}
        self.Genes_actual = {}
        summed_error = 0
        count = 0
        summed_norm = 0
        # global Error
        index = 0
        for x, y in self.test_samples:
            norm = 0
            error = 0
            count = count + 1
            Hadamart_ai_bj = torch.mul(torch.tensor(self.A[x, :][:]), torch.tensor(self.B[y, :][:]))
            Fiber_actual = self.Dict[(x, y)]
            # print("Matrix C ", torch.tensor(self.C))
            # print("Hadamart Shape", Hadamart_ai_bj.shape)
            # print("Fiber Actual", Fiber_actual.shape)
            predicted = torch.tensor(self.C) @ Hadamart_ai_bj
            self.Genes_predicted[index] = predicted
            self.Genes_actual[index] = Fiber_actual
            sub = torch.sub(Fiber_actual, predicted, alpha=1)
            error += torch.norm(sub)
            norm += torch.norm(Fiber_actual)
            # print("Fiber diff:",sub.shape)
            # print("Norm", torch.norm(sub))
            # predicted=np.matmul(np.multiply(self.A[x, :][:], self.B[y, :][:]), self.C[z,:][:])
            # error += pow(self.Dict[(x,y,z)] - predicted, 2)
            # norm +=pow(self.Dict[(x,y,z)], 2)
            summed_error = summed_error + error
            summed_norm = summed_norm + norm
            self.Error[(x, y)] = np.sqrt(error.numpy()) / np.sqrt(norm.numpy())
            index = index + 1
        return np.sqrt(summed_error.numpy()) / np.sqrt(summed_norm.numpy())

    def sgd(self, iterations):
        """
        Perform stochastic gradient descent
        """
        if (self.mode == 0):
            m = 0
            # print('Alpha:', self.alpha)
        if (self.mode == 1):
            self.alpha = float(1 / (iterations))
            # print('Alpha:', self.alpha)
        np.random.seed(iterations + 0)
        np.random.shuffle(self.training_samples)
        mini_samples = self.training_samples[:1000]
        for i, j in mini_samples:
            for each in range(1):
                # Computer prediction and error
                # prediction = self.get_rating(i, j ,k)
                # e = (r - prediction)

                # Create copy of row of P since we need to update it but use older values for update on Q
                A_i = self.A[i, :][:]
                B_j = self.B[j, :][:]
                # C_k = self.C[k, :][:]

                Ce = self.C

                Hadamart_ai_bj = torch.mul(torch.tensor(self.A[i, :][:]), torch.tensor(self.B[j, :][:]))
                Fiber_actual = self.Dict[(i, j)]
                predicted = torch.tensor(self.C) @ Hadamart_ai_bj
                sub = torch.sub(Fiber_actual, torch.tensor(self.C) @ Hadamart_ai_bj, alpha=1)

                Grad_A = - torch.mul(torch.tensor(self.C).T @ sub, torch.tensor(self.B[j, :][:]))
                # Grad_A2 = - torch.tensor(self.C).T@torch.mul(sub,torch.tensor(self.B[j, :][:]))
                Grad_B = - torch.mul(torch.tensor(self.C).T @ sub, torch.tensor(self.A[i, :][:]))
                # Grad_B2 = - torch.tensor(self.C).T@torch.mul(sub,torch.tensor(self.A[i, :][:]))

                transpose = Hadamart_ai_bj.T
                # print(transpose.unsqueeze(0).shape)
                # print(sub.unsqueeze(1).shape)
                # print(torch.tensor(self.C).shape)
                Grad_C = - sub.unsqueeze(1) @ transpose.unsqueeze(0)
                # print("Gradient A", Grad_A.shape)
                # print("Gradient B", Grad_B.shape)
                # print("Gradient C", Grad_C)
                # ac=np.multiply(A_i,C_k)
                # bc=np.multiply(B_j,C_k)
                # ab=np.multiply(A_i,B_j)

                # innerproduct=np.matmul(np.multiply(self.A[i, :][:], self.B[j, :][:]), self.C[k,:][:])
                # n= innerproduct-self.Dict[(i,j,k)]
                self.A[i, :] -= self.alpha * Grad_A.numpy()
                self.B[j, :] -= self.alpha * Grad_B.numpy()
                self.C -= self.alpha * Grad_C.numpy()
                # self.C[k, :] -= self.alpha * ( ab*n)

    def AdaGrad(self, iterations):
        """
        Perform stochastic gradient descent with AdaGrad update
        """
        # self.alpha=0.08
        # self.alpha=float(1/(iterations*10))
        print('Alpha:', self.alpha)

        gti_a = np.zeros(self.K)
        gti_b = np.zeros(self.K)
        gti_c = np.zeros((self.depth, self.K))

        np.random.shuffle(self.training_samples)
        mini_samples = self.training_samples
        for i, j in mini_samples:
            # Create copy of row of P since we need to update it but use older values for update on Q
            A_i = self.A[i, :][:]
            B_j = self.B[j, :][:]
            # C_k = self.C[k, :][:]

            Ce = self.C

            Hadamart_ai_bj = torch.mul(torch.tensor(self.A[i, :][:]), torch.tensor(self.B[j, :][:]))
            Fiber_actual = self.Dict[(i, j)]
            predicted = torch.tensor(self.C) @ Hadamart_ai_bj
            sub = torch.sub(Fiber_actual, torch.tensor(self.C) @ Hadamart_ai_bj, alpha=1)

            Grad_A = - torch.mul(torch.tensor(self.C).T @ sub, torch.tensor(self.B[j, :][:]))
            Grad_B = - torch.mul(torch.tensor(self.C).T @ sub, torch.tensor(self.A[i, :][:]))

            transpose = Hadamart_ai_bj.T
            Grad_C = - sub.unsqueeze(1) @ transpose.unsqueeze(0)

            # ac=np.multiply(A_i,C_k)
            # bc=np.multiply(B_j,C_k)
            # ab=np.multiply(A_i,B_j)

            # innerproduct=np.matmul(np.multiply(self.A[i, :][:], self.B[j, :][:]), self.C[k,:][:])
            # n= innerproduct-self.Dict[(i,j,k)]

            fudge_factor = 1e-6

            # grad_a=( bc*n)
            # print(Grad_A.shape)
            # print(torch.tensor(Grad_A**2).shape)
            gti_a += (Grad_A.numpy() ** 2)
            adjusted_grad_a = Grad_A.numpy() / np.sqrt(fudge_factor + (gti_a))

            # grad_b=(  ac*n)
            gti_b += (Grad_B.numpy() ** 2)
            adjusted_grad_b = Grad_B.numpy() / np.sqrt(fudge_factor + (gti_b))

            # grad_c=(  ab*n)
            gti_c += (Grad_C.numpy() ** 2)
            adjusted_grad_c = Grad_C.numpy() / np.sqrt(fudge_factor + (gti_c))

            self.A[i, :] -= self.alpha * adjusted_grad_a
            self.B[j, :] -= self.alpha * adjusted_grad_b
            self.C -= self.alpha * adjusted_grad_c

    def Adam(self, iterations):
        """
        Perform stochastic gradient descent with AdaGrad update
        """
        self.alpha = 0.001
        # Beta values are exponential decay rates for the function
        beta_1 = 0.9
        beta_2 = 0.999
        e = 1e-8
        # Uncomment the following line for selecting diminishing alpha
        # self.alpha=float(1/(iterations*10))
        # print('Alpha:', self.alpha)

        np.random.shuffle(self.training_samples)
        mini_samples = self.training_samples[:40000]
        for i, j, k, r in mini_samples:
            # Initialize the first moment vector and the second raw moment vector to zero vector
            m1_a = np.zeros(self.K)
            m1_b = np.zeros(self.K)
            m1_c = np.zeros(self.K)

            v1_a = np.zeros(self.K)
            v1_b = np.zeros(self.K)
            v1_c = np.zeros(self.K)

            # Create copy of row of A,B,C since we need to update it but use older values for update on B,C
            A_i = self.A[i, :][:]
            B_j = self.B[j, :][:]
            C_k = self.C[k, :][:]

            ac = np.multiply(A_i, C_k)
            bc = np.multiply(B_j, C_k)
            ab = np.multiply(A_i, B_j)

            innerproduct = np.matmul(np.multiply(self.A[i, :][:], self.B[j, :][:]), self.C[k, :][:])
            n = innerproduct - r
            # Get gradients wrt stochastic objective at timestep t
            gradient_a = bc * n
            gradient_b = ac * n
            gradient_c = ab * n
            # Update biased first moment estimate
            m1_a = beta_1 * (m1_a) + (1 - beta_1) * gradient_a
            m1_b = beta_1 * (m1_b) + (1 - beta_1) * gradient_b
            m1_c = beta_1 * (m1_c) + (1 - beta_1) * gradient_c
            # Update biased second raw moment estimate
            v1_a = beta_2 * (v1_a) + (1 - beta_2) * (gradient_a @ gradient_a.T)
            v1_b = beta_2 * (v1_b) + (1 - beta_2) * (gradient_b @ gradient_b.T)
            v1_c = beta_2 * (v1_c) + (1 - beta_2) * (gradient_c @ gradient_c.T)
            # Compute bias-corrected first moment estimate
            m1_a = m1_a / (1 - (beta_1) ** (iterations))
            m1_b = m1_b / (1 - (beta_1) ** (iterations))
            m1_c = m1_c / (1 - (beta_1) ** (iterations))
            # Compute bias-corrected second moment estimate
            v1_a = v1_a / (1 - (beta_2) ** (iterations))
            v1_b = v1_b / (1 - (beta_2) ** (iterations))
            v1_c = v1_c / (1 - (beta_2) ** (iterations))

            # Update parameters
            self.A[i, :] -= self.alpha * (m1_a / (np.sqrt(v1_a) + e))
            self.B[j, :] -= self.alpha * (m1_b / (np.sqrt(v1_b) + e))
            self.C[k, :] -= self.alpha * (m1_c / (np.sqrt(v1_c) + e))

    def get_rating(self, i, j, k):
        """
        Get the predicted rating of user i and item j
        """
        predicted = np.matmul(np.multiply(self.A[i, :][:], self.B[j, :][:]), self.C[k, :][:])
        return predicted

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    seed=int(sys.argv[1])
    # Read the dataset to a ListVector

    readRDS = ro.r['readRDS']
    df = readRDS('./dataset.rds')
    df = ro.conversion.rpy2py(df)

    # Convert ListVector to Dictionary

    Dict = dict(zip(df.names, map(list, list(df))))

    number_of_nans=0
    counter=0
    Set={}

    for i,rows in enumerate(Dict.values()):
        for j,fiber in enumerate(rows):
            Set[(i,j)]=[]
            counter=0
            for k,value in enumerate(fiber):
                if (np.isnan(value)):
                    if counter==0:
                        Set.pop((i,j))
                        counter=1
                    number_of_nans=number_of_nans+1
                else:
                    Set[(i,j)].append(value)


    print("Number of nans",  number_of_nans)
    print("Set size of ij s", len(Set))
    print("Non zero entries", len(list(Set.keys())))

    for key in Set.keys():
        Set[key]= torch.FloatTensor(Set[key])

    #Create test and training samples with a fixed seed 0
    random.seed(0)

    list_of_keys = list(Set.keys())
    random.shuffle(list_of_keys)

    test_list = list_of_keys[20000:]
    train_list = list(Set.keys())[:20000]

    print("Test data size:", len(test_list))
    print("Training data size:", len(train_list))

    # For each gene find the transcription values vector
    GeneVector = {}
    gene_index = 0
    for gene_index in range(978):
        # print(gene_index)
        GeneVector[gene_index] = []
        for key, values in Set.items():
            if key not in test_list:
                GeneVector[gene_index].append(values[gene_index])

    data = list(GeneVector.values())
    an_array = np.array(data)
    kmeans = KMeans(n_clusters=2, random_state=seed).fit(an_array)


    print("Zero Indices")
    print(np.where(kmeans.labels_ == 0)[0])
    print("One Indices")
    print(np.where(kmeans.labels_ == 1)[0])


    Genes0 = {}
    Genes1 = {}
    for index, label in enumerate(kmeans.labels_):
        if label == 1:
            Genes1[list(GeneVector.keys())[index]] = GeneVector[list(GeneVector.keys())[index]]
        else:
            Genes0[list(GeneVector.keys())[index]] = GeneVector[list(GeneVector.keys())[index]]

    data = list(Genes0.values())
    an_array = np.array(data)

    kmeans0 = KMeans(n_clusters=2, random_state=seed).fit(an_array)
    print("K-means 0")
    print("Zero Indices")
    print(np.where(kmeans0.labels_ == 0)[0])
    print("One Indices")
    print(np.where(kmeans0.labels_ == 1)[0])

    data = list(Genes1.values())

    an_array = np.array(data)

    kmeans1 = KMeans(n_clusters=2, random_state=seed).fit(an_array)

    print(kmeans1.labels_)
    print("K-means 1")
    print("Zero Indices")
    print(np.where(kmeans1.labels_ == 0)[0])
    print("One Indices")
    print(np.where(kmeans1.labels_ == 1)[0])

    Genes00 = {}
    Genes01 = {}
    for index, label in enumerate(kmeans0.labels_):
        if label == 1:
            Genes01[list(Genes0.keys())[index]] = Genes0[list(Genes0.keys())[index]]
        else:
            Genes00[list(Genes0.keys())[index]] = Genes0[list(Genes0.keys())[index]]
    Genes10 = {}
    Genes11 = {}
    for index, label in enumerate(kmeans1.labels_):
        if label == 1:
            Genes11[list(Genes1.keys())[index]] = Genes1[list(Genes1.keys())[index]]
        else:
            Genes10[list(Genes1.keys())[index]] = Genes1[list(Genes1.keys())[index]]


    data = list(Genes00.values())

    an_array = np.array(data)

    kmeans00 = KMeans(n_clusters=2, random_state=seed).fit(an_array)
    print("K-means 00")
    print("Zero Indices")
    print(np.where(kmeans00.labels_ == 0)[0])
    print("One Indices")
    print(np.where(kmeans00.labels_ == 1)[0])

    data = list(Genes01.values())

    an_array = np.array(data)

    kmeans01 = KMeans(n_clusters=2, random_state=seed).fit(an_array)
    print("K-means 01")
    print("Zero Indices")
    print(np.where(kmeans01.labels_ == 0)[0])
    print("One Indices")
    print(np.where(kmeans01.labels_ == 1)[0])

    data = list(Genes10.values())

    an_array = np.array(data)

    kmeans10 = KMeans(n_clusters=2, random_state=seed).fit(an_array)
    print("K-means 10")
    print("Zero Indices")
    print(np.where(kmeans10.labels_ == 0)[0])
    print("One Indices")
    print(np.where(kmeans10.labels_ == 1)[0])

    data = list(Genes11.values())

    an_array = np.array(data)

    kmeans11 = KMeans(n_clusters=2, random_state=seed).fit(an_array)
    print("K-means 11")
    print("Zero Indices")
    print(np.where(kmeans11.labels_ == 0)[0])
    print("One Indices")
    print(np.where(kmeans11.labels_ == 1)[0])

    print("K-means 00")
    zeros = list(kmeans00.labels_).count(0)
    ones = list(kmeans00.labels_).count(1)
    print("Zeros: ",zeros)
    print("Ones: ", ones)
    print("K-means 01")
    zeros = list(kmeans01.labels_).count(0)
    ones = list(kmeans01.labels_).count(1)
    print("Zeros: ", zeros)
    print("Ones: ", ones)
    print("K-means 10")
    zeros = list(kmeans10.labels_).count(0)
    ones = list(kmeans10.labels_).count(1)
    print("Zeros: ", zeros)
    print("Ones: ", ones)
    print("K-means 11")
    zeros = list(kmeans11.labels_).count(0)
    ones = list(kmeans11.labels_).count(1)
    print("Zeros: ",zeros)
    print("Ones: ", ones)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
