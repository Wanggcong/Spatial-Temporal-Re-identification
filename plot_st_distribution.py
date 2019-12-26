import scipy.io
import torch
import numpy as np
import time
import os
import math

import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

def gaussian_func(x, u, o=50):
    if (o == 0):
        print("In gaussian, o shouldn't equel to zero")
        return 0
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(math.pow(x - u, 2)) / (2 * math.pow(o, 2))
    return temp1 * math.exp(temp2)

def gaussian_func2(x, u, o=50):
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(o, 2))
    return temp1 * np.exp(temp2)

def gauss_smooth(arr):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i
    # gaussian_vect= gaussian_func2(vect,0,1)
    gaussian_vect= gaussian_func2(vect,0,50)
    matrix = np.zeros((hist_num,hist_num))
    # matrix = np.eye(hist_num)
    for i in range(hist_num):
        for j in range(i,hist_num):
            matrix[i][j]=gaussian_vect[j-i]    
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2
    # for i in range(hist_num):
    #     for j in range(i):
    #         matrix[i][j]=gaussian_vect[j]     
    xxx = np.dot(matrix,arr)
    return xxx

# faster gauss_smooth
def gauss_smooth2(arr):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i
    # gaussian_vect= gaussian_func2(vect,0,1)
    o=5
    approximate_delta = 6*o     #  when x-u>approximate_delta, e.g., 6*o, the gaussian value is approximately equal to 0.
    gaussian_vect= gaussian_func2(vect,0,o)
    matrix = np.zeros((hist_num,hist_num))
    for i in range(hist_num):
        k=0
        for j in range(i,hist_num):
            if k>approximate_delta:
                continue
            matrix[i][j]=gaussian_vect[j-i] 
            k=k+1  
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2
    # for i in range(hist_num):
    #     for j in range(i):
    #         matrix[i][j]=gaussian_vect[j]     
    xxx = np.dot(matrix,arr)
    return xxx


result2 = scipy.io.loadmat('model/'+'ft_ResNet50_market_pcb'+'/'+'pytorch_result2.mat')
distribution = result2['distribution']

#############################################################
for i in range(0,8):
    for j in range(0,8):
        print("gauss "+str(i)+"->"+str(j))
        # gauss_smooth(distribution[i][j])
        distribution[i][j][:]=gauss_smooth2(distribution[i][j][:])


eps = 0.0000001
sum_ = np.sum(distribution,axis=2)
for i in range(1):
    for j in range(8):
        distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)   

# plot:

for i in range(1):
    for j in range(8):
        one_distr=distribution[i][j][:]
        x=range(0,len(one_distr))
        plt.figure(1)
        plt.plot(x, one_distr)
        # plt.savefig(str(i)+'-'+str(j)+'.jpg')
        # plt.show()
    # plt.axis([0, 3000, 0, 0.05])
    plt.xlim((0,1000))
    plt.show()
