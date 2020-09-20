"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import os
import csv
import numpy
from sklearn import preprocessing  
import urllib

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def load_data(path,type_,size,dataset): # loading the data
    reg = np.random.rand(400, 35, 35)    # here put view1
    mal = np.random.rand(400, 35, 35)  # here put view 2
    data = np.zeros((reg.shape[0], reg.shape[1], reg.shape[2], 2))
    for i in range(reg.shape[0]):
        data[i, :, :, 0] = reg[i]
        data[i, :, :, 1] = mal[i]
    return data

def load_data1(size,dataset):
    reg = np.random.rand(400, 35, 35)
    data = np.zeros((reg.shape[0], reg.shape[1], reg.shape[2], 2))
    for i in range(reg.shape[0]):
        data[i, :, :, 0] = reg[i]
        data[i, :, :, 1] = reg[i]
    return data


def load_data_test(size,dataset):


    reg = np.load('./Vtest1.npy')

    data = np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
    for i in range(reg.shape[0]):
        data[i,:,:,0]=reg[i]
        data[i,:,:,1]=reg[i]
        return data



