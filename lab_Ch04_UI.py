#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:42:29 2021

@author: umreenimam
"""

"""
IMPORTING PACKAGES
"""
import os
import math
import numpy as np 
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

from pyeeg import bin_power, spectral_entropy
from docx import Document
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

"""
Global variables
"""
bands = [0.5, 4, 7, 12, 30]
fs = 1024
array_length = 1024
state_labels = ['Pre', 'Med', 'Post']

"""
Question 1 - Load the data in each of the Matlab files into a data frame
"""
# Setting file names
os.chdir('/Users/umreenimam/Documents/Masters/Masters_Classes/CS_5310/Week_4/lab_ch4')
prefile = 'Pre(2).mat'
medfile = 'Med(2).mat'
postfile = 'Post(2).mat'

# Loading data 
predata = loadmat(prefile)
meddata = loadmat(medfile)
postdata = loadmat(postfile)

pre = predata['data']
med = meddata['data']
post = postdata['data']

# Transpose data and create dataframes 
pre_transposed = pd.DataFrame(pre.T)
med_transposed = pd.DataFrame(med.T)
post_transposed = pd.DataFrame(post.T)