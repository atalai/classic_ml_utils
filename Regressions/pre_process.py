#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Various pre-processing tools

# def libs
import os
import scipy 
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

# def func
def scale_data_row(X, lower_scale, upper_scale):
    '''Takes an N*M data and scales each row based on lower and upper scale and returns
        said array'''
    data = [np.interp(X[i], (X[i].min(), X[i].max()), (lower_scale, upper_scale)) for i in range(0,X.shape[0])]
    return np.array(data).astype('float32')

def normalize_data_row(X):
    '''Takes an N*M data and centers it around 0, then divide by the max to scale it to [−1,1].'''
    data = [(X[i] - X[i].mean()) / (X[i] - X[i].mean()).max() for i in range(0,X.shape[0])]
    return np.array(data).astype('float32')

def l1_norm(X):
    '''Takes an N*M data and makes each element in a row add up to 1. it's
    just adding all the values in the array and dividing each of it using the sum.'''
    data = [normalize(X[i].reshape(1, -1), norm="l1") for i in range(0,X.shape[0])]
    return np.array(data).astype('float32')

def l2_norm(X):
    '''Takes an N*M data and divides each element in a row by δ. Where δ is the square root of the sum of all the squared values in that row.'''
    data = [normalize(X[i].reshape(1, -1), norm="l2") for i in range(0,X.shape[0])]
    return np.array(data).astype('float32')

def map_2_uniform(X):
    '''Maps N*M data from any distribution to as close to a G a uniform distribution with values between 0 and 1'''
    quantile_transformer = QuantileTransformer(random_state=1993)
    data = [quantile_transformer.fit_transform((X[i].reshape(1, -1))) for i in range(0,X.shape[0])]
    return np.array(data).astype('float32')

def map_2_gaussian(X, mapping_method):
    '''Maps N*M data from any distribution to as close to a Gaussian distribution as possible in order to stabilize variance and minimize skewness.
    mapping method either 'box-cox' or 'yeo-johnson, standardize = False will apply zero-mean, unit-variance normalization to the transformed output by default.'''
    pt = PowerTransformer(method=mapping_method, standardize=False)
    data = [pt.fit_transform(X[i].reshape(1, -1)) for i in range(0,X.shape[0])]
    return np.array(data).astype('float32')
