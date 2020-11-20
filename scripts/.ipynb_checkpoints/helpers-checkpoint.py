# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import pandas as pd

def load_csv_data(data_path_features, data_path_labels, K):
    """Loads data.
    return
        y(class labels), tX (features) and ids (event ids).
    """
    y = np.genfromtxt(
        data_path_labels, delimiter=",", skip_header=1, dtype=str, usecols=2)
    x = np.genfromtxt(
        data_path_features, delimiter=",", skip_header=1)
    
    #removing the undisered features 
    if (len(K) > 0):
        for k in K:
            X=np.delete(x, k, axis=1)
    return X , y

def analyse(X):
    feature_details = np.zeros([7, X.shape[1]])
    for i in range(X.shape[1]):
        feature_details[0, i] = np.nanmean(X[:,i])
        feature_details[1, i] = np.nanvar(X[:,i])
        feature_details[2, i] = np.nanstd(X[:,i])
        feature_details[3, i] = np.nanmin(X[:,i])
        feature_details[4, i] = np.nanmax(X[:,i])
        feature_details[5, i] = np.isnan(X[:,i]).sum()
        feature_details[6, i] = np.nanmedian(X[:,i])

    #print(feature_details)
    df = pd.DataFrame(feature_details)
    df.index = ['Mean', 'Variance', 'Std', 'min', 'max', 'n-NaNs', 'median']
    df.index.name = 'Statistics'
    return df

def standardize(X):
    feature_details = np.zeros([7, X.shape[1]])
    for i in range(X.shape[1]):
        feature_details[0, i] = np.nanmean(X[:,i])
        feature_details[1, i] = np.nanvar(X[:,i])
        feature_details[2, i] = np.nanstd(X[:,i])
        feature_details[3, i] = np.nanmin(X[:,i])
        feature_details[4, i] = np.nanmax(X[:,i])
        feature_details[5, i] = np.isnan(X[:,i]).sum()
        feature_details[6, i] = np.nanmedian(X[:,i])

    for k in range (X.shape[1]):
        X[:,k]-=feature_details[0,k]
        X[:,k]/=feature_details[2,k]
                 
    return feature_details


    
    

