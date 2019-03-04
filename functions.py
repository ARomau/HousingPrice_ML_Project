#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:30:13 2019

@author: stellakim
"""
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))


def standardize(input):
    return (input - input.mean()) / input.std()
    
    
def unstandardize(input):
    return (input*train_fe[input.columns].std() + train_fe[input.columns].mean())


def rmse_kaggle(actual, pred):
    return np.sqrt(np.mean((np.log(actual) - np.log(pred))**2))