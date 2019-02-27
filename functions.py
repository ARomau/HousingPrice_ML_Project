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