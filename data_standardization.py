#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:57:22 2019

@author: stellakim
"""
from scipy import stats
from astropy.table import Table
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

###############################################################################
########################## BOX-COX TRANSFORMATIONS ############################
###############################################################################
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html
#http://dataunderthehood.com/2018/01/15/box-cox-transformation-with-python/
def normtesttab(x):
    nm_value, nm_p = stats.normaltest(x)
    jb_value, jb_p = stats.jarque_bera(x)
    data_rows = [('D’Agostino-Pearson', nm_value, nm_p),
                 ('Jarque-Bera', jb_value, jb_p)]
    t = Table(rows=data_rows, names=('Test name', 'Statistic', 'p-value'), 
              meta={'name': 'normal test table'},
          dtype=('S25', 'f8', 'f8'))
    print(t)

normtesttab(train_fe['GarageScore'])
normtesttab(train_fe['TotalSF'])
normtesttab(train_fe['GrLivArea'])
normtesttab(train_fe['SalePrice'])

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
xt, maxlog, interval = stats.boxcox(train_fe['GarageScore'] + abs(min(train_fe['GarageScore']))+1, alpha=0.05)
print("lambda = {:g}".format(maxlog))
# Power transform with 0.75
train_fe['GarageScore'] = (train_fe['GarageScore'] + abs(min(train_fe['GarageScore']))+1)**0.75

xt, maxlog, interval = stats.boxcox(train_fe['TotalSF'], alpha=0.05)
print("lambda = {:g}".format(maxlog))
# Power transform with 0.25
train_fe['TotalSF'] = (train_fe['TotalSF'] + abs(min(train_fe['GarageScore']))+1)**0.25

xt, maxlog, interval = stats.boxcox(train_fe['GrLivArea'], alpha=0.05)
print("lambda = {:g}".format(maxlog))
# Power transform with -0.25
#train_fe['GrLivArea'] = (train_fe['GrLivArea'] + abs(min(train_fe['GarageScore']))+1)**-0.25

xt, maxlog, interval = stats.boxcox(train_fe['SalePrice'], alpha=0.05)
print("lambda = {:g}".format(maxlog))
# Power transform with -0.5
train_fe['SalePrice'] = (train_fe['SalePrice'] + abs(min(train_fe['GarageScore']))+1)**-0.1


###############################################################################
############################# STANDARDIZATION #################################
###############################################################################

scaler = StandardScaler()

scaled_features = pd.DataFrame(scaler.fit_transform(train_fe),
             columns = train_fe.columns)

#scaled_features.hist(figsize=(25, 25))

import pandas as pd
#train_fe.to_csv("data/cleaned_standardized_fe.csv", index = False)