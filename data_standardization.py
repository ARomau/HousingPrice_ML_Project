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
import pandas as pd

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
#normtesttab(train_fe['SalePrice'])

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
xt, maxlog, interval = stats.boxcox(train_fe['GarageScore'] + abs(min(train_fe['GarageScore']))+1, alpha=0.05)
print("lambda = {:g}".format(maxlog))
# Power transform with 0.75
train_fe['GarageScore'] = (train_fe['GarageScore'] + abs(min(train_fe['GarageScore']))+1)**0.75

xt, maxlog, interval = stats.boxcox(train_fe['TotalSF'], alpha=0.05)
print("lambda = {:g}".format(maxlog))
# Power transform with 0.25
train_fe['TotalSF'] = (train_fe['TotalSF'] + abs(min(train_fe['GarageScore']))+1)**0.25

xt, maxlog, interval = stats.boxcox(train_fe['SalePrice'], alpha=0.05)
print("lambda = {:g}".format(maxlog))
# Power transform with -0.5
train_fe['SalePrice'] = (train_fe['SalePrice'] + abs(min(train_fe['GarageScore']))+1)**-0.1


###############################################################################
############################# STANDARDIZATION #################################
###############################################################################

to_scale = train_fe.drop(columns = ['SalePrice'])
colnames = to_scale.columns
scaler = StandardScaler()
scaler.fit(to_scale)
scaled_features = pd.DataFrame(scaler.transform(to_scale),
             columns = to_scale.columns)
scaler2 = StandardScaler()
scaler2.fit(train_fe[['SalePrice']])
scaled_price = pd.DataFrame(scaler2.transform(train_fe[['SalePrice']]),
                            columns = ['SalePrice'])

scaled_features = pd.concat([scaled_features, scaled_price], axis = 1)


#scaled_features.to_csv("data/cleaned_standardized_fe.csv", index = False)




# this is just for the test set, to get the same # of columns to transform

#train.columns.tolist()
#dir(train.columns)

#train.columns.where(~train.columns.isin(train_fe.columns)).dropna()
#housestyle = pd.DataFrame(np.zeros(len(test)), columns = ["HouseStyle_2.5Fin"])
#train_fe = pd.concat([train_fe, housestyle], axis = 1)
#train_fe = train_fe.reindex(columns = colnames)
#
#
#scaled_features = pd.DataFrame(scaler.transform(train_fe),
#             columns = train_fe.columns)
#
#
##scaled_features.hist(figsize=(25, 25))
#scaled_features.to_csv("data/cleaned_standardized_TEST.csv", index = False)

