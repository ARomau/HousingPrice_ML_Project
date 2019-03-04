#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 18:06:44 2019

@author: stellakim
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/
test_data = pd.read_csv("data/test.csv")
test_data = test_data.drop("Id", axis = 1)

test_data.shape

test_data.dtypes

# Columns with missing data
test_data.columns[test_data.isnull().any(axis = 0)]

# Missing MSZoning filled with most common "RL"
test_data['MSZoning'].fillna("RL", inplace = True)

## Filling in "NaN" in Alley with "None"
test_data['Alley'].fillna("NA", inplace = True)

## Filling in missing Utilities with AllPub (most common)
test_data['Utilities'].value_counts()
test_data['Utilities'].fillna("AllPub", inplace = True)

## Filling in with most common values
test_data['Exterior1st'].value_counts()
test_data['Exterior1st'].fillna("VinylSd", inplace = True)

test_data['Exterior2nd'].value_counts()
test_data['Exterior2nd'].fillna("VinylSd", inplace = True)

# MasVnrType and MasVnrArea: assume "None" and "0"
test_data['MasVnrType'].fillna("NA", inplace = True)
test_data['MasVnrArea'].fillna(0, inplace = True)

# BsmtQual: None
test_data['BsmtQual'].fillna("NA", inplace = True)

# BsmtCond: None
test_data[test_data['BsmtCond'].isna()].index
test_data[test_data['TotalBsmtSF'] == 0].index
test_data['BsmtCond'].fillna("NA", inplace = True)

# BsmtExposure: None
test_data['BsmtExposure'].fillna("NA", inplace = True)

# BsmtFinType1: None
test_data['BsmtFinType1'].fillna("NA", inplace = True)

# BsmtFinType2: None
test_data['BsmtFinType2'].fillna("NA", inplace = True)

# No basement
test_data['BsmtFinSF1'].fillna(0, inplace = True)

# No basement
test_data['BsmtFinSF2'].fillna(0, inplace = True)

# No basement
test_data['BsmtUnfSF'].fillna(0, inplace = True)

# No basement
test_data['TotalBsmtSF'].fillna(0, inplace = True)

# No basement
test_data['BsmtFullBath'].fillna(0, inplace = True)

# No basement
test_data['BsmtHalfBath'].fillna(0, inplace = True)

# No kitchen
test_data['KitchenQual'].fillna("NA", inplace = True)

# Functional: fill mode
test_data['Functional'].fillna("Typ", inplace = True)

# No fireplace
test_data['FireplaceQu'].fillna("NA", inplace = True)
test_data['Fireplaces'].value_counts()


# No garage
test_data['GarageArea'].value_counts()
test_data['GarageType'].fillna("NA", inplace = True)
test_data['GarageYrBlt'].fillna("NA", inplace = True)
test_data['GarageFinish'].fillna("NA", inplace = True)
test_data['GarageQual'].fillna("NA", inplace = True)
test_data['GarageCond'].fillna("NA", inplace = True)
test_data['GarageCars'].fillna(0, inplace = True)
test_data['GarageArea'].fillna(0, inplace = True)

# No pool
sum(test_data['PoolQC'].isna())
test_data['PoolArea'].value_counts()
test_data['PoolQC'].fillna("NA", inplace = True)

# No fence
test_data['Fence'].fillna("NA", inplace = True)

# Fill in MiscFeature
test_data['MiscFeature'].fillna("NA", inplace = True)

# Fill in with most common SaleType
test_data['SaleType'].fillna("WD", inplace = True)




#################### Filling in LotFrontage with log(LotArea) #################
#We can impute the missing data with the log-transform of lot area, correlation ~0.65
test_data[['LotFrontage','LotArea']].corr()
pd.concat([test_data['LotFrontage'], np.log(test_data['LotArea'])], axis = 1).corr()

test_data['LotFrontage'] = np.log(test_data['LotArea'])