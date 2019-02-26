#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:50:48 2019

@author: stellakim
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/
train_data = pd.read_csv("data/train.csv")
train_data = train_data.drop("Id", axis = 1)

train_data.shape

train_data.dtypes

train_data.isnull().any(axis = 0)
train_data.isnull().any(axis = 1)

# Columns with missing data
train_data.columns[train_data.isnull().any(axis = 0)]
train_data.index[train_data.isnull().any(axis = 1)]


## MISSINGNESS IN COLUMNS
train_data.isnull().sum(axis = 0).sort_values(ascending = False)
# Percent of missingness in each column
train_data.isnull().sum(axis = 0).sort_values(ascending = False)/train_data.shape[0]

## MISSINGNESS IN ROWS
train_data.isnull().sum(axis = 1).sort_values(ascending = False)
# Percent of missingness in each row
train_data.isnull().sum(axis = 1).sort_values(ascending = False)/train_data.shape[1]


train_data.columns
train_data.columns[train_data.isnull().any(axis = 0)]
## Filling in "NaN" in Alley with "None"
train_data['Alley'].fillna("NA", inplace = True)

# MasVnrType: NaN values are the same for type and area -- assume "None" and "0"
train_data[train_data['MasVnrType'].isna()].index
train_data[train_data['MasVnrArea'].isna()].index
train_data['MasVnrType'].fillna("NA", inplace = True)
train_data['MasVnrArea'].fillna(0, inplace = True)

# Empty BsmtQual = no basement (0 sq ft)
train_data[train_data['BsmtQual'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
train_data['BsmtQual'].fillna("NA", inplace = True)

# Empty BsmtCond = no basement (0 sq ft)
train_data[train_data['BsmtCond'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
train_data['BsmtCond'].fillna("NA", inplace = True)


# No basement
train_data[train_data['BsmtFinType1'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
train_data['BsmtFinType1'].fillna("NA", inplace = True)

# No basement
train_data[train_data['BsmtFinType1'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
train_data['BsmtFinType1'].fillna("NA", inplace = True)

# No fireplace
train_data['FireplaceQu'].fillna("NA", inplace = True)
train_data['Fireplaces'].value_counts()


# No garage
train_data['GarageArea'].value_counts()
train_data['GarageType'].fillna("NA", inplace = True)
train_data['GarageYrBlt'].fillna("NA", inplace = True)
train_data['GarageFinish'].fillna("NA", inplace = True)
train_data['GarageQual'].fillna("NA", inplace = True)
train_data['GarageCond'].fillna("NA", inplace = True)

# No pool
sum(train_data['PoolQC'].isna())
train_data['PoolArea'].value_counts()
train_data['PoolQC'].fillna("NA", inplace = True)

# No fence
train_data['Fence'].fillna("NA", inplace = True)


# Fill in MiscFeature
train_data[train_data['MiscFeature'].isna()].index
train_data[train_data['MiscVal'] == 0].index
pd.Index.difference(train_data[train_data['MiscVal'] == 0].index,
        train_data[train_data['MiscFeature'].isna()].index)
train_data.iloc[873][['MiscVal', 'MiscFeature']]
train_data.iloc[1200][['MiscVal', 'MiscFeature']]
train_data['MiscFeature'].fillna("NA", inplace = True)


# We decided to fill in the onemissing "Electrical" value with "SBrkr" since this is the most common one
train_data['Electrical'].value_counts()
train_data['Electrical'].fillna("SBrkr", inplace = True)

# Empty BsmtQual = no basement (0 sq ft)
# Index 948 "has a basement" but we will just fill this with "None" for simplicity
train_data['BsmtExposure'].value_counts()
train_data['BsmtExposure'].fillna("NA", inplace = True)

# For indices wtih no basement, we fill in BsmtFinType2 with "None"
# Index 332 "has a basement" but we will just fill this with "None" for simplicity
train_data['BsmtFinType2'].value_counts()
train_data['BsmtFinType2'].fillna("NA", inplace = True)



#################### Filling in LotFrontage with log(LotArea) #################
#We can impute the missing data with the log-transform of lot area, correlation ~0.65
train_data[['LotFrontage','LotArea']].corr()
pd.concat([train_data['LotFrontage'], np.log(train_data['LotArea'])], axis = 1).corr()

train_data['LotFrontage'] = np.log(train_data['LotArea'])


#import seaborn as sns
#import matplotlib.pyplot as plt

#np.std(train_data['LotFrontage'])

#sns.regplot('LotFrontage', np.log(train_data['LotArea']), data = train_data)




###############################################################################
########################## KNN IMPUTATION #####################################
###############################################################################

# First, standardize our dataset
num_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF',
 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']

num_features = train_data[num_features]
scaler = StandardScaler()
scaled_LF = pd.DataFrame(scaler.fit_transform(pd.DataFrame(num_features['LotFrontage'][num_features['LotFrontage'].notna()])),
             columns = ['LotFrontage'],
             index = num_features[num_features['LotFrontage'].notna()].index)
na_LF = pd.DataFrame(num_features['LotFrontage'][num_features['LotFrontage'].isna()],
                                  columns = ['LotFrontage'],
                                  index = num_features[num_features['LotFrontage'].isna()].index)
scaled_LF = scaled_LF.append(na_LF).sort_index()

scaled_rest = pd.DataFrame(scaler.fit_transform(num_features.loc[:, "LotArea":]),
             columns = ['LotArea', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF',
 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice'])

scaled_num_features = pd.concat([scaled_LF, scaled_rest], axis = 1)

# Replacing train data with scaled train data
for col in scaled_num_features.columns:
    if col in train_data.columns:
        train_data[col] = scaled_num_features[col]

del (col, num_features, scaled_LF, na_LF, scaled_rest, scaled_num_features)


train_data.columns
train_data.columns[train_data.isnull().any(axis = 0)]