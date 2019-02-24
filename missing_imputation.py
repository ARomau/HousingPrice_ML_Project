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

# cd Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/
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
train_data['Alley'].fillna("None", inplace = True)

# MasVnrType: NaN values are the same for type and area -- assume "None" and "0"
train_data[train_data['MasVnrType'].isna()].index
train_data[train_data['MasVnrArea'].isna()].index
train_data['MasVnrType'].fillna("None", inplace = True)
train_data['MasVnrArea'].fillna(0, inplace = True)

# Empty BsmtQual = no basement (0 sq ft)
train_data[train_data['BsmtQual'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
train_data['BsmtQual'].fillna("None", inplace = True)

# Empty BsmtCond = no basement (0 sq ft)
train_data[train_data['BsmtCond'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
train_data['BsmtCond'].fillna("None", inplace = True)


# No basement
train_data[train_data['BsmtFinType1'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
train_data['BsmtFinType1'].fillna("None", inplace = True)

# No basement
train_data[train_data['BsmtFinType1'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
train_data['BsmtFinType1'].fillna("None", inplace = True)

# No fireplace
train_data['FireplaceQu'].fillna("None", inplace = True)
train_data['Fireplaces'].value_counts()


# No garage
train_data['GarageArea'].value_counts()
train_data['GarageType'].fillna("None", inplace = True)
train_data['GarageYrBlt'].fillna("None", inplace = True)
train_data['GarageFinish'].fillna("None", inplace = True)
train_data['GarageQual'].fillna("None", inplace = True)
train_data['GarageCond'].fillna("None", inplace = True)

# No pool
sum(train_data['PoolQC'].isna())
train_data['PoolArea'].value_counts()
train_data['PoolQC'].fillna("None", inplace = True)

# No fence
train_data['Fence'].fillna("None", inplace = True)


# Fill in MiscFeature
train_data[train_data['MiscFeature'].isna()].index
train_data[train_data['MiscVal'] == 0].index
pd.Index.difference(train_data[train_data['MiscVal'] == 0].index,
        train_data[train_data['MiscFeature'].isna()].index)
train_data.iloc[873][['MiscVal', 'MiscFeature']]
train_data.iloc[1200][['MiscVal', 'MiscFeature']]
train_data['MiscFeature'].fillna("None", inplace = True)


# We decided to fill in the onemissing "Electrical" value with "SBrkr" since this is the most common one
train_data['Electrical'].value_counts()
train_data['Electrical'].fillna("SBrkr", inplace = True)

# Empty BsmtQual = no basement (0 sq ft)
# For the indices where there is no basement, we filled the missing BsmtQual with "None"
train_data['BsmtExposure'].value_counts()
train_data[train_data['BsmtExposure'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
pd.Index.difference(train_data[train_data['BsmtExposure'].isna()].index,
                              train_data[train_data['TotalBsmtSF'] == 0].index)

indices = train_data[train_data['BsmtExposure'].isna()].index.drop(948).tolist()

for idx in indices:
    train_data.loc[idx, 'BsmtExposure'] = "None"


# For indices wtih no basement, we fill in BsmtFinType2 with "None"
train_data['BsmtFinType2'].value_counts()
train_data[train_data['BsmtFinType2'].isna()].index
train_data[train_data['TotalBsmtSF'] == 0].index
pd.Index.difference(train_data[train_data['BsmtFinType2'].isna()].index,
                               train_data[train_data['TotalBsmtSF'] == 0].index)

indices = train_data[train_data['BsmtFinType2'].isna()].index.drop(332).tolist()

for idx in indices:
    train_data.loc[idx, 'BsmtFinType2'] = "None"

del (idx, indices)



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



# Next, use KNN to impute the missing LotFrontage values
k = round(np.sqrt(sum(train_data['LotFrontage'].notna())))
knn_r = KNeighborsRegressor(n_neighbors = k)

x_columns = ['LotArea']
y_columns = ['LotFrontage']

X = train_data[train_data['LotFrontage'].notna()]['LotArea'].values.reshape(1, -1)
y = train_data[train_data['LotFrontage'].notna()]['LotFrontage'].values.reshape(1,-1)

X = train_data[train_data['LotFrontage'].notna()][x_columns]
y = train_data[train_data['LotFrontage'].notna()][y_columns]

knn_r.fit(X, y)
knn_r.predict(X)
X_predict = np.array(train_data[train_data['LotFrontage'].isna()]['LotArea']).reshape(1,-1)
knn_r.predict(X_predict)








# For the remaining missing basement values, we will be using KNeighborsClassifier and 
# the columns ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
# 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'] to impute the missing value
k = round(np.sqrt(sum(train_data['BsmtExposure'].notna())))
knn_c = KNeighborsClassifier(n_neighbors = k)


## For BsmtExposure
bsmt = train_data[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                   'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]

X = pd.get_dummies(bsmt[bsmt.index != 948], columns = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2'], drop_first = True)
X.drop('BsmtExposure', axis = 1, inplace = True)
y = pd.get_dummies(bsmt[bsmt.index != 948]['BsmtExposure'])

knn_c.fit(X,y)
knn_c.predict(X)


basement_target = train_data.loc[948]['BsmtExposure']
















# For the missing value in LotFrontage, we will be using KNeighborsRegressor
# using the LotFrontage and LotArea



train_data.columns
train_data.columns[train_data.isnull().any(axis = 0)]