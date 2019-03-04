#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:30:26 2019

@author: stellakim
"""

import numpy as np
import pandas as pd
from scipy import stats
from astropy.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
# cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/
###############################################################################
############################### READ IN DATA ##################################
###############################################################################

df = pd.read_csv("data/train.csv")
train_id = df["Id"]
actual_price = df["SalePrice"]
df2 = pd.read_csv("data/test.csv")
test_id = df2["Id"]
submission_id = df2["Id"]
df2 = pd.concat([df2, pd.DataFrame(np.zeros(len(df2)), columns = ['SalePrice'])], axis = 1)
df = df.merge(df2, on = df.columns.tolist(), how = "outer")
df.set_index("Id", inplace = True)
del(df2)



###############################################################################
############################### DATA ANALYSIS #################################
###############################################################################
df.sample(10)
df.shape
df.dtypes
df['MSSubClass'] = df['MSSubClass'].astype(str)

df.describe()
# https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa
# Skewness of the data
df.drop(index = test_id).skew()
df.drop(index = test_id).skew()[(df.drop(index = test_id).skew() > 1) | (df.drop(index = test_id).skew() < -1)]

#Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.
df.drop(index = test_id).kurt()
df.drop(index = test_id).kurt()[df.drop(index = test_id).kurt() > 3]

sns.distplot(df.drop(index = test_id)['SalePrice'])

# Correlation
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
train_correlation = df.drop(index = test_id).corr()

mask = np.zeros_like(train_correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(20, 20))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(train_correlation, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

df.drop(index = test_id)[['SalePrice', 'GrLivArea']].corr()
df.drop(index = test_id)[['SalePrice', 'OverallQual']].corr()

###############################################################################
############################ FILL IN MISSING DATA #############################
###############################################################################
# Columns with missing data
df.columns[df.isnull().any(axis = 0)]

## MISSINGNESS IN COLUMNS
df.isnull().sum(axis = 0).sort_values(ascending = False)
# Percent of missingness in each column
df.isnull().sum(axis = 0).sort_values(ascending = False)/df.shape[0]
missingness = df.isnull().sum(axis = 0).sort_values(ascending = False)/df.shape[0] * 100
missingness = missingness[missingness > 0]
missingness_names = missingness.index.tolist()
sns.barplot(missingness_names, y = missingness)
plt.xticks(rotation=90)
missingness

## MISSINGNESS IN ROWS
df.isnull().sum(axis = 1).sort_values(ascending = False)
# Percent of missingness in each row
df.isnull().sum(axis = 1).sort_values(ascending = False)/df.shape[1]

df.columns
df.columns[df.isnull().any(axis = 0)]

## MSZoning: Fill in with most common value
df['MSZoning'].value_counts()
sum(df['MSZoning'].isna())
df['MSZoning'].fillna("RL", inplace = True)
df.columns[df.isnull().any(axis = 0)]

## Filling in "NaN" in Alley with "None"
df['Alley'].value_counts()
sum(df['Alley'].isna())
df['Alley'].fillna("NA", inplace = True)
df.columns[df.isnull().any(axis = 0)]

## Filling in missing Utilities with AllPub (most common)
df['Utilities'].value_counts()
sum(df['Utilities'].isna())
df['Utilities'].fillna("AllPub", inplace = True)
df.columns[df.isnull().any(axis = 0)]

## Filling in Exterior1st and Exterior2nd with most common values
df['Exterior1st'].value_counts()
sum(df['Exterior1st'].isna())
df['Exterior1st'].fillna("VinylSd", inplace = True)
df['Exterior2nd'].value_counts()
sum(df['Exterior2nd'].isna())
df['Exterior2nd'].fillna("VinylSd", inplace = True)
df.columns[df.isnull().any(axis = 0)]

# MasVnrType: NaN values are the same for type and area -- assume "None" and "0"
mas_cols = ['MasVnrType', 'MasVnrArea', 'Exterior1st', 'Exterior2nd', 'YearBuilt', 'YearRemodAdd', 'Neighborhood']
mason = df[mas_cols]
# One observation where there is MasVnrArea, but no MasVnrType
df[df['MasVnrType'].isna()].index.difference(df[df['MasVnrArea'].isna()].index)
mason.loc[2611]
# Group by the year built, determine the most common
mason[mason['MasVnrType'] != 'None'].groupby(['YearBuilt']).agg(lambda x:x.value_counts().index[0])
# Alternatively, find MasVnrAreas that are within around the same range and find the most common
df[(df['MasVnrArea'] > 175) & (df['MasVnrArea'] < 225)]['MasVnrType'].value_counts()
# In both cases, the most common is "BrkFace"
df.loc[2611, 'MasVnrType'] == "BrkFace"
df['MasVnrType'].value_counts()
df['MasVnrType'].fillna("NA", inplace = True)
df['MasVnrArea'].fillna(0, inplace = True)
df.columns[df.isnull().any(axis = 0)]
del(mason, mas_cols)

# We decided to fill in the onemissing "Electrical" value with "SBrkr" since this is the most common one
df['Electrical'].value_counts()
df['Electrical'].fillna("SBrkr", inplace = True)
df.columns[df.isnull().any(axis = 0)]

# Kitchen
kitchen_cols = ['KitchenAbvGr', 'KitchenQual', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'Neighborhood']
kitchen = df[kitchen_cols]
sum(df['KitchenQual'].isna())
kitchen[kitchen['KitchenQual'].isna()]['Neighborhood']
kitchen.groupby(['Neighborhood']).agg(lambda x:x.value_counts().index[0]).loc['ClearCr']
# Fill with most common KitchenQual value in the neighborhood
df['KitchenQual'].fillna("NA", inplace = True)
df.columns[df.isnull().any(axis = 0)]
del(kitchen ,kitchen_cols)

# Functional: fill mode
df['Functional'].fillna("Typ", inplace = True)
df.columns[df.isnull().any(axis = 0)]

# No fireplace
df['Fireplaces'].value_counts()
sum(df['FireplaceQu'].isna())
df['FireplaceQu'].fillna("NA", inplace = True)
df.columns[df.isnull().any(axis = 0)]

# Pool
pool_cols = ['PoolArea', 'PoolQC', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'Neighborhood']
sum(df['PoolQC'].isna())
pool = df[df['PoolArea'] > 0][pool_cols]
#pool['OverallQual'].hist()
#pool['OverallCond'].hist()
# Looking at the pools column, if another neighborhood has a pool just fill in using the rating found in that neighborhood.
# Neighborhoods NAmes and Mitchel
df.loc[2421, 'PoolQC'] = "Ex"
df.loc[2600, 'PoolQC'] = "Gd"
# For the remaining value, fill in based on the overall condition
pool.groupby(['OverallCond']).agg(lambda x:x.value_counts().index[0])
df.loc[2504,'PoolQC'] = "Ex"
df['PoolArea'].value_counts()
# Fill in remaining with NA
df['PoolQC'].fillna("NA", inplace = True)
df.columns[df.isnull().any(axis = 0)]
del (pool, pool_cols)

# No fence
df['Fence'].value_counts()
df['Fence'].fillna("NA", inplace = True)
df.columns[df.isnull().any(axis = 0)]

# Fill in MiscFeature
df['MiscVal'].value_counts()
df[(df['MiscVal'] != 0) & (df['MiscFeature'].isna())].index
df[df['MiscVal'] != 0]['MiscFeature'].value_counts()
df.loc[2550, 'MiscFeature'] = "Shed"
df['MiscFeature'].fillna("NA", inplace = True)
df.columns[df.isnull().any(axis = 0)]

# Fill in with most common SaleType
sum(df['SaleType'].isna())
df['SaleType'].value_counts()
df['SaleType'].fillna("WD", inplace = True)
df.columns[df.isnull().any(axis = 0)]

# Fill in lot frontage with the median lot frontage according to neighborhood
#We can impute the missing data with the log-transform of lot area, correlation ~0.68
lot_dict = df.groupby("Neighborhood").agg({"LotFrontage": "median"}).to_dict()
df['LotFrontage'] = df['LotFrontage'].fillna(df['Neighborhood'].apply(lambda x: lot_dict['LotFrontage'].get(x)))
#df[['LotFrontage', 'Neighborhood']].groupby('Neighborhood').agg("median")
#df[['LotFrontage','LotArea']].corr()
#pd.concat([df['LotFrontage'], np.log(df['LotArea'])], axis = 1).corr()
#df['LotFrontage'] = np.log(df['LotArea'])
df.columns[df.isnull().any(axis = 0)]

# GARAGE
garage_cols = ['GarageArea', 'GarageType', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'GarageFinish',
               'GarageCars', 'GarageQual', 'GarageCond', 'Neighborhood', 'SalePrice']

df[garage_cols].isna().sum(axis = 0)

garage = df[garage_cols]
df[df['GarageArea'].isna()].index
garage.groupby('GarageType').agg(lambda x: x.value_counts().index[0])
garage.groupby('GarageType').agg({'GarageArea': "median", 'GarageCars': "median",
              'GarageFinish': lambda x:x.value_counts().index[0],
              'GarageQual': lambda x:x.value_counts().index[0],
              'GarageCond': lambda x:x.value_counts().index[0]})
df.loc[2577, garage_cols]
df.loc[2577, 'GarageArea'] = 400
df.loc[2577, 'GarageYrBlt'] = 1923
df.loc[2577, 'GarageFinish'] = "Unf"
df.loc[2577, 'GarageCars'] = 2
df.loc[2577, 'GarageQual'] = "TA"
df.loc[2577, 'GarageCond'] = "TA"

df[garage_cols].isna().sum(axis = 0)

df[df['GarageQual'].isna()].index.difference(df[df['GarageType'].isna()].index)
df.loc[2127, garage_cols]
df.loc[2127, 'GarageYrBlt'] = 1910
df.loc[2127, 'GarageFinish'] = "Unf"
df.loc[2127, 'GarageQual'] = "TA"
df.loc[2127, 'GarageCond'] = "TA"

df[garage_cols].isna().sum(axis = 0)

garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    df[col].fillna("NA", inplace = True)

df['GarageYrBlt'].fillna(0, inplace = True)
for row in df.index:
    if (df.loc[row, 'GarageYrBlt'] == 0):
        df.loc[row, 'GarageYrBlt'] = df.loc[row, 'YearBuilt']

df.columns[df.isnull().any(axis = 0)]
del (garage, garage_cols)

# Basement
# BsmtQual: None
basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','OverallQual', 'OverallCond',
                 'BsmtFullBath', 'BsmtHalfBath', 'Neighborhood']
df[basement_cols].isna().sum(axis = 0)
basement = df[basement_cols]
#for col in basement_cols:
#    print("Number of missing rows in {} is {}".format(col, sum(df[col].isna())))


df[df['BsmtFinSF1'].isna()].index
df.loc[2121, basement_cols]
cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for col in cols:
    df.loc[2121, col] = 0
cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']    
for col in cols:
    df.loc[2121, col] = "NA"    
df.loc[2121, basement_cols]

df[basement_cols].isna().sum(axis = 0)

df[df['BsmtFullBath'].isna()].index
df.loc[2189, basement_cols]
cols = ['BsmtFullBath', 'BsmtHalfBath']
for col in cols:
    df.loc[2189, col] = 0
cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']    
for col in cols:
    df.loc[2189, col] = "NA"  

df[basement_cols].isna().sum(axis = 0)

df[df['BsmtHalfBath'].isna()].index

basement.groupby('BsmtQual').agg(lambda x: x.value_counts().index[0])['BsmtCond']
df[df['BsmtCond'].isna()].index.difference(df[df['BsmtFinType1'].isna()].index)
df.loc[2041, basement_cols]
df.loc[2186, basement_cols]
df.loc[2525, basement_cols]

rows = [2041, 2186, 2525]
for row in rows:
    df.loc[row, 'BsmtCond'] = "TA"

df[basement_cols].isna().sum(axis = 0)

basement.groupby('BsmtCond').agg(lambda x: x.value_counts().index[0])['BsmtQual']
df[df['BsmtQual'].isna()].index.difference(df[df['BsmtFinType1'].isna()].index)
df.loc[2218, basement_cols]
df.loc[2219, basement_cols]

rows = [2218, 2219]
for row in rows:
    df.loc[row, 'BsmtQual'] = "TA"

df[basement_cols].isna().sum(axis = 0)

# Walkout / garden level walls - probably depends on area/neighborhood
df[df['BsmtExposure'].isna()].index.difference(df[df['BsmtFinType1'].isna()].index)
basement.groupby('Neighborhood').agg(lambda x: x.value_counts().index[0])[['BsmtExposure']]
df.loc[949, basement_cols]
df.loc[1488, basement_cols]
df.loc[2349, basement_cols]
df.loc[949, 'BsmtExposure'] = "No"
df.loc[1488, 'BsmtExposure'] = "No"
df.loc[2349, 'BsmtExposure'] = "No"

df[basement_cols].isna().sum(axis = 0)

# Fill with most common value in basement type 2 sqft > 450 and < 500
df[df['BsmtFinType2'].isna()].index.difference(df[df['BsmtFinType1'].isna()].index)
basement
df.loc[333, basement_cols]
df[(df['BsmtFinSF2'] > 450) & (df['BsmtFinSF2'] < 500)]['BsmtFinType2'].value_counts()
df.loc[333, 'BsmtFinType2'] = "Rec"

df[basement_cols].isna().sum(axis = 0)

# Fill remaining vaue
df[basement_cols].isna().sum(axis = 0)
cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in cols:
    df[col].fillna("NA", inplace = True)

df[basement_cols].isna().sum(axis = 0)

df.columns[df.isna().any(axis = 0)]

#######################################################
# Dropping BsmtFinSF1, BsmtFinSF2, BsmtUnfSF - this is captured in TotalSF
df.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis = 1, inplace = True)#######################################################
# Combine 1stFlrSF, 2ndFlrSF, and TotalBsmtSF
df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
df = df.drop(['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'], axis = 1)#######################################################
# Combine BsmtFullBath, BsmtHalfBath, FullBath, HalfBath
df['Baths'] = df['BsmtFullBath'] + df['BsmtHalfBath']*0.5 + df['FullBath'] + df['HalfBath']*0.5
df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis = 1, inplace = True)
# Combine WoodDeckSF, OpenPorchSF, 3SsnPorch, ScreenPorch
df['PorchSF'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['3SsnPorch'] + df['ScreenPorch']
# Drop EnclosedPorch
df.drop('EnclosedPorch', axis = 1, inplace = True)

###############################################################################
########################### SPLIT TRAIN AND TEST ##############################
###############################################################################
train = df.drop(index = test_id)
train = pd.concat([train.drop('SalePrice', axis = 1), train['SalePrice']], axis = 1)
#train.reset_index(drop = True, inplace = True)
test = df.drop(index = train_id)
#test.reset_index(drop = True, inplace = True)


num_features_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'TotalSF', 'GrLivArea',
                'GarageArea', 'PorchSF', 'SalePrice']
num_features = train[num_features_columns]
#num_features.hist(figsize=(15,15))
#num_features['LotFrontage']

# https://seaborn.pydata.org/generated/seaborn.pairplot.html
#sns.pairplot(num_features)
#for column in num_features.columns:
#    if column != "SalePrice":
#        sns.jointplot(x = num_features[column], y = num_features['SalePrice'])

z = np.abs(stats.zscore(num_features))
threshold = 3
# Number of features dropped: 95
#len(num_features[(z > threshold).any(axis=1)].index.tolist())
#num_features[(z > threshold).any(axis=1)].index.tolist()
#train.drop(index = num_features[(z > threshold).any(axis=1)].index, inplace = True)

#sns.pairplot(train[num_features_columns])
#for column in num_features.columns:
#    if column != "SalePrice":
#        sns.jointplot(x = train[column], y = train['SalePrice'])
outliers = []
# Manual removal of outliers
# LotFrontage
#sns.jointplot(x = train['LotFrontage'], y = train['SalePrice'])
train[['LotFrontage']].sort_values('LotFrontage', ascending = False)
train[np.abs(stats.zscore(train['LotFrontage'])) > 3]['LotFrontage'].sort_values(ascending = False)
#sns.jointplot(x = train['LotFrontage'].drop(index = [935, 1299]), y = train['SalePrice'].drop(index = [934,1298]))
outliers.extend([935, 1299])
outliers

# LotArea
#sns.jointplot(x = train['LotArea'], y = train['SalePrice'])
train[['LotArea']].sort_values('LotArea', ascending = False)
train[np.abs(stats.zscore(train['LotArea'])) > 3]['LotArea'].sort_values(ascending = False)
# Remove index 313, 335, 249, 706
sns.jointplot(x = train['LotArea'].drop(index = [314, 336, 250, 707]), y = train['SalePrice'].drop(index = [313, 335, 249, 706]))
outliers.extend([314, 336, 250, 707])

# MasVnrArea - will not remove anything for now
#sns.jointplot(x = train['MasVnrArea'], y = train['SalePrice'])
train[['MasVnrArea']].sort_values('MasVnrArea', ascending = False)
train[np.abs(stats.zscore(train['MasVnrArea'])) > 3]['MasVnrArea'].sort_values(ascending = False)

# TotalSF
#sns.jointplot(x = train['TotalSF'], y = train['SalePrice'])
train[['TotalSF']].sort_values('TotalSF', ascending = False)
train[np.abs(stats.zscore(train['TotalSF'])) > 3]['TotalSF'].sort_values(ascending = False)
#sns.jointplot(x = train['TotalSF'].drop(index = [524, 1299]), y = train['SalePrice'].drop(index = [524, 1298]))
outliers.extend([524, 1299])

# GrLivArea
#sns.jointplot(x = train['GrLivArea'], y = train['SalePrice'])
train[['GrLivArea']].sort_values('GrLivArea', ascending = False)
train[np.abs(stats.zscore(train['GrLivArea'])) > 3]['GrLivArea'].sort_values(ascending = False)
# Remove index 1298, 523
#sns.jointplot(x = train['GrLivArea'].drop(index = [523, 1298]), y = train['SalePrice'].drop(index = [523, 1298]))

# GarageArea
#sns.jointplot(x = train['GarageArea'], y = train['SalePrice'])
train[['GarageArea']].sort_values('GarageArea', ascending = False)
train[np.abs(stats.zscore(train['GarageArea'])) > 3]['GarageArea'].sort_values(ascending = False)

# PorchSF
#sns.jointplot(x = train['PorchSF'], y = train['SalePrice'])
train[['PorchSF']].sort_values('PorchSF', ascending = False)
train[np.abs(stats.zscore(train['PorchSF'])) > 3]['PorchSF'].sort_values(ascending = False)

# SalePrice
#sns.distplot(train['SalePrice'])
train[['SalePrice']].sort_values('SalePrice', ascending = False)
train[np.abs(stats.zscore(train['SalePrice'])) > 3]['SalePrice'].sort_values(ascending = False)
#sns.distplot(np.log(train['SalePrice']))

outliers = list(set(outliers))

#idx_rm = [934, 1298, 313, 335, 249, 706, 523, 129, 1190, 1328]
idx_rm = outliers
train.drop(index = idx_rm, inplace = True)
actual_price.drop(index = idx_rm, inplace = True)
train_id = train.index


###############################################################################
########################## DATA ANALYSIS ROUND 2 ##############################
###############################################################################
# Skewness of the data
train[num_features_columns].skew()
train[num_features_columns].skew()[(train[num_features_columns].skew() > 1) | (train[num_features_columns].skew() < -1)]

#Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.
train[num_features_columns].kurt()
train[num_features_columns].kurt()[train[num_features_columns].kurt() > 3]

########################## BOX-COX TRANSFORMATIONS ############################
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
    print(x.name, "\n\n", t, "\n\n")


for col in num_features_columns:
    normtesttab(train[col])

for col in num_features_columns[:-1]:
    if ((min(train[col]) <= 0) & (min(test[col]) <= 0)):
        xt, maxlog, interval = stats.boxcox(train[col] + abs(min(train[col]))+1, alpha=0.05)
        print(col, ": Non-positive values found. Train lambda = {:g}".format(maxlog))
        xt, maxlog, interval = stats.boxcox(test[col] + abs(min(test[col]))+1, alpha=0.05)
        print(col, ": Non-positive values found. Test lambda = {:g}".format(maxlog), "\n")
    else:
        xt, maxlog, interval = stats.boxcox(train[col], alpha=0.05)
        print("All values in", col, "are positive. Train lambda = {:g}".format(maxlog))
        xt, maxlog, interval = stats.boxcox(test[col], alpha=0.05)
        print("All values in", col, "are positive. Test lambda = {:g}".format(maxlog), "\n")
        
        

# TRANSFORMATIONS
sns.distplot(train['LotFrontage'])
sns.distplot(test['LotFrontage'])
xt, lf_maxlog, interval = stats.boxcox(train['LotFrontage'], alpha=0.05)
xt, lf2_maxlog, interval = stats.boxcox(test['LotFrontage'], alpha=0.05)
print("lambda = {:g}".format(lf_maxlog))
print("lambda = {:g}".format(lf2_maxlog))
lf_lambda = ((lf_maxlog+lf2_maxlog)/2)
train['LotFrontage'] = (train['LotFrontage'])**lf_lambda
test['LotFrontage'] = (test['LotFrontage'])**lf_lambda
sns.distplot(train['LotFrontage'])
sns.distplot(test['LotFrontage'])


sns.distplot(train['LotArea'])
sns.distplot(test['LotArea'])
xt, la_maxlog, interval = stats.boxcox(train['LotArea'], alpha=0.05)
xt, la2_maxlog, interval = stats.boxcox(test['LotArea'], alpha=0.05)
print("lambda = {:g}".format(la_maxlog))
print("lambda = {:g}".format(la2_maxlog))
la_lambda = ((la_maxlog+la2_maxlog)/2)
train['LotArea'] = (train['LotArea'])**la_lambda
test['LotArea'] = (test['LotArea'])**la_lambda
sns.distplot(train['LotArea'])
sns.distplot(test['LotArea'])


xt, ts_maxlog, interval = stats.boxcox(train['TotalSF'], alpha=0.05)
xt, ts2_maxlog, interval = stats.boxcox(test['TotalSF'], alpha=0.05)
print("lambda = {:g}".format(ts_maxlog))
print("lambda = {:g}".format(ts2_maxlog))
ts_lambda = ((ts_maxlog+ts2_maxlog)/2)

train['SalePrice'] = np.log(train['SalePrice'])
train_columns = train.columns

#df = pd.concat([train, test], axis = 0)[train_columns]

#train_id = df[df['SalePrice'] != 0].index
#test_id = df[df['SalePrice'] == 0].index

###############################################################################
############################# STANDARDIZE DATA ################################
###############################################################################
train_columns = train.columns
to_scale = train[num_features_columns].drop(columns = "SalePrice")
colnames = to_scale.columns
scaler = StandardScaler()
scaler.fit(to_scale)
scaled_features = pd.DataFrame(scaler.transform(to_scale),
             columns = to_scale.columns, index = train.index)
scaler2 = StandardScaler()
scaler2.fit(train[['SalePrice']])
scaled_price = pd.DataFrame(scaler2.transform(train[['SalePrice']]),
                            columns = ['SalePrice'], index = train.index)

train_num = pd.concat([scaled_features, scaled_price], axis = 1)

train.drop(columns = num_features_columns, inplace = True)

train = pd.concat([train_num, train], axis = 1)[train_columns]

# Standardize test set
test_num = test[num_features_columns[:-1]]
test_num = pd.DataFrame(scaler.transform(test_num),
             columns = test_num.columns, index = test.index)


test.drop(columns = num_features_columns[:-1], inplace = True)

test = pd.concat([test_num, test], axis = 1)[train_columns]

df = train.merge(test, on = test.columns.tolist(), how = "outer")
ids = pd.DataFrame(pd.concat([pd.Series(train_id), pd.Series(test_id)], axis = 0), columns = ["Id"])
ids.reset_index(drop = True, inplace = True)
df.reset_index(drop = True, inplace = True)
df = pd.concat([df, ids], axis = 1).set_index("Id")



###############################################################################
########################## CATEGORICAL  VARIABLES #############################
###############################################################################
df_fe = df.copy()
#######################################################
#PoolArea
df_fe['HasPool'] = [1 if x > 0 else 0 for x in df_fe['PoolArea'] ]
df_fe.drop('PoolArea', axis = 1, inplace = True)

# http://www.insightsbot.com/blog/2AeuRL/chi-square-feature-selection-in-python
class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

cat_features_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
                'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
                'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
                'GarageQual', 'PavedDrive', 'HasPool', 'PoolQC', 'Fence', 'MoSold',
                'YrSold', 'SaleType']

cat_features = df_fe[cat_features_columns]





###############################################################################
############################# DUMMIFY VARIABLES ###############################
###############################################################################
#######################################################
#MSSubClass
#sns.violinplot(df_fe['MSSubClass'], y = actual_price)
sns.countplot(x='MSSubClass', data = df_fe)
dummy_df = pd.get_dummies(df_fe['MSSubClass'], drop_first=True, prefix = 'MSSubClass')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('MSSubClass', axis = 1)
#######################################################
#MSZoning
#sns.violinplot(x='MSZoning', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='MSZoning', data = df_fe)
dummy_df = pd.get_dummies(df_fe['MSZoning'], drop_first=True, prefix = 'MSZoning')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('MSZoning', axis = 1)
#######################################################
#Lot Frontage
#######################################################
#Area
#######################################################
#Street
#sns.violinplot(df_fe['Street'], y = actual_price)
df.Street.value_counts()
df.drop(index=test_id).groupby('Street').agg({'SalePrice': 'mean'})
sns.countplot(x='Street', data = df_fe)
dummy_df = pd.get_dummies(df_fe['Street'], drop_first=True, prefix = 'Street')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Street', axis = 1)
#######################################################
#Alley
sns.violinplot(df_fe['Alley'], y = actual_price)
df.drop(index=test_id).groupby('Alley').agg({'SalePrice': 'mean'})
df.Alley.value_counts()
sns.countplot(x='Alley', data = df_fe)
dummy_df = pd.get_dummies(df_fe['Alley'], drop_first=True, prefix = 'Alley')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Alley', axis = 1)
#######################################################
#LotShape
sns.violinplot(df_fe['LotShape'], actual_price)
sns.countplot(x='LotShape', data = df_fe)
dummy_df = pd.get_dummies(df_fe['LotShape'], drop_first=True, prefix = 'LotShape')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('LotShape', axis = 1)
#######################################################
#LandContour
sns.violinplot(x='LandContour', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='LandContour', data = df_fe)
dummy_df = pd.get_dummies(df_fe['LandContour'], drop_first=True, prefix = 'LandContour')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('LandContour', axis = 1)
#######################################################
#Utilities
sns.violinplot(x='Utilities', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='Utilities', data = df_fe)
df.Utilities.value_counts()
df_fe = df_fe.drop('Utilities', axis = 1)
#######################################################
#LotConfig
sns.violinplot(x='LotConfig', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='LotConfig', data = df_fe)
dummy_df = pd.get_dummies(df_fe['LotConfig'], drop_first=True, prefix = 'LotConfig')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('LotConfig', axis = 1)
#######################################################
#LandSlope
sns.violinplot(x='LandSlope', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='LandSlope', data = df_fe)
dummy_df = pd.get_dummies(df_fe['LandSlope'], drop_first=True, prefix = 'LandSlope')
df_fe = pd.concat([df_fe, dummy_df], axis = 1)
df_fe = df_fe.drop('LandSlope', axis = 1)
#######################################################
#Neighborhood
sns.violinplot(x='Neighborhood', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='Neighborhood', data = df_fe)
dummy_df = pd.get_dummies(df_fe['Neighborhood'], drop_first=True, prefix = 'Neighborhood')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Neighborhood', axis = 1) 
#######################################################
#Condition1
sns.violinplot(x='Condition1', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='Condition1', data = df_fe)
dummy_df = pd.get_dummies(df_fe['Condition1'], drop_first=True, prefix = 'Condition1')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Condition1', axis = 1)
#######################################################
#Condition2
sns.violinplot(x='Condition2', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='Condition2', data = df_fe)
dummy_df = pd.get_dummies(df_fe['Condition2'], drop_first=True, prefix = 'Condition2')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Condition2', axis = 1)
#######################################################
#BldgType
sns.violinplot(x='BldgType', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='BldgType', data = df_fe)
dummy_df = pd.get_dummies(df_fe['BldgType'], drop_first=True, prefix = 'BldgType')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('BldgType', axis = 1)
#######################################################
#HouseStyle
sns.violinplot(x='HouseStyle', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='HouseStyle', data = df_fe)
dummy_df = pd.get_dummies(df_fe['HouseStyle'], drop_first = True, prefix = "HouseStyle")
df_fe = pd.concat([df_fe, dummy_df], axis = 1) 
df_fe = df_fe.drop('HouseStyle', axis = 1)
#######################################################
#OverallQual
sns.violinplot(x='OverallQual', y = 'SalePrice', data = df_fe.drop(index = test_id))
#######################################################
#OverallCond
sns.violinplot(x='OverallCond', y = 'SalePrice', data = df_fe.drop(index = test_id))
#######################################################
#YearBuilt
sns.violinplot(x='YearBuilt', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.jointplot(x='YearBuilt', y = 'SalePrice', data = df_fe.drop(index = test_id))
#######################################################
#YearRemodAdd
sns.violinplot(x='YearRemodAdd', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.jointplot(x='YearRemodAdd', y = 'SalePrice', data = df_fe.drop(index = test_id))
#######################################################
#RoofStyle
sns.violinplot(x='RoofStyle', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='RoofStyle', data = df_fe)
dummy_df = pd.get_dummies(df_fe['RoofStyle'], drop_first=True, prefix = 'RoofStyle')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('RoofStyle', axis = 1)
#######################################################
#RoofMatl
sns.violinplot(x='RoofMatl', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.countplot(x='RoofMatl', data = df_fe)
df.RoofMatl.value_counts()
df.drop(index = test_id).groupby('RoofMatl').agg({'SalePrice':'mean'})
dummy_df = pd.get_dummies(df_fe['RoofMatl'], drop_first=True, prefix = 'RoofMatl')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('RoofMatl', axis = 1)
#######################################################
#Exterior1st
sns.violinplot(x='Exterior1st', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['Exterior1st'], drop_first=True, prefix = 'Exterior1st')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Exterior1st', axis = 1)
#######################################################
#Exterior2nd
sns.violinplot(x='Exterior2nd', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['Exterior2nd'], drop_first=True, prefix = 'Exterior2nd')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Exterior2nd', axis = 1)
#######################################################
#MasVnrType
sns.violinplot(x='MasVnrType', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['MasVnrType'], drop_first=True, prefix = 'MasVnrType')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('MasVnrType', axis = 1)
#######################################################
#MasVnrArea
#######################################################
#ExterQual
sns.violinplot(x='ExterQual', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['ExterQual'] = df_fe['ExterQual'].map(ord_map)
#######################################################
#ExterCond
sns.violinplot(x='ExterCond', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['ExterCond'] = df_fe['ExterCond'].map(ord_map)
#######################################################
#Foundation
sns.violinplot(x='Foundation', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['Foundation'], drop_first=True, prefix = 'Foundation')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Foundation', axis = 1)
#######################################################
#BsmtQual
sns.violinplot(x='BsmtQual', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['BsmtQual'] = df_fe['BsmtQual'].map(ord_map)
#######################################################
#BsmtCond
sns.violinplot(x='BsmtCond', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['BsmtCond'] = df_fe['BsmtCond'].map(ord_map)
#######################################################
#BsmtExposure
sns.violinplot(x='BsmtExposure', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['BsmtExposure'], drop_first=True, prefix = 'BsmtExposure')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('BsmtExposure', axis = 1)
#######################################################
#BsmtFinType1
sns.violinplot(x='BsmtFinType1', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['BsmtFinType1'], drop_first=True, prefix = 'BsmtFinType1')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('BsmtFinType1', axis = 1)
#######################################################
#BsmtFinType2
sns.violinplot(x='BsmtFinType2', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['BsmtFinType2'], drop_first=True, prefix = 'BsmtFinType2')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('BsmtFinType2', axis = 1)    
#######################################################
#Heating
sns.violinplot(x='Heating', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['Heating'], drop_first=True, prefix = 'Heating')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Heating', axis = 1)
#######################################################
#HeatingQC
sns.violinplot(x='HeatingQC', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['HeatingQC'] = df_fe['HeatingQC'].map(ord_map)
#######################################################
#CentralAir
sns.violinplot(x='CentralAir', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['CentralAir'], drop_first=True, prefix = 'CentralAir')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('CentralAir', axis = 1)
#######################################################
#Electrical
sns.violinplot(x='Electrical', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['Electrical'], drop_first=True, prefix = 'Electrical')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Electrical', axis = 1)   
#######################################################
#LowQualFinSF
df_fe.drop('LowQualFinSF', axis = 1, inplace = True)
#######################################################
#GrLivArea
#######################################################
#BedroomAbvGr
#######################################################
#KitchenAbvGr
#######################################################
#KitchenQual
sns.violinplot(x='KitchenQual', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['KitchenQual'] = df_fe['KitchenQual'].map(ord_map)
#######################################################
#TotRmsAbvGrd
#######################################################
#Functional
sns.violinplot(x='Functional', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['Functional'], drop_first=True, prefix = 'Functional')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Functional', axis = 1)  
#######################################################
#Fireplaces
#######################################################
#FireplaceQu
sns.violinplot(x='FireplaceQu', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
df_fe['FireplaceQu'] = df_fe['FireplaceQu'].map(ord_map)
#######################################################
#GarageType
sns.violinplot(x='GarageType', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['GarageType'], drop_first=True, prefix = 'GarageType')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('GarageType', axis = 1)
#######################################################
#GarageYrBlt
sns.violinplot(x='GarageYrBlt', y = 'SalePrice', data = df_fe.drop(index = test_id))
sns.jointplot(x='GarageYrBlt', y = 'SalePrice', data = df_fe.drop(index = test_id))
#######################################################
#GarageFinish
sns.violinplot(x='GarageFinish', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['GarageFinish'], drop_first=True, prefix = 'GarageFinish')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('GarageFinish', axis = 1)   
#######################################################
#GarageCars
#######################################################
#GarageArea
#######################################################
#GarageQual
sns.violinplot(x='GarageQual', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['GarageQual'] = df_fe['GarageQual'].map(ord_map)
#######################################################
#GarageCond
sns.violinplot(x='GarageCond', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['GarageCond'] = df_fe['GarageCond'].map(ord_map)
#######################################################
#PavedDrive
sns.violinplot(x='PavedDrive', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['PavedDrive'], drop_first=True, prefix = 'PavedDrive')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('PavedDrive', axis = 1)
#######################################################
#PoolQC
sns.violinplot(x='PoolQC', y = 'SalePrice', data = df_fe.drop(index = test_id))
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['PoolQC'] = df_fe['PoolQC'].map(ord_map)
#######################################################   
#Fence
sns.violinplot(x='Fence', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['Fence'], drop_first=True, prefix = 'Fence')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Fence', axis = 1)  
#######################################################
#MiscFeature
sns.violinplot(x='MiscFeature', y = 'SalePrice', data = df_fe.drop(index = test_id))
df_fe.drop(index = test_id)['MiscFeature'].value_counts()
df_fe.drop(index = train_id)['MiscFeature'].value_counts()
#dummy_df = pd.get_dummies(df_fe['MiscFeature'], drop_first=True, prefix = 'MiscFeature')
#df_fe = pd.concat([df_fe, dummy_df], axis=1)
#df_fe = df_fe.drop('MiscFeature', axis = 1) 
df_fe.drop('MiscFeature', axis = 1, inplace = True)
#######################################################
#MiscVal
sns.jointplot(x='MiscVal', y = 'SalePrice', data = df_fe.drop(index = test_id))
df_fe.drop('MiscVal', axis = 1, inplace = True)
#######################################################
#MoSold
#Dummify
sns.jointplot(x='MoSold', y = 'SalePrice', data = df_fe.drop(index = test_id))
#######################################################
#YrSold
#Dummify
sns.jointplot(x='YrSold', y = 'SalePrice', data = df_fe.drop(index = test_id))
#######################################################
#SaleType
sns.violinplot(x='SaleType', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['SaleType'], drop_first=True, prefix = 'SaleType')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('SaleType', axis = 1) 
#######################################################
#SaleCondition
sns.violinplot(x='SaleCondition', y = 'SalePrice', data = df_fe.drop(index = test_id))
dummy_df = pd.get_dummies(df_fe['SaleCondition'], drop_first=True, prefix = 'SaleCondition')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('SaleCondition', axis = 1) 
#######################################################
#SalePrice

del (dummy_df, ord_map)

df_fe.columns[df_fe.isnull().any(axis = 0)]


###############################################################################
########################## DATA ANALYSIS ROUND 2 ##############################
###############################################################################
num_features_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'TotalSF', 'GrLivArea',
                'GarageArea', 'PorchSF', 'SalePrice']

df_fe[num_features_columns].describe()
# https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa
# Skewness of the data
df_fe[num_features_columns].drop(index = test_id).skew()
df_fe[num_features_columns].drop(index = test_id).skew()[(df_fe[num_features_columns].drop(index = test_id).skew() > 1) | (df_fe[num_features_columns].drop(index = test_id).skew() < -1)]

#Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.
df_fe[num_features_columns].drop(index = test_id).kurt()
df_fe[num_features_columns].drop(index = test_id).kurt()[df_fe[num_features_columns].drop(index = test_id).kurt() > 3]

# Correlation
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
fe_train_correlation = df_fe[num_features_columns].drop(index = test_id).corr()

mask = np.zeros_like(fe_train_correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(10, 10))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(fe_train_correlation, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

train = df_fe.drop(index = test_id)
test = df_fe.drop(columns = 'SalePrice', index = train_id)

#train.to_csv("data/train_clean_std_full.csv", index = False)
#test.to_csv("data/test_clean_std_full.csv", index = False)