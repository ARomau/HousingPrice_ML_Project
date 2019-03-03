#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:01:31 2019

@author: stellakim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:25:13 2019

@author: stellakim
"""
import numpy as np
import pandas as pd
from scipy import stats
from astropy.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/
###############################################################################
############################### READ IN DATA ##################################
###############################################################################

df = pd.read_csv("data/train.csv")
train_id = df["Id"]
df2 = pd.read_csv("data/test.csv")
test_id = df2["Id"]
df2 = pd.concat([df2, pd.DataFrame(np.zeros(len(df2)), columns = ['SalePrice'])], axis = 1)
#df = df.drop("Id", axis = 1)
df = df.merge(df2, on = df.columns.tolist(), how = "outer")
df.set_index("Id", inplace = True)
del(df2)

df.shape

df.dtypes

# Columns with missing data
df.columns[df.isnull().any(axis = 0)]

## MISSINGNESS IN COLUMNS
df.isnull().sum(axis = 0).sort_values(ascending = False)
# Percent of missingness in each column
df.isnull().sum(axis = 0).sort_values(ascending = False)/df.shape[0]

## MISSINGNESS IN ROWS
df.isnull().sum(axis = 1).sort_values(ascending = False)
# Percent of missingness in each row
df.isnull().sum(axis = 1).sort_values(ascending = False)/df.shape[1]

df.columns
df.columns[df.isnull().any(axis = 0)]

###############################################################################
############################ FILL IN MISSING DATA ##############################
###############################################################################

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



###############################################################################
############################# DUMMIFY VARIABLES ###############################
###############################################################################
df_fe = df.copy()

#Begin feature extraction
#######################################################
#Col ID
#Remove Col Id
#df_fe = df_fe.drop('Id', axis = 1)
#######################################################
#MSSubClass
dummy_df = pd.get_dummies(df_fe['MSSubClass'], drop_first=True, prefix = 'MSSubClass')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('MSSubClass', axis = 1)
#######################################################
#MSZoning
dummy_df = pd.get_dummies(df_fe['MSZoning'], drop_first=True, prefix = 'MSZoning')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('MSZoning', axis = 1)
#######################################################
#Lot Frontage
#######################################################
#Area
#######################################################
#Street
#Remove Column
dummy_df = pd.get_dummies(df_fe['Street'], drop_first=True, prefix = 'Street')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Street', axis = 1)
#######################################################
#Alley
dummy_df = pd.get_dummies(df_fe['Alley'], drop_first=True, prefix = 'Alley')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Alley', axis = 1)
#######################################################
#LotShape
dummy_df = pd.get_dummies(df_fe['LotShape'], drop_first=True, prefix = 'LotShape')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('LotShape', axis = 1)
#######################################################
#LandContour
dummy_df = pd.get_dummies(df_fe['LandContour'], drop_first=True, prefix = 'LandContour')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('LandContour', axis = 1)
#######################################################
#Utilities
dummy_df = pd.get_dummies(df_fe['Utilities'], drop_first=True, prefix = 'Utilities')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Utilities', axis = 1)
#######################################################
#LotConfig
dummy_df = pd.get_dummies(df_fe['LotConfig'], drop_first=True, prefix = 'LotConfig')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('LotConfig', axis = 1)
#######################################################
#LandSlope
dummy_df = pd.get_dummies(df_fe['LandSlope'], drop_first=True, prefix = 'LandSlope')
df_fe = pd.concat([df_fe, dummy_df], axis = 1)
df_fe = df_fe.drop('LandSlope', axis = 1)
#######################################################
#Neighborhood
dummy_df = pd.get_dummies(df_fe['Neighborhood'], drop_first=True, prefix = 'Neighborhood')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Neighborhood', axis = 1) 
#######################################################
#Condition1
dummy_df = pd.get_dummies(df_fe['Condition1'], drop_first=True, prefix = 'Condition1')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Condition1', axis = 1)
#######################################################
#Condition2
dummy_df = pd.get_dummies(df_fe['Condition2'], drop_first=True, prefix = 'Condition2')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Condition2', axis = 1)
#######################################################
#BldgType
dummy_df = pd.get_dummies(df_fe['BldgType'], drop_first=True, prefix = 'BldgType')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('BldgType', axis = 1)
#######################################################
#HouseStyle
dummy_df = pd.get_dummies(df_fe['HouseStyle'], drop_first = True, prefix = "HouseStyle")
df_fe = pd.concat([df_fe, dummy_df], axis = 1) 
df_fe = df_fe.drop('HouseStyle', axis = 1)
#######################################################
#OverallQual
##Create interaction feature OverallQualCond = OverallQual*OverallCond
#if (flag == 1):
#    df_fe['OverallQualCond'] = df_fe['OverallQual']*df_fe['OverallCond']
#    #Remove Column
#    df_fe = df_fe.drop('OverallQual', axis = 1)
#######################################################
#OverallCond
##Remove Column 
#if (flag == 1):
#    df_fe = df_fe.drop('OverallCond', axis = 1)
#######################################################
#YearBuilt
#Create interaction feature Age = YearSold-YearBuilt
#######################################################
#YearRemodAdd
#Create interaction feature Age = YearSold-YearBuilt
#######################################################
#RoofStyle
#Combine classes - 'Flat','Gambrel','Mansard','Shed' into a single class "Other"
dummy_df = pd.get_dummies(df_fe['RoofStyle'], drop_first=True, prefix = 'RoofStyle')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('RoofStyle', axis = 1)
#######################################################
#RoofMatl
#Combine classes - 'Tar&Grv','WdShngl','WdShake','Roll','Membran','Metal','ClyTile' 
#into a single class "NotShingle"
dummy_df = pd.get_dummies(df_fe['RoofMatl'], drop_first=True, prefix = 'RoofMatl')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('RoofMatl', axis = 1)
#######################################################
#Exterior1st
dummy_df = pd.get_dummies(df_fe['Exterior1st'], drop_first=True, prefix = 'Exterior1st')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Exterior1st', axis = 1)
#######################################################
#Exterior2nd
dummy_df = pd.get_dummies(df_fe['Exterior2nd'], drop_first=True, prefix = 'Exterior2nd')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Exterior2nd', axis = 1)
#######################################################
#MasVnrType
dummy_df = pd.get_dummies(df_fe['MasVnrType'], drop_first=True, prefix = 'MasVnrType')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('MasVnrType', axis = 1)
#######################################################
#MasVnrArea
#######################################################
#ExterQual
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['ExterQual'] = df_fe['ExterQual'].map(ord_map)
#######################################################
#ExterCond
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['ExterCond'] = df_fe['ExterCond'].map(ord_map)
#######################################################
#Foundation
#Dummify
dummy_df = pd.get_dummies(df_fe['Foundation'], drop_first=True, prefix = 'Foundation')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Foundation', axis = 1)
#######################################################
#BsmtQual
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['BsmtQual'] = df_fe['BsmtQual'].map(ord_map)
#######################################################
#BsmtCond
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['BsmtCond'] = df_fe['BsmtCond'].map(ord_map)
#######################################################
#BsmtExposure
dummy_df = pd.get_dummies(df_fe['BsmtExposure'], drop_first=True, prefix = 'BsmtExposure')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('BsmtExposure', axis = 1)
#######################################################
#BsmtFinType1
dummy_df = pd.get_dummies(df_fe['BsmtFinType1'], drop_first=True, prefix = 'BsmtFinType1')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('BsmtFinType1', axis = 1)
#######################################################
#BsmtFinType2
dummy_df = pd.get_dummies(df_fe['BsmtFinType2'], drop_first=True, prefix = 'BsmtFinType2')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('BsmtFinType2', axis = 1)    
#######################################################
# Dropping BsmtFinSF1, BsmtFinSF2, BsmtUnfSF - this is captured in TotalSF
df_fe.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis = 1, inplace = True)
#######################################################
#Heating
dummy_df = pd.get_dummies(df_fe['Heating'], drop_first=True, prefix = 'Heating')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Heating', axis = 1)
#######################################################
#HeatingQC
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['HeatingQC'] = df_fe['HeatingQC'].map(ord_map)
#######################################################
#CentralAir
#Create Binary
ls = ['Y']
df_fe['CentralAir_Bin'] = [1 if x in ls else 0 for x in df_fe['CentralAir'] ]
df_fe = df_fe.drop('CentralAir', axis = 1)
#######################################################
#Electrical
dummy_df = pd.get_dummies(df_fe['Electrical'], drop_first=True, prefix = 'Electrical')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Electrical', axis = 1)   
#######################################################
# Combine 1stFlrSF, 2ndFlrSF, and TotalBsmtSF
df_fe['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
df_fe = df_fe.drop(['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'], axis = 1)
#######################################################
#LowQualFinSF
df_fe.drop('LowQualFinSF', axis = 1, inplace = True)
#######################################################
#GrLivArea
#######################################################
# Combine BsmtFullBath, BsmtHalfBath, FullBath, HalfBath
df_fe['Baths'] = df['BsmtFullBath'] + df['BsmtHalfBath']*0.5 + df['FullBath'] + df['HalfBath']*0.5
df_fe.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis = 1, inplace = True)
#######################################################
#BedroomAbvGr
#######################################################
#KitchenAbvGr
#######################################################
#KitchenQual
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['KitchenQual'] = df_fe['KitchenQual'].map(ord_map)
#######################################################
#TotRmsAbvGrd
#######################################################
#Functional
dummy_df = pd.get_dummies(df_fe['Functional'], drop_first=True, prefix = 'Functional')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Functional', axis = 1)  
#######################################################
#Fireplaces
#######################################################
#FireplaceQu
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
df_fe['FireplaceQu'] = df_fe['FireplaceQu'].map(ord_map)
#######################################################
#GarageType
dummy_df = pd.get_dummies(df_fe['GarageType'], drop_first=True, prefix = 'GarageType')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('GarageType', axis = 1)
#######################################################
#GarageYrBlt
#######################################################
#GarageFinish
dummy_df = pd.get_dummies(df_fe['GarageFinish'], drop_first=True, prefix = 'GarageFinish')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('GarageFinish', axis = 1)   
#######################################################
#GarageCars
#######################################################
#GarageArea
#######################################################
#GarageQual
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['GarageQual'] = df_fe['GarageQual'].map(ord_map)
#######################################################
#GarageCond
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['GarageCond'] = df_fe['GarageCond'].map(ord_map)
#######################################################
#PavedDrive
dummy_df = pd.get_dummies(df_fe['PavedDrive'], drop_first=True, prefix = 'PavedDrive')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('PavedDrive', axis = 1)
#######################################################
# Combine WoodDeckSF, OpenPorchSF, 3SsnPorch, ScreenPorch
df_fe['PorchSF'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['3SsnPorch'] + df['ScreenPorch']
# Drop EnclosedPorch
df_fe.drop('EnclosedPorch', axis = 1, inplace = True)
#######################################################
#PoolArea
df_fe['HasPool'] = [1 if x > 0 else 0 for x in df_fe['PoolArea'] ]
df_fe.drop('PoolArea', axis = 1, inplace = True)
#######################################################
#PoolQC
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
df_fe['PoolQC'] = df_fe['PoolQC'].map(ord_map)
#######################################################   
#Fence
dummy_df = pd.get_dummies(df_fe['Fence'], drop_first=True, prefix = 'Fence')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('Fence', axis = 1)  
#######################################################
#MiscFeature
#dummy_df = pd.get_dummies(df_fe['MiscFeature'], drop_first=True, prefix = 'MiscFeature')
#df_fe = pd.concat([df_fe, dummy_df], axis=1)
#df_fe = df_fe.drop('MiscFeature', axis = 1) 
df_fe.drop('MiscFeature', axis = 1, inplace = True)
#######################################################
#MiscVal
df_fe.drop('MiscVal', axis = 1, inplace = True)
#######################################################
#MoSold
#Dummify
dummy_df = pd.get_dummies(df_fe['MoSold'], drop_first=True, prefix = 'MoSold')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('MoSold', axis = 1) 
#######################################################
#YrSold
#Dummify
#dummy_df = pd.get_dummies(df_fe['YrSold'], drop_first=True, prefix = 'YrSold')
#df_fe = pd.concat([df_fe, dummy_df], axis=1)
#df_fe = df_fe.drop('YrSold', axis = 1) 
#######################################################
#SaleType
#Dummify
dummy_df = pd.get_dummies(df_fe['SaleType'], drop_first=True, prefix = 'SaleType')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('SaleType', axis = 1) 
#######################################################
#SaleCondition
#Dummify
dummy_df = pd.get_dummies(df_fe['SaleCondition'], drop_first=True, prefix = 'SaleCondition')
df_fe = pd.concat([df_fe, dummy_df], axis=1)
df_fe = df_fe.drop('SaleCondition', axis = 1) 
#######################################################
#SalePrice

del (dummy_df, ls, ord_map)

df_fe.columns[df_fe.isnull().any(axis = 0)]

###############################################################################
########################### SPLIT TRAIN AND TEST ##############################
###############################################################################

train = df_fe.drop(index = test_id)
train = pd.concat([train.drop('SalePrice', axis = 1), train['SalePrice']], axis = 1)
train.reset_index(drop = True, inplace = True)
test = df_fe.drop(index = train_id).drop('SalePrice', axis = 1)
test.reset_index(drop = True, inplace = True)


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

# Manual removal of outliers
# LotFrontage
#sns.jointplot(x = train['LotFrontage'], y = train['SalePrice'])
train[['LotFrontage']].sort_values('LotFrontage', ascending = False)
train[np.abs(stats.zscore(train['LotFrontage'])) > 3]['LotFrontage'].sort_values(ascending = False)
# Remove index 934, 1298, since these are heavy outliers
#sns.jointplot(x = train['LotFrontage'].drop(index = [934, 1298]), y = train['SalePrice'].drop(index = [934,1298]))

# LotArea
#sns.jointplot(x = train['LotArea'], y = train['SalePrice'])
train[['LotArea']].sort_values('LotArea', ascending = False)
train[np.abs(stats.zscore(train['LotArea'])) > 3]['LotArea'].sort_values(ascending = False)
# Remove index 313, 335, 249, 706
#sns.jointplot(x = train['LotArea'].drop(index = [313, 335, 249, 706]), y = train['SalePrice'].drop(index = [313, 335, 249, 706]))

# MasVnrArea - will not remove anything for now
#sns.jointplot(x = train['MasVnrArea'], y = train['SalePrice'])
train[['MasVnrArea']].sort_values('MasVnrArea', ascending = False)
train[np.abs(stats.zscore(train['MasVnrArea'])) > 3]['MasVnrArea'].sort_values(ascending = False)

# TotalSF
#sns.jointplot(x = train['TotalSF'], y = train['SalePrice'])
train[['TotalSF']].sort_values('TotalSF', ascending = False)
train[np.abs(stats.zscore(train['TotalSF'])) > 3]['TotalSF'].sort_values(ascending = False)
# Remove index 1298, 523
#sns.jointplot(x = train['TotalSF'].drop(index = [523, 1298]), y = train['SalePrice'].drop(index = [523, 1298]))

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
# Remove index 1298, 523, maybe 1190
#sns.jointplot(x = train['GarageArea'].drop(index = [1190, 523, 1298]), y = train['SalePrice'].drop(index = [1190, 523, 1298]))

# PorchSF
#sns.jointplot(x = train['PorchSF'], y = train['SalePrice'])
train[['PorchSF']].sort_values('PorchSF', ascending = False)
train[np.abs(stats.zscore(train['PorchSF'])) > 3]['PorchSF'].sort_values(ascending = False)
# Remove index 1298, 523, maybe 1190
#sns.jointplot(x = train['PorchSF'].drop(index = [1328]), y = train['SalePrice'].drop(index = [1328]))


# SalePrice
#sns.distplot(train['SalePrice'])
train[['SalePrice']].sort_values('SalePrice', ascending = False)
train[np.abs(stats.zscore(train['SalePrice'])) > 3]['SalePrice'].sort_values(ascending = False)
# I don't think we should drop any sale prices, we are transforming to remove skew so I don't think it would be beneficial to remove outliers and it would make it harder to predict higher priced homes
#sns.distplot(np.log(train['SalePrice']))
np.log(train[np.abs(stats.zscore(np.log(train['SalePrice']))) > 3]['SalePrice']).sort_values(ascending = False)


idx_rm = [934, 1298, 313, 335, 249, 706, 523, 129, 1190, 1328]
train.drop(index = idx_rm, inplace = True)
#sns.pairplot(train[num_features_columns])
#for column in num_features.columns:
#    if column != "SalePrice":
#        sns.jointplot(x = train[column], y = train['SalePrice'])
#
#actual_price = df['SalePrice'].drop(index = test_id).drop(index = idx_rm)

#x = df['1stFlrSF'].reset_index(drop = True).drop(index = [197, 523, 1298])
#y = num_features['SalePrice'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)
#
#x = df['2ndFlrSF'].reset_index(drop = True).drop(index = [197, 523, 1298])
#y = num_features['SalePrice'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)
#
#np.where(df['1stFlrSF']+df['2ndFlrSF'] == max(df['1stFlrSF']+df['2ndFlrSF']))
#sns.jointplot((df['1stFlrSF']+df['2ndFlrSF']), num_features['SalePrice'])
#x = (df['1stFlrSF']+df['2ndFlrSF']).reset_index(drop = True).drop(index = [197, 523, 1298])
#df.loc[1298, '1stFlrSF']
#y = num_features['SalePrice'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x, y)
#
#
#x = df['BsmtFinSF'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x, y)
#
#x = (df['1stFlrSF']+df['2ndFlrSF'] + df['TotalBsmtSF']).reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x, y)


#z = np.abs(stats.zscore(num_features))
#x[stats.zscore(df['LotArea'].reset_index(drop = True).drop(index = [197, 523, 1298])) > 3]
#
#x = df['LotArea'].reset_index(drop = True).drop(index = [197, 523, 1298])
#x[stats.zscore(df['LotArea'].reset_index(drop = True).drop(index = [197, 523, 1298])) > 3]
#x[stats.zscore(df['LotArea'].reset_index(drop = True).drop(index = [197, 523, 1298])) > 3].sort_values(ascending = False)
#sns.jointplot(x.drop(index = [313, 335, 249, 706]), y.drop(index = [313, 335, 249, 706, 451]))
##sns.jointplot(x.drop(index = 934), y.drop(index = 934))
#
#sns.jointplot(df['PoolArea'].reset_index(drop=True).drop(index=[197, 523, 1298]), y)



########################################################
##WoodDeckSF
#x = df['WoodDeckSF'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)
########################################################
##OpenPorchSF
#x = df['OpenPorchSF'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)
########################################################
##EnclosedPorch
#x = df['EnclosedPorch'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)
########################################################
##3SsnPorch
#x = df['3SsnPorch'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)
########################################################
##ScreenPorch
#x = df['ScreenPorch'].reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)



#x = (df['WoodDeckSF'] + df['OpenPorchSF'] + df['3SsnPorch'] + df['ScreenPorch']).reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)
## remove enclosed porch
#
#
## I think we can drop MiscVal and MiscFeature
#x = (df['MiscVal']).reset_index(drop = True).drop(index = [197, 523, 1298])
#sns.jointplot(x,y)
#
#x = (df['MiscFeature']).reset_index(drop = True).drop(index = [197, 523, 1298])
#x.value_counts()
#sns.boxplot(x, y)
#df[df['MiscFeature'] == "NA"]
#
#x = df.reset_index(drop = True).drop(index = [197, 523, 1298])
#a = x[(x['MiscFeature'] == "NA") & (x['SalePrice'] > 0)]['SalePrice']
#b = x[(x['MiscFeature'] != "NA") & (x['SalePrice'] > 0)]['SalePrice']
#sns.distplot( a , color="skyblue", label="NA")
#sns.distplot( b , color="red", label="Not NA")
#plt.legend()


#sns.pairplot(num_features[['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2']])
#sns.pairplot(num_features[['BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF']])
#sns.pairplot(num_features[['GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']])
#sns.pairplot(num_features[['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2']])
#sns.pairplot(num_features[['3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']])

######## OUTLIER REMOVAL########
# TotalBsmtSF
#num_features['TotalBsmtSF'].std()
#num_features['TotalBsmtSF'].mean()
#num_features['TotalBsmtSF'].max()
#num_features['TotalBsmtSF'].min()
#
#sns.boxplot(num_features['TotalBsmtSF'])
#num_features['TotalBsmtSF'].sort_values(ascending = False)
#np.where(stats.zscore(num_features['TotalBsmtSF']) > 3)
#np.where(num_features['TotalBsmtSF'] == max(num_features['TotalBsmtSF']))
#sns.jointplot(x = num_features['TotalBsmtSF'], y = num_features['SalePrice'])
#sns.jointplot(num_features['TotalBsmtSF'].drop(index = [523, 1298]), num_features['SalePrice'].drop(index = [523, 1298]))
#
## 1stFlrSF
#num_features['1stFlrSF'].std()
#num_features['1stFlrSF'].mean()
#num_features['1stFlrSF'].max()
#num_features['1stFlrSF'].min()
#
#sns.boxplot(num_features['1stFlrSF'])
#np.where(stats.zscore(num_features['1stFlrSF']) > 3)
#np.where(num_features['1stFlrSF'] == max(num_features['1stFlrSF']))
#sns.jointplot(x = num_features['1stFlrSF'], y = num_features['SalePrice'])
#sns.jointplot(num_features['1stFlrSF'].drop(index = [523, 1298]), num_features['SalePrice'].drop(index = [523, 1298]))
#
## GrLivArea
#num_features['GrLivArea'].std()
#num_features['GrLivArea'].mean()
#num_features['GrLivArea'].max()
#num_features['GrLivArea'].min()
#
#sns.boxplot(num_features['GrLivArea'])
#num_features['GrLivArea'].sort_values(ascending = False)
#np.where(stats.zscore(num_features['GrLivArea']) > 3)
#np.where(num_features['GrLivArea'] == max(num_features['GrLivArea']))
#sns.jointplot(x = num_features['GrLivArea'], y = num_features['SalePrice'])
#num_features[['GrLivArea','SalePrice']].loc[523]
#sns.jointplot(num_features['GrLivArea'].drop(index = [523, 1298]), num_features['SalePrice'].drop(index = [523, 1298]))
#
#
## EnclosedPorch
#num_features['EnclosedPorch'].std()
#num_features['EnclosedPorch'].mean()
#num_features['EnclosedPorch'].max()
#num_features['EnclosedPorch'].min()
#
#sns.boxplot(num_features['EnclosedPorch'])
#np.where(stats.zscore(num_features['EnclosedPorch']) > 3)
#np.where(num_features['EnclosedPorch'] == max(num_features['EnclosedPorch']))
#sns.jointplot(x = num_features['EnclosedPorch'], y = num_features['SalePrice'])
#sns.jointplot(num_features['EnclosedPorch'].drop(index = 197), num_features['SalePrice'].drop(index = 197))
#num_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2',
#                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',#'LowQualFinSF',
#                'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
#                '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']
#sns.pairplot(num_features, x_vars = num_features.drop(columns = 'SalePrice').columns.tolist(), y_vars = 'SalePrice')
#sns.pairplot(num_features, x_vars = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'], y_vars = 'SalePrice')
#sns.pairplot(num_features, x_vars = ['1stFlrSF', '2ndFlrSF'], y_vars = 'SalePrice')
#sns.pairplot(num_features, x_vars = ['GrLivArea', 'GarageArea'], y_vars = 'SalePrice')
#sns.pairplot(num_features, x_vars = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea'], y_vars = 'SalePrice')
#sns.pairplot(num_features, x_vars = ['LotFrontage', 'LotArea', 'MasVnrArea'], y_vars = 'SalePrice')
#
#
#sns.jointplot(x = num_features['1stFlrSF'], y = num_features['SalePrice'])
#sns.jointplot(x = num_features['2ndFlrSF'], y = num_features['SalePrice'])
#sns.jointplot(x = num_features['1stFlrSF'], y = num_features['2ndFlrSF'])


#train
#
#
#
#
#
#
#
#
#
#
#sns.boxplot(df.TotRmsAbvGrd, y)
#sns.boxplot(df.BsmtFullBath, y)
#sns.boxplot(df.BsmtHalfBath, y)
#sns.boxplot(df.FullBath, y)
#sns.boxplot(df.HalfBath, y)
#
#df.BsmtFullBath + (df.BsmtHalfBath)*0.5 + df.FullBath + (df.HalfBath)*0.5
#
########################################################
##FullBath
#df_fe['FullBath'] = df_fe['FullBath'] +df_fe['BsmtFullBath']
#df_fe.drop('BsmtFullBath', axis = 1, inplace = True)
########################################################
##HalfBath
#df_fe['HalfBath'] = df_fe['HalfBath'] +df_fe['BsmtHalfBath']
#df_fe.drop('BsmtHalfBath', axis = 1, inplace = True)
########################################################
##BedroomAbvGr
########################################################
##KitchenAbvGr
########################################################
#
#
#
#x = df[df['SalePrice'] != 0][['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'SalePrice']].corr()
#
#import seaborn as sns
#sns.heatmap(x, 
#            xticklabels=x.columns.values,
#            yticklabels=x.columns.values)
#
#
#baths = pd.DataFrame(df['BsmtFullBath'].drop(index = test_id) + (df['BsmtHalfBath'].drop(index = test_id))*0.5 + df['FullBath'].drop(index = test_id) + (df['HalfBath'].drop(index = test_id))*0.5, columns = ["Baths"])
#baths = pd.concat([baths, df['SalePrice'].drop(index = test_id)], axis = 1)
#baths
#baths.corr()



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
        print(col, ": Non-positive found. Test lambda = {:g}".format(maxlog), "\n")
    else:
        xt, maxlog, interval = stats.boxcox(train[col], alpha=0.05)
        print("All values in", col, "are positive. Train lambda = {:g}".format(maxlog))
        xt, maxlog, interval = stats.boxcox(test[col], alpha=0.05)
        print("All values in", col, "are positive. Test lambda = {:g}".format(maxlog), "\n")

        
# TRANSFORMATIONS
#sns.jointplot((train['PorchSF'] + abs(min(train['PorchSF']))+1)**porch_maxlog, train['SalePrice'])
xt, lf_maxlog, interval = stats.boxcox(train['LotFrontage'], alpha=0.05)
print("lambda = {:g}".format(lf_maxlog))
train['LotFrontage'] = (train['LotFrontage'])**lf_maxlog
test['LotFrontage'] = (test['LotFrontage'])**lf_maxlog
    
xt, la_maxlog, interval = stats.boxcox(train['LotArea'], alpha=0.05)
print("lambda = {:g}".format(la_maxlog))
train['LotArea'] = (train['LotArea'])**0.3
test['LotArea'] = (test['LotArea'])**0.3

#xt, mas_maxlog, interval = stats.boxcox(train['MasVnrArea'] + abs(min(train['MasVnrArea']))+1, alpha=0.05)
#print("lambda = {:g}".format(mas_maxlog))
#train['MasVnrArea'] = (train['MasVnrArea'] + abs(min(train['MasVnrArea']))+1)**mas_maxlog
#test['MasVnrArea'] = (test['MasVnrArea'] + abs(min(test['MasVnrArea']))+1)**mas_maxlog

xt, totsf_maxlog, interval = stats.boxcox(train['TotalSF'], alpha=0.05)
print("lambda = {:g}".format(totsf_maxlog))
train['TotalSF'] = (train['TotalSF'])**0.33
test['TotalSF'] = (test['TotalSF'])**0.33
#sns.jointplot((train['PorchSF'] + abs(min(train['PorchSF']))+1)**porch_maxlog, train['SalePrice'])


#xt, gar_maxlog, interval = stats.boxcox(train['GarageArea'] + abs(min(train['GarageArea']))+1, alpha=0.05)
#print("lambda = {:g}".format(gar_maxlog))
#train['GarageArea'] = (train['GarageArea'] + abs(min(train['GarageArea']))+1)**0.8
#train['GarageArea'] = (train['GarageArea'] + abs(min(train['GarageArea']))+1)**0.8

#xt, porch_maxlog, interval = stats.boxcox(train['PorchSF'] + abs(min(train['PorchSF']))+1, alpha=0.05)
#print("lambda = {:g}".format(porch_maxlog))
#train['PorchSF'] = (train['PorchSF'] + abs(min(train['PorchSF']))+1)**porch_maxlog
#train['PorchSF'] = (train['PorchSF'] + abs(min(train['PorchSF']))+1)**porch_maxlog


# Log transform SalePrice
train['SalePrice'] = np.log(train['SalePrice'])
#pd.DataFrame(train['SalePrice']).to_csv("data/price_logtransform_prestd.csv", index = False)
## To untransform sale price, np.exp(prediction price)



###############################################################################
############################# STANDARDIZE DATA ################################
###############################################################################
to_scale = train.drop(columns = ['SalePrice'])
colnames = to_scale.columns
scaler = StandardScaler()
scaler.fit(to_scale)
scaled_features = pd.DataFrame(scaler.transform(to_scale),
             columns = to_scale.columns)
scaler2 = StandardScaler()
scaler2.fit(train[['SalePrice']])
scaled_price = pd.DataFrame(scaler2.transform(train[['SalePrice']]),
                            columns = ['SalePrice'])

train = pd.concat([scaled_features, scaled_price], axis = 1)

#train.to_csv("data/train_clean_std_reduced.csv", index = False)

# Standardize test set
test = pd.DataFrame(scaler.transform(test),
             columns = test.columns)

#test.to_csv("data/test_clean_std_reduced.csv", index = False)