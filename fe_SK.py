#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 22:50:38 2019

@author: stellakim
"""
# cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/
train_fe = train_data.copy()

#Set Feature Engineering Flag
#Saturated Features: Flag = 0
#Engineered Features: Flag = 1
flag = 1

#Begin feature extraction
#######################################################
#Col ID
#Remove Col Id
#train_fe = train_fe.drop('Id', axis = 1)
#######################################################
#MSSubClass
#Remove Column
if (flag == 1):
    train_fe = train_fe.drop('MSSubClass', axis = 1)

#Dummify
#Combine classes - 190,85,75,45,180 & 40 into a single class "Other" 
else:
    ls = [190,85,75,45,180,40]
    train_fe['MSSubClass'] = ['Other' if x in ls else x for x in train_fe['MSSubClass'] ]
    dummy_df = pd.get_dummies(train_fe['MSSubClass'], drop_first=True, prefix = 'MSSubClass')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('MSSubClass', axis = 1)
#######################################################
#MSZoning
#Combine classes - 'RH','RM', 'RH', 'FV' into a single class "Residential" 
if (flag == 1):
    ls = ['RL', 'RM', 'RH', 'FV']
    train_fe['MSZoning'] = ['Residential' if x in ls else x for x in train_fe['MSZoning'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['MSZoning'], drop_first=True, prefix = 'MSZoning')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('MSZoning', axis = 1)
#######################################################
#Lot Frontage
#######################################################
#Area
#######################################################
#Street
#Remove Column
if (flag == 1):
    train_fe = train_fe.drop('Street', axis = 1)
else:
    #Dummify
    dummy_df = pd.get_dummies(train_fe['Street'], drop_first=True, prefix = 'Street')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('Street', axis = 1)

#######################################################
#Alley
#Remove Column
if (flag == 1):
    train_fe = train_fe.drop('Alley', axis = 1)
else:
    #Dummify
    dummy_df = pd.get_dummies(train_fe['Alley'], drop_first=True, prefix = 'Alley')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('Alley', axis = 1)

#######################################################
#LotShape
#Combine classes - 'IR1', 'IR2','IR3' into a single class "Other"
if (flag == 1):
    ls = ['IR1', 'IR2','IR3']
    train_fe['LotShape'] = ['Other' if x in ls else x for x in train_fe['LotShape'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['LotShape'], drop_first=True, prefix = 'LotShape')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('LotShape', axis = 1)
#######################################################
#LandContour
#Combine classes - 'Bnk', 'HLS', 'Low' into a single class "NotLvl" 
if (flag == 1):
    ls = ['Bnk', 'HLS', 'Low']
    train_fe['LandContour'] = ['NotLvl' if x in ls else x for x in train_fe['LandContour'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['LandContour'], drop_first=True, prefix = 'LandContour')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('LandContour', axis = 1)
#######################################################
#Utilities
if (flag == 1):
    #Remove Column
    train_fe = train_fe.drop('Utilities', axis = 1)
else:
    #Dummify
    dummy_df = pd.get_dummies(train_fe['Utilities'], drop_first=True, prefix = 'Utilities')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('Utilities', axis = 1)

#######################################################
#LotConfig
#Combine classes - 'FR2', 'FR3' into a single class "Other" 
if (flag == 1):
    ls = ['FR2', 'FR3']
    train_fe['LotConfig'] = ['Other' if x in ls else x for x in train_fe['LotConfig'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['LotConfig'], drop_first=True, prefix = 'LotConfig')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('LotConfig', axis = 1)
#######################################################
#LandSlope
#Dummify
if (flag == 1):
    dummy_df = pd.get_dummies(train_fe['LandSlope'], drop_first=True, prefix = 'LandSlope')
    train_fe = pd.concat([train_fe, dummy_df], axis = 1)
    train_fe = train_fe.drop('LandSlope', axis = 1)

#Remove Column
else:
    train_fe = train_fe.drop('LandSlope', axis = 1)
#######################################################
#Condition1
#Combine classes - Norm,Feedr,Artery into a single class "Other" 
if (flag == 1):
    ls1 = ['RRNn','RRAn','RRNe','RRAe']
    ls2 = ['PosN','PosA']
    ls3 = ['Artery', 'Feedr']
    train_fe['Condition1'] = ['RR' if x in ls1 else 'Pos' if x in ls2 else 'HwyStr' if x in ls3 else x for x in train_fe['Condition1'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['Condition1'], drop_first=True, prefix = 'Condition1')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('Condition1', axis = 1)
#######################################################
#Condition2
#Remove Column
if (flag == 1):
    train_fe = train_fe.drop('Condition2', axis = 1)

#######################################################
#BldgType
#Remove Column
if (flag == 1):
    train_fe = train_fe.drop('BldgType', axis = 1)

#######################################################
#HouseStyle
if (flag == 1):
    dummy_df = pd.get_dummies(train_fe['HouseStyle'], drop_first = True, prefix = "HouseStyle")
    train_fe = pd.concat([train_fe, dummy_df], axis = 1) 
    train_fe = train_fe.drop('HouseStyle', axis = 1)

#Remove Column
else:
    train_fe = train_fe.drop('HouseStyle', axis = 1)
#######################################################
#OverallQual
#Create interaction feature OverallQualCond = OverallQual*OverallCond
if (flag == 1):
    train_fe['OverallQualCond'] = train_fe['OverallQual']*train_fe['OverallCond']
    #Remove Column
    train_fe = train_fe.drop('OverallQual', axis = 1)

#######################################################
#OverallCond
#Remove Column 
if (flag == 1):
    train_fe = train_fe.drop('OverallCond', axis = 1)

#######################################################
#YearBuilt
#Create interaction feature Age = YearSold-YearBuilt
if (flag == 1):
    train_fe['Age'] = train_fe['YrSold']-train_fe['YearBuilt']
    #Remove Column
    train_fe = train_fe.drop('YearBuilt', axis = 1)

#######################################################
#YearRemodAdd
#Create interaction feature Age = YearSold-YearBuilt
if (flag == 1):
    train_fe['Rem_Age'] = train_fe['YrSold']-train_fe['YearRemodAdd']
    #Remove Column
    train_fe = train_fe.drop('YearRemodAdd', axis = 1)

#######################################################
#RoofStyle
#Combine classes - 'Flat','Gambrel','Mansard','Shed' into a single class "Other"
if (flag == 1):
    ls = ['Flat','Gambrel','Mansard','Shed']
    train_fe['RoofStyle'] = ['Other' if x in ls else x for x in train_fe['RoofStyle'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['RoofStyle'], drop_first=True, prefix = 'RoofStyle')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('RoofStyle', axis = 1)
#######################################################
#RoofMatl
#Combine classes - 'Tar&Grv','WdShngl','WdShake','Roll','Membran','Metal','ClyTile' 
#into a single class "NotShingle"
if (flag == 1):
    ls = ['Tar&Grv','WdShngl','WdShake','Roll','Membran','Metal','ClyTile']
    train_fe['RoofMatl'] = ['NotShingle' if x in ls else x for x in train_fe['RoofMatl'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['RoofMatl'], drop_first=True, prefix = 'RoofMatl')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('RoofMatl', axis = 1)
#######################################################
#Exterior1st
#Combine classes - 'Wd Sdng','Plywood','WdShing' into a single class "wood"
if (flag == 1):
    ls = ['Wd Sdng','Plywood','WdShing']
    train_fe['Exterior1st'] = ['Wood' if x in ls else x for x in train_fe['Exterior1st'] ]

#Combine classes - 'Stucco','AsbShng','Stone','BrkComm','AsphShn','ImStucc','CBlock' 
#into a single class "other"
if (flag == 1):
    ls = ['Stucco','AsbShng','Stone','BrkComm','AsphShn','ImStucc','CBlock']
    train_fe['Exterior1st'] = ['Other' if x in ls else x for x in train_fe['Exterior1st'] ]


#Dummify
dummy_df = pd.get_dummies(train_fe['Exterior1st'], drop_first=True, prefix = 'Exterior1st')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('Exterior1st', axis = 1)
#######################################################
#Exterior2nd
#Combine classes - 'Wd Sdng','Plywood','WdShing' into a single class "wood"
if (flag == 1):
    ls = ['Wd Sdng','Plywood','WdShing']
    train_fe['Exterior2nd'] = ['Wood' if x in ls else x for x in train_fe['Exterior2nd'] ]

#Combine classes - 'Stucco','AsbShng','Stone','BrkComm','AsphShn','ImStucc','CBlock' 
#into a single class "other"
if (flag == 1):
    ls = ['Stucco','AsbShng','Stone','BrkComm','AsphShn','ImStucc','CBlock']
    train_fe['Exterior2nd'] = ['Other' if x in ls else x for x in train_fe['Exterior2nd'] ]


#Dummify
dummy_df = pd.get_dummies(train_fe['Exterior2nd'], drop_first=True, prefix = 'Exterior2nd')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('Exterior2nd', axis = 1)
#######################################################
#MasVnrType
#Combine classes - 'BrkFace','BrkCmn'
#into a single class "brick"
if (flag == 1):
    ls = ['BrkFace','BrkCmn']
    train_fe['MasVnrType'] = ['Brick' if x in ls else x for x in train_fe['MasVnrType'] ]


#Dummify
dummy_df = pd.get_dummies(train_fe['MasVnrType'], drop_first=True, prefix = 'MasVnrType')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('MasVnrType', axis = 1)
#######################################################
#MasVnrArea
#######################################################
#ExterQual
#Set Ordinal Mapping
if (flag == 1):
    ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
    train_fe['ExterQual'] = train_fe['ExterQual'].map(ord_map)

#######################################################
#ExterCond
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
train_fe['ExterCond'] = train_fe['ExterCond'].map(ord_map)
#Create interaction feature: ExterQualCond = ExterQual*ExterCond
if (flag == 1):
    train_fe['ExterQualCond'] = train_fe['ExterQual']*train_fe['ExterCond']
    #Remove Columns
    train_fe = train_fe.drop(['ExterQual','ExterCond'], axis = 1)

#######################################################
#Foundation
#Dummify
if (flag == 1):
    ls = ['Slab', 'Stone', 'Wood']
    train_fe['Foundation'] = ['Other' if x in ls else x for x in train_fe['Foundation']]

dummy_df = pd.get_dummies(train_fe['Foundation'], drop_first=True, prefix = 'Foundation')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('Foundation', axis = 1)
#######################################################
#BsmtQual
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
train_fe['BsmtQual'] = train_fe['BsmtQual'].map(ord_map)
#######################################################
#BsmtCond
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
train_fe['BsmtCond'] = train_fe['BsmtCond'].map(ord_map)
#Create interaction feature: BsmtQualCond = BsmtQual*BsmtCond
if (flag == 1):
    train_fe['BsmtQualCond'] = train_fe['BsmtQual']*train_fe['BsmtCond']
    #Remove Columns
    train_fe = train_fe.drop(['BsmtQual','BsmtCond'], axis = 1)

#######################################################
#BsmtExposure
if (flag == 1):
    #Remove Column
    train_fe = train_fe.drop('BsmtExposure', axis = 1)
else:
    #Dummify
    dummy_df = pd.get_dummies(train_fe['BsmtExposure'], drop_first=True, prefix = 'BsmtExposure')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('BsmtExposure', axis = 1)

#######################################################
#BsmtFinType1
if (flag == 1):
    #Remove Column
    train_fe = train_fe.drop('BsmtFinType1', axis = 1)
else:
    #Dummify
    dummy_df = pd.get_dummies(train_fe['BsmtFinType1'], drop_first=True, prefix = 'BsmtFinType1')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('BsmtFinType1', axis = 1)

#######################################################
#BsmtFinSF1
if (flag == 1):
    #Remove Column
    train_fe = train_fe.drop('BsmtFinSF1', axis = 1)

#######################################################
#BsmtFinType2
if (flag == 1):
    #Remove Column
    train_fe = train_fe.drop('BsmtFinType2', axis = 1)
else:
    #Dummify
    dummy_df = pd.get_dummies(train_fe['BsmtFinType2'], drop_first=True, prefix = 'BsmtFinType2')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('BsmtFinType2', axis = 1)    

#######################################################
#BsmtFinSF2
if (flag == 1):
    #Remove Column
    train_fe = train_fe.drop('BsmtFinSF2', axis = 1)

#######################################################
#BsmtUnfSF
#Create interaction feature: Bsmt_fin_perc = (TotalBsmtSF - BsmtUnfSF)/TotalBsmtSF
if (flag == 1):
    train_fe['Bsmt_fin_perc'] = ((train_fe['TotalBsmtSF']-train_fe['BsmtUnfSF'])/train_fe['TotalBsmtSF']).fillna(0)
    #Remove Columns
    train_fe = train_fe.drop('BsmtUnfSF', axis = 1)

#######################################################
#TotalBsmtSF
#Create interaction feature: Bsmt_Score = BsmtQualCond*Bsmt_fin_perc*TotalBsmtSF
if (flag == 1):
    train_fe['Bsmt_Score'] = train_fe['BsmtQualCond']*train_fe['Bsmt_fin_perc']*train_fe['TotalBsmtSF']
    #Remove Columns
    train_fe = train_fe.drop(['BsmtQualCond','Bsmt_fin_perc'], axis = 1)

#######################################################
#Heating
#Combine classes - 'GasW','Grav','Wall','OthW','Floor' into a single class "NotGasA"
if (flag == 1):
    ls = ['GasW','Grav','Wall','OthW','Floor']
    train_fe['Heating'] = ['NotGasA' if x in ls else x for x in train_fe['Heating'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['Heating'], drop_first=True, prefix = 'Heating')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('Heating', axis = 1)
#######################################################
#HeatingQC
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
train_fe['HeatingQC'] = train_fe['HeatingQC'].map(ord_map)
#######################################################
#CentralAir
#Create Binary
if (flag == 1):
    ls = ['Y']
    train_fe['CentralAir_Bin'] = [1 if x in ls else 0 for x in train_fe['CentralAir'] ]
    train_fe = train_fe.drop('CentralAir', axis = 1)

#######################################################
#Electrical
#Combine classes - 'FuseA','FuseF','FuseP','Mix' into a single class "Fuse"
if (flag == 1):
    ls = ['FuseA','FuseF','FuseP','Mix']
    train_fe['Electrical'] = ['Fuse' if x in ls else x for x in train_fe['Electrical'] ]

#Dummify
dummy_df = pd.get_dummies(train_fe['Electrical'], drop_first=True, prefix = 'Electrical')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('Electrical', axis = 1)   
#######################################################
#1stFlrSF
#2ndFlrSF
#Create interaction feature: TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF
if (flag == 1):
    train_fe['TotalSF'] = train_fe['1stFlrSF']+train_fe['2ndFlrSF']+train_fe['TotalBsmtSF']
    #Remove Column
    train_fe = train_fe.drop(['1stFlrSF','2ndFlrSF','TotalBsmtSF'], axis = 1)

#######################################################
#LowQualFinSF
#######################################################
#GrLivArea
#######################################################
# Combining all full baths and all half baths, but keeping them separate
if (flag == 1):
    train_fe['FullBath'] = train_fe['FullBath'] + train_fe['BsmtFullBath']
    train_fe['HalfBath'] = train_fe['HalfBath'] + train_fe['BsmtHalfBath']
    train_fe = train_fe.drop(['BsmtFullBath', 'BsmtHalfBath'], axis = 1)
    #######################################################

#BedroomAbvGr
#######################################################
#KitchenAbvGr
#######################################################
#KitchenQual
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
train_fe['KitchenQual'] = train_fe['KitchenQual'].map(ord_map)
if (flag == 1):
    #Create interaction feature KitchenScore = KitchenQual*KitchenAbvGr
    train_fe['KitchenScore'] = train_fe['KitchenQual']*train_fe['KitchenAbvGr']
    #Remove Column
    train_fe = train_fe.drop(['KitchenQual','KitchenAbvGr'], axis = 1)

#######################################################
#TotRmsAbvGrd
#######################################################
#Functional
#Combine classes -  (1) Typ (2) Min1 Min2 (3) Mod (4) Maj1 Maj2 Sev
if (flag == 1):
    ls1 = ['Min2','Min1']
    ls2 = ['Maj1', 'Maj2', 'Sev']
    train_fe['Functional'] = ['Min' if x in ls1 else 'Maj' if x in ls2 else x for x in train_fe['Functional'] ]
    ord_map = {'Typ': 4, 'Min': 3, 'Mod': 2, 'Maj': 1, 'NA':0}
    train_fe['Functional'] = train_fe['Functional'].map(ord_map)
else: 
    ls = ['Min2','Min1','Mod','Maj1','Maj2','Sev']
    train_fe['Functional'] = ['nonFunc' if x in ls else x for x in train_fe['Functional'] ]
    #Dummify
    dummy_df = pd.get_dummies(train_fe['Functional'], drop_first=True, prefix = 'Functional')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('Functional', axis = 1)  
#######################################################
#Fireplaces
#######################################################
#FireplaceQu
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
train_fe['FireplaceQu'] = train_fe['FireplaceQu'].map(ord_map)
if (flag == 1):
    #Create interaction feature KitchenScore = KitchenQual*KitchenAbvGr
    train_fe['Fireplace_Score'] = train_fe['FireplaceQu']*train_fe['Fireplaces']
    #Remove Column
    train_fe = train_fe.drop(['FireplaceQu','Fireplaces'], axis = 1)
#######################################################
#GarageType
#Combine classes - 'BuiltIn' into a single class "Attchd"
if (flag == 1):
    ls = ['BuiltIn']
    train_fe['GarageType'] = ['Attchd' if x in ls else x for x in train_fe['GarageType'] ]
#Combine classes - 'Basment','CarPort','2Types' into a single class "other"
if (flag == 1):
    ls = ['Basment','CarPort','2Types']
    train_fe['GarageType'] = ['Other' if x in ls else x for x in train_fe['GarageType'] ]
    
#Dummify
dummy_df = pd.get_dummies(train_fe['GarageType'], drop_first=True, prefix = 'GarageType')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('GarageType', axis = 1)
#######################################################
#GarageYrBlt
#Create interaction feature Gar_Age = YearSold-GarageYrBlt
if (flag == 1):
    train_fe['Gar_Age'] = train_fe['YrSold']-train_fe['GarageYrBlt'].replace("NA", np.nan).astype('float')
    train_fe['Gar_Age'].fillna(0, inplace = True)
    #Remove Column
    train_fe = train_fe.drop('GarageYrBlt', axis = 1)
#######################################################
#GarageFinish
#Dummify
dummy_df = pd.get_dummies(train_fe['GarageFinish'], drop_first=True, prefix = 'GarageFinish')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('GarageFinish', axis = 1)   
#######################################################
#GarageCars
#######################################################
#GarageArea
#######################################################
#GarageQual
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
train_fe['GarageQual'] = train_fe['GarageQual'].map(ord_map)
#######################################################
#GarageCond
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
train_fe['GarageCond'] = train_fe['GarageCond'].map(ord_map)
#Create interaction feature: GarageScore = GarageQual*GarageCond*GarageArea
if (flag == 1):
    train_fe['GarageScore'] = train_fe['GarageQual']*train_fe['GarageCond']*train_fe['GarageArea']
    #Remove Columns
    train_fe = train_fe.drop(['GarageQual','GarageCond','GarageArea'], axis = 1)
#######################################################
#PavedDrive
#Dummify
train_fe['PavedDrive'].replace("P", "Y", inplace=True)
dummy_df = pd.get_dummies(train_fe['PavedDrive'], drop_first=True, prefix = 'PavedDrive')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('PavedDrive', axis = 1)
#######################################################
#WoodDeckSF
#######################################################
#OpenPorchSF
#######################################################
#EnclosedPorch
#######################################################
#3SsnPorch
#######################################################
#ScreenPorch
if (flag == 1):
    #Create interaction feature: Porch_Deck_SF = WoodDeckSF+OpenPorchSF+
    #EnclosedPorch+3SsnPorch+ScreenPorch
    train_fe['Porch_Deck_SF'] = (train_fe['WoodDeckSF']+train_fe['OpenPorchSF']+
                                 train_fe['EnclosedPorch']+train_fe['3SsnPorch']+
                                 train_fe['ScreenPorch'])
    #Remove Columns
    train_fe = train_fe.drop(['WoodDeckSF','OpenPorchSF',
                              'EnclosedPorch','3SsnPorch','ScreenPorch'], axis = 1)
#######################################################
#PoolArea
#######################################################
#PoolQC
#Set Ordinal Mapping
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'NA':0}
train_fe['PoolQC'] = train_fe['PoolQC'].map(ord_map)
#Create interaction feature: PoolScore = PoolQC*PoolArea
if (flag == 1):
    train_fe['PoolScore'] = train_fe['PoolQC']*train_fe['PoolArea']
    #Remove Columns
    train_fe = train_fe.drop(['PoolQC','PoolArea'], axis = 1)
#######################################################   
#Fence
if (flag == 1):
    #Remove Column
    train_fe = train_fe.drop('Fence', axis = 1)
else:
    #Dummify
    dummy_df = pd.get_dummies(train_fe['Fence'], drop_first=True, prefix = 'Fence')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('Fence', axis = 1)  
#######################################################
#MiscFeature
if (flag == 1):
    ls = ['Shed']
    train_fe['Shed'] = [1 if x in ls else 0 for x in train_fe['MiscFeature'] ]
    train_fe = train_fe.drop('MiscFeature', axis = 1) 
else:
    #Dummify
    dummy_df = pd.get_dummies(train_fe['MiscFeature'], drop_first=True, prefix = 'MiscFeature')
    train_fe = pd.concat([train_fe, dummy_df], axis=1)
    train_fe = train_fe.drop('MiscFeature', axis = 1) 
#######################################################
#MiscVal
if (flag == 1):
    #Remove Column
    train_fe = train_fe.drop('MiscVal', axis = 1) 
#######################################################
#MoSold
#Dummify
dummy_df = pd.get_dummies(train_fe['MoSold'], drop_first=True, prefix = 'MoSold')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('MoSold', axis = 1) 
#######################################################
#YrSold
#Dummify
dummy_df = pd.get_dummies(train_fe['YrSold'], drop_first=True, prefix = 'YrSold')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('YrSold', axis = 1) 
#######################################################
#SaleType
#Dummify
train_fe['SaleType'].replace(["WD","CWD","VWD"], "WD", regex = True, inplace = True)
train_fe['SaleType'].replace(["Con","ConLw","ConLI", "ConLD"], "Con", regex = True, inplace = True)
dummy_df = pd.get_dummies(train_fe['SaleType'], drop_first=True, prefix = 'SaleType')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('SaleType', axis = 1) 
#######################################################
#SaleCondition
#Dummify
train_fe['SaleCondition'].replace(["Family", "Alloca", "AdjLand"], "Other", regex = True, inplace = True)
dummy_df = pd.get_dummies(train_fe['SaleCondition'], drop_first=True, prefix = 'SaleCondition')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('SaleCondition', axis = 1) 
#######################################################
#SalePrice
#######################################################
#Neighborhood
#Dummify
dummy_df = pd.get_dummies(train_fe['Neighborhood'], drop_first=True, prefix = 'Neighborhood')
train_fe = pd.concat([train_fe, dummy_df], axis=1)
train_fe = train_fe.drop('Neighborhood', axis = 1) 


del (dummy_df, flag, ls, ls1, ls2, ls3, ord_map)

train_fe.columns[train_fe.isnull().any(axis = 0)]