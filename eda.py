
"""
The following code is for exploratory data analysis of
Housing Price Data
"""

#Import Libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Load Data
train = pd.read_csv('./data/train.csv')
#train.info()
#print('Number of Rows:', max(df.count()))
#Print Missing Rows for each column
#print(df.isna().sum())

continuous_cols = train._get_numeric_data().columns
#print(continuous_cols)
X = train[continuous_cols]
X = X.dropna()
#y = train['SalePrice']
#train_cont.to_csv('train_cont.csv',encoding='utf-8')


#sns.set()
#sns.pairplot(train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','TotRmsAbvGrd', 'YearBuilt','YearRemodAdd']], size = 3)
#plt.show();

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif = vif.round(1)
mask = [True if x > 5 else False for x in vif['VIF Factor']]
colinear = vif[mask]
print(colinear)
colinear.to_csv('colinear.csv',encoding='utf-8')

