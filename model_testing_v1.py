#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:41:58 2019

@author: stellakim
"""


# cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

train = pd.read_csv("data/cleaned_standardized_fe.csv")
test = pd.read_csv("data/cleaned_standardized_TEST.csv")

###############################################################################
######################## Multiple Linear Regression ###########################
###############################################################################
# Using LinearRegression from scikit-learn package
mlr = LinearRegression()
mlr

features = train.drop(columns = 'SalePrice')
price = train['SalePrice']

features_test = test

#X_train.columns[~X_train.columns.isin(X_test.columns)]
#X_test['HouseStyle_2.5Fin'] = np.zeros(1459)

mlr.fit(features, price)
# Beta coefficients for our model
mlr.coef_
# Intercept for model
mlr.intercept_
# R^2 score of our linear model
mlr.score(features, price)
# We should plot actual and predicted Y values
#mlr.predict(X_train)



# Residuals
residuals = price - mlr.predict(features)
residuals.hist(bins = 50)
#len(residuals)

# Using statsmodels
X_add_constant = sm.add_constant(features)
mlr_stats = sm.OLS(price, X_add_constant)
mlr_stats.fit().summary()

####### Checking multicollinearity of continuous features

features.hist(figsize=(25,25))




















































































#Run Model
from sklearn.linear_model import LinearRegression
features = train.drop('SalePrice', axis = 1)
price = train['SalePrice'] 
lm = LinearRegression()
lm.fit(features, price)
residuals = price - lm.predict(features)
#plt.hist(residuals, bins = 100)
#residuals
lm.score(features,price)

print('R^2 is equal to %.3f' %lm.score(features, price))
print('RSS is equal to %.3f' %np.sum((lm.predict(features) - price) ** 2))
print('The intercept is %.3f' %lm.intercept_)
print('The slopes are %s' %lm.coef_)

import statsmodels.api as sm
X_add_constant = sm.add_constant(features)
mlr_stats = sm.OLS(price, X_add_constant)
mlr_stats.fit().summary()



#Ridge
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn import datasets
ridge = Ridge()
ridge.fit(features, price)
print('the ridge intercept is: %.2f' %(ridge.intercept_))
pd.Series(ridge.coef_, index=features.columns)

alphas = np.linspace(0.01, 10, 100)
ridge.set_params(normalize=False)
coefs  = []
scores = []
for alpha in alphas:
        ridge.set_params(alpha=alpha)
        ridge.fit(features, price)  
        coefs.append(ridge.coef_)
        scores.append(ridge.score(features, price))
coefs = pd.DataFrame(coefs, index = alphas, columns = features.columns)  

#plt.rcParams['figure.figsize'] = (10,5)
#for name in coefs.columns:
#    plt.plot(coefs.index, coefs[name], label=name)
#plt.legend(loc=4)   
#plt.xlabel(r'hyperparameter $\lambda$')
#plt.ylabel(r'slope values')
#
#plt.plot(alphas, scores, c='b', label=r'$R^2$')
#plt.legend(loc=1)
#plt.title(r'$R^2$ Drops with Increaing Regularizations')
#plt.xlabel(r'hyperparameter $\lambda$')
#plt.ylabel(r'$R^2$')

max(scores)


#Lasso
lasso  = Lasso()
alphas = np.linspace(0.01, 5, 100)
lasso.set_params(normalize=False)
coefs_lasso  = []

for alpha in alphas:
        lasso.set_params(alpha=alpha)
        lasso.fit(features, price)  
        coefs_lasso.append(lasso.coef_)

coefs_lasso = pd.DataFrame(coefs_lasso, index = alphas, columns = features.columns)  


#for name in coefs_lasso.columns:
#    plt.plot(coefs_lasso.index, coefs_lasso[name], label=name)
#plt.xlabel(r'hyperparameter $\lambda$')
#plt.ylabel(r'slope values')
#plt.legend(loc=1)  
#
#alphas = np.linspace(.0001, .01, 100)
#lasso_cv = LassoCV(alphas = alphas, cv = 10, max_iter = 10000, normalize = False)
#lasso_cv.fit(features,price)
#plt.plot(lasso_cv.alphas_, lasso_cv.mse_path_)
#plt.xlim((0, .01)) 

max(scores)
#Use code below for test set
#lasso.set_params(alpha = lasso_cv.alpha_)
#lasso.fit(features,price)
#mean_squared_error(price_test, lasso.predict(features))






