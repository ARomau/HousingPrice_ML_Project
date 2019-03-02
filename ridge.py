#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:39:44 2019

@author: stellakim
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from functions import rmse, rmse_kaggle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

################################################################################
################################# READ DATA ####################################
################################################################################

## cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/

actual_price = pd.read_csv('data/train.csv')
actual_price = np.array(actual_price['SalePrice'].drop(index = [197, 523, 1298]).astype(float))
scaler2 = StandardScaler()
scaler2.fit(np.log(actual_price).reshape(-1,1))

train = pd.read_csv("data/train_clean_std_full.csv")
test = pd.read_csv("data/test_clean_std_full.csv")

###############################################################################
################################# Ridge CV ####################################
###############################################################################

##https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
## Use RidgeCV instead of GridSearchCV to find the optimal alpha

#Ridge
features = train.drop(columns = 'SalePrice')
price = train['SalePrice']
features_test = test

ridge = Ridge()
ridge.fit(features, price)
#print('the ridge intercept is: %.2f' %(ridge.intercept_))
#pd.Series(ridge.coef_, index=features.columns)
#
alphas = np.logspace(-1, 6, 100)
ridge.set_params(normalize=False)
#coefs  = []
#scores = []
#for alpha in alphas:
#        ridge.set_params(alpha=alpha)
#        ridge.fit(features, price)  
#        coefs.append(ridge.coef_)
#        scores.append(ridge.score(features, price))
#coefs = pd.DataFrame(coefs, index = alphas, columns = features.columns)  
#
#plt.rcParams['figure.figsize'] = (10,5)
#for name in coefs.columns:
#    plt.xscale('log')
#    plt.plot(coefs.index, coefs[name], label=name)
##plt.legend(loc=4)
#plt.title('Ridge coefficients as a function of the regularization')
#plt.xlabel(r'hyperparameter $\lambda$')
#plt.ylabel(r'slope values')

#plt.plot(alphas, scores, c='b', label=r'$R^2$')
#plt.legend(loc=1)
#plt.title(r'$R^2$ Drops with Increaing Regularizations')
#plt.xlabel(r'hyperparameter $\lambda$')
#plt.ylabel(r'$R^2$')

ridge_cv = RidgeCV()
ridge_cv.set_params(alphas = alphas, cv = 5)
ridge_cv.fit(features, price)
ridge_cv.score(features, price)
ridge_cv.alpha_
ridge_cv.predict(features)

ridge.set_params(alpha = ridge_cv.alpha_)
ridge.fit(features, price)
ridge.predict(features)

rmse(ridge_cv.predict(features), price)
rmse_kaggle(actual_price, np.exp(scaler2.inverse_transform(ridge.predict(features))))

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(scaler2.inverse_transform(ridge.predict(features))) , color="red", label="Predicted Price")
plt.legend()


############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
x = scaler2.inverse_transform(ridge.predict(features_test))
# (2) np.exp()
ridge_pred = np.exp(x)
#actual_price.hist(bins = 30)
#ridge_pred.hist(bins = 30)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(scaler2.inverse_transform(ridge.predict(features))) , color="red", label="Predicted Train Price")
sns.distplot( ridge_pred , color="green", label="Predicted Test Price")
plt.legend()


ridge_pred = pd.DataFrame(ridge_pred, columns = ['SalePrice'])
ridge_cv_submission = pd.concat([test_id, ridge_pred], axis = 1)
#ridge_cv_submission.to_csv("data/full_197_1298_outliersrmvd_ridge_cv5_submission.csv", index = False)