#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:50:48 2019

@author: stellakim
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from functions import rmse, rmse_kaggle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

################################################################################
################################# READ DATA ####################################
################################################################################

## cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/

actual_price = pd.read_csv('data/train.csv')
actual_price = np.array(actual_price['SalePrice'].astype(float))
scaler2 = StandardScaler()
scaler2.fit(np.log(actual_price).reshape(-1,1))

train = pd.read_csv("data/train_clean_std_full.csv")
test = pd.read_csv("data/test_clean_std_full.csv")

features = train.drop(columns = 'SalePrice')
price = train['SalePrice']
features_test = test


###############################################################################
############################# Elastic Net CV ##################################
###############################################################################
#ElasticNet
elastic_net = ElasticNet()
elastic_net.fit(features, price)
#print('the elastic_net intercept is: %.2f' %(elastic_net.intercept_))
pd.Series(elastic_net.coef_, index=features.columns)

alphaSize  = 40
rhoSize    = 30
alphas = np.linspace(1e-2, 10, alphaSize)
rhos   = np.linspace(0.001, 0.5, rhoSize) #avoid very small rho by setting 0.1
elastic_net.set_params(normalize=False)
coefs  = np.zeros((alphaSize, rhoSize, 254))
scores = np.zeros((alphaSize, rhoSize))
for alphaIdx, alpha in enumerate(alphas):
    for rhoIdx, rho in enumerate(rhos):
        elastic_net.set_params(alpha = alpha, l1_ratio = rho)
        elastic_net.fit(features, price)  
        coefs[alphaIdx, rhoIdx, :] = elastic_net.coef_
        scores[alphaIdx, rhoIdx] = elastic_net.score(features, price)
net_scores = pd.DataFrame(scores, index = alphas, columns = rhos)
max(net_scores.idxmax())


#plt.rcParams['figure.figsize'] = (10,5)
#for name in coefs.columns:
#    plt.xscale('log')
#    plt.plot(coefs.index, coefs[name], label=name)
##plt.legend(loc=4)
#plt.title('Elastic Net coefficients as a function of the regularization')
#plt.xlabel(r'hyperparameter $\lambda$')
#plt.ylabel(r'slope values')

#plt.plot(alphas, scores, c='b', label=r'$R^2$')
#plt.legend(loc=1)
#plt.title(r'$R^2$ Drops with Increaing Regularizations')
#plt.xlabel(r'hyperparameter $\lambda$')
#plt.ylabel(r'$R^2$')

elastic_net_cv = ElasticNetCV()
elastic_net_cv.set_params(alphas = alphas, l1_ratio = rhos)
elastic_net_cv.fit(features, price)
elastic_net_cv.score(features, price)
elastic_net_cv.alpha_
elastic_net_cv.l1_ratio_
elastic_net_cv.predict(features)
rmse(elastic_net_cv.predict(features), price)

elastic_net_cv.predict(features_test)

elastic_net.set_params(alpha = elastic_net_cv.alpha_, l1_ratio = elastic_net_cv.l1_ratio_)
elastic_net.fit(features, price)
elastic_net.predict(features)

rmse(elastic_net.predict(features), price)
rmse_kaggle(actual_price, np.exp(scaler2.inverse_transform(elastic_net.predict(features))))

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(scaler2.inverse_transform(elastic_net.predict(features))) , color="red", label="Predicted Price")
plt.legend()

############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
x = scaler2.inverse_transform(elastic_net.predict(features_test))
# (2) np.exp()
elastic_net_pred = np.exp(x)
#actual_price.hist(bins = 30)
#elastic_net_pred.hist(bins = 30)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(scaler2.inverse_transform(elastic_net.predict(features))) , color="red", label="Predicted Train Price")
sns.distplot( elastic_net_pred , color="green", label="Predicted Test Price")
plt.legend()


elastic_net_pred = pd.DataFrame(elastic_net_pred, columns = ['SalePrice'])
elastic_net_cv_submission = pd.concat([test_id, elastic_net_pred], axis = 1)
#elastic_net_cv_submission.to_csv("data/full_197_1298_outliersrmvd_elastic_net_cv5_submission.csv", index = False)
