#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:21:22 2019

@author: stellakim
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from functions import rmse, rmse_kaggle

features = train.drop(columns = 'SalePrice')
price = train['SalePrice']
features_test = test

Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=5)
grid_search.fit(features, price)
grid_search.best_params_

svr.set_params(C = 10, gamma = 0.001)
svr.fit(features, price)
svr.predict(features)

rmse(svr.predict(features), price)
rmse_kaggle(actual_price, np.exp(scaler2.inverse_transform(svr.predict(features))))

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(scaler2.inverse_transform(svr.predict(features))) , color="red", label="Predicted Price")
plt.legend()




#sns.distplot( actual_price , color="skyblue", label="Actual Price")
#sns.distplot( np.exp(scaler2.inverse_transform(ridge.predict(features))) , color="red", label="Predicted Price")
#plt.legend()
#
#sns.distplot( actual_price , color="skyblue", label="Actual Price")
#sns.distplot( np.exp(scaler2.inverse_transform(lasso.predict(features))) , color="green", label="Predicted Price")
#plt.legend()
#
#sns.distplot( actual_price , color="skyblue", label="Actual Price")
#sns.distplot( np.exp(scaler2.inverse_transform(elastic_net.predict(features))) , color="purple", label="Predicted Price")
#plt.legend()
#
#sns.distplot( actual_price , color="skyblue", label="Actual Price")
#sns.distplot( np.exp(scaler2.inverse_transform(svr.predict(features))) , color="yellow", label="Predicted Price")
#plt.legend()




#rmse_kaggle(actual_price, np.exp(scaler2.inverse_transform(ridge.predict(features))))
#rmse_kaggle(actual_price, np.exp(scaler2.inverse_transform(lasso.predict(features))))
#rmse_kaggle(actual_price, np.exp(scaler2.inverse_transform(elastic_net.predict(features))))
rmse_kaggle(actual_price, np.exp(scaler2.inverse_transform(svr.predict(features))))




############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
x = scaler2.inverse_transform(svr.predict(features_test))
# (2) np.exp()
svr_pred = np.exp(x)
#actual_price.hist(bins = 30)
#elastic_net_pred.hist(bins = 30)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(scaler2.inverse_transform(svr.predict(features))) , color="red", label="Predicted Train Price")
sns.distplot( svr_pred , color="green", label="Predicted Test Price")
plt.legend()


svr_pred = pd.DataFrame(elastic_net_pred, columns = ['SalePrice'])
svr_cv_submission = pd.concat([test_id, elastic_net_pred], axis = 1)
#svr_cv_submission.to_csv("data/reduced_outliersrmvd_svr_cv5_submission.csv", index = False)
