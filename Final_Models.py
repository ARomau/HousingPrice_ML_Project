#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:44:06 2019

@author: stellakim
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV,  Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from functions import rmse, rmse_kaggle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy import stats
import warnings
warnings.filterwarnings('ignore') 

################################################################################
################################# READ DATA ####################################
################################################################################

## cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/

scaler2 = StandardScaler()
scaler2.fit(np.log(actual_price).reshape(-1,1))

###############################################################################
################################## Ridge ######################################
###############################################################################

##https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
## Use RidgeCV instead of GridSearchCV to find the optimal alpha

#Ridge
#features = train.drop(columns = 'SalePrice', index = [197, 523, 1298])
#price = train['SalePrice'].drop(index = [197, 523, 1298])
#features_test = test

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
ridge_pred_train = ridge.predict(features)

rmse(ridge_cv.predict(features), price)
rmse_kaggle(actual_price, np.exp(ridge.predict(features)))

#np.sqrt(-cross_val_score(ridge, features, price, cv=5, scoring = "neg_mean_squared_error")).mean()

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(ridge.predict(features)) , color="red", label="Predicted Price")
plt.legend()

sns.residplot(price, ridge_pred_train)
residuals = price - ridge_pred_train
outliers = residuals[np.abs(stats.zscore(residuals)) > 3].index.tolist()
# Outliers where the residuals are more than 3 standard deviations away.

sns.jointplot(actual_price, np.exp(ridge_pred_train))



############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
# (2) np.exp()
ridge_pred = np.exp(ridge.predict(features_test))
#actual_price.hist(bins = 30)
#ridge_pred.hist(bins = 30)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(ridge.predict(features)) , color="red", label="Predicted Train Price")
sns.distplot( ridge_pred , color="green", label="Predicted Test Price")
plt.legend()

ridge_pred = pd.DataFrame(ridge_pred, columns = ['SalePrice'])

ridge_cv_submission = pd.concat([submission_id, ridge_pred], axis = 1)
#ridge_cv_submission.to_csv("data/ridge_some_outliers_Mar3.csv", index = False)

###############################################################################
################################# Lasso CV ####################################
###############################################################################

#https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

# Use LassoCV instead of GridSearchCV to find the optimal alpha

#Lasso
lasso = Lasso()
lasso.fit(features, price)
#print('the lasso intercept is: %.2f' %(lasso.intercept_))
#pd.Series(lasso.coef_, index=features.columns)

alphas = np.logspace(-3, 0.25, 100)


lasso_cv = LassoCV()
lasso_cv.set_params(alphas = alphas, cv = 5)
lasso_cv.fit(features, price)
lasso_cv.score(features, price)
lasso_cv.alpha_
lasso_cv.predict(features)

lasso.set_params(alpha = lasso_cv.alpha_)
lasso.fit(features, price)
lasso_pred_train = lasso.predict(features)

rmse(lasso.predict(features), price)
rmse_kaggle(actual_price, np.exp(lasso.predict(features)))

cross_val_score(lasso, features, price, cv=5)


sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(lasso.predict(features)) , color="red", label="Predicted Price")
plt.legend()

sns.residplot(price, lasso_pred_train)
residuals = price - lasso_pred_train
outliers.extend(residuals[np.abs(stats.zscore(residuals)) > 3].index.tolist())
# Outliers where the residuals are more than 3 standard deviations away.



sns.jointplot(actual_price, np.exp(scaler2.inverse_transform(lasso_pred_train)))


lasso_coef = pd.DataFrame(lasso.coef_, index = features.columns, columns = ["Coefficients"])
lasso_coef.abs()
lasso_coef.sort_values("Coefficients", inplace = True)
top_coef = pd.concat([lasso_coef.head(15), lasso_coef.tail(15)], axis = 0)
top_coef.plot(kind = "barh", figsize=(15,15), title = "Feature Importance")

############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
# (2) np.exp()
lasso_pred = np.exp(lasso.predict(features_test))
#actual_price.hist(bins = 30)
#lasso_pred.hist(bins = 30)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(lasso.predict(features)) , color="red", label="Predicted Train Price")
sns.distplot( lasso_pred , color="green", label="Predicted Test Price")
plt.legend()


lasso_pred = pd.DataFrame(lasso_pred, columns = ['SalePrice'])
lasso_cv_submission = pd.concat([test_id, lasso_pred], axis = 1)
#lasso_cv_submission.to_csv("data/lasso_some_outliers_Mar3.csv", index = False)

###############################################################################
############################## ELASTIC NET ####################################
###############################################################################

#ElasticNet
elastic_net = ElasticNet()
#print('the elastic_net intercept is: %.2f' %(elastic_net.intercept_))
#pd.Series(elastic_net.coef_, index=features.columns)

alphaSize  = 40
rhoSize    = 30
alphas = np.linspace(1e-2, 10, alphaSize)
rhos   = np.linspace(0.001, 0.5, rhoSize) #avoid very small rho by setting 0.1
#elastic_net.set_params(normalize=False)
#coefs  = np.zeros((alphaSize, rhoSize, 254))
#scores = np.zeros((alphaSize, rhoSize))
#for alphaIdx, alpha in enumerate(alphas):
#    for rhoIdx, rho in enumerate(rhos):
#        elastic_net.set_params(alpha = alpha, l1_ratio = rho)
#        elastic_net.fit(features, price)  
#        coefs[alphaIdx, rhoIdx, :] = elastic_net.coef_
#        scores[alphaIdx, rhoIdx] = elastic_net.score(features, price)
#net_scores = pd.DataFrame(scores, index = alphas, columns = rhos)
#max(net_scores.idxmax())


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

elastic_net.set_params(alpha = elastic_net_cv.alpha_, l1_ratio = elastic_net_cv.l1_ratio_)
elastic_net.fit(features, price)
elastic_net_pred_train = elastic_net.predict(features)

rmse(elastic_net.predict(features), price)
rmse_kaggle(actual_price, np.exp(elastic_net.predict(features)))



cross_val_score(elastic_net, features, price, cv=5)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(elastic_net.predict(features)) , color="red", label="Predicted Price")
plt.legend()

sns.residplot(price, elastic_net_pred_train)
residuals = price - elastic_net_pred_train
outliers.extend(residuals[np.abs(stats.zscore(residuals)) > 3].index.tolist())


############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
# (2) np.exp()
elastic_net_pred = np.exp(elastic_net.predict(features_test))
#actual_price.hist(bins = 30)
#elastic_net_pred.hist(bins = 30)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(elastic_net.predict(features)) , color="red", label="Predicted Train Price")
sns.distplot( elastic_net_pred , color="green", label="Predicted Test Price")
plt.legend()


elastic_net_pred = pd.DataFrame(elastic_net_pred, columns = ['SalePrice'])
elastic_net_cv_submission = pd.concat([submission_id, elastic_net_pred], axis = 1)
#elastic_net_cv_submission.to_csv("data/elastic_net_some_outlier_Mar3.csv", index = False)




###############################################################################
################################### SVR #######################################
###############################################################################
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=5)
grid_search.fit(features, price)
grid_search.best_params_

svr.set_params(C = 1, gamma = 0.001)
svr.fit(features, price)
svr_pred_train = svr.predict(features)

rmse(svr.predict(features), price)
rmse_kaggle(actual_price, np.exp(svr.predict(features)))


cross_val_score(svr, features, price, cv=5)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(svr.predict(features)) , color="red", label="Predicted Price")
plt.legend()

sns.residplot(price, svr_pred_train)
residuals = price - svr_pred_train
np.where(np.abs(stats.zscore(residuals)) > 3)
# Outliers where the residuals are more than 3 standard deviations away.

sns.jointplot(actual_price, np.exp(scaler2.inverse_transform(svr_pred_train)))

rmse_kaggle(actual_price, np.exp(scaler2.inverse_transform(svr.predict(features))))
############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
x = scaler2.inverse_transform(svr.predict(features_test))
# (2) np.exp()
svr_pred = np.exp(x)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(scaler2.inverse_transform(svr.predict(features))) , color="red", label="Predicted Train Price")
sns.distplot( svr_pred , color="green", label="Predicted Test Price")
plt.legend()

svr_pred = pd.DataFrame(svr_pred, columns = ['SalePrice'])
svr_cv_submission = pd.concat([test_id, svr_pred], axis = 1)
#svr_cv_submission.to_csv("data/svr_some_outliers_Mar3.csv", index = False)





###############################################################################
############################ GRADIENT BOOSTING ################################
###############################################################################




gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=3, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10).fit(features, price)

gbr_pred_train = gbr.predict(features)


sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(gbr.predict(features)) , color="red", label="Predicted Price")
plt.legend()




sns.residplot(price, gbr_pred_train)
residuals = price - gbr_pred_train
outliers.extend(residuals[np.abs(stats.zscore(residuals)) > 3].index.tolist())







###############################################################################
########################## Outlier Detection ##################################
###############################################################################
sns.residplot(price, ridge_pred_train)
sns.residplot(price, lasso_pred_train)
sns.residplot(price, svr_pred_train)

outliers = list(set(outliers))

features = features.drop(index = outliers)
price = price.drop(index = outliers)
actual_price = actual_price.drop(index = [x-1 for x in outliers])


## Ridge
ridge_cv.fit(features, price)
ridge_cv.score(features, price)
ridge_cv.alpha_
ridge_cv.predict(features)

ridge.set_params(alpha = ridge_cv.alpha_)
ridge.fit(features, price)
ridge_pred_train = ridge.predict(features)

rmse(ridge_cv.predict(features), price)
rmse_kaggle(actual_price, np.exp(ridge.predict(features)))

cross_val_score(ridge, features, price, cv=5)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(ridge.predict(features)) , color="red", label="Predicted Price")
plt.legend()

sns.jointplot(actual_price, np.exp(ridge_pred_train))

# To transform our predicted price BACK:
# (2) np.exp()
ridge_pred = np.exp(ridge.predict(features_test))
#actual_price.hist(bins = 30)
#ridge_pred.hist(bins = 30)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(ridge.predict(features)) , color="red", label="Predicted Train Price")
sns.distplot( ridge_pred , color="green", label="Predicted Test Price")
plt.legend()

ridge_pred = pd.DataFrame(ridge_pred, columns = ['SalePrice'])

ridge_cv_submission = pd.concat([submission_id, ridge_pred], axis = 1)
#ridge_cv_submission.to_csv("data/non_std_Ridge_zscore_removal_Mar3.csv", index = False)



##Lasso
lasso_cv = LassoCV()
lasso_cv.set_params(alphas = alphas, cv = 5)
lasso_cv.fit(features, price)
lasso_cv.score(features, price)
lasso_cv.alpha_
lasso_cv.predict(features)

lasso.set_params(alpha = lasso_cv.alpha_)
lasso.fit(features, price)
lasso_pred_train = lasso.predict(features)

rmse(lasso.predict(features), price)
rmse_kaggle(actual_price, np.exp(lasso.predict(features)))

cross_val_score(lasso, features, price, cv=5)


sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(lasso.predict(features)) , color="red", label="Predicted Price")
plt.legend()

sns.residplot(price, lasso_pred_train)
residuals = price - lasso_pred_train
# Outliers where the residuals are more than 3 standard deviations away.



sns.jointplot(actual_price, np.exp(lasso_pred_train))

############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
# (2) np.exp()
lasso_pred = np.exp(lasso.predict(features_test))
#actual_price.hist(bins = 30)
#lasso_pred.hist(bins = 30)

sns.distplot( actual_price , color="skyblue", label="Actual Price")
sns.distplot( np.exp(lasso.predict(features)) , color="red", label="Predicted Train Price")
sns.distplot( lasso_pred , color="green", label="Predicted Test Price")
plt.legend()


lasso_pred = pd.DataFrame(lasso_pred, columns = ['SalePrice'])
lasso_cv_submission = pd.concat([submission_id, lasso_pred], axis = 1)
#lasso_cv_submission.to_csv("data/non_std_Lasso_zscore_removal_Mar3.csv", index = False)