#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:41:58 2019

@author: stellakim
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from functions import rmse

# cd ~/Documents/NYC\ Data\ Science\ Academy/HousingPrice_ML_Project/

###############################################################################
############################# STANDARDIZATION #################################
###############################################################################
# Standardize training set
train_fe = pd.read_csv("data/transformed_pre-standardized_fe.csv")

to_scale = train_fe.drop(columns = ['SalePrice'])
colnames = to_scale.columns
scaler = StandardScaler()
scaler.fit(to_scale)
scaled_features = pd.DataFrame(scaler.transform(to_scale),
             columns = to_scale.columns)
scaler2 = StandardScaler()
scaler2.fit(train_fe[['SalePrice']])
scaled_price = pd.DataFrame(scaler2.transform(train_fe[['SalePrice']]),
                            columns = ['SalePrice'])

train = pd.concat([scaled_features, scaled_price], axis = 1)


#scaled_features.to_csv("data/cleaned_standardized_fe.csv", index = False)

# Standardize test set
test_fe = pd.read_csv("data/transformed_pre-standardized_fe_TEST.csv")
housestyle = pd.DataFrame(np.zeros(len(test_fe)), columns = ["HouseStyle_2.5Fin"])
test_fe = pd.concat([test_fe, housestyle], axis = 1)
test_fe = test_fe.reindex(columns = colnames)

test = pd.DataFrame(scaler.transform(test_fe),
             columns = test_fe.columns)


#scaled_features.hist(figsize=(25, 25))
#scaled_features.to_csv("data/cleaned_standardized_TEST.csv", index = False)


############################## IMPORTANT NOTE #################################
# To transform our predicted price BACK:
# (1) Undo standardization using scaler2 object (see line below)
# x = scaler2.inverse_transform(mlr.predict(features_test))
# (2) Raise inverse-standardized to -10 (see line below)
# x**-10


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
#mlr.predict(features)
rmse(mlr.predict(features), price)

#mlr.predict(features)
#mlr.predict(features_test)
#scaler2.inverse_transform(mlr.predict(features_test))
#x = scaler2.inverse_transform(mlr.predict(features_test))
#pred_price = pd.DataFrame(scaler2.inverse_transform(mlr.predict(features_test))**-10, columns = ['SalePrice'])
#
#train_data['SalePrice'].hist()
#pred_price.hist()
#submission = pd.read_csv("data/test.csv")
#submission = submission[['Id']]
#submission = pd.concat([submission, pred_price], axis = 1)
#submission.to_csv("data/submission_try1.csv", index = False)


# Residuals
residuals = price - mlr.predict(features)
residuals.hist(bins = 50)
#len(residuals)

# Using statsmodels
X_add_constant = sm.add_constant(features)
mlr_stats = sm.OLS(price, X_add_constant)
ans = mlr_stats.fit()
ans.summary()
#######
# try fitting a model with just coefficients with p-value cutoff of 0.05 or less
table = pd.DataFrame(ans.summary().tables[1].data[1:])
table.columns = ['name','coef','std err','t value','p value','2.5% confidence','97.5% confidence']
table = table.astype({'name':str,'coef':float,'std err':float, 't value':float, 'p value':float,'2.5% confidence':float, '97.5% confidence':float})
reduced_features_columns = table[table['p value']<0.05]['name']

reduced_features = features[reduced_features_columns]
reduced_features_test = features_test[reduced_features_columns]
mlr2 = LinearRegression()
mlr2

mlr2.fit(reduced_features, price)
# Beta coefficients for our model
mlr2.coef_
# Intercept for model
mlr2.intercept_
# R^2 score of our linear model
mlr2.score(reduced_features, price)
# We should plot actual and predicted Y values
#mlr.predict(features)
rmse(mlr2.predict(reduced_features), price)

# Bonferroni correction, alpha/N-tests aka alpha/features = 0.05/122 = 0.0004
table[table['p value']<0.0004]['name']
reduced_features2_columns = table[table['p value']<0.0004]['name']

reduced_features2 = features[reduced_features2_columns]
reduced_features2_test = features_test[reduced_features2_columns]
mlr3 = LinearRegression()
mlr3

mlr3.fit(reduced_features2, price)
# Beta coefficients for our model
mlr3.coef_
# Intercept for model
mlr3.intercept_
# R^2 score of our linear model
mlr3.score(reduced_features2, price)
# We should plot actual and predicted Y values
#mlr.predict(features)
rmse(mlr3.predict(reduced_features2), price)

x = scaler2.inverse_transform(mlr3.predict(reduced_features2_test))
### running into a problem here
### we should write a function that manually standardizes and unstandardizes columns, instead of using scaler
### we have 39 columns here, instead of the original 122
pred_price = pd.DataFrame(x**-10, columns = ['SalePrice'])

train_data['SalePrice'].hist()
pred_price.hist()
submission = pd.read_csv("data/test.csv")
submission = submission[['Id']]
submission = pd.concat([submission, pred_price], axis = 1)
submission.to_csv("data/submission_try3_reduced_lm_bonferroni.csv", index = False)









######## Checking multicollinearity of continuous features
#
#feats = features[['Age', 'BedroomAbvGr', 'Gar_Age', 'GarageScore', 'GrLivArea',
#          'KitchenScore', 'LotFrontage','MasVnrArea', 'OverallQualCond',
#          'Porch_Deck_SF', 'Rem_Age', 'TotRmsAbvGrd', 'TotalSF']]
#scores = {}
#ols2 = LinearRegression()
#for feature_name in feats:
#                df2     = features.copy()
#                feature = df2[feature_name].copy()
#                df2.drop(feature_name, axis=1, inplace=True)
#                ols2.fit(df2, feature) #### R^2 score, seeing how model fits without the specific feature
#                scores[feature_name] = ols2.score(df2, feature)   
#scores
#
#sns.barplot(x='index', y='R2', data=pd.DataFrame(scores, index=['R2']).T.reset_index())
#plt.xticks(rotation=45)
#plt.title('$R^2$ of a continuous feature against the other features')
#
#
#features.corr()
#












































































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
alphas = np.linspace(0.001, 0.9, 1000)
#alphas_lasso = np.logspace(-3, 0.1, 1000)
lasso.set_params(normalize=False)
coefs_lasso  = []
intercepts_lasso = []

for alpha in alphas_lasso:
        lasso.set_params(alpha=alpha)
        lasso.fit(features, price)  
        coefs_lasso.append(lasso.coef_)
        intercepts_lasso.append(lasso.intercept_)

coefs_lasso = pd.DataFrame(coefs_lasso, index = alphas_lasso, columns = features.columns)  


title = 'Lasso coefficients as a function of the regularization'
coefs_lasso.plot(semilogx, logx = True, title = title, legend = False)
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'slope values')

plt.plot(intercepts_lasso)


#alphas = np.linspace(.0001, .01, 100)
lasso_cv = LassoCV(alphas = alphas, cv = 10, max_iter = 10000, normalize = False)
lasso_cv.fit(features,price)
plt.plot(lasso_cv.alphas_, lasso_cv.mse_path_)
plt.xlim((0, .01)) 

max(scores)
#Use code below for test set
#lasso.set_params(alpha = lasso_cv.alpha_)
#lasso.fit(features,price)
#mean_squared_error(price_test, lasso.predict(features))






