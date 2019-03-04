
#####################################################
############ Multi Linear Regression ################
#####################################################
train = read.csv('train_clean_std_full.csv')

library(corrplot)
library(dplyr)

#Basic numerical EDA for states dataset.
#summary(train)
#sapply(train, sd)
#corr = cor(train)

#Basic graphical EDA for the train dataset.
#plot(train)
#corrplot(corr, method="circle")

#Creating a saturated model (a model with all variables included).
model.saturated = lm(SalePrice ~ ., data = train)

#summary(model.saturated) #Many predictor variables are not significant, yet the
#overall regression is significant.

plot(model.saturated) #Assessing the assumptions of the model.

library(car) #Companion to applied regression.
influencePlot(model.saturated)

#alias(model.saturated)
#vif(model.saturated) #Assessing the variance inflation factors for the variables
#in our model.
#avPlots(model.saturated)

#We can use stepwise regression to help automate the variable selection process.
#Here we define the minimal model, the full model, and the scope of the models
#through which to search:
model.empty = lm(SalePrice ~ 1, data = train) #The model with an intercept ONLY.
model.full = lm(SalePrice ~ ., data = train) #The model with ALL variables.
scope = list(lower = formula(model.empty), upper = formula(model.full))

library(MASS) #The Modern Applied Statistics library.

#Stepwise regression using AIC as the criteria (the penalty k = 2).
#forwardAIC = step(model.empty, scope, direction = "forward",k = 2)
#backwardAIC = step(model.full, scope, direction = "backward", k = 2)
#bothAIC.empty = step(model.empty, scope, direction = "both", k = 2)
bothAIC.full = step(model.full, scope, direction = "both", k = 2)

#Stepwise regression using BIC as the criteria (the penalty k = log(n)).
#forwardBIC = step(model.empty, scope, direction = "forward", k = log(50))
#backwardBIC = step(model.full, scope, direction = "backward", k = log(50))
#bothBIC.empty = step(model.empty, scope, direction = "both", k = log(50))
#bothBIC.full = step(model.full, scope, direction = "both", k = log(50))

#Checking the model summary and assumptions of the reduced model.
summary(bothAIC.full)
#plot(forwardAIC)
#influencePlot(forwardAIC)
#vif(forwardAIC)
#avPlots(forwardAIC)
#confint(forwardAIC)

#Predicting new observations.
bothAIC.full$fitted.values #Returns the fitted values.
test = read.csv('./data/test_clean_std_full.csv')

predict_test = predict(bothAIC.full, test, interval = "confidence") #Construct confidence intervals

write.csv(predict_test, file = "both_backward_submission.csv")
#for the average value of an
#outcome at a specific point.
