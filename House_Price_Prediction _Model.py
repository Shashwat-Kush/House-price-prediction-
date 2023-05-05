#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
#IMPORTING THE REQUIRED LIBRARIES


# In[65]:


#IMPORTING THE CLEANED-UP DATASET

df = pd.read_csv("house_prediction_train_data.csv")


# In[66]:


df.head()


# In[67]:


df.shape


# In[68]:


X = df.drop(['SalePrice'],axis=1)

#DEFINING THE TARGET VARIABLE
y= df['SalePrice']


# In[69]:


X


# In[70]:


#CREATING A LINEAR REGRESSION MODEL

linear_model = LinearRegression()


# In[71]:


#EVALUATING THE MODEL PERFORMANCE USING MSE CRITERIA

mse = cross_val_score(linear_model,X,y,scoring='neg_mean_squared_error',cv=5,error_score= 'raise')


# In[72]:


print(mse)


# In[73]:


#CREATING A RIDGE MODEL

ridge = Ridge()


# In[74]:


#DEFINING A SET OF REGULARIZATION PARAMETERS USED IN RIDGE

params = {'alpha':[1e-15,1e-10,1e-5,1e-2,1,2,5,10,25,50,100,150,200,250,300,500,1000]}


# In[75]:


#GRID-SEARCHCV RETURNS AN ESTIMATOR WITH THE BEST ALPHA PARAMETER VALUE

ridge_regressor = GridSearchCV(estimator = ridge,param_grid = params,scoring = 'neg_mean_squared_error',cv =5)

#ESTIMATOR IS FIT

ridge_regressor.fit(X,y)


# In[76]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[77]:


lasso = Lasso()
params = {'alpha':[1e-15,1e-10,1e-5,1e-2,1,2,5,10,25,50,100,150,200,250,300,500,1000]}
lasso_regressor = GridSearchCV(estimator = lasso, param_grid= params ,scoring = "neg_mean_squared_error",cv=5)
lasso_regressor.fit(X,y)


# In[78]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[79]:


#SPLITTING THE DATASET INTO TRAIN AND TEST DATA

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size =0.3,random_state =0)


# In[80]:


#PREDICTIONS ARE STORED IN THESE VARIABLES

predict_ridge = ridge_regressor.predict(xtest)
predict_lasso = lasso_regressor.predict(xtest)


# In[81]:


#DIFFERENCE BETWEEN BOTH ACTUAL VALUES AND PREDICTED VALUES IS CONVERTED INTO A GRAPH 

#IT SHOWS HOW MANY TIMES IS THE MODEL DEVIATING FROM THE ACTUAL VALUE BY SHOWING THE DIFFERENCES AND COUNT PLOT
sns.displot(ytest-predict_ridge)


# In[82]:


sns.displot(ytest-predict_lasso)


# In[83]:


#SIMILARLY HERE THE MEDIAN VALUE OF DEVIATION FROM THE ACTUAL VALUE IS SHOWN.

print((ytest-predict_ridge).median())


# In[84]:


print((ytest-predict_lasso).median())


# In[ ]:




