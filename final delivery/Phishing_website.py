#!/usr/bin/env python
# coding: utf-8

# In[47]:



'''
Download the dataset.
Preprocess or clean the data.
Analyze the pre-processed data.
Train the machine with preprocessed data using an appropriate machine learning algorithm.
Save the model and its dependencies.
Build a Web application using a flask that integrates with the model built.
'''


# In[48]:


from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import Image
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import seaborn as sns


# In[50]:


#df = pd.read_csv('../input/mytest.csv')
df = pd.read_csv('dataset_website.csv')

print (df.shape)

#df.dtypes


# In[51]:


# Load data
df.head(3)


# In[52]:


df.info()


# In[53]:


df.isnull().sum()


# In[54]:


df.duplicated().sum()


# In[55]:


df['Google_Index'].value_counts()


# In[56]:


df.mean()


# In[57]:


# filling na values with mean
data = df.fillna(df.mean())

data.head(3)


# In[58]:


data.isnull().any()


# In[59]:


y = df["Result"].value_counts()
#print (y)
sns.barplot(y.index, y.values)


# In[60]:


y_True = df["Result"][df["Result"] == 1]
print ("Result Percentage = "+str( (y_True.shape[0] / df["Result"].shape[0]) * 100 ))


# In[61]:


df.describe()


# In[62]:


df.groupby(["having_IPhaving_IP_Address", "Result"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30,10)) 


# In[63]:


df.groupby(["URLURL_Length", "Result"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 


# In[64]:


y = df['Result'].to_numpy().astype(np.int)
y.size


# In[65]:


df.drop(["Result"], axis = 1, inplace=True)


# In[66]:


X = df.to_numpy().astype(np.float)


# In[67]:


X


# In[68]:


X.shape


# In[69]:


scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


# In[70]:


X


# In[71]:


def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    # ii -> train
    # jj -> test indices
    for ii, jj in stratified_k_fold: 
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred


# In[72]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[73]:


xtrain,ytrain,xtest,ytest=train_test_split(X,y,test_size=0.25,random_state=123)

rf=ensemble.RandomForestClassifier(max_depth=8,n_estimators=5)
rf_cv_score=cross_val_score(estimator=rf,X=xtrain,y=xtest,cv=5)
print(rf_cv_score)


# In[74]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain, xtest)
LR_score=cross_val_score(estimator=lr,X=xtrain,y=xtest,cv=5)
print(LR_score)


# In[75]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR_score=cross_val_score(estimator=LR,X=xtrain,y=xtest,cv=5)
print(LR_score)


# In[76]:


from sklearn.neighbors import KNeighborsClassifier
knc =  KNeighborsClassifier()
knc.fit(xtrain, xtest)
knn_score=cross_val_score(estimator=knc,X=xtrain,y=xtest,cv=5)
print(knn_score)


# In[77]:


from sklearn.svm import LinearSVC
SVC=  LinearSVC()
svc_score=cross_val_score(estimator=SVC,X=xtrain,y=xtest,cv=5)
print(svc_score)


# In[78]:


from sklearn.tree import DecisionTreeClassifier
decTree = DecisionTreeClassifier(max_depth=6, random_state=0)
dt_cv_score=cross_val_score(estimator=decTree,X=xtrain,y=xtest,cv=5)
print(dt_cv_score)


# In[79]:


print("DT", dt_cv_score)
print("RF", rf_cv_score)
print("SCM", svc_score)
print("kNN ", knn_score)
print("LR", LR_score)


# In[80]:


import pickle
with open('model_pkl_knc', 'wb') as files:
    pickle.dump(knc, files)


# In[81]:


with open('model_pkl_knc' , 'rb') as f:
    lr = pickle.load(f)


# In[82]:


X1 = df.to_numpy().astype(np.float)
t = X1[10]
lr.predict([t]) 


# In[ ]:




