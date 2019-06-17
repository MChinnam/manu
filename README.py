#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[17]:


data=pd.read_csv("C:/Users/manoc/Downloads/50_Startups (1).csv")
data.columns=['RND','ADMIN','MKT','STATE','PROFIT']

#data.head()


# In[76]:


import seaborn as sb
sb.pairplot(data)


# In[55]:


from sklearn.model_selection import train_test_split
X=data[['RND']]
Y=data[['PROFIT']]


# In[56]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=25)


# In[58]:


from  sklearn.linear_model import LinearRegression
lm=LinearRegression()
model=lm.fit(X_train,Y_train)


# In[66]:


import statsmodels.api as sm

X2=sm.add_constant(X_train)


# In[74]:


out=sm.OLS(Y_train,X2)
fin=out.fit()
print(fin.summary())


# In[103]:


import matplotlib.pyplot as plt
plt.scatter(data.RND,data.PROFIT)
#dir(plt)
plt.xlabel('RND')
plt.ylabel('PROFIT')


# In[108]:


plt.plot(X_train.RND,Y_train.PROFIT,'*', label='original data')
plt.plot(X_train.RND, intercept + slope*X_train.RND, label='REGRESSION LINE')
plt.legend()
plt.show()
print(intercept)

