#!/usr/bin/env python
# coding: utf-8

# # TASK : 2 SUPERVISED LEARNING

# # In this supervised learning task we have to predict the marks scored by a student is based upon the number of hours they studied.
# 

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score


# # 1. IMPORTING DATA

# In[2]:


url="http://bit.ly/w-data"
students = pd.read_csv(url)


# In[3]:



students.head()


# In[4]:


students.describe()


# # 2.MISSING VALUE

# In[5]:


students.isnull().sum()


# # 3.SHUFFLING & CREATING TRAIN AND TEST SET

# In[6]:


from sklearn.utils import shuffle

students=shuffle(students,random_state=42)
div=int(students.shape[0]/4)
train=students.loc[:3*div+1,:]
test=students.loc[3*div+1:]

train.shape,test.shape


# In[7]:


train.head()


# In[8]:


test.head()


# # 4.SIMPLE MODE

# In[9]:


test["simple_mode"]=train["Scores"].mode()[0]
test["simple_mode"].head()


# In[10]:


simple_mode_accuracy=accuracy_score(test["Scores"],test["simple_mode"])
simple_mode_accuracy


# In[11]:


x=students.drop(["Scores"],axis=1)
y=students["Scores"]
x.shape,y.shape


# # 5.PREPARING THE DATA

# In[12]:



X = students['Hours'].values.reshape(-1,1) 
y = students['Scores'].values


# # 6.SPLITTING TRAINING &TEST SET

# In[13]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# # 7.IMPLEMENTING LINEAR REGRESSION

# In[14]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[15]:



X_test 
y_pred = lr.predict(X_test)


# In[16]:


students = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
students


# In[17]:


students.plot.bar(figsize=(10,8))


# # 8. PREDICTION BASED ON HOUR

# In[18]:


hours = 9.25
pred = lr.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))


# # 9.PREDICTING ERROR

# In[19]:


from sklearn.metrics import mean_absolute_error as mae


# In[20]:


# prdiction  over trains set and calculatin error
train_predict=lr.predict(X_train)
k=mae(train_predict,y_train)
print("Train Mean Absolute Error",k)


# In[21]:


# prdiction  over trains set and calculatin error
test_predict=lr.predict(X_test)
k=mae(test_predict,y_test)
print("Test Mean Absolute Error",k)


# # 10.DATA VISUALISATION

# In[22]:


# TRAINING SET
plt.scatter(X_train,y_train, color = 'r')
plt.plot(X_train, lr.predict(X_train), color = 'g')
plt.title('Hours vs Score')
plt.xlabel('STUDYING HOURS')
plt.ylabel('GAINED SCORES')
plt.show()


# In[23]:


#TESTING SET
plt.scatter(X_test,y_test ,color = 'r')
plt.plot(X_test, lr.predict(X_test), color = 'g')
plt.title('Hours vs Score ')
plt.xlabel('STUDYING HOURS')
plt.ylabel('GAINED SCORES')
plt.show()


# In[24]:


# Plotting the regression line
line = lr.coef_*X+lr.intercept_

# Plotting for the test data
plt.scatter(X, y,color="orange")
plt.plot(X, line);
plt.show()


# In[25]:


lr.coef_


# In[26]:


lr.intercept_


# In[27]:


lr.score(X_test,y_test) 


# In[ ]:




