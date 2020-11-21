#!/usr/bin/env python
# coding: utf-8

# ## Name : Vidhi Aggarwal
# 
# 
# ### GRIP - The Spark Foundation
# 
# ### Data Science and Buisness Analytics Intern
# 
# 
# 

# ### Task 1: Prediction Using Supervised Learning
# 
# Predict the percentage of an student based on the no. of study hours.
# 
# We can use R, Python, SAS Enterprise Miner or any other tool. 
# I am going to use python for this task
# 

# ### Step 1: Importing Libraries
# ##### Step 1 involves importing the libraries

# In[3]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Step 2: Import the data

# In[57]:


data = pd.read_csv("http://bit.ly/w-data")
print("Data Imported Successfully\n")
data.head()


# ### Information about the dataset

# In[6]:


data.describe()


# In[7]:


data.shape


# In[8]:


data.info()


# ### Step 3: Cleanning the data

# In[9]:


data.isnull().values.sum()


# #### Since the values are non null, so we will be starting our prediction 

# This is a simple linear regression task as it involves just 2 variables. 

# ### Step 4: Plotting the data

# In[39]:


plt.figure()
sns.scatterplot(x = data["Hours"], y = data["Scores"], COLOR = 'r')
plt.title("Hours v/s Percentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Scored")


# ### Step 5: Prepare the data
# 
# In this step we will spilt the data into two arrays, X and y. Each element of X will be a hours, and the corresponding element of y will be the associated percentage.

# In[11]:


X = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[12]:


print(X)


# ### Splitting testing and training data

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:



X_train,X_test, y_train, y_test =train_test_split(X,y, test_size = 0.2, random_state = 0)


# ### Here we can see the values of our split data

# In[44]:


print(X_train)


# In[45]:


print(X_test)


# In[46]:


print(y_train)


# In[47]:


print(y_test)


# ### We initialize a linear regression model and fit it on the Train dataset.

# In[48]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# ### Step 6: Plotting the Regression Line

# In[49]:



print(model.coef_*X)


# In[50]:


# Plotting the regression line
line = model.coef_*X+model.intercept_
plt.scatter(X,y,color = 'r')
plt.plot(X,line)
plt.show()


# ### Step 7: Predicting the Values
# 
# Now the model has been succesuffly build. We will use this model for predicting the values of the dependent variable in the test dataset

# In[51]:


print(X_test)
y_pred = model.predict(X_test)


# ### Actual values

# In[52]:


print(y_test)


# ### Predicted Values

# In[53]:


print(y_pred)


# In[54]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# ### Step 8:Testing the model
# 
# You can test with your data also. But, here we are finding the score of a student if he/she studies for 9.25 hours

# In[55]:


predicted = model.predict([[9.25]])
print("Number of hours studied : 9.25")
print("Predicted Score :{}".format(predicted[0]))


# ### Step 9: Evaluating the model 
# 
# 
# - **R Square** : It measures how much of variability in dependent variable can be explained by the model
# - **Mean Square Error** : It gives you an absolute number on how much your predicted results deviate from the actual number.
# - **Mean absolute error** : It is calculated by sum of absolute value of error.

# In[56]:


from sklearn import metrics
print("Mean Squared Error : ", metrics.mean_squared_error(y_test,model.predict(X_test))) # mean_squared_error command to calculate MSE 
print("r^2 score error: ", metrics.r2_score(y_test,model.predict(X_test))) # Coefficient of Determination (R2)
print("Mean absolute error: ", metrics.mean_absolute_error(y_test, model.predict(X_test)))


# In[ ]:




