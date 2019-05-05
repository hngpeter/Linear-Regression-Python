#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression
# 
# A linear regression is a linear approximation of a causal relationship between two or more variables.
# 
# - $Y$ = Dependent variable = predicted
# - $x_1$, $x_2$, $x_3$, $x_k$ = Independent variable = predictors
# 
# The dependent variable Y is a function of the independent variables x1 to k:
# - $Y$ = F($x_1, x_2, x_3,…,x_k$)
# 
# Simple Linear Regression:
# - $y = β_0 + β_1x_1 + ε $
# - $β_1$ = quantifies the effect of $x_1$ on $Y$
# - $β_0$ = constant (for example minimum salary) 
# - ε = error of estimation (residual)
# 
# **NOTE**: Correlation is does not imply causation! 
# - Correlation measures the degree of relationship between 2 variables, also movement together: p(x,y) = p(y,x), graphical presentation is different: single point.
# - Regression measures how one variable affects another, also it only 1 way, a line as a graph.
# 
# 

# In[1]:


from IPython.display import Image
PATH = "/Users/peterhng/Desktop/"
Image(filename = PATH + "Képernyőfotó 2019-04-12 - 18.06.33.png",width=500, height=500)


# ### The whole code 

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sm 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


DF = pd.read_csv("/Users/peterhng/Dropbox/1.01. Simple linear regression.csv")


# In[4]:


DF.head()


# In[5]:


y = DF["GPA"]
x1 = DF["SAT"]


# #### Explore the data 

# In[6]:


plt.scatter(x1,y)
plt.xlabel("SAT", fontsize=20)
plt.ylabel("GPA", fontsize=20)


# In[7]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit
results.summary()


# In[8]:


plt.scatter(x1,y)
yhat = 0.0017*x1+0.275
fig = plt.plot(x1, yhat, lw=2, c="orange")
plt.xlabel("SAT", fontsize=20)
plt.ylabel("GPA", fontsize=20)


# ### ANOVA table
# 
# - Sum of Squares Total 
# - Sum of Squares Regression 
# - Sum of Square Error
# 
# SST = Sum of Squares Total = Observed variables minus the its mean. Measure the total variability of the dataset:
# 
# $\sum\limits_{i=1}^{n} = (y_i-y)^2$
# 
# SSR = Sum of Squares Regression = Total differences of the predicted value and the mean dependent value, measures the explained variablity by your line
# 
# $\sum\limits_{i=1}^{n} = (y_i-y)^2$
# 
# SSE = Sum of Square Error = Measures the unexplained variability by the regression. 
# 
# $\sum\limits_{i=1}^{n} = ε_i^2$
# 
# SST = SSR + SSE
# 
# OLS = Ordinary Least Square --> min SSE
# 
# $R_2$ = $SSR/SST$ 

# In[9]:


PATH = "/Users/peterhng/Desktop/"
Image(filename = PATH + "Képernyőfotó 2019-04-12 - 21.02.14.png",width=500, height=500)


# #### Advanced Method

# In[10]:


y = DF["GPA"]
X = DF["SAT"]
from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[78]:


from sklearn.linear_model import LinearRegression
linmodel = LinearRegression()
linmodel.fit(X_train,y_train)


# In[82]:


predictions = linmodel.predict(X_test)


# In[83]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


confusion_matrix(y_test, predictions)
classification_report(y_test,predictions)

