#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn import metrics


# In[3]:


df = pd.read_csv(r"C:/Users/NIVEDITHA/Downloads/rainfall.csv")
df.describe()


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df=df.fillna(df.mean(numeric_only=True).round(1))


# In[7]:


group = df.groupby('SUBDIVISION')['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
data=group.get_group(('TAMIL NADU'))
data.head()


# In[8]:


data=pd.melt(df, id_vars =['YEAR'], value_vars =['JAN','FEB','MAR','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
data.tail()


# In[9]:


data= data[['YEAR','variable','value']].sort_values(by=['YEAR'])
data.head()


# In[10]:


data.columns=['Year','Month','Rainfall']


# In[11]:


Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
data['Month']=data['Month'].map(Month_map)
data.head(12)


# In[12]:


X=np.asanyarray(data[['Month']]).astype('int')
y=np.asanyarray(data['Rainfall']).astype('int')
print(X.shape)
print(y.shape)


# In[13]:


# splitting the dataset into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[14]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[15]:


from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=5000)
random_forest_model.fit(X_train, y_train)


# In[16]:


y_test_predict=random_forest_model.predict(X_test)
print("-------Test Data--------")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))


# In[17]:


from sklearn.linear_model import Lasso

reg = Lasso(alpha=0.001)
reg.fit(X_train, y_train)


# In[18]:


y_test_predict=reg.predict(X_test)
print("-------Test Data--------")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))


# In[19]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


# In[20]:


# evaluate an ridge regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute


# In[21]:


# use automatically configured the ridge regression algorithm
from numpy import arange

from sklearn.linear_model import RidgeCV


# In[22]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
# fit model
model.fit(X_train, y_train)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)


# In[23]:


model = Ridge(alpha=0.0)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
# evaluate model
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[24]:


model = Ridge(alpha=0.0000)
# fit model
model.fit(X_train, y_train)


# In[25]:


y_test_predict=model.predict(X_test)
print("-------Test Data--------")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))


# In[26]:


from sklearn.tree import DecisionTreeRegressor 
  
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 10) 
  
# fit the regressor with X and Y data
regressor.fit(X_train, y_train)


# In[27]:


y_test_predict=regressor.predict(X_test)
print("-------Test Data--------")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))


# In[28]:


ans=random_forest_model.predict([[4]])
print(ans)


# In[ ]:


X_train.shape


# In[29]:


import pickle


# In[30]:



file = open("model.pkl","wb")
pickle.dump(random_forest_model,file)
file.close()
# print(y_predict)


# In[31]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
dt = pd.read_csv(r"C:/Users/NIVEDITHA/Downloads/Crop_recommendation.csv")
  
# Create feature and target arrays
train=dt[['rainfall']]
target=dt['label']
train=np.array(train)
target=np.array(target)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             train,target, test_size = 0.3, random_state=42)
  
knn = GaussianNB()
  
knn.fit(X_train.reshape(-1,1), y_train)
pred=knn.predict(X_test.reshape(-1,1))
print(accuracy_score(y_test,pred))
  

