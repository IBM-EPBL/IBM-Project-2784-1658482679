#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='vGMdyWin9RVDOvyXW5phzQHH19I0YcnA6UhAU79AUuk6',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'rainfallanalysis-donotdelete-pr-6zopw5u2diq0e7'
object_key = 'rainfall.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head()


# In[3]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='vGMdyWin9RVDOvyXW5phzQHH19I0YcnA6UhAU79AUuk6',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'rainfallanalysis-donotdelete-pr-6zopw5u2diq0e7'
object_key = 'Crop_recommendation.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dt= pd.read_csv(body)
dt.head()


# In[4]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

  
# Create feature and target arrays
train=dt['rainfall']
target=dt['label']
train=np.array(train)
target=np.array(target)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             train,target, test_size = 0.3, random_state=1)
  
knn = GaussianNB()
  
knn.fit(X_train.reshape(-1,1), y_train)
pred=knn.predict(X_test.reshape(-1,1))
print(accuracy_score(y_test,pred))
  


# In[5]:


from sklearn import metrics


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


df=df.fillna(df.mean(numeric_only=True).round(1))


# In[9]:


group = df.groupby('SUBDIVISION')['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
data=group.get_group(('TAMIL NADU'))
data.head()


# In[10]:


data=pd.melt(df, id_vars =['YEAR'], value_vars =['JAN','FEB','MAR','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
data.tail()


# In[11]:


data= data[['YEAR','variable','value']].sort_values(by=['YEAR'])
data.head()


# In[12]:


data.columns=['Year','Month','Rainfall']


# In[13]:


Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
data['Month']=data['Month'].map(Month_map)
data.head(12)


# In[14]:


X=np.asanyarray(data[['Month']]).astype('int')
y=np.asanyarray(data['Rainfall']).astype('int')
print(X.shape)
print(y.shape)


# In[15]:


# splitting the dataset into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[16]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[17]:


from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=5000)
random_forest_model.fit(X_train, y_train)


# In[18]:


y_test_predict=random_forest_model.predict(X_test)
print("-------Test Data--------")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))


# In[19]:


ans=random_forest_model.predict([[4]])
print(ans)


# In[20]:


X_train.shape


# In[45]:


dt1=group.get_group(('WEST BENGAL'))
# data.head()

df1=dt1.melt(['YEAR']).reset_index()
# df.head()

df1= df1[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
# df.head()

df1.columns=['Index','Year','Month','Avg_Rainfall']
Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
df1['Month']=df1['Month'].map(Month_map)
# df.head(12)

df1.drop(columns="Index",inplace=True)

X1=np.asanyarray(df1[['Month']]).astype('int')
y1=np.asanyarray(df1['Avg_Rainfall']).astype('int')

from sklearn.model_selection import train_test_split
X_train1, X_test, y_train1, y_test = train_test_split(X1, y1, test_size=0.3, random_state=10)

random_forest_model1 = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                      min_samples_split=10, n_estimators=800)
random_forest_model1.fit(X_train, y_train)


# In[22]:


dt5=group.get_group(('JAMMU & KASHMIR'))
# data.head()

df5=dt5.melt(['YEAR']).reset_index()
# df.head()

df5= df5[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
# df.head()

df5.columns=['Index','Year','Month','Avg_Rainfall']
Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
df5['Month']=df5['Month'].map(Month_map)
# df.head(12)

df5.drop(columns="Index",inplace=True)

X5=np.asanyarray(df5[['Month']]).astype('int')
y5=np.asanyarray(df5['Avg_Rainfall']).astype('int')

X_train, X_test, y_train, y_test = train_test_split(X5, y5, test_size=0.3, random_state=10)


random_forest_model2 = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                      min_samples_split=10, n_estimators=800)
random_forest_model2.fit(X_train, y_train)


# In[23]:


dt3=group.get_group(("PUNJAB"))
# data.head()

df3=dt3.melt(['YEAR']).reset_index()
# df.head()

df3= df3[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
# df.head()

df3.columns=['Index','Year','Month','Avg_Rainfall']
Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
df3['Month']=df3['Month'].map(Month_map)
# df.head(12)

df3.drop(columns="Index",inplace=True)

X3=np.asanyarray(df3[['Month']]).astype('int')
y3=np.asanyarray(df3['Avg_Rainfall']).astype('int')

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.3, random_state=10)


random_forest_model3 = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                      min_samples_split=10, n_estimators=800)
random_forest_model3.fit(X_train, y_train)
#-------------------------------------------


# In[24]:


pwd


# In[25]:


pip install -U ibm-watson-machine-learning


# In[26]:


from ibm_watson_machine_learning import APIClient
import json


# In[27]:


#Authencate and set space
val_credentials ={
    "apikey" :"6LfbCZ72apZnwHTs9njb0wj2UlpLWNDPIk8wEN8ayRAH",
    "url":"https://us-south.ml.cloud.ibm.com"
}


# In[28]:


val_client = APIClient(val_credentials)
val_client.spaces.list()


# In[29]:


SPACE_ID = "e891031e-546a-4c0c-8e62-af4a620fdea0"


# In[30]:


val_client.set.default_space(SPACE_ID)


# In[31]:


val_client.software_specifications.list(500)


# In[32]:


import sklearn
sklearn.__version__


# In[33]:


MODEL_NAME ='Rainfall_Prediction'
DEPLOYMENT_NAME = 'Rainfall_Deploy'
DEMO_MODEL =random_forest_model


# In[34]:


#set python version
software_spec_uid = val_client.software_specifications


# In[35]:


#set python version

software_spec_uid = val_client.software_specifications.get_id_by_name('runtime-22.1-py3.9')



# In[36]:


#setup model meta
model_props = {
val_client.repository.ModelMetaNames.NAME : MODEL_NAME,
val_client.repository.ModelMetaNames.TYPE : 'scikit-learn_1.0',
val_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID : software_spec_uid
}


# In[37]:


model_details = val_client.repository.store_model(
model = DEMO_MODEL,
meta_props = model_props,
training_data =X_train,
training_target = y_train
)


# In[38]:


#check
model_details


# In[39]:


#model id

model_id = val_client.repository.get_model_id(model_details)
model_id


# In[40]:


#set meta
deployment_props ={
val_client.deployments.ConfigurationMetaNames.NAME :DEPLOYMENT_NAME,
val_client.deployments.ConfigurationMetaNames.ONLINE :{}
}


# In[41]:


#deploy
deployment = val_client.deployments.create(
	artifact_uid = model_id,
	meta_props = deployment_props
)


# In[43]:


MODEL_NAME1 ='Westbengal'
DEMO_MODEL1 =random_forest_model1
DEPLOYMENT_NAME1="RainfallWestBengal"


# In[ ]:


#setup model meta
model_props1 = {
val_client.repository.ModelMetaNames.NAME : MODEL_NAME1,
val_client.repository.ModelMetaNames.TYPE : 'scikit-learn_1.0',
val_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID : software_spec_uid
}


#save model
model_details1 = val_client.repository.store_model(
model = DEMO_MODEL1,
meta_props = model_props1,
training_data =X_train1,
training_target = y_train1
)



#check
model_details1



#model id

model_id1 = val_client.repository.get_model_id(model_details1)
model_id1


#set meta
deployment_props1 ={
val_client.deployments.ConfigurationMetaNames.NAME :DEPLOYMENT_NAME1,
val_client.deployments.ConfigurationMetaNames.ONLINE :{}
}


#deploy
deployment1 = val_client.deployments.create(
	artifact_uid = model_id1,
	meta_props = deployment_props1
)


