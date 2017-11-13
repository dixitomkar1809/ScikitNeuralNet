
# coding: utf-8

# In[50]:

import numpy as np
import sklearn
import urllib
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[51]:

#car evaluation database 
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"


# In[87]:

#reading the database from the url, and then naming the attributes
Data = pandas.read_csv(filepath_or_buffer=url,names=['buying','maint','doors','persons','lug_boot','safety','class'])


# In[53]:

LabelEncoder = preprocessing.LabelEncoder()


# In[88]:

#converts the strings into numerical values
LabelEncoder.fit(Data['buying'])
LabelEncoder.fit(Data['maint'])
LabelEncoder.fit(Data['doors'])
LabelEncoder.fit(Data['persons'])
LabelEncoder.fit(Data['lug_boot'])
LabelEncoder.fit(Data['safety'])
LabelEncoder.fit(Data['class'])


# In[55]:

Data['buying'] = LabelEncoder.fit_transform(Data['buying'])
Data['maint'] = LabelEncoder.fit_transform(Data['maint'])
Data['doors'] = LabelEncoder.fit_transform(Data['doors'])
Data['persons'] = LabelEncoder.fit_transform(Data['persons'])
Data['lug_boot'] = LabelEncoder.fit_transform(Data['lug_boot'])
Data['safety'] = LabelEncoder.fit_transform(Data['safety'])
Data['class'] = LabelEncoder.fit_transform(Data['class'])


# In[56]:

X = Data.loc[:, Data.columns != 'class']


# In[57]:

y = Data['class']


# In[58]:

#splitting into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[59]:

scaler = StandardScaler()


# In[60]:

scaler.fit(X_train)


# In[61]:

X_train = scaler.transform(X_train) 


# In[62]:

X_test = scaler.transform(X_test)


# In[68]:

mlp = MLPClassifier(hidden_layer_sizes=(6,5,4))


# In[69]:

mlp.fit(X_train, y_train)


# In[80]:

#predictions for testing
predictions = mlp.predict(X_test)


# In[81]:

print(confusion_matrix(y_test,predictions))


# In[82]:

print(classification_report(y_test,predictions))


# In[89]:

#predictions for training
predictions = mlp.predict(X_train)


# In[90]:

print(confusion_matrix(y_train,predictions))


# In[91]:

print(classification_report(y_train,predictions))

