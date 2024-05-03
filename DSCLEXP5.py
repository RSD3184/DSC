#!/usr/bin/env python
# coding: utf-8

# In[169]:


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")


# In[170]:


df=pd.read_csv("C:\\Users\\srmve\\Downloads\\Twitter.csv")
df.head()


# In[171]:


df.info()


# In[172]:


df.describe()


# In[173]:


df.isnull().sum()


# In[174]:


df['text'].fillna("",inplace=True)
df['text']


# In[175]:


df['location'].fillna("", inplace=True)
df['location']


# In[176]:


df=df.drop(['status_id','created_at'],axis=1)
df.head()


# In[177]:


x=df.iloc[:,:-1]
x


# In[178]:


y=df.iloc[:,-1]
y


# In[179]:


from sklearn. model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split (x, y, random_state=0)


# In[180]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x_train_text = vectorizer.fit_transform(x_train['text'])
x_train_text


# In[181]:


x_test_text = vectorizer. transform(x_test['text'])
x_test_text


# In[182]:


from sklearn. naive_bayes import MultinomialNB
model = MultinomialNB ()
model.fit (x_train_text, y_train)


# In[183]:


y_pred = model. predict(x_test_text)
y_pred


# In[187]:


# Evaluate the model
from sklearn. metrics import accuracy_score, classification_report
accuracy = accuracy_score (y_test, y_pred) * 100
print ("Accuracy of the model is={:.2f}". format(accuracy))


# In[188]:


from sklearn. metrics import classification_report
class_report = classification_report (y_test, y_pred)
print (f"\nClassification Report:\n{class_report}")


# In[ ]:




