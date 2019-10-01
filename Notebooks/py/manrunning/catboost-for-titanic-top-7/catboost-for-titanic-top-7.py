#!/usr/bin/env python
# coding: utf-8

# Hi, every one. I am almost new to the kaggle, and I do the ML and DL jobs during my work. I find the kaggle as best for me to make it done. With no more words , I used the catboost and Tensorflow to make the Titanic prediction, and I got the top 7% for it, and I want to share with others what I have done and if is there any better suggestion for it , I want to discuss with you! Here we go!

# In[1]:


#import 
import numpy as np
import pandas as pd
import hyperopt
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#get the train and test data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[3]:


#show the train data
train_df.info()


# In[4]:


#show how many the null value for each column
train_df.isnull().sum()


# In[5]:


#for the train data ,the age ,fare and embarked has null value,so just make it -999 for it
#and the catboost will distinguish it
train_df.fillna(-999,inplace=True)
test_df.fillna(-999,inplace=True)


# In[6]:


#now we will get the train data and label
x = train_df.drop('Survived',axis=1)
y = train_df.Survived


# In[7]:


#show what the dtype of x, note that the catboost will just make the string object to categorical 
#object inside
x.dtypes


# In[8]:


#choose the features we want to train, just forget the float data
cate_features_index = np.where(x.dtypes != float)[0]


# In[9]:


#make the x for train and test (also called validation data) 
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=.85,random_state=1234)


# In[10]:


#let us make the catboost model, use_best_model params will make the model prevent overfitting
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)


# In[11]:


#now just to make the model to fit the data
model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))


# In[12]:


#for the data is not so big, we need use the cross-validation(cv) for the model, to find how
#good the model is ,I just use the 10-fold cv
cv_data = cv(model.get_params(),Pool(x,y,cat_features=cate_features_index),fold_count=10)


# In[13]:


#show the acc for the model
print('the best cv accuracy is :{}'.format(np.max(cv_data["b'Accuracy'_test_avg"])))


# In[14]:


#show the model test acc, but you have to note that the acc is not the cv acc,
#so recommend to use the cv acc to evaluate your model!
print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))


# In[15]:


#last let us make the submission,note that you have to make the pred to be int!
pred = model.predict(test_df)
pred = pred.astype(np.int)
submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':pred})


# In[16]:


#make the file to yourself's directory
submission.to_csv('catboost.csv',index=False)


# Above all the catboost have done! catboost is based on boosting method and it is really amazing for the imputer data! 
# At last, If there is any problem in the code, you can tell me, I want to discuss all the possible to better solve the problem.  I also use the feature engineering for my work and use the Tensorflow to build a MLP to solve the problem, maybe because of my pool engineering feature, it may be not a better score(80.5%). I also use the ML method to solve it, and i find it is useful to standard the data to gaussian distribute(mean 0,std =1) but  min_max to 0-1 may not be better.
