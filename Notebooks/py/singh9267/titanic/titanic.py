#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

get_ipython().magic(u'matplotlib inline')


# In[ ]:


data_1=pd.read_csv("../input/gender_submission.csv")
data_2=pd.read_csv("../input/train.csv")
data_3=pd.read_csv("../input/test.csv")


# In[ ]:


data_2.columns
data_2["Age"].fillna(value=29.69).head(10)
data_3["Age"].fillna(value=14.181209).head(10)


# In[ ]:


def handle_non_numerical_data(data_3):
    columns=data_3.columns.values
    
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        if data_3[column].dtype!=np.int64 and data_3[column].dtype!=np.float64:
            column_contents=data_3[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
            data_3[column]=list(map(convert_to_int,data_3[column]))
    return data_3
data_3=handle_non_numerical_data(data_3)
data_3.head()


# In[ ]:


def handle_non_numerical_data(data_2):
    columns=data_2.columns.values
    
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        if data_2[column].dtype!=np.int64 and data_2[column].dtype!=np.float64:
            column_contents=data_2[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
            data_2[column]=list(map(convert_to_int,data_2[column]))
    return data_2
data_2=handle_non_numerical_data(data_2)
data_2.head()


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


my_imputer = Imputer()
x= my_imputer.fit_transform(x)
x_test= my_imputer.transform(x_test)


# In[ ]:


x=data_2[['PassengerId', 'Pclass','Sex' ,'Age', 'SibSp',
       'Parch','Ticket','Fare','Cabin','Embarked']]
y=data_2['Survived']
x_test=data_3[['PassengerId', 'Pclass','Sex', 'Age', 'SibSp',
       'Parch','Ticket','Fare','Cabin','Embarked']]
y_test=data_1['Survived']


# In[ ]:


from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(x, y, verbose=False)


# In[ ]:


predictions = my_model.predict(x_test)
print(my_model.score(x_test,y_test)*100)


# In[ ]:


my_model = XGBRegressor(n_estimators=1000, learning_rate=1)
my_model.fit(x, y, early_stopping_rounds=5, 
             eval_set=[(x_test, y_test)], verbose=False)


# In[ ]:


my_model.predict(x_test)
print(my_model.score(x_test,y_test)*100)


# In[ ]:


logrec=LogisticRegression()
logrec.fit(x,y)
array=logrec.predict(x_test)
acc=logrec.score(x_test,y_test)
"Accuracy:{:.2f}%".format(acc*100)


# In[ ]:


my_submission=pd.DataFrame({'PassengerId':data_3.PassengerId,'Survived':array})
my_submission.to_csv('solution.csv',index=False)

