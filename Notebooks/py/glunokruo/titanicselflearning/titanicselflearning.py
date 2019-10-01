#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv('../input/train.csv')


# First of all, lets start inspecting our data:

# In[ ]:


dataset.info()


# As we can see from the data, there are missing values for Age, Cabin and Embarked features. For Age feature, we will use mean value for NaN values. For Embarked feature, since it is only 2 rows, we will assign random values for this 2 value. For the Cabin Feature we will try to investigate it in further detail because we have lots of missing data.  

# In[ ]:


from random import choice
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['Embarked'] = dataset['Embarked'].fillna(choice(['S','C','Q']))



# Cabin Features are consists of some letter followed by number. We can say that following numbers dont have an effect on our objective. However, letter in the cabin value can have an effect Pclass feature; therefore, in our objective. For this reason, we will remove number parts of Cabin values and only keep letter parts. Although some Cabin values consists of several letter-number combinations such as "C23 C25 C27'', we will only take first letter of each value for simplicity.

# In[ ]:


dataset['Cabin2']=dataset[dataset['Cabin'].notnull()]['Cabin'].apply(lambda x: str(x)[0])


# In[ ]:


a = dataset.loc[:,['Pclass','Ticket','Cabin2']]
a[a['Cabin2'].notnull()].sort_values(by='Pclass').values
cabins_for_classes =[]
cabins_for_classes.append([a.loc[a['Pclass']==1]['Cabin2'].value_counts()])

cabins_for_classes.append([a.loc[a['Pclass']==2]['Cabin2'].value_counts()])
cabins_for_classes.append([a.loc[a['Pclass']==3]['Cabin2'].value_counts()])
cabins_for_classes

#a.loc(a['Pclass']==1)

#a[a['Cabin'].notnull()].sort_values(by='Ticket').head(30)#cabindeki verilerin ilk harflerini al


# Cabin letters are most probably given according to structure of the ship and since passengers are staying in different areas of the ship because of their ticket class, by looking the values above we can at least assume that 2nd class passengers wil stay at D or E or F cabin and 3rd class passengers will stay at E or F or G. Thus, in the next step, we will randomly assign cabin values based on the Pclass according to the distribution we have. 

# In[ ]:


first_class_sample=['C']*59+['B']*47+['D']*29+['E']*25+['A']*15+['T']
second_class_sample=['F']*8+['E']*4+['D']*4
third_class_sample =['F']*5+['G']*4+['E']*3
for i in range(len(dataset)):
    if str(dataset['Cabin2'][i]) == 'nan':
        if dataset['Pclass'][i] == 1:
            dataset['Cabin2'][i]=choice(first_class_sample)
        elif dataset['Pclass'][i]== 2:
            dataset['Cabin2'][i]=choice(second_class_sample)
        elif dataset['Pclass'][i]== 3:
            dataset['Cabin2'][i]=choice(third_class_sample)

dataset


# In[ ]:


dataset.info()


# In[ ]:


dataset.corr()


# We handled the missing values. Next, we will work on categorical features. Categorical features are Pclass, Sex, Embarked and Cabin2. For each feature, we will add (unique_value-1) dummy variables.  

# In[ ]:


Pclass_dummy=pd.get_dummies(dataset['Pclass'])
sex_dummy=pd.get_dummies(dataset['Sex'])
embarked_dummy=pd.get_dummies(dataset['Embarked'])
cabin2_dummy=pd.get_dummies(dataset['Cabin2'])


# Now, we will remove one dummy column from each of them.

# In[ ]:


Pclass_dummy = Pclass_dummy.drop(columns = [3])
sex_dummy = sex_dummy.drop(columns = ['male'])
embarked_dummy = embarked_dummy.drop(columns = ['S'])
cabin2_dummy = cabin2_dummy.drop(columns= ['T'])


# In[ ]:


Pclass_dummy=Pclass_dummy.rename(columns={ Pclass_dummy.columns[0]: "first_class",Pclass_dummy.columns[1]:'second_class' })
embarked_dummy=embarked_dummy.rename(columns={ embarked_dummy.columns[0]: "Cherbourg",embarked_dummy.columns[1]:'Queenstown' })


# Now our dataset will be ready after we remove unecessary columns and add dummy columns to our dataset. 

# In[ ]:


dataset


# In[ ]:


dataset2= dataset.drop(columns=['PassengerId','Pclass','Name','Sex','Ticket','Cabin','Embarked','Cabin2'])
dataset2


# In[ ]:


dataset3=pd.concat([dataset2, Pclass_dummy, sex_dummy, embarked_dummy, cabin2_dummy],axis=1)
dataset3


# Now, lets split our dataset into train and test sets.

# In[ ]:


X = dataset3.iloc[:,1:]
y= dataset3.iloc[:,0]


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


# In[ ]:


Next we will apply linear regression to our data.


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Now we trained our model, lets see the predictions of this model.

# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred2 = np.around(y_pred,decimals = 0).astype(int)


# Our prediction is completed. now we will observe confusion matrix for our results.

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred2)


# In[ ]:


cm


# In[ ]:


accuracy = (90+52)/(90+15+22+52)


# In[ ]:


accuracy


# In[ ]:


We can see that by using linear regression model, we got 0.79 accuracy.

