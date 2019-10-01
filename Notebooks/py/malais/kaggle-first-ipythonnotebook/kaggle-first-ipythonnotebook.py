#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import ensemble



# In[ ]:


train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
train.head()


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# In[ ]:


train.dtypes


# In[ ]:


train['Cabin'].describe()


# In[ ]:


train['Embarked'].describe()


# In[ ]:


train = train.drop(['Ticket','Cabin'], axis=1)
train = train.dropna() 





# In[ ]:


train.describe()


# In[ ]:


# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

# lets us plot many diffrent shaped graphs together 
ax1 = plt.subplot2grid((2,3),(0,0))
# plots a bar graph of those who surived vs those who did not.               
train.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
ax1.set_xlim(-1, 2)
# puts a title on our graph
plt.title("Distribution of Survival, (1 = Survived)")    

plt.subplot2grid((2,3),(0,1))
plt.scatter(train.Survived, train.Age, alpha=alpha_scatterplot)
# sets the y axis lablea
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survial by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
train.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
ax3.set_ylim(-1, len(train.Pclass.value_counts()))
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel desnsity estimate of the subset of the 1st class passanges's age
train.Age[train.Pclass == 1].plot(kind='kde')    
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

ax5 = plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(train.Embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")


# In[ ]:


fig = plt.figure(figsize=(18,6))

# create a plot of two subsets, male and female, of the survived variable.
# After we do that we call value_counts() so it can be easily plotted as a bar graph. 
# 'barh' is just a horizontal bar graph
ax1 = fig.add_subplot(121)
train.Survived[train.Sex == 'male'].value_counts().plot(kind='barh',label='Male')
train.Survived[train.Sex == 'female'].value_counts().plot(kind='barh', color='#FA2379',label='Female')
ax1.set_ylim(-1, 2) 
plt.title("Who Survived? with respect to Gender, (raw value counts) "); plt.legend(loc='best')


# adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(122)
(train.Survived[train.Sex == 'male'].value_counts()/float(train.Sex[train.Sex == 'male'].size)).plot(kind='barh',label='Male')  
(train.Survived[train.Sex == 'female'].value_counts()/float(train.Sex[train.Sex == 'female'].size)).plot(kind='barh', color='#FA2379',label='Female')
ax2.set_ylim(-1, 2)
plt.title("Who Survived proportionally? with respect to Gender"); plt.legend(loc='best')


# In[ ]:


fig = plt.figure(figsize=(18,4), dpi=1600)
alpha_level = 0.65

# building on the previous code, here we create an additional subset with in the gender subset 
# we created for the survived variable. I know, thats a lot of subsets. After we do that we call 
# value_counts() so it it can be easily plotted as a bar graph. this is repeated for each gender 
# class pair.
ax1=fig.add_subplot(141)
female_highclass = train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female highclass', color='#FA2479', alpha=alpha_level)
ax1.set_xticklabels(["Survived", "Died"], rotation=0)
ax1.set_xlim(-1, len(female_highclass))
plt.title("Who Survived? with respect to Gender and Class"); plt.legend(loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
female_lowclass = train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)
ax2.set_xticklabels(["Died","Survived"], rotation=0)
ax2.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
male_lowclass = train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=alpha_level)
ax3.set_xticklabels(["Died","Survived"], rotation=0)
ax3.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
male_highclass = train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male highclass', alpha=alpha_level, color='steelblue')
ax4.set_xticklabels(["Died","Survived"], rotation=0)
ax4.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')


# In[ ]:


train = train.drop(['Name','PassengerId'], axis=1)


# In[ ]:


train['Pclass'] = train['Pclass'].astype(object)
train['Survived'] = train['Survived'].astype(int)


# In[ ]:


train.dtypes


# In[ ]:


pclass_frame  = pd.get_dummies(train['Pclass'],prefix='Class')
pclass_frame.dtypes
pclass_frame.head()



# In[ ]:


pclass_frame = pclass_frame.drop(['Class_3'], axis=1)
train = train.join(pclass_frame)


# In[ ]:


sex_frame  = pd.get_dummies(train['Sex'])
sex_frame = sex_frame.drop(['male'], axis=1)




# In[ ]:


sex_frame.head()


# In[ ]:





# In[ ]:


train = train.drop(['Sex','Pclass','Embarked'], axis=1)


# In[ ]:


train.head()


# In[ ]:


labels = train.Survived.values


# In[ ]:


train = train.drop(['Survived'], axis=1)


# In[ ]:


train_Array = train.ix[:,:]


# In[ ]:


clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=10)
clf.fit(train_Array, labels)


# In[ ]:


logreg = lm.LogisticRegression(C=1e5)
logreg.fit(train_Array,labels)


# In[ ]:


test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# In[ ]:


predictors = ['Age','SibSp','Parch','Fare','Class_1','Class_2','female']


# In[ ]:


test_sex_frame  = pd.get_dummies(test['Sex'])
test_sex_frame = test_sex_frame.drop(['male'], axis=1)
test = test.join(test_sex_frame)
test_pclass_frame  = pd.get_dummies(test['Pclass'],prefix='Class')
test_pclass_frame = test_pclass_frame.drop(['Class_3'], axis=1)
test = test.join(test_pclass_frame)


# In[ ]:


test = test.drop(['Pclass','Sex'],axis = 1)


# In[ ]:


test = test.fillna(0)


# In[ ]:


predicted_probability = clf.predict(test[predictors])


# In[ ]:


predicted_probability


# In[ ]:


# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predicted_probability
    })


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("malai_submission.csv", index=False)


# In[ ]:




