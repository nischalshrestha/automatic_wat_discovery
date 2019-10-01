#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hello everyone. This is my first time to compete in a competition here.
# I've learned a lot, and I know that I have still much more to learn.
# I'm glad to share my code with you to have your advices. Any comments with regards to how I can improve is welcome!!! 

# # The point
# 
# 
# Here is some point important in my code need attention:
# 
# 1. I analysed the variable «name», I got the title of people and I classified the title rare the the same class «Autre» (Idea come from [Anisotropic](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python))
# 2. Because of the characteristics of the algorithme in python, I transformed the categorical variable into one-hot-matrix

# # Loading data

# In[1828]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy import stats
import re

from scipy.stats import chi2_contingency

from collections import Counter
import seaborn as sns


# In[1829]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_y = pd.read_csv("../input/gender_submission.csv")


# # Describe data

# In[1830]:


train.head(10)


# ### «SibSp» «Parch» :
# 
# New variable who contains «SibSp» and «Parch»
# 

# In[1831]:


train["FamilleMember"] = train["SibSp"]+train["Parch"]


# In[1832]:


train = train.drop(["SibSp","Parch"],axis=1)


# ### «Name»
# 
# Variable «Name» contians many informations, we want to classify these informations in detail

# In[1833]:


for i in range(train.shape[0]):
    l = re.split("[,|.|()]",train.Name[i])
    train.loc[i,"First Name"] = l[0].strip()
    train.loc[i,"Title"] = l[1].strip()
    train.loc[i,"Last Name"] = l[2].strip()
    try:
        train.loc[i,"Commentaire"] = l[3].strip()
    except:
        train.loc[i,"Commentaire"] = float("nan")


# In[1834]:


train.head()


# In[1835]:


train.describe()


# In[1836]:


train.describe(include='O')


# In[1837]:


Counter(train.Title)


# ### Drop variable
# 
# - As we can see, variables «PassengerId», «Name», «Ticket», «First Name», «Last Name» and «Commentaire» have to many units. Variable «Cabin» have too many miss data and also many units. So we will drop them.
# - «Tivle» contians many units, we will classify the title rare in the same class
#     - Mlle, Lady => Miss
#     - Mme, Ms => Mrs
#     - Les restes => Autre

# In[1838]:


train = train.drop(["PassengerId","Name","Ticket","First Name","Last Name","Commentaire","Cabin"],axis=1)


# In[1839]:


train.Title = train.Title.map({'Sir':'Autre', 'Lady':'Miss', 'Dr':'Autre', 'Jonkheer':'Autre', 'the Countess':'Autre', 'Don':'Autre', 'Mme':'Mrs', 'Mlle':'Miss', 'Major':'Autre', 'Col':'Autre', 'Ms':'Mrs', 'Rev':'Autre', 'Capt':'Autre','Miss':'Miss','Mr':'Mr','Master':'Master','Mrs':'Mrs'})


# In[1840]:


Counter(train.Title)


# ### Let us take a view to the relationship between «Title» and «Survived»

# In[1841]:


t = pd.crosstab(train.Survived,train.Title)


# In[1842]:


x = [1,2,3,4,5]
y_non_survived = t.iloc[0,:]
y_survived = t.iloc[1,:]
p1 = plt.bar(x,y_non_survived,alpha=0.5)
plt.xticks(x,t.columns)
p2 = plt.bar(x,y_survived,bottom=y_non_survived,alpha=0.5)
plt.xticks(x,t.columns)
plt.legend((p1[0],p2[0]),("Not Survived","Survived"))
plt.show()


# In[1843]:


x = [1,2,3,4,5]
y_non_survived = t.iloc[0,:]/t.sum()
y_survived = t.iloc[1,:]/t.sum()
p1 = plt.bar(x,y_non_survived,alpha=0.5)
plt.xticks(x,t.columns)
p2 = plt.bar(x,y_survived,bottom=y_non_survived,alpha=0.5)
plt.xticks(x,t.columns)
plt.legend((p1[0],p2[0]),("Not Survived","Survived"))
plt.show()


# ## Imputing the missing data
# 
# ### «Age»
# 
# Check out if «Age» came from a normal distribution:
#     - If yes, we can impute the missing data with a normal distribution
#     - If no, we need to find other method:
#         1. Use function "random" to have the value
#         2. Create a model who predict the missing value ( this method is complicated, and after have compared with the first method, this one have a worse predict, so I won't show it here. But if you are interested, please let me know, I can put it here later.

# #### Check out «Age»

# In[1844]:


from scipy import stats


# In[1845]:


stats.normaltest(train.loc[train.Age.notna(),"Age"])


# p_value is very small (< 0.05)
# It is unlikely that the variable «Age» came from a normal distribution.

# #### Use function "random" to have the value

# In[1846]:


np.random.seed(12345)
age_avg = train['Age'].mean()
age_std = train['Age'].std()
age_null_count = train['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)


# Here is the distribution of the random value for the missing data and the distribution of data «Age»

# In[1847]:


plt.hist(age_null_random_list,alpha=0.7)
plt.xlim(0,80)
plt.show()


# In[1848]:


train.Age.hist(alpha=0.7)
plt.xlim(0,80)
plt.show()


# In[1849]:


train['Age'][np.isnan(train['Age'])] = age_null_random_list
train['Age'] = train['Age'].astype(int)


# ### «Embarked»
# The missing data have juste two, we use the class with the highest frequency of occurrence : "S"

# In[1850]:


Counter(train.Embarked)


# In[1851]:


train.loc[train.Embarked.isna(),"Embarked"] = "S"


# ### Have a look at the data

# In[1852]:


import seaborn as sns


# In[1853]:


sns.pairplot(train, diag_kind="kde",hue="Survived")
plt.title("Distribution of the variable")
plt.show()


# # Classification

# ## Prepare the training data

# In[1854]:


# Colonnes à supprimer : Cabin, Commentaire, Last Name, Ticket, First Name, Name, PassengerId
train.head()


# In[1855]:


train.describe(include='O')


# In[1856]:


train.describe()


# ### Important : 
# In python, we need to transform the data "str" to "numeric" if we want to use it in algoritme classification.
# Bur when we transform a categorical variable, who contains A,B and C, to 1,2,3, therefore the ordering is altered: 1 < 2 < 3 implies A < B < C, which is not true. So I transform the categorical variable to one-hot-matrix to avoid this problem.

# In[1857]:


train["Survived"] = train.Survived.astype(int)


# In[1858]:


train["Pclass_1"] = train["Pclass"] == 1
train["Pclass_2"] = train["Pclass"] == 2
train["Pclass_3"] = train["Pclass"] == 3


# In[1859]:


train["Sex"] = train["Sex"].map({"male":1,"female":0})
Counter(train.Sex)


# In[1860]:


Counter(train.Embarked)


# In[1861]:


train["Embarked_1"] = train["Embarked"] == 'C'
train["Embarked_2"] = train["Embarked"] == 'Q'
train["Embarked_3"] = train["Embarked"] == 'S'


# In[1862]:


Counter(train.FamilleMember)


# In[1863]:


train["FamilleMember"] = train["FamilleMember"].map({0:'0',1:'1',2:'2',3:'3',4:'>4',5:'>4',6:'>4',7:'>4',10:'>4'})


# In[1864]:


train["Famille_0"] = train["FamilleMember"] == '0'
train["Famille_1"] = train["FamilleMember"] == '1'
train["Famille_2"] = train["FamilleMember"] == '2'
train["Famille_3"] = train["FamilleMember"] == '3'
train["Famille_4"] = train["FamilleMember"] == '>4'


# In[1865]:


Counter(train.Title)


# In[1866]:


train["Title_master"] = train["Title"] == 'Master'
train["Title_miss"] = train["Title"] == 'Miss'
train["Title_mr"] = train["Title"] == 'Mr'
train["Title_mrs"] = train["Title"] == 'Mrs'
train["Title_autre"] = train["Title"] == 'Autre'


# In[1867]:


train["Age_1"] = train["Age"]<4
train["Age_2"] = (4 <= train["Age"])&(train["Age"]<15)
train["Age_3"] = (15 <= train["Age"])&(train["Age"]<30)
train["Age_4"] = (30 <= train["Age"])&(train["Age"]<45)
train["Age_5"] = (45 <= train["Age"])&(train["Age"]<60)
train["Age_6"] = (60 <= train["Age"])


# In[1868]:


train["Fare_1"] = train["Fare"]<20
train["Fare_2"] = (20 <= train["Fare"])&(train["Fare"]<40)
train["Fare_3"] = (40 <= train["Fare"])&(train["Fare"]<100)
train["Fare_4"] = (100 <= train["Fare"])


# ## Prepare test data

# In[1869]:


test.describe()


# In[1870]:


test.describe(include='O')


# ### Impute missing test data

# In[1871]:


np.random.seed(12345)
age_avg = test['Age'].mean()
age_std = test['Age'].std()
age_null_count = test['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
test['Age'][np.isnan(test['Age'])] = age_null_random_list
test['Age'] = test['Age'].astype(int)


# In[1872]:


test["Fare"][test.Fare.isna()] = test["Fare"].mean()


# ### Transform categorical variable to one-hot-matrix

# In[1873]:


test["Pclass_1"] = test["Pclass"] == 1
test["Pclass_2"] = test["Pclass"] == 2
test["Pclass_3"] = test["Pclass"] == 3


# In[1874]:


test["Sex"] = test["Sex"].map({"male":1,"female":0})


# In[1875]:


test["Embarked_1"] = test["Embarked"] == 'C'
test["Embarked_2"] = test["Embarked"] == 'Q'
test["Embarked_3"] = test["Embarked"] == 'S'


# In[1876]:


test["FamilleMember"] = test["SibSp"]+test["Parch"]


# In[1877]:


Counter(test.FamilleMember)


# In[1878]:


test["FamilleMember"] = test["FamilleMember"].map({0:'0',1:'1',2:'2',3:'3',4:'>4',5:'>4',6:'>4',7:'>4',10:'>4'})
test["Famille_0"] = test["FamilleMember"] == '0'
test["Famille_1"] = test["FamilleMember"] == '1'
test["Famille_2"] = test["FamilleMember"] == '2'
test["Famille_3"] = test["FamilleMember"] == '3'
test["Famille_4"] = test["FamilleMember"] == '>4'


# In[1879]:


for i in range(test.shape[0]):
    l = re.split("[,|.|()]",test.Name[i])
    test.loc[i,"First Name"] = l[0].strip()
    test.loc[i,"Title"] = l[1].strip()
    test.loc[i,"Last Name"] = l[2].strip()
    try:
        test.loc[i,"Commentaire"] = l[3].strip()
    except:
        test.loc[i,"Commentaire"] = float("nan")


# In[1880]:


test.Title = test.Title.map({'Sir':'Autre', 'Lady':'Miss', 'Dr':'Autre', 'Jonkheer':'Autre', 'the Countess':'Autre', 'Don':'Autre', 'Mme':'Mrs', 'Mlle':'Miss', 'Major':'Autre', 'Col':'Autre', 'Ms':'Mrs', 'Rev':'Autre', 'Capt':'Autre','Miss':'Miss','Mr':'Mr','Master':'Master','Mrs':'Mrs'})

test["Title_master"] = test["Title"] == 'Master'
test["Title_miss"] = test["Title"] == 'Miss'
test["Title_mr"] = test["Title"] == 'Mr'
test["Title_mrs"] = test["Title"] == 'Mrs'
test["Title_autre"] = test["Title"] == 'Autre'


# In[1881]:


test["Age_1"] = test["Age"]<4
test["Age_2"] = (4 <= test["Age"])&(test["Age"]<15)
test["Age_3"] = (15 <= test["Age"])&(test["Age"]<30)
test["Age_4"] = (30 <= test["Age"])&(test["Age"]<45)
test["Age_5"] = (45 <= test["Age"])&(test["Age"]<60)
test["Age_6"] = (60 <= test["Age"])


# In[1882]:


test["Fare_1"] = test["Fare"]<20
test["Fare_2"] = (20 <= test["Fare"])&(test["Fare"]<40)
test["Fare_3"] = (40 <= test["Fare"])&(test["Fare"]<100)
test["Fare_4"] = (100 <= test["Fare"])


# ## Take a look at the correlation before the analyst

# Variables «Pclass», «Age», «Fare» are all transformed to one-hot-matrix, so we don't need them now. 

# In[1883]:


train = train.loc[:,['Survived','Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_1', 'Embarked_2','Embarked_3'
                     ,'Famille_0', 'Famille_1', 'Famille_2', 'Famille_3', 'Famille_4'
                     ,'Title_master', 'Title_miss', 'Title_mr', 'Title_mrs','Title_autre'
                     ,'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6'
                     ,'Fare_1', 'Fare_2', 'Fare_3', 'Fare_4']]


# It's «Survived» that we want to predict, so we can juste look at the first columns.
# We find that : variables «Embarked_2», «Title_autre», «Age_4», «Age_5», «Age_6», «Fare_2» have the weak correlation with «Survived». For avoid too many variables, we can try to drop them to have a better prediction.

# In[1884]:


colormap = plt.cm.RdBu
x = [i for i in range(train.shape[1])]
plt.figure(figsize=(15,10))
plt.bar(x,train.corr()["Survived"],alpha=0.5)
plt.xticks(x,train.columns,rotation=-45)
plt.show()


# In[1885]:


columns_complete = ['Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_1', 'Embarked_2','Embarked_3'
                     ,'Famille_0', 'Famille_1', 'Famille_2', 'Famille_3', 'Famille_4'
                     ,'Title_master', 'Title_miss', 'Title_mr', 'Title_mrs','Title_autre'
                     ,'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6'
                     ,'Fare_1', 'Fare_2', 'Fare_3', 'Fare_4']

columns_reduced = ['Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_1', 'Embarked_3'
                     ,'Famille_0', 'Famille_1', 'Famille_2', 'Famille_3', 'Famille_4'
                     ,'Title_master', 'Title_miss', 'Title_mr', 'Title_mrs'
                     ,'Age_1', 'Age_2', 'Age_3'
                     ,'Fare_1', 'Fare_3', 'Fare_4']


# 
# ## Decision Tree

# In[1886]:


from sklearn.metrics import accuracy_score


# In[1887]:


from sklearn import tree


# In[1888]:


train_y = train["Survived"].ravel()


# ### With columns complete

# In[1889]:


classifier = tree.DecisionTreeClassifier()
dt_complete = classifier.fit(train.loc[:,columns_complete],train_y)
dt_complete_pred = dt_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],dt_complete_pred)


# ### With columns reduced
# 

# In[1890]:


classifier = tree.DecisionTreeClassifier()
dt_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
dt_reduce_pred = dt_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],dt_reduce_pred)


# ## GaussianNB

# In[1891]:


from sklearn.naive_bayes import GaussianNB


# ### With columns complete

# In[1892]:


classifier = GaussianNB()
nb_complete = classifier.fit(train.loc[:,columns_complete],train_y)
nb_complete_pred = nb_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],nb_complete_pred)


# ### With columns reduced

# In[1893]:


classifier = GaussianNB()
nb_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
nb_reduce_pred = nb_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],nb_reduce_pred)


# ## RandomForest

# In[1894]:


from sklearn.ensemble import RandomForestClassifier


# ### With columns complete

# In[1895]:


classifier = RandomForestClassifier()
rf_complete = classifier.fit(train.loc[:,columns_complete],train_y)
rf_complete_pred = rf_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],rf_complete_pred)


# ### With columns reduced

# In[1896]:


classifier = RandomForestClassifier()
rf_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
rf_reduce_pred = rf_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],rf_reduce_pred)


# ## SVM

# In[1897]:


from sklearn import svm


# ### With columns complete

# #### kernel: rbf

# In[1898]:


classifier = svm.SVC() #rbf
svm_complete = classifier.fit(train.loc[:,columns_complete],train_y)
svm_complete_pred_rbf = svm_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],svm_complete_pred_rbf)


# #### kernel: poly

# In[1899]:


classifier = svm.SVC(kernel='poly') #rbf
svm_complete = classifier.fit(train.loc[:,columns_complete],train_y)
svm_complete_pred_poly = svm_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],svm_complete_pred_poly)


# ### With columns reduced

# #### kernel: rbf

# In[1900]:


classifier = svm.SVC() #rbf
svm_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
svm_reduce_pred_rbf = svm_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],svm_reduce_pred_rbf)


# #### kernel: poly

# In[1901]:


classifier = svm.SVC(kernel='poly') #rbf
svm_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
svm_reduce_pred_poly = svm_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],svm_reduce_pred_poly)


# In[1902]:


data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':svm_reduce_pred_rbf
})
data_to_submit.to_csv("output_2.csv",index=False)


# Thank for your time!!

# In[ ]:




