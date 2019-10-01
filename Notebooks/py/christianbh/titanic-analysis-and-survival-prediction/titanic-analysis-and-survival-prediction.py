#!/usr/bin/env python
# coding: utf-8

# ![](https://www.revell.de/fileadmin/_processed_/csm_05210__I_RMS_TITANIC_347cc45361.jpg)
#                 *Image Credit: https://www.revell.de/en/products/model-building/ships/civilian-ships/id/05210.html*

# Introduction
# ====
# I will be doing a exploratory analysis on the Titanic dataset, and predicting which passengers survived based on the given features in the dataset.

# In[30]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk # models
import seaborn as sns# visualizations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data_train=pd.read_csv('../input/train.csv') #Read train data
data_test=pd.read_csv('../input/test.csv')#Read test data


# In[31]:


print("SHAPE")
print("Training data: ", data_train.shape) #Examine shape of data
print("Testing data: ", data_test.shape)#Examine shape of data
print()

#Examine first 10 rows of data
data_train.head(10)


# In[32]:


data_test.info()


# In[33]:


#Check to see how many null values are in dataframe for each column.
print("NUMBER OF NULLS IN COLUMNS data_train: ")
data_train.isnull().sum()#Takes all null values and displays ammount for each coloumn


# In[34]:


#Check to see how mnay null values are in dataframe for each column.
print("NUMBER OF NULLS IN COLUMNS data_test: ")
data_test.isnull().sum()


#  There is only null/NaN values in three columns. 
#  
# *  For **Age** I will probably take the average age on the ship and fill in those values.
# *  For **Embarked** I will most likely just drop those rows considering there are only two...
# * Im not quite sure what is best to do with **Cabin**. Part of me thinks I should try to do something about the missing values, but there are quite a few missing values and that usually means it should be dropped... I will assess it further.
# 

# ![Cabin Layout](http://www.visualisingdata.com/blog/wp-content/uploads/2013/04/TITANIC.jpg)

# *The approximate time that the Titanic struck the iceberg was at 11:40pm, it then sank entirely at 2:20am. At these times majority of people would be in bed so I feel like it is a crucial to obtain some sort of values for **Cabin**. As you can see the first class was on the upper decks. As you go further down you have majority of second and third class on the lower decks. I think the best approach for dealing with the missing values in the **Cabin** column could potentially be to instead have the deck level 1st, 2nd, or 3rd corresponding to Upper, Middle, and Lower decks.*
# 
# **EDIT: It seems that there is to much data missing in the Cabin feature for it to be beneicial. I initially thought that I could do some sort of feature engineering to fix this, but it did not seem to be benefical. I will remove the Cabin feature and see how much it improves.**
# 1. Dropped Cabin Feature.
# 
# **EDIT2: A new idea came to mind while going through other peoples kernels. It seems that the cabin numbers could be associated with a higher class, and more likely to survive. Thank you** @Nadin Tamer
# 1. Create a boolean column named **CabinBool** showing whether or not each row inside **Cabin** column with a value survived or died.
# 
# 
# 
# 
# 
# My next thought is that there is probably a relationship between **Fare** + **Pclass**.  I first want to visually check for any obvious outliers in a plot...

# In[35]:


# Create CabinBool feature
data_train["CabinBool"] = (data_train["Cabin"].notnull().astype('int'))
data_test["CabinBool"] = (data_test["Cabin"].notnull().astype('int'))


# In[36]:


sns.lmplot(x="PassengerId", y="Fare", data=data_train, fit_reg=True)


# In[37]:


data_train.loc[data_train['Fare'] > 300] #Show all passengers that paid more than 300


# Not sure why they paid so much more... They all paid the same amount as well. Weird. Either way they are outliers in this data and have some unorderly information so they must go.

# In[38]:


data_train = data_train[data_train.Fare < 300]


# In[39]:


sns.lmplot(x="PassengerId", y="Fare", data=data_train, fit_reg=True)


# There we go that looks better!
# 
# Now I will go ahead and drop the **Cabin** column as it has to many missing values.
# I will also go ahead and drop the **PassengerID** column from the training set as it is not benefical for the prediction.

# In[40]:


data_train.drop('Cabin', axis = 1, inplace = True)
data_test.drop('Cabin', axis = 1, inplace = True)

data_train.head() # Check to see if the replacement worked...


# Now I have to replace the empty **Age** column values with a reasonable input. Lets look at the heatmap to see what effects **Age** the most.

# In[41]:


#Calculate correlations
corr=data_train.corr()

#Heatmap
sns.heatmap(corr, cmap="Blues")


# Heatmap analysis
# ========
# The correlations that stand out to me the most in relation to the **Survived** column is **Survived**+**Fare** and **Survived**+**Parch**.

# In[42]:


data_train["Age"].fillna(data_train.groupby("Sex")["Age"].transform("mean"), inplace=True)
data_test['Age'].fillna(data_test.groupby('Sex')['Age'].transform("mean"), inplace=True)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
data_train['AgeGroup'] = pd.cut(data_train["Age"], bins, labels = labels)
data_test['AgeGroup'] = pd.cut(data_test["Age"], bins, labels = labels)

# Map each age value into a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
data_train['AgeGroup'] = data_train['AgeGroup'].map(age_mapping)
data_test['AgeGroup'] = data_test['AgeGroup'].map(age_mapping)


# Drop Age column from each dataset now that new column 'FareGroups' has been made.
data_train = data_train.drop(['Age'], axis = 1)
data_test = data_test.drop(['Age'], axis = 1)


# Above I determined the missing **Age** values by taking the mean of each **Sex** value and filling them in.
# 
# Now its time to fix the values that are missing in the **Embarked** and **Fare** columns in the train & test datatsets...
# I will fill the **Embarked** value with "S" because it is the most reoccuring value in that column. For missing value in the **Fare** column I will replace it with the mean value.

# In[43]:


data_train.loc[data_train.Embarked.isnull()]


# In[44]:


data_train['Embarked'].fillna("S", inplace = True)
data_test['Fare'].fillna(data_test['Fare'].mean(), inplace = True)
                                                            
#Check to see how many null values are in dataframe for each column.
print("NUMBER OF NULLS IN COLUMNS: ")
data_train.isnull().sum()


# *Next I will go ahead and split the **Fare** feature into groupings.
# This idea was originally presented to me by Nadin Tamer, Thank you! (https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner)*
# 
# **EDIT**: This actually wound up lowering my previous score. Will keep it in for future use if needed, but did not actually implement what was mentioned above. **Fare** feature remains the same as prior

# In[45]:


# Split Fare column in each dataset into four different labels.
data_train['FareGroups'] = pd.qcut(data_train['Fare'], 4, labels = [1, 2, 3, 4])
data_test['FareGroups'] = pd.qcut(data_test['Fare'], 4, labels = [1, 2, 3, 4])


# Drop Fare column from each dataset now that new column 'FareGroups' has been made.
data_train = data_train.drop(['Fare'], axis = 1)
data_test = data_test.drop(['Fare'], axis = 1)


# There we go! No empty or out of the ordinary data.
# 
# The last thing we need to do is convert the **Embarked** and **Sex** columns into numerical values.
# I will also be dropping the **PassengerID, Name,** and **Ticket** features as they have do not have a large coorelation to the survival rate.

# In[46]:


data_train = pd.get_dummies(data_train, columns=['Sex', 'Embarked'], drop_first=True)
data_test = pd.get_dummies(data_test, columns=['Sex', 'Embarked'], drop_first=True)

data_train = data_train.drop(["PassengerId","Name","Ticket"], axis=1)
data_test = data_test.drop(['Name','Ticket'], axis=1)
data_test.tail()


# I am going to go ahead and take one last look at each of the datasets before we move on to the modeling to make sure everything looks good.

# In[47]:


data_train.head()


# In[48]:


data_test.tail()


# Now it is time to begin testing different models!
# I need to split my training data into different variables and create a variable for my test data for fitting to the models.

# In[49]:


X_train= data_train.drop(["Survived"], axis=1)
Y_train= data_train.Survived
X_test= data_test.drop(['PassengerId'], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape


# In[50]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[51]:


random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[52]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[53]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_sub = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[54]:


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[55]:


naive_bayes = GaussianNB()
naive_bayes.fit(X_train, Y_train)
Y_pred = naive_bayes.predict(X_test)
acc_naive_bayes = round(naive_bayes.score(X_train, Y_train) * 100, 2)
acc_naive_bayes


# In[56]:


#from xgboost import XGBClassifier

#xgb = XGBClassifier(n_estimators=200)
#xgb.fit(X_train, Y_train)
#Y_pred = xgb.predict(X_test)
#acc_xgb = round(xgb.score(X_train, Y_train)*100, 2)
#acc_xgb


# In[57]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Decision Tree'], 
    
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_naive_bayes, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)


# Looking at the models it looks like **Decision Tree** scores the best! With **Random Forest** close behind it. 
# In previous submissions I have used **Decision Tree**, but for this submission I will try **Random Forest** to see if it will result in a higher prediction score.

# Time to submit.

# In[58]:


submission = pd.DataFrame({"PassengerId": data_test["PassengerId"],
                           "Survived": Y_pred_sub
                          })
submission.to_csv('submit.csv', index=False)


# Finished!
# =====
