#!/usr/bin/env python
# coding: utf-8

# ### Hey there, this is my very first kernel on Kaggle!
# I've been procrastinating for the longest time ever. I've never used matplotlib and seaborn, so I'll be trying to explore this dataset with them. In this notebook, I'll play around with the attributes that I intuitively think would help to predict the survival of a Titanic passenger; **Pclass, Age, Sex, sibsp** and **parch**. Do let me know if there's anything I can improve on! 

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
train = pd.read_csv("../input/train.csv")
print("Number of Rows = " + str(len(train)))
print()
print("Number of Rows with missing values by column:")
pd.isnull(train).sum()


# "Age" and "Cabin (cabin number") have the highest number of observations with missing columns. Since the dataset only contains 891 rows, it wouldn't make sense to remove rows with missing cabin values. <br>
# 
# However, we should dig deeper into the survival of passengers by **Age** before deciding to remove them. If age isn't useful in explaining the survival rate, it wouldn't make sense to remove the observations as it would remove potentially useful information from other attributes.

# ## Distribution of Survival Rates by Age

# In[ ]:


age_survived = train.loc[train['Survived']==1, "Age"]
age_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")
age_did_not_survive = train.loc[train['Survived']==0, "Age"]
age_did_not_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")
plt.xlabel("Age")
plt.ylabel("Percentage of Passengers")
plt.legend(loc='upper right')
plt.title("Distribution of Age of Survivors and Non-Survivors")


# * More than half of children between 0-10 year of Age on the Titanic **survived**. 
# * Passengers between 15-30 years of Age have a significantly **higher percentage of non-survivors** as compared to survivors.
# 
# A passenger's age might be useful in predicting his/her survival. Let's remove observations with missing values for Age.

# In[ ]:


train = train[np.isfinite(train['Age'])]
print("Number of Rows = " + str(len(train)))


# ## Survival Rates by Gender and Ticket Class

# In[ ]:


# Survived 
num_males = len(train.loc[train["Sex"]=="male",])
num_females = len(train.loc[train["Sex"]=="female",])

rates = train.loc[train["Survived"]==1, ["Pclass","Sex","PassengerId"]]
rates = pd.DataFrame(rates.groupby(["Pclass","Sex"]).count())
rates.reset_index(inplace=True)  
rates["Percentage"]=0

# adding a percentage column to show the percentage of males and females
for row in range(len(rates)):
    if rates.loc[row,"Sex"]=="male":
        rates.loc[row, "Percentage"] = round((rates.loc[row,"PassengerId"]/num_males)*100,2)
    else:
        rates.loc[row, "Percentage"] = round((rates.loc[row,"PassengerId"]/num_females)*100,2)

sns.set_style("whitegrid")
sns.barplot(x="Pclass", y="Percentage", hue="Sex", data=rates).set_title("Percentage of Survivors by Class")


# In[ ]:


# Did not Survive
num_males = len(train.loc[train["Sex"]=="male",])
num_females = len(train.loc[train["Sex"]=="female",])

rates = train.loc[train["Survived"]==0, ["Pclass","Sex","PassengerId"]]
rates = pd.DataFrame(rates.groupby(["Pclass","Sex"]).count())
rates.reset_index(inplace=True)  
rates["Percentage"]=0 

# adding a percentage column to show the percentage of males and females
for row in range(len(rates)):
    if rates.loc[row,"Sex"]=="male":
        rates.loc[row, "Percentage"] = round((rates.loc[row,"PassengerId"]/num_males)*100,2)
    else:
        rates.loc[row, "Percentage"] = round((rates.loc[row,"PassengerId"]/num_females)*100,2)

sns.set_style("whitegrid")
sns.barplot(x="Pclass", y="Percentage", hue="Sex", data=rates).set_title("Percentage of Non-Survivors by Class")


# What's obvious: 
# * There are highest percentage of **females among survivors**, and **males among non-survivors**. 
# * The higher the class, the more survivors. 
# 
# What's not so obvious:
# * In Pclass 1 and 2, a **higher** percentage of **male** passengers did **not** survive (>10%), and **higher** percentage of **female** passengers survived (>25%).
# * In Pclass 3, a **higher** percentage of both male and female passengers did not survive. 
# 
# After looking at the survival rates for Age, Gender and Ticket Class, it is apparent that females stand a higher chance of survival. However, we should **find out if Ticket Class has precendence over Age**. Did the children survive because they were all in first class, or were they deliberately rescued? Did most of the youths who did not survive come from the lower ticket classes?

# ## Does Ticket Class matter more than Age?

# In[ ]:


num_passengers = len(train)
classes = pd.DataFrame(train.groupby(["Pclass"]).count())
classes.reset_index(inplace=True)
classes["Percentage"] = round(classes["PassengerId"].div(num_passengers)*100,2)

sns.set_style("darkgrid")
sns.pointplot(x="Pclass", y="Percentage", data=classes).set_title("Proportion of Passengers by Ticket Classes")


# ~50% of the Titanic passengers were in the Third Class. 
# 
# ### Number of Survivors and Non-Survivors by Ticket Class

# In[ ]:


survivors_1 = train.loc[(train["Survived"]==1)&(train["Pclass"]==1), "Age"]
survivors_2 = train.loc[(train["Survived"]==1)&(train["Pclass"]==2), "Age"]
survivors_3 = train.loc[(train["Survived"]==1)&(train["Pclass"]==3), "Age"]

survivors_1.plot.hist(fc=(0, 0, 1, 0.5), label="Class 1")
survivors_2.plot.hist(fc=(1, 0, 0, 0.5), label="Class 2")
survivors_3.plot.hist(fc=(0, 1, 0, 0.5), label="Class 3")
plt.xlabel("Age")
plt.legend(loc='upper right')
plt.title("Age Distribution of Survivors by Ticket Class")


# Although only ~25% of passengers were from first class, there is a significantly **higher number of survivors** than the rest of the classes. <br>
# 
# It may also seem ironic that the third class has a higher number of survivors than the second class, but this is explained by the higher proportion of passengers from the third class.

# In[ ]:


survivors_1 = train.loc[(train["Survived"]==0)&(train["Pclass"]==1), "Age"]
survivors_2 = train.loc[(train["Survived"]==0)&(train["Pclass"]==2), "Age"]
survivors_3 = train.loc[(train["Survived"]==0)&(train["Pclass"]==3), "Age"]

survivors_1.plot.hist(fc=(0, 0, 1, 0.5), label="Class 1")
survivors_2.plot.hist(fc=(1, 0, 0, 0.5), label="Class 2")
survivors_3.plot.hist(fc=(0, 1, 0, 0.5), label="Class 3")
plt.xlabel("Age")
plt.legend(loc='upper right')
plt.title("Age Distribution of Non-Survivors by Ticket Class")


# The third class has the highest number of non-survivors, while the first class has the least. However, the previous plots are **based on frequency**, and are not relative to the number of passengers in each ticket class or age groups. <br>
# 
# ### Proportion of Survivors and Non-Survivors per Ticket Class
# As we are looking at absolute numbers so far (frequency instead of percentage), I decided to look at the number of survivors and non-survivors **within each class** in the histograms below.

# In[ ]:


first_class_survived = train.loc[(train["Pclass"]==1)&(train["Survived"]==1), "Age"]
first_class_didnt_survive = train.loc[(train["Pclass"]==1)&(train["Survived"]==0), "Age"]
first_class_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")
first_class_didnt_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")
plt.xlabel("Age")
plt.ylabel("Proportion of Class 1 Passengers")
plt.legend(loc='upper right')
plt.title("Distribution of Age of Passengers in First Class")


# In[ ]:


second_class_survived = train.loc[(train["Pclass"]==2)&(train["Survived"]==1), "Age"]
second_class_didnt_survive = train.loc[(train["Pclass"]==2)&(train["Survived"]==0), "Age"]
second_class_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")
second_class_didnt_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")
plt.xlabel("Age")
plt.ylabel("Proportion of Class 2 Passengers")
plt.legend(loc='upper right')
plt.title("Distribution of Age of Passengers in Second Class")


# In[ ]:


third_class_survived = train.loc[(train["Pclass"]==3)&(train["Survived"]==1), "Age"]
third_class_didnt_survive = train.loc[(train["Pclass"]==3)&(train["Survived"]==0), "Age"]
third_class_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")
third_class_didnt_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")
plt.xlabel("Age")
plt.ylabel("Proportion of Class 3 Passengers")
plt.legend(loc='upper right')
plt.title("Distribution of Age of Passengers in Third Class")


# * A high proportion of children below 15 years old in the *second and third* class survived. 
# * In the *first* class, there isn't a clear relationship between age and survival. Youths (18-25 years of age) and adults in their mid 30s have the highest survival rates.
# <br>
# 
#  **Conclusion: Age has a precedence over Ticket Class (children are given a priority)...most of the time.**  

# ## Survival Rates by Number of Family Members
# 
# 
# Sibsp = number of siblings/spouses on board <br>
# Parch = number of parents/children on board <br>
# <br>
# As siblings, spouses, parents and children are considered family members, I created a new column **nFamily**, which indicates the number of family members a passenger has on board.

# In[ ]:


train["nFamily"] = train["SibSp"] + train["Parch"]
train.head()


# In[ ]:


count = train.groupby(["Survived","nFamily"]).count()
count.reset_index(inplace=True)
count = count[["Survived", "nFamily", "PassengerId"]]

# get percentage
num_passengers = len(train)
count["Percentage"] = round(count["PassengerId"].div(num_passengers),2)

fig,ax = plt.subplots()

for i in range(2):
    ax.plot(count[count.Survived==i].nFamily, count[count.Survived==i].Percentage, label="Survived = "+ str(i))

ax.set_xlabel("Number of Family Members")
ax.set_ylabel("Proportion of Passengers")
ax.legend(loc='best')


# * Almost **twice** of passengers with no family members on board **did not survive**. 
# * Passenges with **smaller** families (1-3 family members) have a **slightly higher** chance of survival.
# * Passengers with **larger** families (number of family members > 4) have a **slightly lower** chance of survival.

# ## Logistic Regression

# In[ ]:


# Converting "Sex" into a numerical column; 1 for male, 0 for female
train["Sex"] = train["Sex"].astype("category")
train["Gender"] = train["Sex"].cat.codes

train_y = train.Survived
predictor_cols = ['Age', 'Pclass', 'Gender', 'SibSp', 'Parch']
train_X = train[predictor_cols]

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
classifier = LogisticRegression(random_state=0)
classifier.fit(train_X, train_y)


# I realized there were missing values in the test data. I decided to use the approach in I,Coder's very comprehensive notebook (https://www.kaggle.com/ash316/eda-to-prediction-dietanic).

# In[ ]:


# Read the test data
test = pd.read_csv('../input/test.csv')
test["Sex"] = test["Sex"].astype("category")
test["Gender"] = test["Sex"].cat.codes

test['Initial']=0
for i in test:
    test['Initial']=test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',
                         'Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs',
                                          'Other','Other','Other','Mr','Mr','Mr'],inplace=True)

test.loc[(test.Age.isnull())&(test.Initial=='Mr'),'Age']=33
test.loc[(test.Age.isnull())&(test.Initial=='Mrs'),'Age']=36
test.loc[(test.Age.isnull())&(test.Initial=='Master'),'Age']=5
test.loc[(test.Age.isnull())&(test.Initial=='Miss'),'Age']=22
test.loc[(test.Age.isnull())&(test.Initial=='Other'),'Age']=46

test_X = test[predictor_cols]
test_X = scaler.transform(test_X)
predictions = classifier.predict(test_X)


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




