#!/usr/bin/env python
# coding: utf-8

# * [2. Join train and test set](#join_data)
# * [3. Check null value](#check_null)
# * [4. Data analysis](#data_analysis)
# * [5. Analyze how to fill NA values](#fill_na)
# * [6. Analyze how to do feature engineering](#engineering)
# * [7. Reference kernel](#reference)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)


# ## 1. Load data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print('train data set, shape: ', train.shape)
print('test data set, shape: ', test.shape)


# <a id='join_data'></a>
# ## 2. Join train and test set

# In[ ]:


dataset =  pd.concat(objs=[train, test], axis=0, sort=False).reset_index(drop=True)
print('total data set, shape: ', dataset.shape)


# In[ ]:


dataset.head(5)


# In[ ]:


dataset.tail(5)


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train.describe(include=['O'])


# <a id='check_null'></a>
# ## 3. Check null value

# In[ ]:


dataset.isnull().sum()


# In[ ]:


train.isnull().sum()


# <a id='data_analysis'></a>
# ## 4. Data analysis

# ### The distribution of single variable

# In[ ]:


# The distribution of Survived
sns.countplot(dataset['Survived'])


# In[ ]:


# The distribution of Embarked
sns.countplot(dataset['Embarked'])


# ### Explore Pclass vs Survived

# In[ ]:


# The distribution of Survived on Pclass
sns.countplot(dataset['Pclass'], hue=dataset['Survived'])


# ### Explore Sex  vs Survived
# It seems that famale passengers have higher survival rate.

# In[ ]:


# The distribution of Survived on Sex
sns.countplot(dataset['Sex'], hue=dataset['Survived'])


# In[ ]:


# Survival probability on Sex
g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")


# In[ ]:


train[["Sex","Survived"]].groupby('Sex').mean()


# In[ ]:


train[["Sex","Survived"]].groupby('Sex').std()


# ### Explore Embarked vs Survived
# It seems that passengers who embarked at Southampton have higher survival rate.

# In[ ]:


# The distribution of Survived on Embarked
sns.countplot(dataset['Embarked'], hue=dataset['Survived'])


# In[ ]:


#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")


# In[ ]:


# Survival probability on Embarked
g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g = g.set_ylabels("survival probability")


# ### Explore Age vs Survived
# It seems that young passengers have higher survival rate.

# In[ ]:


# The distribution of Age on Survived 0 or 1: distplot
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'Age', kde=False)


# In[ ]:


# The distribution of Age on Survived 0 or 1: kdeplot
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# ### Explore Fare vs Survived

# In[ ]:


# The distribution of Fare on Survived 0 or 1 
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'Fare', kde=False)


# In[ ]:


#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


# In[ ]:


# Explore Fare distribution 
g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# In[ ]:


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


# Explore Fare distribution after log
g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# ### Explore SibSp vs Survived

# In[ ]:


# The distribution of SibSp on Survived 0 or 1 
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'SibSp', kde=False)


# In[ ]:


# Survival probability on SibSp 
g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , palette = "muted")
g = g.set_ylabels("survival probability")


# ### Explore Parch vs Survived

# In[ ]:


# The distribution of Parch on Survived 0 or 1 
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'Parch', kde=False)


# In[ ]:


# Survival probability on Parch
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 ,palette = "muted")
g = g.set_ylabels("survival probability")


# ### Combine Parch and SibSp into Family_Size

# In[ ]:


dataset['Family_Size'] = dataset['Parch'] + dataset['SibSp']


# In[ ]:


# The distribution of Family_Size on Survived 0 or 1 
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'Family_Size', kde=False)


# ### Explore Pclass vs Survived by Sex

# In[ ]:


# Survival probability on Pclass by Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g = g.set_ylabels("survival probability")


# In[ ]:


train[["Pclass", "Sex","Survived"]].groupby(["Pclass", "Sex"]).mean()


# ### Explore Pclass, Embarked, Sex, Survived

# In[ ]:


# The distribution of Pclass on Embarked
g = sns.factorplot("Pclass", col="Embarked",  data=train,
                   size=6, kind="count", palette="muted")
g = g.set_ylabels("Count")


# In[ ]:


# Survival probability on Pclass by Sex, Embarked
g = sns.factorplot(x="Pclass", y="Survived", col="Embarked", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g = g.set_ylabels("survival probability")


# ### Correlation matrix between numerical values and Survived
# 

# In[ ]:


g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# <a id='fill_na'></a>
# ## 5. Analyze how to fill NA values

# ### 5.1 Analyze Age
# It seems the distribution of Age is not related to Sex, so Sex is not informative to predict Age.

# In[ ]:


# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box", size=6)
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box", size=6)
g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box", size=6)
g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box", size=6)


# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


# In[ ]:


g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)


# In[ ]:


# In order to fill Age null value, I pick out samples whose Age value is null. 
# Then I pick out samples(Samples_A) whose SibSp, Parch, Pclass values are the same as these values of samples whose Age value is null.
# Finally I use median of Age value of Samples_A to fill Age null value.
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][(
            (dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (
            dataset['Parch'] == dataset.iloc[i]["Parch"]) & (
                    dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med


# In[ ]:


# Explore Age vs Survived after filling NA
g = sns.factorplot(x="Survived", y = "Age",data = dataset, kind="box")
g = sns.factorplot(x="Survived", y = "Age",data = dataset, kind="violin")


# In[ ]:


dataset[["Survived","Age"]].groupby('Survived').median()


# ### 5.2 Analyze Cabin

# In[ ]:


dataset["Cabin"].unique()


# In[ ]:


# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
dataset["Cabin"].unique()


# In[ ]:


g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])


# In[ ]:





# In[ ]:


train[["Survived","Age"]].groupby('Survived').median()


# In[ ]:


# Explore Cabin vs Survived after filling NA
g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")


# ### 5.3 Analyze Fare

# In[ ]:


#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


# In[ ]:


# Explore Fare distribution before log transformation
g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# In[ ]:


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


# Explore Fare distribution after log transformation
g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# <a id='engineering'></a>
# ## 6. Analyze how to do feature engineering

# ### 6.1 Feature engineering - Name/Title
# Extract Title from Name

# In[ ]:


dataset["Name"].head()


# In[ ]:


# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].unique()


# In[ ]:


# The distribution of Title
g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)


# In[ ]:


# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[ ]:


# The distribution of Title: less classes
g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


# In[ ]:


# Explore Title vs Survived
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# ### 6.2 Feature engineering - SibSp/Parch/Fsize
# Create a family size from SibSp and Parch

# In[ ]:


dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[ ]:


# Explore Fize vs Survived
g = sns.factorplot(x="Fsize",y="Survived",data = dataset, kind='bar')
g = g.set_ylabels("Survival Probability")


# In[ ]:


# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


# Explore new feature vs Survived
g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")


# ### 6.3 Feature engineering - Ticket

# In[ ]:


dataset["Ticket"].head()


# In[ ]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[ ]:


dataset["Ticket"].describe()


# In[ ]:


dataset["Ticket"].value_counts()


# <a id='reference'></a>
# ## 7. Reference kernel

# 1. [Yassine Ghouzam: Titanic Top 4% with ensemble modeling](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)
# 2. [Yeh James: Titanic survival prediction](https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC4-1%E8%AC%9B-kaggle%E7%AB%B6%E8%B3%BD-%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E7%94%9F%E5%AD%98%E9%A0%90%E6%B8%AC-%E5%89%8D16-%E6%8E%92%E5%90%8D-a8842fea7077)

# In[ ]:




