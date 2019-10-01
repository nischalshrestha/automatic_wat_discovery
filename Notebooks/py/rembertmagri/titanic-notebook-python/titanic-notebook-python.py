#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# [Kaggle](https://www.kaggle.com/c/titanic/)
# 
# **The goal is to predict Survived (yes or no)**
# 
# 

# ## 1st Step: loading the training data

# In[ ]:


import pandas as pd

df = pd.read_csv("../input/train.csv")

print(type(df)) # df type: pandas.core.frame.DataFrame

df.head()


# ## 2nd Step: checking the domain of the input data

# In[ ]:


print(df.columns)
for column in df.columns:
    print("* "+column+":", df[column].unique(), sep="\n")


# ## 3rd Step: checking the domain of relevant columns

# In[ ]:


print("* Survived:", df["Survived"].unique(), sep="\n")
print("\n* Pclass:", df["Pclass"].unique(), sep="\n")
print("\n* Sex:", df["Sex"].unique(), sep="\n")
print("\n* Age:", df["Age"].unique(), sep="\n")
print("\n* SibSp:", df["SibSp"].unique(), sep="\n")
print("\n* Parch:", df["Parch"].unique(), sep="\n")
print("\n* Ticket:", df["Ticket"].unique(), sep="\n")
print("\n* Fare:", df["Fare"].unique(), sep="\n")
print("\n* Cabin:", df["Cabin"].unique(), sep="\n")
print("\n* Embarked:", df["Embarked"].unique(), sep="\n")


# In[ ]:


get_ipython().magic(u'matplotlib inline')

hist = df['Survived'].hist()
hist.set_ylabel("# of passangers")
hist.set_xlabel("Survived")


# In[ ]:


import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')

fig, axs = plt.subplots(1,2)

hist = df['Pclass'].hist(ax=axs[0])
hist.set_ylabel("# of passangers")
hist.set_xlabel("Pclass")

hist = df.groupby('Survived').Pclass.hist(alpha=0.4)


# In[ ]:


get_ipython().magic(u'matplotlib inline')

hist = df['Sex'].value_counts().plot(kind='bar')
hist.set_ylabel("# of passangers")
hist.set_xlabel("Sex")


# In[ ]:


get_ipython().magic(u'matplotlib inline')

hist = df['Age'].hist()
hist.set_ylabel("# of passangers")
hist.set_xlabel("Age")


# In[ ]:


get_ipython().magic(u'matplotlib inline')

hist = df['SibSp'].hist()
hist.set_ylabel("# of passangers")
hist.set_xlabel("SibSp")


# In[ ]:


get_ipython().magic(u'matplotlib inline')

hist = df['Parch'].hist()
hist.set_ylabel("# of passangers")
hist.set_xlabel("Parch")


# In[ ]:


get_ipython().magic(u'matplotlib inline')

hist = df['Fare'].hist()
hist.set_ylabel("# of passangers")
hist.set_xlabel("Fare")


# In[ ]:


get_ipython().magic(u'matplotlib inline')

hist = df['Embarked'].value_counts().plot(kind='bar')
hist.set_ylabel("# of passangers")
hist.set_xlabel("Embarked")


# ## 4th Step: data cleansing

# In[ ]:


print(len(df))
df = df.dropna(subset=['Age']) # drops rows where age=NaN
print(len(df))
# TODO: Don't simply drop the rows where there is a NaN value. Treat this situations better to imporve the quality of the training data


# In[ ]:


print(df['Embarked'].isnull().sum())
df['Embarked'] = df['Embarked'].fillna('S') # fills Embarked=NaN with the most common option (S)
print(df['Embarked'].isnull().sum())


# ## 4.5th Step: Preprocessing

# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Preprocessing Sex 
lb = LabelEncoder()
print(df["Sex"])
df["Sex"] = lb.fit_transform(df["Sex"]) # Substitutes 'male' and 'female' by float values to allow the decision tree to run
print(df["Sex"])

# Preprocessing Embarked 
lb = LabelEncoder()
print(df["Embarked"])
df["Embarked"] = lb.fit_transform(df["Embarked"]) # Substitutes 'S', 'C' and 'Q' by float values to allow the decision tree to run
print(df["Embarked"])

# Features selection
features = [df.columns[2], df.columns[4], df.columns[5], df.columns[6], df.columns[7], df.columns[11]] # TODO: include the non-numeric columns
print("Features: ",features)
y = df["Survived"]
X = df[features]


# ## 5th Step: Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X, y)

dt.score(X, y)


# In[ ]:


# Visualize the decision tree
from sklearn.tree import export_graphviz
import graphviz

export_graphviz(dt, out_file="mytree.dot")
with open("mytree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# ## 5th Step: Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

rf.score(X, y)


# ## 6th Step: Load the test data

# In[ ]:


df_test = pd.read_csv("../input/test.csv")

df_test.head()


# ## 6.5th Step: Preprocessing

# In[ ]:


# Preprocessing Sex
lb = LabelEncoder()
print(df_test["Sex"])
df_test["Sex"] = lb.fit_transform(df_test["Sex"]) # Substitutes 'male' and 'female' by float values to allow the decision tree to run
print(df_test["Sex"])

# Preprocessing Embarked 
lb = LabelEncoder()
print(df_test["Embarked"])
df_test["Embarked"] = lb.fit_transform(df_test["Embarked"]) # Substitutes 'S', 'C' and 'Q' by float values to allow the decision tree to run
print(df_test["Embarked"])

# Treating the test data
treated_df = df_test.drop("Name",axis=1).drop("Ticket",axis=1).drop("Fare",axis=1).drop("Cabin",axis=1)
print(len(treated_df))
treated_df = treated_df.fillna(0) #treated_df = treated_df.dropna()
print(len(treated_df))
treated_df.head()


# ## 7th Step: Predict using Decision Tree

# In[ ]:


pred = dt.predict(treated_df.drop("PassengerId",axis=1))

print(pred)

submission = pd.DataFrame({
        "PassengerId": treated_df["PassengerId"],
        "Survived": pred
    })
submission.to_csv('prediction_dt.csv', index=False)


# ## 7th Step: Predict using Random Forest

# In[ ]:


pred = rf.predict(treated_df.drop("PassengerId",axis=1))

print(pred)

submission = pd.DataFrame({
        "PassengerId": treated_df["PassengerId"],
        "Survived": pred
    })
submission.to_csv('prediction_rf.csv', index=False)

