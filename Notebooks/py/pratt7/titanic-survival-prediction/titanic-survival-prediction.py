#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


# In[ ]:


data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")


# In[ ]:


data_train.Age.fillna(data_train.Age.mean(), inplace = True)


# Lets try to find out the statistics about Survival based on the Gender in the accident. This is a clear factor that Female are most likely to be survived

# In[ ]:


sex_pivot = data_train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()


# Since the response to ages is expected to be different in comparison to fares (in that, while people who paid more would be thought to have been priorities, people of both very small and very big ages would be thought to have been considered as an evacuation priority similarly). So, for this metric, let's check the distribution of survivors wrt age instead.

# In[ ]:


ns = data_train.loc[data_train["Survived"] == False] # Didn't survive :(
s  = data_train.loc[data_train["Survived"] == True]  # Survivors


# In[ ]:


sns.distplot(ns["Age"])
sns.distplot(s["Age"])
plt.show()


# For the survivors, there's a small bump in the number of children. The distribution of the survivors, as expected, is also more variant than that of those who did not due to the age bias. This too, seems to be an important estimator.

# We won't require the PassengerId, Name, Ticketnumber, Cabin and Embarked for our prediction so dropping them from the dataset

# In[ ]:


data_train.drop("PassengerId", axis = 1, inplace = True)
data_train.drop("Name", axis=1, inplace = True)
data_train.drop("Ticket", axis=1, inplace = True)
data_train.drop("Cabin", axis=1, inplace = True)
data_train.drop("Embarked", axis=1, inplace = True)


data_test.drop("PassengerId", axis = 1, inplace = True)
data_test.drop("Name", axis=1, inplace = True)
data_test.drop("Ticket", axis=1, inplace = True)
data_test.drop("Cabin", axis=1, inplace = True)
data_test.drop("Embarked", axis=1, inplace = True)


# In[ ]:


data_train.head()


# In[ ]:


data_train.isnull().any()


# In[ ]:


data_train.isnull().any()


# In[ ]:


data_test.isnull().any()


# In[ ]:


data_test.Age.fillna(data_test.Age.mean(), inplace = True)
data_test.Fare.fillna(data_test.Fare.mean(), inplace = True)


# In[ ]:


data_test.isnull().any()


# In[ ]:


data_train.Sex.replace(['male', 'female'], [0, 1], inplace=True)


# In[ ]:


data_test.Sex.replace(['male', 'female'], [0, 1], inplace=True)


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


data_train.describe()


# In[ ]:


data_test.describe()


# In[ ]:


data_whole = pd.concat([data_train, data_test])


# Concatenated both Train and Test data inorder to scale the dataset

# In[ ]:


data_whole.describe()


# In[ ]:


data_whole.tail()


# In[ ]:


del data_whole['Survived']
data_whole.describe()


# In[ ]:


from sklearn import preprocessing


# In[ ]:


data_scaled = pd.DataFrame(preprocessing.scale(data_whole))
data_scaled.describe()


# In[ ]:


titanic_train_x = data_whole.iloc[0:891,:]
titanic_test_x = data_whole.iloc[891:1309,:]
titanic_train_y = data_train.iloc[:,0]

titanic_train_x = titanic_train_x.values
titanic_test_x = titanic_test_x.values
titanic_train_y = titanic_train_y.values


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
model_LR = clf.fit(titanic_train_x, titanic_train_y)


# In[ ]:


clf.score(titanic_train_x, titanic_train_y)


# In[ ]:


output = clf.predict(titanic_test_x)


# In[ ]:


df = pd.DataFrame(output)


# In[ ]:


data_test_df = pd.read_csv("../input/test.csv")
df["PassengerId"] = data_test_df["PassengerId"]
df.head()


# In[ ]:


df.columns = ["Survived", "PassengerId"]


# In[ ]:


result = df.reindex(columns = ["PassengerId", "Survived"])


# In[ ]:


result.to_csv("titanic_lg.csv", header=True, index=False,  )


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model = RandomForestClassifier(n_estimators=1000)
model.fit(titanic_train_x, titanic_train_y)


# In[ ]:


from sklearn import cross_validation
scores = cross_validation.cross_val_score(model, titanic_train_x, titanic_train_y, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# It seems from these values that the Logistical Regression and Random Forest Classifier are giving the similar performance.

# In[ ]:


predictions = model.predict(titanic_test_x)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": data_test_df["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('titanic_rf.csv', index=False)


# In[ ]:




