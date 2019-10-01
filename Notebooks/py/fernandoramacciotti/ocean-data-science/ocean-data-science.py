#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivors

# This is a classification problem and our model must predict if a passenger died (0) or survived (1), given the *PassengerId*.
# 
# Evaluation metric is simple the percentage of correctly predicted passengers, i.e. accuracy.
# 
# Let's use as benchmark a simple "coin toss" model, i.e. 50% of survival change for any given passenger and let's see how the models developed here are compared to it.
# 

# ## Get the data
# 
# First let's import the necessary data.

# In[74]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['figure.figsize'] = (4, 2.7)
rcParams['figure.dpi'] = 150
#rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = '#eeeeee'
rcParams['font.size'] = 9
rcParams['patch.edgecolor'] = 'none'


# Let's keep a copy of the original data and the split it into train-test so we can evaluate and validate our models

# In[75]:


train_original = pd.read_csv("../input/train.csv")

# X = features, y = labels

X = train_original[train_original.columns.drop("Survived")]
y = train_original["Survived"]
X.head()


# In[76]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
train = pd.concat([y_train, X_train], axis=1)
test = pd.concat([y_test, X_test], axis=1)
train.head()


# In[77]:


train.info()


# We have 12 columns and 891 rows:
# * passenger id;
# * survived column, indicating if the corresponding passenger survived or not;
# * 10 informational variables

# ### Explore the data

# Let's explore a bit more the data.

# #### Survival distribution

# ##### Survival vs. Sex
# Let's check if more men or women survived

# In[78]:


feat = "Sex"
train[["Survived", feat]].groupby(by=feat).mean().plot.bar()
plt.title("Survival rate vs. Sex", )


# Survival rate for female is __3.5x__ higher than male's, i.e ~70% vs. 20%

# ##### Survival vs. Sex and Siblings/Spouse

# In[79]:


feat = ["Sex", "SibSp", "Parch"]
df_perc = train[["Survived", "Sex", "SibSp", "Parch"]].groupby(by=feat).mean().reset_index()
pd.crosstab(df_perc.Sex, df_perc.SibSp, margins=True, 
            normalize=True).style.background_gradient(cmap="summer_r", axis=1)


# The lesser the number of siblings/spouses, the higher chance of survival. However, those with __one sibling/spouse__ had, on average, the best survival rate.

#  ##### Survival vs. Sex and Parents/Children

# In[80]:


pd.crosstab(df_perc.Sex, df_perc.Parch, margins=True, 
            normalize=True).style.background_gradient(cmap="summer_r", axis=1)


# Here, those with __2 parents/children__ have the highest survival rate. Maybe they are children with both parents on board.
# 
# Let's check it.

# ##### Survival vs. SibSp and Age

# In[81]:


feat = ["Age", "Parch"]
df_perc = train[["Survived", "Age", "Parch"]].groupby(by=feat).mean().reset_index()
# group the ages at every 10 years
df_perc["age_group"] = np.where(df_perc["Age"] < 10, "0-10",
                               np.where(df_perc["Age"] < 20, "11-20",
                                       np.where(df_perc["Age"] < 30, "21-30",
                                               np.where(df_perc["Age"] > 60, "61+",
                                                       "31-60"))))

pd.crosstab(df_perc.age_group, df_perc.Parch, margins=True, 
            normalize=True).style.background_gradient(cmap="summer_r", axis=0)


# No clear pattern here, but some interesting insights.
# 
# First, the percentages don't add up with the previoues table - maybe `Age` has some missing values.
# 
# Second, people between 31 and 60 years old have survived more.
# 
# Third, our hypothesis that those with 2 parents/children survived more because they were supposedly children is not really confirmed - the %s are kind of the same for all groups with 2  `Parch` (but 61+ group that has no one in this sample)

# ##### Survival vs. Pclass

# In[82]:


feat = "Pclass"
#df_perc = train[["Survived", "Pclass"]].groupby(by=feat).mean().reset_index()
pd.crosstab(train.Survived, train[feat], margins=True, 
            normalize=True).style.background_gradient(cmap="summer_r", axis=1)


# Among those who died, people in the lower class survived less and among those who survived, 1st and 3rd class survived more.

# ##### Survival vs. Fare

# In[83]:


feat = "Fare"
df_perc = train[["Survived", "Fare"]].groupby(by=feat).mean().reset_index()
# group the fare at every 100
df_perc["fare_group"] = np.where(df_perc[feat] < 100, "0-100",
                               np.where(df_perc[feat] < 200, "101-200",
                                       np.where(df_perc[feat] < 300, "201-300",
                                                np.where(df_perc[feat] < 400, "301-400",
                                                         np.where(df_perc[feat] < 500, "401-500",
                                                                    "500+")))))
df_perc[["Survived", "fare_group"]].groupby(by=["fare_group"]).mean().plot()


# 500+ fares have a pretty high survival rate and 101-200 fare group has 80% survival rate (maybe there are more people?)

# ##### Survival vs. Cabin

# In[84]:


feat = "Cabin"
#df_perc = train[["Survived", "Pclass"]].groupby(by=feat).mean().reset_index()
train["Cabin_group"] = train[feat].str[0]
feat = "Cabin_group"
pd.crosstab(train.Survived, train[feat], margins=True, 
            normalize=True).style.background_gradient(cmap="summer_r", axis=0)


# Cabins B, E and D have the highest survival rate.

# ##### Survival vs. Embarked

# In[85]:


feat = "Embarked"
pd.crosstab(train.Survived, train[feat], margins=True, 
            normalize=True).style.background_gradient(cmap="summer_r", axis=1)


# Those who embarked at port S survived more

# ---
# #### Missing values
# Let's check if there is any missing value in the columns.a
# 

# In[86]:


def create_missing_df(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ["feature", "missing_count"]
    missing_df = missing_df[missing_df.missing_count > 0]
    missing_df["missing_perc"] = 100 * missing_df["missing_count"] / train.shape[0]
    missing_df = missing_df.sort_values(by="missing_perc", ascending=False)
    return missing_df


# In[87]:


missing_df = create_missing_df(train)
missing_df.head()


# There are 3 variables with missing values: the cabin number, age and port of embarkation with 77%, 19.9% and 0.2% of missing values, respectively. We confirm, then, that the percentages didn't match because `Age` has ~20% of missing values.

# ---

# ## Model

# ### Benchmark - coin toss model
# our benchmark will be a simple random assigment for each `PassengerId` of test data.
# 

# In[88]:


from sklearn.metrics import accuracy_score

np.random.seed(0)
y_pred_bench = np.random.randint(0, 2, size=y_test.size)

acc_bench = accuracy_score(y_pred=y_pred_bench, y_true=y_test)
print("Benchmark: %.2f" % acc_bench)


# Our benchmark is 54% of accuracy.

# In[89]:


### function to get Cabin Class
def cabin_class(df, col_cabin="Cabin"):
    df[col_cabin] = df[col_cabin].str[0].fillna("No Cabin Assign")
    return df


# In[90]:


#X_train = cabin_class(X_train.drop("PassengerId", axis=1))
#X_test = cabin_class(X_test.drop("PassengerId", axis=1))

X_train_2 = pd.get_dummies(X_train.drop(["Cabin", "Ticket", "PassengerId", "Name"], axis=1), prefix_sep="_")
X_test_2 = pd.get_dummies(X_test.drop(["Cabin", "Ticket", "PassengerId", "Name"], axis=1), prefix_sep="_")


# In[91]:


### Age nan imputer
from sklearn.preprocessing import Imputer

imp = Imputer(strategy='mean')
imp.fit(X_train_2["Age"].reshape(-1, 1))
X_train_2["Age"] = imp.transform(X_train_2["Age"].reshape(-1, 1))
X_test_2["Age"] = imp.transform(X_test_2["Age"].reshape(-1, 1))

#X_train = X_train.dropna()
#X_test = X_test.dropna()


# ### Decision Tree

# In[92]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train_2, y_train)
y_pred_dtree = dtree.predict(X_test_2)

acc_dtree = accuracy_score(y_pred=y_pred_dtree, y_true=y_test)
print("Decision Tree: %.2f" % acc_dtree)


# ### Random Forest

# In[93]:


from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier(n_estimators=300, random_state=0)
rforest.fit(X_train_2, y_train)
y_pred_rforest = rforest.predict(X_test_2)

acc_rforest = accuracy_score(y_pred=y_pred_rforest, y_true=y_test)
print("Random Forest: %.2f" % acc_rforest)


# ### Logistic Regression

# In[94]:


from sklearn.linear_model import LogisticRegressionCV

lreg = LogisticRegressionCV(cv=10, random_state=0)
lreg.fit(X_train_2, y_train)
y_pred_lreg = lreg.predict(X_test_2)

acc_lreg = accuracy_score(y_pred=y_pred_lreg, y_true=y_test)
print("Logistic Regression: %.2f" % acc_lreg)


# ### KNN Classifier

# In[95]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20, weights='distance', p=1)
knn.fit(X_train_2, y_train)
y_pred_knn = knn.predict(X_test_2)

acc_knn = accuracy_score(y_pred=y_pred_knn, y_true=y_test)
print("KNN: %.2f" % acc_knn)


# ### Voting and submitting

# In[96]:


from sklearn.ensemble import VotingClassifier

rforest = RandomForestClassifier(n_estimators=500, random_state=0)
lreg = LogisticRegressionCV(cv=10, random_state=0)
knn = KNeighborsClassifier(n_neighbors=20, weights='distance', p=1)

voting = VotingClassifier(estimators=[('rforest', rforest), ('lreg', lreg), ('knn', knn)], 
                         voting='soft')
voting.fit(X_train_2, y_train)
y_pred_voting = voting.predict(X_test_2)

acc_voting = accuracy_score(y_pred=y_pred_voting, y_true=y_test)
print("Voting: %.2f" % acc_voting)


# In[97]:


test_data = pd.read_csv("../input/test.csv")
X_test_data = pd.get_dummies(test_data.drop(["Cabin", "Ticket", "PassengerId", "Name"], axis=1), prefix_sep="_")
X_test_data["Age"] = imp.transform(X_test_data["Age"].reshape(-1, 1))

out = pd.DataFrame({"PassengerId": test_data["PassengerId"].values, 
                    "Survived": voting.predict(X_test_data.fillna(0))})


# In[98]:


out.to_csv("voting.csv", index=False)

