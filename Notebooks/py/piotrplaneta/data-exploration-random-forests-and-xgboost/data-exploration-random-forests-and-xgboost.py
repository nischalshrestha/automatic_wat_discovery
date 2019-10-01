#!/usr/bin/env python
# coding: utf-8

# # Data exploration, random forests and XGBoost

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.describe(include=["O"])


# # Simple preprocessing
#  - Set index
#  - Downcase column names

# In[ ]:


all_data_df = pd.concat([train_df, test_df])
train_df.columns = map(lambda name: name.lower(), train_df.columns)
train_df = train_df.set_index("passengerid")

test_df.columns = map(lambda name: name.lower(), test_df.columns)
test_df = test_df.set_index("passengerid")


# # Pivoting observations
# * Higher class -> higher survival chance
# * Females -> higher survival chance
# * SibSp/Parch -> 1 sibling and between 1 to 3 parents/children has the highest survival rate
#     * Family size between 1 and 3 has the highest survival chance
#     * Men from big families (4+ has super low survive ratio)
# * Embarked -> C higher survival chance

# In[ ]:


train_df.groupby("pclass")["survived"].mean()


# In[ ]:


train_df.groupby("sex")["survived"].mean()


# In[ ]:


train_df.groupby("sibsp")["survived"].mean().plot()


# In[ ]:


train_df.groupby("parch")["survived"].mean().plot()


# In[ ]:


train_df.groupby("embarked")["survived"].mean()


# ## Family size

# In[ ]:


train_df["family_size"] = train_df["sibsp"] + train_df["parch"]
test_df["family_size"] = test_df["sibsp"] + test_df["parch"]

train_df.groupby("family_size")["survived"].mean().plot()


# In[ ]:


plt.figure(figsize=(15, 4))
plt.subplot(121)
plt.title("Male")
plt.plot(train_df[train_df["sex"] == "male"].groupby("family_size")["survived"].mean())
plt.subplot(122)
plt.title("Female")
plt.plot(train_df[train_df["sex"] == "female"].groupby("family_size")["survived"].mean())


# In[ ]:


train_df["family_size_category"] = pd.cut(train_df["family_size"], [-1, 0.5, 3, 10])
test_df["family_size_category"] = pd.cut(test_df["family_size"], [-1, 0.5, 3, 10])

train_df.groupby("family_size_category")["survived"].mean().plot()


# # Age observations
# * Age < 4 -> higher chance of survival (but no direct correlation between age and survival)
# * Oldest passengers survived
# * Most in 15-35 did not survive
# * Most of passengers are in 15-50

# In[ ]:


plot_grid = sns.FacetGrid(train_df, col="survived")
plot_grid.map(plt.hist, "age", bins=20)


# In[ ]:


train_df["age"] = train_df["age"].fillna(train_df["age"].mean())
test_df["age"] = test_df["age"].fillna(test_df["age"].mean())

train_df["age_category"] = pd.cut(train_df["age"], 6)
test_df["age_category"] = pd.cut(test_df["age"], 6)


# In[ ]:


plt.figure(figsize=(10, 3))
train_df.groupby("age_category")["survived"].mean().plot()


# # Pclass observations
# * Higher class -> higher chance of survival
# * Almost all women from 1st and 2nd class survived

# In[ ]:


plot_grid = sns.FacetGrid(train_df, col="survived", row="pclass", size=4, aspect=2)
plot_grid.map(plt.hist, "age", bins=20)


# In[ ]:


plot_grid = sns.FacetGrid(train_df, col="survived", row="sex", size=4, aspect=2)
plot_grid.map(plt.hist, "pclass", bins=20)


# # Fare observations
# * The higher fare, the higher survival rate

# In[ ]:


train_df["fare"] = train_df["fare"].fillna(train_df["fare"].mean())
test_df["fare"] = test_df["fare"].fillna(test_df["fare"].mean())


# In[ ]:


train_df["fare"].hist(bins=50)


# In[ ]:


train_df['fare_category'] = pd.cut(train_df['fare'], [-1, 6, 7.5, 8, 14, 20, 30, 60, 90, 600])
test_df['fare_category'] = pd.cut(test_df['fare'], [-1, 6, 7.5, 8, 14, 20, 30, 60, 90, 600])

train_df.groupby("fare_category")["survived"].mean().plot()
train_df["fare_category"].value_counts()


# # Cabin observations
# * Since the cabins are defined mostly for first class, there is a lot of nulls, and no visible correlation to survival, we should drop it

# In[ ]:


#Fill NaNs with fake data
train_df["cabin"] = train_df["cabin"].fillna("Z1")
test_df["cabin"] = test_df["cabin"].fillna("Z1")


# In[ ]:


train_df[train_df["cabin"] != "Z1"]["pclass"].value_counts()


# In[ ]:


test_df[test_df["cabin"] != "Z1"]["pclass"].value_counts()


# In[ ]:


train_df["cabin_type"] = train_df["cabin"].str[0]


# In[ ]:


train_df.groupby("cabin_type")["survived"].mean().plot()


# # Name observations
# * Most of the ordinary "Mr." didn't survive
# * A lot of correlation in the titles, especially between Miss and Mrs
# * A lot of different titles, let's combine them

# In[ ]:


train_df["title"] = train_df["name"].str.split(",").str[1].str.strip().str.split().str[0]
test_df["title"] = test_df["name"].str.split(",").str[1].str.strip().str.split().str[0]

train_df["title"].value_counts()


# In[ ]:


plt.figure(figsize=(15, 3))
title_data = train_df.groupby("title")["survived"].mean()
plt.xticks(np.arange(len(title_data)), title_data, rotation=90)
title_data.plot()


# In[ ]:


def preprocess_titles(given_df):
    df = given_df.copy()
    df.loc[df["title"].isin(["Rev.", "Capt.", "Don."]), "title"] = "not_survived"
    df.loc[df["title"].isin(["Dona.", "Mlle."]), "title"] = "Miss."
    df.loc[df["title"].isin(["Ms.", "Mme."]), "title"] = "Mrs."
    title_counts = df["title"].value_counts()
    non_frequent_titles = np.setdiff1d(title_counts[title_counts < 10].index.values, ["not_survived"])
    df.loc[df["title"].isin(non_frequent_titles), "title"] = "non_frequent"
    return df


# In[ ]:


train_df = preprocess_titles(train_df)
test_df = preprocess_titles(test_df)
train_df["title"].value_counts()


# In[ ]:


title_data = train_df.groupby("title")["survived"].mean()
title_data.plot()


# # Preparing model
# * Keep just wanted columns
# * Encode labels
# * Create train and validation sets

# In[ ]:


train_preprocessed_df = train_df[["survived", "pclass", "sex", "embarked", "family_size_category", "age_category", "fare_category", "title"]]
test_preprocessed_df = test_df[["pclass", "sex", "embarked", "family_size_category", "age_category", "fare_category", "title"]]

train_preprocessed_df.head()


# In[ ]:


test_preprocessed_df.info()


# ## Encode labels

# In[ ]:


merged_data = pd.concat([train_preprocessed_df, test_preprocessed_df])


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[ ]:


merged_data["sex"] = le.fit_transform(merged_data["sex"])
merged_data["embarked"] = le.fit_transform(merged_data["embarked"].fillna("Z"))
merged_data["age_category"] = le.fit_transform(merged_data["age_category"])
merged_data["family_size_category"] = le.fit_transform(merged_data["family_size_category"])
merged_data["fare_category"] = le.fit_transform(merged_data["fare_category"])
merged_data["title"] = le.fit_transform(merged_data["title"])


# In[ ]:


train_encoded_df = merged_data[merged_data.index.isin(train_df.index)]
test_encoded_df = merged_data[merged_data.index.isin(test_df.index)]


# ## Create train and validation sets

# In[ ]:


X_train = train_encoded_df[train_preprocessed_df.columns.difference(["survived"])]
y_train = train_encoded_df["survived"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3, random_state=0)


# # First model - decision tree

# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

clf.score(X_validation, y_validation)


# # Random forests

# In[ ]:


from sklearn import ensemble
clf = ensemble.RandomForestClassifier()
clf = clf.fit(X_train, y_train)

clf.score(X_validation, y_validation)


# # Train random forest on the whole set and prepare submission

# In[ ]:


X_whole_train = train_encoded_df[train_preprocessed_df.columns.difference(["survived"])]
y_whole_train = train_encoded_df["survived"]


# In[ ]:


clf = ensemble.RandomForestClassifier()
clf = clf.fit(X_whole_train, y_whole_train)


# In[ ]:


test_encoded_df = test_encoded_df[test_encoded_df.columns.difference(["survived"])]

prediction = clf.predict(test_encoded_df)
prediction = [int(p) for p in prediction]


# In[ ]:


submission = pd.DataFrame({"PassengerId": test_df.index, "Survived": prediction})
submission.to_csv('random_forest_submission.csv', index=False)


# # XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)


# In[ ]:


model.score(X_validation, y_validation)


# In[ ]:


model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_whole_train, y_whole_train)

prediction = model.predict(test_encoded_df)
prediction = [int(p) for p in prediction]


# In[ ]:


submission = pd.DataFrame({"PassengerId": test_df.index, "Survived": prediction})
submission.to_csv('xgboost_submission.csv', index=False)

