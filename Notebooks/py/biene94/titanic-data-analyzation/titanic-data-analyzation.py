#!/usr/bin/env python
# coding: utf-8

# # Load Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# # First Look

# In[ ]:


train_data.head(10)


# **Notes**
# * Following features are categorical: Pclass, Sex, Embarked
# * Name includes title

# In[ ]:


train_data["Cabin"][train_data["Cabin"].notnull()].head(100)


# **Notes**
# * All cabins contain letters (A, B, C, D, E, F or G)
# * Some data sets contain multiple cabins (but same letter)

# In[ ]:


train_data["Name"].head(100)


# **Notes**
# * I see following titles: "Mr.", "Mrs.", "Miss." and "Master." but there could be more

# In[ ]:


train_data["Ticket"].head(100)


# **Note**
# * There are some letters that may be a code or something, but I don't really know, if I could categorize them...

# In[ ]:


train_data.info()


# **Notes**
# * There are 891 data sets
# * There are missing values at Age (177), Cabin (687) and Embarked (2)
# * I may drop Cabin, because there are so many values missing
# * Following features are not numerical: Name, Sex, Ticket, Cabin, Embarked

# In[ ]:


test_data.info()


# **Notes**
# * There are 418 data sets
# * There are missing values at Age (86), Cabin (327) and Fare (1)
# * I may drop Cabin, because there are so many values missing

# # Find Categories

# In[ ]:


train_data["Pclass"].value_counts()


# **Notes**
# * There are three categories: 1, 2 and 3

# In[ ]:


train_data["Sex"].value_counts()


# **Notes**
# * There are two categories: male and female

# In[ ]:


train_data["Embarked"].value_counts()


# **Notes**
# * There are three categories: S, C and Q
# * I know from the data describtion, that these are: Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


cabin_letters = ["A", "B", "C", "D", "E", "F", "G"]
not_recorded_cabin_letters = []

for cabin in train_data["Cabin"][train_data["Cabin"].notnull()]:
    found_letter = False
    for cabin_letter in cabin_letters:
        if(cabin_letter in cabin) :
            found_letter = True
            break
    if(found_letter is False) :
        not_recorded_cabin_letters.append(cabin)

print(not_recorded_cabin_letters)


# **Notes**
# * One cabin only has a T as the value

# In[ ]:


titles = ["Mr.", "Mrs.", "Miss.", "Master."]
not_recorded_titles = []

for name in train_data["Name"]:
    found_title = False
    for title in titles:
        if(title in name) :
            found_title = True
            break
    if(found_title is False) :
        not_recorded_titles.append(name)

print(not_recorded_titles)


# **Notes**
# * I found following titles: "Don.", "Rev.", "Dr.", "Mme.", "Ms.", "Major.", "Lady.", "Sir.", "Mlle.", "Col.", "Capt.", "Countess." and "Jonkheer."

# # Fill missing Values

# In[ ]:


# Fill Age with median
median = train_data["Age"].median()
train_data["Age"].fillna(median, inplace=True)

# Fill Embarked with most occuring value
train_data["Embarked"].fillna(2, inplace=True)


# # One-Hot encode categorical Features

# In[ ]:


# Sex
train_data["Female"] = 0
train_data["Male"] = 0
train_data.loc[train_data["Sex"] == "female", "Female"] = 1
train_data.loc[train_data["Sex"] == "male", "Male"] = 1
train_data = train_data.drop("Sex", axis=1)

# Embarked
train_data["Cherbourg"] = 0
train_data["Queenstown"] = 0
train_data["Southampton"] = 0
train_data.loc[train_data["Embarked"] == "C", "Cherbourg"] = 1
train_data.loc[train_data["Embarked"] == "Q", "Queenstown"] = 1
train_data.loc[train_data["Embarked"] == "Q", "Southampton"] = 1
train_data = train_data.drop("Embarked", axis=1)

# Pclass
train_data["PclassOne"] = 0
train_data["PclassTwo"] = 0
train_data["PclassThree"] = 0
train_data.loc[train_data["Pclass"] == 1, "PclassOne"] = 1
train_data.loc[train_data["Pclass"] == 2, "PclassTwo"] = 1
train_data.loc[train_data["Pclass"] == 3, "PclassThree"] = 1
train_data = train_data.drop("Pclass", axis=1)

# Cabin Letters
train_data["CabinLetterA"] = 0
train_data["CabinLetterB"] = 0
train_data["CabinLetterC"] = 0
train_data["CabinLetterD"] = 0
train_data["CabinLetterE"] = 0
train_data["CabinLetterF"] = 0
train_data["CabinLetterG"] = 0
train_data["CabinLetterT"] = 0
train_data.loc[train_data["Cabin"].notnull() & train_data["Cabin"].str.contains("A"), "CabinLetterA"] = 1
train_data.loc[train_data["Cabin"].notnull() & train_data["Cabin"].str.contains("B"), "CabinLetterB"] = 1
train_data.loc[train_data["Cabin"].notnull() & train_data["Cabin"].str.contains("C"), "CabinLetterC"] = 1
train_data.loc[train_data["Cabin"].notnull() & train_data["Cabin"].str.contains("D"), "CabinLetterD"] = 1
train_data.loc[train_data["Cabin"].notnull() & train_data["Cabin"].str.contains("E"), "CabinLetterE"] = 1
train_data.loc[train_data["Cabin"].notnull() & train_data["Cabin"].str.contains("F"), "CabinLetterF"] = 1
train_data.loc[train_data["Cabin"].notnull() & train_data["Cabin"].str.contains("G"), "CabinLetterG"] = 1
train_data.loc[train_data["Cabin"].notnull() & train_data["Cabin"].str.contains("T"), "CabinLetterT"] = 1
train_data = train_data.drop("Cabin", axis=1)

# Titles
train_data["TitleMr"] = 0
train_data["TitleMrs"] = 0
train_data["TitleMiss"] = 0
train_data["TitleMaster"] = 0
train_data["TitleDon"] = 0
train_data["TitleRev"] = 0
train_data["TitleDr"] = 0
train_data["TitleMme"] = 0
train_data["TitleMs"] = 0
train_data["TitleMajor"] = 0
train_data["TitleLady"] = 0
train_data["TitleSir"] = 0
train_data["TitleMlle"] = 0
train_data["TitleCol"] = 0
train_data["TitleCapt"] = 0
train_data["TitleCountess"] = 0
train_data["TitleJonkheer"] = 0
train_data.loc[train_data["Name"].str.contains("Mr."), "TitleMr"] = 1
train_data.loc[train_data["Name"].str.contains("Mrs."), "TitleMrs"] = 1
train_data.loc[train_data["Name"].str.contains("Miss."), "TitleMiss"] = 1
train_data.loc[train_data["Name"].str.contains("Master."), "TitleMaster"] = 1
train_data.loc[train_data["Name"].str.contains("Don."), "TitleDon"] = 1
train_data.loc[train_data["Name"].str.contains("Rev."), "TitleRev"] = 1
train_data.loc[train_data["Name"].str.contains("Dr."), "TitleDr"] = 1
train_data.loc[train_data["Name"].str.contains("Mme."), "TitleMme"] = 1
train_data.loc[train_data["Name"].str.contains("Ms."), "TitleMs"] = 1
train_data.loc[train_data["Name"].str.contains("Major."), "TitleMajor"] = 1
train_data.loc[train_data["Name"].str.contains("Lady."), "TitleLady"] = 1
train_data.loc[train_data["Name"].str.contains("Sir."), "TitleSir"] = 1
train_data.loc[train_data["Name"].str.contains("Mlle."), "TitleMlle"] = 1
train_data.loc[train_data["Name"].str.contains("Col."), "TitleCol"] = 1
train_data.loc[train_data["Name"].str.contains("Capt."), "TitleCapt"] = 1
train_data.loc[train_data["Name"].str.contains("Countess."), "TitleCountess"] = 1
train_data.loc[train_data["Name"].str.contains("Jonkheer."), "TitleJonkheer"] = 1
train_data = train_data.drop("Name", axis=1)


# # Find new Features

# In[ ]:


# Underage Passengers
train_data["Underage"] = 0
train_data.loc[train_data["Age"] < 18, "Underage"] = 1

# Age Groups
train_data["Baby"] = 0
train_data["Child"] = 0
train_data["Teenager"] = 0
train_data["YoungAdult"] = 0
train_data["Adult"] = 0
train_data["Senior"] = 0
train_data.loc[train_data["Age"] < 4, "Baby"] = 1
train_data.loc[(train_data["Age"] >= 4) & (train_data["Age"] < 13), "Child"] = 1
train_data.loc[(train_data["Age"] >= 13) & (train_data["Age"] < 18), "Teenager"] = 1
train_data.loc[(train_data["Age"] >= 18) & (train_data["Age"] < 30), "YoungAdult"] = 1
train_data.loc[(train_data["Age"] >= 30) & (train_data["Age"] < 60), "Adult"] = 1
train_data.loc[train_data["Age"] >= 60, "Senior"] = 1


# # Visualize Data

# In[ ]:


corr_matrix = train_data.corr()
survived_correlation = corr_matrix["Survived"].sort_values(ascending=True)
survived_correlation = survived_correlation.drop("Survived")

colors = []
positive_survived_correlation = abs(survived_correlation.values)
max_difference = 1 - positive_survived_correlation.max()

for survived in positive_survived_correlation:
    green = 1 - (survived + max_difference)
    blue = 0
    red = 1
    if(survived<=positive_survived_correlation.mean()) :
        blue = 1
        red = 0
    colors.append((red, green, blue, 1))

features = survived_correlation.axes[0]
plt.figure(figsize=(10,20))
plt.barh(range(len(features)), survived_correlation, color=colors)
plt.yticks(range(len(features)), features)
plt.grid()
plt.show()


# **Notes**
# * Following features seem to be very relevant: TitleMrs, TitleMr, TitleMiss, PclassThree, PclassOne, Male, Female, Fare
# * The age especially if it's a child or not seems a little bit irrelevant compared to the values above
# * I could try to categorize the Fare, because it's relevant and the differences between the values are huge...

# In[ ]:


train_data["Fare"].hist(bins=50, figsize=(10,5))


# In[ ]:


train_data["Fare"][train_data["Fare"] < 100].hist(bins=20, figsize=(10,5))


# **Notes**
# * Possible categories: Below20, Below40, Below100, Below200, Below300, Above300

# In[ ]:


# Fare Categories
train_data["FareBelow20"] = 0
train_data["FareBetween20And40"] = 0 
train_data["FareBetween40And100"] = 0 
train_data["FareBetween100And200"] = 0
train_data["FareBetween200And300"] = 0
train_data["FareAbove300"] = 0
train_data.loc[train_data["Fare"] < 20, "FareBelow20"] = 1
train_data.loc[(train_data["Fare"] >= 20) & (train_data["Fare"] < 40), "FareBetween20And40"] = 1
train_data.loc[(train_data["Fare"] >= 40) & (train_data["Fare"] < 100), "FareBetween40And100"] = 1
train_data.loc[(train_data["Fare"] >= 100) & (train_data["Fare"] < 200), "FareBetween100And200"] = 1
train_data.loc[(train_data["Fare"] >= 200) & (train_data["Fare"] < 300), "FareBetween200And300"] = 1
train_data.loc[train_data["Fare"] >= 300, "FareAbove300"] = 1


# In[ ]:


corr_matrix = train_data.corr()
survived_correlation = corr_matrix["Survived"].sort_values(ascending=True)
survived_correlation = survived_correlation.drop("Survived")

colors = []
positive_survived_correlation = abs(survived_correlation.values)
max_difference = 1 - positive_survived_correlation.max()

for survived in positive_survived_correlation:
    green = 1 - (survived + max_difference)
    blue = 0
    red = 1
    if(survived<=positive_survived_correlation.mean()) :
        blue = 1
        red = 0
    colors.append((red, green, blue, 1))

features = survived_correlation.axes[0]
plt.figure(figsize=(10,20))
plt.barh(range(len(features)), survived_correlation, color=colors)
plt.yticks(range(len(features)), features)
plt.grid()
plt.show()


# **Notes**
# * Passengers who have paid less than 20 are likely to die

# # FamilySize Category

# In[ ]:


train_data["FamilySize"] = train_data["Parch"] + train_data["SibSp"] + 1


# In[ ]:


train_data["FamilySize"].hist(bins=10, figsize=(10,5))


# **Notes**
# * Possible categories: Below2, Below 4, 4To8, Above8

# In[ ]:


# FamilySize Categories
train_data["FamilySizeBelow2"] = 0
train_data["FamilySizeBetween2And4"] = 0 
train_data["FamilySizeBetween4And8"] = 0 
train_data["FamilySizeAbove8"] = 0
train_data.loc[train_data["FamilySize"] < 2, "FamilySizeBelow2"] = 1
train_data.loc[(train_data["FamilySize"] >= 2) & (train_data["FamilySize"] < 4), "FamilySizeBetween2And4"] = 1
train_data.loc[(train_data["FamilySize"] >= 4) & (train_data["FamilySize"] < 8), "FamilySizeBetween4And8"] = 1
train_data.loc[train_data["FamilySize"] >= 8, "FamilySizeAbove8"] = 1


# In[ ]:


corr_matrix = train_data.corr()
survived_correlation = corr_matrix["Survived"].sort_values(ascending=True)
survived_correlation = survived_correlation.drop("Survived")

colors = []
positive_survived_correlation = abs(survived_correlation.values)
max_difference = 1 - positive_survived_correlation.max()

for survived in positive_survived_correlation:
    green = 1 - (survived + max_difference)
    blue = 0
    red = 1
    if(survived<=positive_survived_correlation.mean()) :
        blue = 1
        red = 0
    colors.append((red, green, blue, 1))

features = survived_correlation.axes[0]
plt.figure(figsize=(10,20))
plt.barh(range(len(features)), survived_correlation, color=colors)
plt.yticks(range(len(features)), features)
plt.grid()
plt.show()


# **Notes**
# * FamilySize 2-4 has a bigger chance to survive
# * FamilySize below 2 has a smaller chance to surviv

# # Fare Correlations

# In[ ]:


corr_matrix = train_data.corr()
survived_correlation = corr_matrix["Fare"].sort_values(ascending=True)
survived_correlation = survived_correlation.drop("Fare")

colors = []
positive_survived_correlation = abs(survived_correlation.values)
max_difference = 1 - positive_survived_correlation.max()

for survived in positive_survived_correlation:
    green = 1 - (survived + max_difference)
    blue = 0
    red = 1
    if(survived<=positive_survived_correlation.mean()) :
        blue = 1
        red = 0
    colors.append((red, green, blue, 1))

features = survived_correlation.axes[0]
plt.figure(figsize=(10,20))
plt.barh(range(len(features)), survived_correlation, color=colors)
plt.yticks(range(len(features)), features)
plt.grid()
plt.show()


# In[ ]:


print("First Class")
print("Median", train_data["Fare"][(train_data["PclassOne"] == 1) & (train_data["Fare"] > 0)].median())
print("Max", train_data["Fare"][(train_data["PclassOne"] == 1) & (train_data["Fare"] > 0)].max())
print("Min", train_data["Fare"][(train_data["PclassOne"] == 1) & (train_data["Fare"] > 0)].min())

print("Second Class")
print("Median", train_data["Fare"][(train_data["PclassTwo"] == 1) & (train_data["Fare"] > 0)].median())
print("Max", train_data["Fare"][(train_data["PclassTwo"] == 1) & (train_data["Fare"] > 0)].max())
print("Min", train_data["Fare"][(train_data["PclassTwo"] == 1) & (train_data["Fare"] > 0)].min())

print("Third Class")
print("Median", train_data["Fare"][(train_data["PclassThree"] == 1) & (train_data["Fare"] > 0)].median())
print("Max", train_data["Fare"][(train_data["PclassThree"] == 1) & (train_data["Fare"] > 0)].max())
print("Min", train_data["Fare"][(train_data["PclassThree"] == 1) & (train_data["Fare"] > 0)].min())


# **Notes**
# * I should check the Pclass from the data sets with missing Fare in the test data. Most first class passengers paid more than 20 and thats a important feature.
