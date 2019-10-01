#!/usr/bin/env python
# coding: utf-8

# In this notebook I try to engineer a few features. I also try and apply several machine learning algorithms.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Part 1: Exploring the data adn Feature Engineering

# In[ ]:


df = pd.read_csv("../input/train.csv")
df.index = df.PassengerId
df.drop('PassengerId', axis = 1, inplace = True)
df.head()


# The "train.csv" was read into the kernel and it is called df. The Passenger ID column was made the index.
# 
# ## Overall boxplot for all numeric variables
# This is a quick view of how the numeric features vary with "Survived" or "Not Survived"
# 
# P.S. There is a problem with these graphs. The y axis is same in all graphs, so the graph does not tell us as much as it should. Still, its a good first level visualization.

# In[ ]:


df.boxplot(by = 'Survived', figsize = (10, 20))
plt.show()


# ## Playing around with features.

# In[ ]:


df[['Age', 'Survived']].boxplot(by = "Survived")
plt.show()


# In[ ]:


ax = df[['Age', 'Survived', 'Sex']].groupby('Sex').boxplot(by = "Survived")
plt.show()


# #### inference: We dont see a very clear pattern in age only. 
# 
# #### But age coupled with gender may be of much more predictive value.

# ## Exploring socio economics

# In[ ]:


df[['Fare', 'Survived']].boxplot(by = "Survived")
plt.show()


# ### How survival rates change with Pclass

# In[ ]:


df[['Pclass', 'Survived']].groupby('Pclass').mean()


# ## Exploring family effects

# In[ ]:


df[['SibSp', 'Survived']].boxplot(by = 'Survived')
plt.show()


# In[ ]:


tempdf1 = df[['SibSp', 'Survived']].groupby('SibSp').count().merge(df[['SibSp', 'Survived']].groupby('SibSp').mean(), right_index = True, left_index = True)
tempdf1.columns = ['Count', 'Prob. Survived']
tempdf1


# In[ ]:


tempdf2 = df[['SibSp', 'Survived']].groupby('SibSp').count().merge(df[['SibSp', 'Survived']].groupby('SibSp').sum(), right_index = True, left_index = True)
tempdf2.columns = ['Count', 'Survived']
tempdf2.plot.bar(figsize = (10, 8))
plt.show()


# In[ ]:


tempdf3 = df[['Parch', 'Survived']].groupby('Parch').count().merge(df[['Parch', 'Survived']].groupby('Parch').mean(), right_index = True, left_index = True)
tempdf3.columns = ['Count', 'Ratio. Survived']
tempdf3


# In[ ]:


df['Family_Size'] = df.Parch + df.SibSp + 1


# In[ ]:


tempdf4 = df[['Family_Size', 'Survived']].groupby('Family_Size').count().merge(df[['Family_Size', 'Survived']].groupby('Family_Size').mean(), right_index = True, left_index = True)
tempdf4.columns = ['Count', 'Ratio. Survived']
tempdf4


# In[ ]:


tempdf6 = df[['Family_Size', 'Survived']].groupby('Family_Size').count().merge(df[['Family_Size', 'Survived']].groupby('Family_Size').sum(), right_index = True, left_index = True)
tempdf6.columns = ['Count', 'Survived']
tempdf6.plot.bar(figsize = (8, 5))
plt.show()


# ### There seems to be a clear pattern in the survival rate by family size. Hence it can be an important feature
# #### Since family size seems to be more predictive than 'Sibsp' and 'Parch', we frop the latter two.

# In[ ]:


df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)


# ## Exploring Names

# In[ ]:


df['Title'] = df['Name'].apply(lambda x: x.split(",")[1].split(" ")[1])
df.head()


# In[ ]:


tempdf5 = df[['Title', 'Survived']].groupby('Title').count().merge(df[['Title', 'Survived']].groupby('Title').mean(), right_index = True, left_index = True)
tempdf5.columns = ['Count', 'Ratio. Survived']
tempdf5


# In[ ]:


df.drop('Name', inplace = True, axis = 1)
df.head()


# ### Since we have already extracted the most important features from the names column, we can drop it.

# ## About cabins and decks

# In[ ]:


df['Cabin'] = df['Cabin'].fillna('No')
# Since all 3rd class passengers didnt have cabins
df.head()


# ### Extracting Cabin Deck and Number

# In[ ]:


df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split(" ")[-1][0] if x != "No" else "No")
df['Cabin_number'] = df['Cabin'].apply(lambda x: 0 if len(x) == 1 else int(x.split(" ")[-1][1:]) if x != "No" else 0)
df.head()


# In[ ]:


tempdf7 = df[['Cabin_deck', 'Survived']].groupby('Cabin_deck').count().merge(df[['Cabin_deck', 'Survived']].groupby('Cabin_deck').mean(), right_index = True, left_index = True)
tempdf7.columns = ['Count', 'Ratio. Survived']
tempdf7


# In[ ]:


tempdf8 = df[['Cabin_number', 'Survived']].groupby('Cabin_number').count().merge(df[['Cabin_number', 'Survived']].groupby('Cabin_number').mean(), right_index = True, left_index = True)
tempdf8.columns = ['Count', 'Ratio. Survived']
tempdf8


# ## Since cabin Number is not very informative in itself, it is changed to a range:

# In[ ]:


df['Cabin_numeric_range'] = df['Cabin_number'].apply(lambda x: str(int(x/10)) + "0 to " + str(int(x/10 + 1)) + "0" if x != 0 else "No Cabin")
df.head()


# In[ ]:


tempdf9 = df[['Cabin_numeric_range', 'Survived']].groupby('Cabin_numeric_range').count().merge(df[['Cabin_numeric_range', 'Survived']].groupby('Cabin_numeric_range').mean(), right_index = True, left_index = True)
tempdf9.columns = ['Count', 'Ratio Survived']
tempdf9


# ## This seems that it might add to the predictive value.

# In[ ]:


df.drop(['Cabin', 'Cabin_number'], inplace = True, axis = 1)
df.head()


# In[ ]:


df.drop('Ticket', inplace = True, axis = 1)
df.head()


# In[ ]:


tempdf10 = df[['Embarked', 'Survived']].groupby('Embarked').count().merge(df[['Embarked', 'Survived']].groupby('Embarked').mean(), right_index = True, left_index = True)
tempdf10.columns = ['Count', 'Ratio. Survived']
tempdf10


# ## Passangers boarding from C has a higher chance of survival compared to the rest. So we keep this feature.

# # Part 2: Processing all the features

# In[ ]:


df['Male'] = df['Sex'].apply(lambda x: 1 if x == "male" else 0)
df.drop('Sex', inplace = True, axis = 1)
df.head()


# ## Getting rid of NaNs

# In[ ]:


df['Age'].fillna(np.mean(df.Age), inplace = True)


# ## Making Dummy variables

# In[ ]:


ndf = pd.get_dummies(df, columns = ['Embarked', 'Title', 'Cabin_deck', 'Cabin_numeric_range'])
ndf.head()


# In[ ]:


ndf.drop(['Cabin_numeric_range_No Cabin', 'Cabin_deck_No'], inplace = True, axis = 1)


# In[ ]:


ndf.columns


# In[ ]:


survived = ndf.Survived
ndf.drop('Survived', inplace = True, axis = 1)


# In[ ]:


ndf.head()


# ## Making the train test splits

# In[ ]:


from sklearn.cross_validation import train_test_split as ttspl
df_train, df_test, out_train, out_test = ttspl(ndf, survived, test_size = 0.25)


# # Part 3: Machine Learning

# ## KNN Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as KNN
for i in range(1, 20):
    knn = KNN(n_neighbors = i)
    knn.fit(df_train, out_train)
    print("Neighbors = " + str(i) + "\t Score: ",)
    print(knn.score(df_test, out_test))


# ### The KNN results show an accuracy of around 70%, which to be very honest is nothing close to what I was looking for. SO I will try other methods as well

# ## Naive Bayes Classifier

# ### Gaussian NB

# In[ ]:


from sklearn.naive_bayes import GaussianNB as GNB
gnb = GNB()
gnb.fit(df_train, out_train)
gnb.score(df_test, out_test)


# ### Not good

# In[ ]:


from sklearn.naive_bayes import MultinomialNB as MNB
mnb = MNB()
mnb.fit(df_train, out_train)
mnb.score(df_test, out_test)


# ### Not good yet

# In[ ]:


from sklearn.naive_bayes import BernoulliNB as BNB
bnb = BNB()
bnb.fit(df_train, out_train)
bnb.score(df_test, out_test)


# ### Better, but still not close to 90%

# #### So the Naive Bayes Classifier also doesnot do a good job in this task.
# #### Now I will try and experiment with Tree based classifiers.

# ## Decision Trees

# In[ ]:


from sklearn.cross_validation import cross_val_score as cvs
from sklearn.tree import DecisionTreeClassifier as dtree
tr = dtree()
cvs(tr, df_train, out_train, cv = 10)


# ### Playing around with the tree parameters
# 
# #### Effect of changing the max depth

# In[ ]:


for i in range(2, 20):
    tr = dtree(max_depth= i)
    print("Max Depth = " + str(i) + "\t Score: ")
    print(np.mean(cvs(tr, df_train, out_train, cv = 10)))
    print("\n")


# ### Visualization of effects of max depth

# In[ ]:


x = []
y = []
for i in range(2, 20):
    x.append(i)
    tr = dtree(max_depth= i)
    y.append(np.mean(cvs(tr, df_train, out_train, cv = 10)))
    
p = plt.plot(x, y)
plt.show()
    


# #### Effect of changing the max leaf nodes

# In[ ]:


for i in range(2, 40):
    tr = dtree(max_leaf_nodes = i)
    print("Max Leaf Nodes = " + str(i) + "\t Score: ")
    print(np.mean(cvs(tr, df_train, out_train, cv = 10)))


# ### Visualization of effects of leaf_nodes

# In[ ]:


x = []
y = []
for i in range(2, 100, 2):
    x.append(i)
    tr = dtree(max_leaf_nodes = i)
    y.append(np.mean(cvs(tr, df_train, out_train, cv = 10)))
    
p = plt.plot(x, y)
plt.show()


# Seems that max_leaf_nodes = 40 gives good results.

# ### Trying the parameters on the test_split data###

# In[ ]:


tr = dtree(max_leaf_nodes = 40)
tr.fit(df_train, out_train)
tr.score(df_test, out_test)


# ### The out of sample accuracy is still around 80%. This is good. But still, more improvements are needed. I need to reach out of sample accuracy of around 90%
