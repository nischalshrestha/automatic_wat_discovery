#!/usr/bin/env python
# coding: utf-8

# **The strategy is to use several machine learning method and compare them. Therefore I import all modules required in the future:**

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


print("modules for data visualization imported")


# ***1. Prepare for model training***

# **It is time to acquire data from our datasets. There are 3 datasets, one for training, one for testing and one as a submission example. I will ignore the example and acquire the training and testing datasets into Pandas variables and combine them incase we have to do some operation on them together.**

# In[ ]:


train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
print("train and test dataset loaded")


# **Take a look at the column names in our dataset so I can know the datatype of  features:**

# In[ ]:


print("features in train dataset \n")
train_dataset.info()
print("features in test dataset \n")
test_dataset.info()


# **I will analyse the column names first to get an idea of the property of features, like if they are categorical or numerical etc.
# And I will also preview the first few rows and last few rows of the dataset so I can know roughly is there any incomplete or incorrect cells
# **

# In[ ]:


train_dataset.head()


# In[ ]:


train_dataset.tail()


# **Based on the movie I watched, I think they were saying "Female and children first!",  now let's create a graph of sex distribution among survivors to prove this:**

# In[ ]:


female_color = "#FA0000"

plt.subplot2grid((3,4), (0,3))
train_dataset.Sex[train_dataset.Survived == 1].value_counts(normalize=True).plot(kind="pie")


# **As expected, female has much bigger survival rate than male, now time to take a look at the influence of age factor. We can prove it by visualizing our data, first let's take a look at the age distribution among different classes**

# In[ ]:


plt.subplot2grid((2, 3), (1, 0), colspan=2)
for x in [1, 2, 3]:
    train_dataset.Age[train_dataset.Pclass == x].plot(kind="kde")
    
plt.title("age distribution among different classes")


# **I can draw some conclusions from this graph, for example, the third class poplulation is younger while the first class population is older etc. Useless? yeah but you never know what information might comes to handy. Next, let's focus on the relationship between age and survive rate:**

# In[ ]:


g = sb.FacetGrid(train_dataset, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g.map(plt.plot, "Age", "Survived", marker=".", color="red")


# **We can see from the graph above, some of the older people died and some of the younger people survived more. But the most dense part of survived and died is basically on the same range between 20 to 45. It makes sense because the older people they have less mobility to save themselves in a disaster like this and they tend to sacrifice themselves because the next generation is the future.**

# **Class would also be a factor as people with higher class get more previledges, it could increase their chance of survival, let's prove my theory: **

# In[ ]:


train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)



# **Let's visualize the data:**

# In[ ]:


plt.subplot2grid((3,4), (1,0), colspan=4)
for x in [1, 2, 3]:
    train_dataset.Survived[train_dataset.Pclass == x].plot(kind="kde")
plt.legend()


# **In the movie, there are two interesting characters, a rich girl and a poor man, it gets me thinking to combine sex and class together, for example, let's find out the survival rate of a rich woman:**

# In[ ]:


plt.subplot2grid((3,4), (2,2))
train_dataset.Survived[(train_dataset.Sex == "female") & (train_dataset.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color="pink")
plt.title("Rich Women Survived")


# **Now it's time to see the survival rate of poor men:**

# In[ ]:


plt.subplot2grid((3,4), (2,2))
train_dataset.Survived[(train_dataset.Sex == "male") & (train_dataset.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color="blue")
plt.title("Poor Men Survived")


# **Therefore I believe if we generate a new feature based on class and sex, our model will learn better from it.**

# ***2. Training the model***

# **First, I will always try to find a model that is intuitive and easy to implement, for example, to predict based purely on sex:**

# In[ ]:


train = train_dataset
train["Hypothesis"] = 0
train.loc[train.Sex == "female", "Hypothesis"] = 1

train["Result"] = 0
train.loc[train.Survived == train["Hypothesis"], "Result"] = 1

print(train["Result"].value_counts(normalize=True))


# **78.67% is not too bad for a simple model like this, however I am looking for something more accurate. Before apply my dataset into different models in scikit-learn library, I have to make some change in our data so it can be accepted. In my opinion, irrelevant factors are nothing but noise, I usually do some logical deduction before apply features in my model. For example, I can not say for sure name has nothing to do with survival rate, because back then some family name could represent class, but it is useless when I already have a class feature. Embarked port can represent the region people live, or live nearby, the difference of environments could affect their behaviour, therefore I will not drop this feature. The cabin cannot represents the physical location when the disaster happened because people tend to gather together after disaster so it is kind meaningless. Same goes for ticket and fare, if you don't add superstitious in your analysis.**

# In[ ]:


def data_process(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    
    
    
    data = data.drop(['Fare'], axis=1)
    data = data.drop(['Ticket'], axis=1)
    data = data.drop(['Cabin'], axis=1)
    freq_port = train_dataset.Embarked.dropna().mode()[0]
#     for dataset in train_dataset:
#         dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
#     for dataset in train_dataset:
#         dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


    data['Embarked'] = data['Embarked'].fillna(freq_port)

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    
    data = data.drop(['Name'], axis=1)
    
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
      
    data.loc[ data['Age'] <= 16, 'Age'] = int(0)
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age']
    
    return data
    
    
    
    
    
# train_dataset = data_process(train_dataset)
# train_dataset.head()
    
    


# **Now since I have my dataset prepared, it's time to apply machine learning method on them and observe the accuracy, let's start with linear regression:**

# In[ ]:


import utils


train_dataset = data_process(train_dataset)
train_dataset.head()




# In[ ]:


from sklearn import linear_model, preprocessing

target = train_dataset["Survived"].values
features = train_dataset[["Pclass", "Sex", "Age", "SibSp", "Parch"]].values

classfier = linear_model.LogisticRegression()
classifier_ = classfier.fit(features, target)

print(classifier_.score(features, target))


# **Looks good for first algorithm, I think I can still improve it if I use polynomial features instead:**

# In[ ]:


poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

classfier = linear_model.LogisticRegression()
classifier_ = classfier.fit(poly_features, target)

print(classifier_.score(poly_features, target))


# **Now I am going to decision tree classifier:**

# In[ ]:


from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(random_state = 1)
decision_tree_ = decision_tree.fit(features, target)

print(decision_tree_.score(features, target))


# **Random forest:**

# In[ ]:


from sklearn.ensemble import *
random_forest = RandomForestClassifier(n_estimators=100)

Y_train_dataset = train_dataset["Survived"]
X_train_dataset = train_dataset.drop("Survived", axis=1)
X_train_dataset = X_train_dataset.drop("PassengerId", axis=1)
X_train_dataset = X_train_dataset.drop("Hypothesis", axis=1)
X_train_dataset = X_train_dataset.drop("Result", axis=1)


X_train_dataset.head()
random_forest.fit(X_train_dataset, Y_train_dataset)



# **Seems like decision tree and random forest have the same accuracy, but I would prefer random forest since it corrects for decision tree's habit of overfitting their training set, now let's use random forest algorithm on our processed test dataset:**

# In[ ]:


X_test_dataset = data_process(test_dataset)
X_test_dataset = X_test_dataset.drop("PassengerId", axis=1)
X_test_dataset.head()
predicted_value = random_forest.predict(X_test_dataset)


# **Now time to submit our predicted value**

# In[ ]:


test_dataset_copy = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": test_dataset_copy["PassengerId"],
        "Survived": predicted_value
})

submission.to_csv('submission.csv', index=False)

