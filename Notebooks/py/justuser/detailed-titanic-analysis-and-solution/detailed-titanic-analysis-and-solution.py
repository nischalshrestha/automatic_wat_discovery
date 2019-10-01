#!/usr/bin/env python
# coding: utf-8

# Titanic is one of the classical problems in machine learning. There are many solutions with different approaches out there, so here is my take on this problem. I tried to explain every step as detailed as I could, too, so if you're new to ML, this notebook may be helpful for you.
# 
# My solution scored 0.79425. If you have noticed any mistakes or if you have any suggestions, you are more than welcome to leave a comment down below.
# 
# With that being said, let's start with importing libraries that we'll need and take a peek at the data:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[ ]:


filePath = "../input/train.csv"
train = pd.read_csv(filePath)
filePath = "../input/test.csv"
test = pd.read_csv(filePath)


# In[ ]:


train.head()


# At the first glance we can already tell that some data is missing.
# 
# First, let's see how much data do we actually miss:

# In[ ]:


plt.figure(figsize=(14, 12))

# don't forget to set titles
plt.subplot(211)
sns.heatmap(train.isnull(), yticklabels=False)
plt.subplot(212)
sns.heatmap(test.isnull(), yticklabels=False)


# As we can see, both in both test and train datasets we miss quite a lot of values. Some data like Age and Embarked may be filled out, but the Cabin column misses so much values that it can't really be used as a feature. It can be transformed or substituted, but we will do that later.
# 
# Now lets focus on the data in details and see if there are any noticeable correlations. 

# # Initial data exploration

# The first thing we need to explore how survivability depends on different factors, such as Sex, Age (younger people are more fit), Passenger Class (possible higher class priority), and Number of Spouses/Siblings
# 
# Let's explore how survivability depends on these features and if there are any correlation between them.

# In[ ]:


plt.figure(figsize=(14, 12))

plt.subplot(321)
sns.countplot('Survived', data=train)
plt.subplot(322)
sns.countplot('Sex', data=train, hue='Survived')
plt.subplot(323)
sns.distplot(train['Age'].dropna(), bins=25)
plt.subplot(324)
sns.countplot('Pclass', data=train, hue='Survived')
plt.subplot(325)
sns.countplot('SibSp', data=train)
plt.subplot(326)
sns.countplot('Parch', data=train)


# From these plots we can make several conclusions:
# 
# * most people didn't survive the crash. 
# * most passengers were males 
# * survivability of women was much higher than of men. We will have to explore the Sex feature more later and see if there are any other interesting correlations.
# * most passengers were middle aged, but there were also quite a few children aboard
# * most passeners had the third class tickets
# * survivability of first and second class passengers were higher compared to the third class
# * most passengers traveled alone or with one sibling/spouse 
# 
# Now we can take a look at each fature specifically to see if it depends on something else or if there ...

# # Filling in the missing data

# Okay, we could jump into full exploration and maybe even transformation of the data, but as we saw before, we miss quite a lot of data. The easiest aproach would be simply dropping all the missing values, be in this case we risk to lose accuracy of our models or entire features. 
# 
# Instead, we will try to fill the missing values based on some logic. Let's take a look at the training data once again to see which values do we miss

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(train.isnull(), yticklabels=False)


# In current state the train data misses Age, Cabin, and Embarked values. Unfortunatelly, the Cabin column is missing most of its data and we can't really use it as a feature. However, it is not entirely useless, but I'll leave it for later. 
# 
# Age column can be filled in many ways. For example, we could take a look at the mean age of every passenger class and fill it based on that information. But instead, if we take a look at the names of the passengers, we can notice a information that can help us:  

# In[ ]:


train.head()


# Every name has a title (such as Mr., Miss., ets.) and follows the following pattern: Last_Name, Title. First_Name. We can categorise passengers by their titles and set unknown age values to mean value of a corresponding title.
# 
# We will do so by adding a column called 'Title' to the data and fill it out with a new funciton.

# In[ ]:


def get_title(pasngr_name):
    
    index_1 = pasngr_name.find(', ') + 2
    index_2 = pasngr_name.find('. ') + 1
    
    return pasngr_name[index_1:index_2]


# In[ ]:


train['Title'] = train['Name'].apply(get_title)
test['Title'] = test['Name'].apply(get_title)


# In[ ]:


plt.figure(figsize=(16, 10))
sns.boxplot('Title', 'Age', data=train)


# Now that we have all the titles, we can find out a mean value for each of them and use it to fill the gaps in the data.

# In[ ]:


train.Title.unique()


# In[ ]:


age_by_title = train.groupby('Title')['Age'].mean()
print(age_by_title)


# In[ ]:


def fill_missing_ages(cols):
    age = cols[0]
    titles = cols[1]
    
    if pd.isnull(age):
        return age_by_title[titles]
    else:
        return age


# In[ ]:


train['Age'] = train[['Age', 'Title']].apply(fill_missing_ages, axis=1)
test['Age'] = test[['Age', 'Title']].apply(fill_missing_ages, axis=1)

#and one Fare value in the test set
test['Fare'].fillna(test['Fare'].mean(), inplace = True)

plt.figure(figsize=(14, 12))

plt.subplot(211)
sns.heatmap(train.isnull(), yticklabels=False)
plt.subplot(212)
sns.heatmap(test.isnull(), yticklabels=False)


# Okay, now we have the Age column filled entirely. There are still missing values in Cabin and Embarked columns. Unfortunatelly, we miss so much data in Cabin that it would be impossible to fill it as we did with Age, but we are not going to get rid of it for now, it will be usefull for us later.
# 
# In embarked column only one value is missing, so we can set it to the most common value.

# In[ ]:


sns.countplot('Embarked', data=train)


# In[ ]:


train['Embarked'].fillna('S', inplace=True)
sns.heatmap(train.isnull(), yticklabels=False)


# Now we have patched the missing data and can explore the features and correlations between them without worrying that we may miss something.

# # Detailed exploration 

# In this section we will try to explore every possible feature and correlations them. Also, ...

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(train.drop('PassengerId', axis=1).corr(), annot=True)


# Here's a shortened plan that we will follow to evaluate each feature and ...:
# * Age 
# * Sex
# * Passenger classes and Fares
# * **(...)**

# ### Age
# The first feature that comes to my mind is Age. The theory is simple: survivability depends on the age of a passenger, old passengers have less chance to survive, younger passengers are more fit, children either not fit enough to survive, or they have higher chances since adults help them

# In[ ]:


plt.figure(figsize=(10, 8))
sns.violinplot('Survived', 'Age', data=train)


# We can already notice that children had better chance to survive, and the majority of casulties were middle aged passengers (which can be explained by the fact that most of the passengers were middle aged). 
# 
# Let's explore the age, but this time separated by the Sex column.

# In[ ]:


plt.figure(figsize=(10, 8))
sns.violinplot('Sex', 'Age', data=train, hue='Survived', split=True)


# The plot above confirmes our theory for the young boys, but it is rather opposite with young girls: most females under the age of 16 didn't survive. This looks weird at first glance, but maybe it is connected with some other feature. 
# 
# Let's see if the class had influence on survivability of females.

# In[ ]:


grid = sns.FacetGrid(train, col='Pclass', hue="Survived", size=4)
grid = grid.map(sns.swarmplot, 'Sex', 'Age', order=["female"])


# ### Pclass
# Idea here is pretty straightforward too: the higher the class, the better chance to survive. First, let's take a look at the overall situation:

# In[ ]:


plt.figure(figsize=(10, 8))
sns.countplot('Pclass', data=train, hue='Survived')


# We can already see that the class plays a big role in survivability. Most of third class passengers didn't survive the crash, second class had 50/50 chance, and most of first class passengers survived.
# 
# Let's further explore Pclass and try to find any correlations with other features.
# 
# If we go back to the correlation heatmap, we will notice that Age and Fare are strongly correlated with Pclass, so they will be our main suspects.

# In[ ]:


plt.figure(figsize=(14, 6))

plt.subplot(121)
sns.barplot('Pclass', 'Fare', data=train)
plt.subplot(122)
sns.barplot('Pclass', 'Age', data=train)


# As expected, these two features indeed are connected with the class. The Fare was rather expected: the higher a class, the more expencive it is. 
# 
# Age can be explained by the fact that usually older people are wealthier than the younger ones. **(...)**
# 
# Here's the overall picture of Fares depending on Ages separated by Classes:

# In[ ]:


sns.lmplot('Age', 'Fare', data=train, hue='Pclass', fit_reg=False, size=7)


# ### Family size
# 
# This feature will represent the family size of a passenger. We have information about number of Siblings/Spouses (SibSp) and Parent/Children relationships (Parch). Although it might not be full information about families, we can use it to determine a family size of each passenger by summing these two features.

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"]
test["FamilySize"] = test["SibSp"] + test["Parch"]
train.head()


# Now let's see how family size affected survivability of passengers:

# In[ ]:


plt.figure(figsize=(14, 6))

plt.subplot(121)
sns.barplot('FamilySize', 'Survived', data=train)
plt.subplot(122)
sns.countplot('FamilySize', data=train, hue='Survived')


# We can notice a curious trend with family size: **(...)**

# In[ ]:


grid = sns.FacetGrid(train, col='Sex', size=6)
grid = grid.map(sns.barplot, 'FamilySize', 'Survived')


# These two plots only confirm our theory. With family size more than 3 survivability drops severely for both women and men. We also should keep in mind while looking at the plots above that women had overall better chances to survive than men.
# 
# Let's just check if this trend depends on something else, like Pclass, for example:

# In[ ]:


plt.figure(figsize=(10, 8))
sns.countplot('FamilySize',  data=train, hue='Pclass')


# ### Embarked
# 

# In[ ]:


sns.countplot('Embarked', data=train, hue='Survived')


# In[ ]:


sns.countplot('Embarked', data=train, hue='Pclass')


# ### Conclusion: 

# # Additional features

# Now we've analyzed the data and have an idea of what will be relevant. But before we start building our model, there is one thing we can do to improve it even further.
# 
# So far we've worked with features that came with the dataset, but we can also create our own custom features (so far we have FamilySize as a custom, or engineered feature).

# ### Cabin
# Now this is a tricky part. Cabin could be a really important feature, especially if we knew the distribution of cabins on the ship, but we miss so much data that there is almost no practical value in the feature itself. However, there is one trick we can do with it. 
# 
# Let's create a new feature called CabinKnown that represents if a cabin of a certain passenger is known or not. Our theory here is that if the cabin is known, then probably that passenger survived.

# In[ ]:


def has_cabin(pasngr_cabin):
    
    if pd.isnull(pasngr_cabin):
        return 0
    else:
        return 1
    
train['CabinKnown'] = train['Cabin'].apply(has_cabin)
test['CabinKnown'] = test['Cabin'].apply(has_cabin)
sns.countplot('CabinKnown', data=train, hue='Survived')


# Clearly, the corelation here is strong: the survivability rate of those passengers, whose cabin is known is 2:1, while situation in case the cabin is unknown is opposite. This would be a very useful feature to have.
# 
# But there is one problem with this feature. In real life, we wouldn't know in advance whether a cabin would be known or not (we can't know an outcome before an event happened). That's why this feature is rather "artificial". Sure, it can improve the score of our model for this competition, but using it is kinda cheating.
# 
# **(decide what u wanna do with that feature and finish the description)**

# ### Age categories
# 
# ** * (explain why categories) * **
# 
# Let's start with Age. The most logical way is to devide age into age categories: young, adult, and elder. Let's say that passenger of the age of 16 and younger are children, older than 50 are elder, and anyone else is adult.

# In[ ]:


def get_age_categories(age):
    if(age <= 16):
        return 'child'
    elif(age > 16 and age <= 50):
        return 'adult'
    else:
        return 'elder'
    
train['AgeCategory'] = train['Age'].apply(get_age_categories)
test['AgeCategory'] = test['Age'].apply(get_age_categories)


# In[ ]:


sns.countplot('AgeCategory', data=train, hue='Survived')


# ** (...) **

# ### Family size category
# 
# Now lets do the same for the family size: we will separate it into TraveledAlone, WithFamily, and WithLargeFamily (bigger than 3, where the survivability rate changes the most)

# In[ ]:


def get_family_category(family_size):
    
    if(family_size > 3):
        return 'WithLargeFamily'
    elif(family_size > 0 and family_size<= 3):
        return 'WithFamily'
    else:
        return 'TraveledAlone'
    
train['FamilyCategory'] = train['FamilySize'].apply(get_family_category)
test['FamilyCategory'] = test['FamilySize'].apply(get_family_category)


# ** (needs a description depending on whether it will be included or not) ** 

# ### Title category

# In[ ]:


print(train.Title.unique())


# In[ ]:


plt.figure(figsize=(12, 10))
sns.countplot('Title', data=train)


# In[ ]:


titles_to_cats = {
    'HighClass': ['Lady.', 'Sir.'],
    'MiddleClass': ['Mr.', 'Mrs.'],
    'LowClass': []
}


# ### Fare scaling
# 
# If we take a look at the Fare distribution, we will see that it is scattered a lot:

# In[ ]:


plt.figure(figsize=(10, 8))
sns.distplot(train['Fare'])


# # Creating the model:

# Now that we have all the data we need, we can start building the model. 
# 
# First of all, we need to prepare the data for the actual model. Classification algorithms work only with numbers or True/False values. For example, model can't tell the difference in Sex at the moment because we have text in that field. What we can do is transform the values of this feature into True or False (IsMale = True for males and IsMale = False for women).
# 
# For this purpose we will use two methods: transofrmation data into numerical values and dummies.
# 
# Lets start with Sex and transformation:

# In[ ]:


train['Sex'] = train['Sex'].astype('category').cat.codes
test['Sex'] = test['Sex'].astype('category').cat.codes
train[['Name', 'Sex']].head()


# As we see, the Sex column is now binary and takes 1 for males and 0 for females. Now classifiers will be able to work with it. 
# 
# Now we will transform Embarked column, but with a different method:

# In[ ]:


embarkedCat = pd.get_dummies(train['Embarked'])
train = pd.concat([train, embarkedCat], axis=1)
train.drop('Embarked', axis=1, inplace=True)

embarkedCat = pd.get_dummies(test['Embarked'])
test = pd.concat([test, embarkedCat], axis=1)
test.drop('Embarked', axis=1, inplace=True)

train[['Q', 'S', 'C']].head()


# We used dummies, which replaced the Embarked column with three new columns corresponding to the values in the old column. Lets do the same for family size and age categories:

# In[ ]:


# for the train set
familyCat = pd.get_dummies(train['FamilyCategory'])
train = pd.concat([train, familyCat], axis=1)
train.drop('FamilyCategory', axis=1, inplace=True)

ageCat = pd.get_dummies(train['AgeCategory'])
train = pd.concat([train, ageCat], axis=1)
train.drop('AgeCategory', axis=1, inplace=True)

#and for the test
familyCat = pd.get_dummies(test['FamilyCategory'])
test = pd.concat([test, familyCat], axis=1)
test.drop('FamilyCategory', axis=1, inplace=True)

ageCat = pd.get_dummies(test['AgeCategory'])
test = pd.concat([test, ageCat], axis=1)
test.drop('AgeCategory', axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(14,12))
sns.heatmap(train.drop('PassengerId', axis=1).corr(), annot=True)


# # Modelling
# Now we need to select a classification algorithm for the model. There are plenty of decent classifiers, but which is the best for this task and which one should we choose? 
# 
# *Here's the idea:* we will take a bunch of classifiers, test them on the data, and choose the best one.
# 
# In order to do that, we will create a list of different classifiers and see how each of them performs on the training data. To select the best one, we will evaluate them using cross-validation and compare their accuracy scores (percentage of the right answers). I decided to use Random Forest, KNN, SVC, Decision Tree, AdaBoost, Gradient Boost, Extremely Randomized Trees, and Logistic Regression.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

classifiers = [
    RandomForestClassifier(),
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    LogisticRegression()
]


# Now we need to select the features that will be used in the model and drop everything else. Also, the training data has to be split in two parts: *X_train* is the data the classifiers will be trained on, and *y_train* are the answers.

# In[ ]:


X_train = train.drop(['PassengerId', 'Survived', 'SibSp', 'Parch', 'Ticket', 'Name', 'Cabin', 'Title', 'FamilySize'], axis=1)
y_train = train['Survived']

X_final = test.drop(['PassengerId', 'SibSp', 'Parch', 'Ticket', 'Name', 'Cabin', 'Title', 'FamilySize'], axis=1)


# We will use K-Folds as cross-validation. It splits the data into "folds", ** (...) **

# In[ ]:


from sklearn.model_selection import KFold

# n_splits=5
cv_kfold = KFold(n_splits=10)


# Now we evaluate each of the classifiers from the list using K-Folds. The accuracy scores will be stored in a list.
# 
# The problem is that K-Folds evaluates each algorithm several times. As result, we will have a list of arrays with scores for each classifier, which is not great for comparison. 
# 
# To fix it, we will create another list of means of scores for each classifier. That way it will be much easier to compare the algorithms and select the best one.  

# In[ ]:


from sklearn.model_selection import cross_val_score

class_scores = []
for classifier in classifiers:
    class_scores.append(cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv_kfold))
    
class_mean_scores = []
for score in class_scores:
    class_mean_scores.append(score.mean())


# Now that we have the mean accuracy scores, we need to compare them somehow. But since it's just a list of numbers, we can easily plot them. First, let's create a data frame of classifiers names and their scores, and then plot it:

# In[ ]:


scores_df = pd.DataFrame({
    'Classifier':['Random Forest', 'KNeighbors', 'SVC', 'DecisionTreeClassifier', 'AdaBoostClassifier', 
                  'GradientBoostingClassifier', 'ExtraTreesClassifier', 'LogisticRegression'], 
    'Scores': class_mean_scores
})

print(scores_df)
sns.factorplot('Scores', 'Classifier', data=scores_df, size=6)


# Two best classifiers happened to be Gradient Boost and Logistic Regression. Since Logistic Regression got sligthly lower score and is rather easily overfitted, we will use Gradient Boost. 

# ### Selecting the parameters
# Now that we've chosen the algorithm, we need to select the best parameters for it. There are many options, and sometimes it's almost impossible to know the best set of parameters. That's why we will use Grid Search to test out different options and choose the best ones.
# 
# But first let's take a look at all the possible parameters of Gradient Boosting classifier:

# In[ ]:


g_boost = GradientBoostingClassifier()
g_boost.get_params().keys()


# We will test different options for min_samples_leaf, min_samples_split, max_depth, and loss parameters. I will set n_estimators to 100, but it can be increased since Gradient Boosting algorithms generally don't tend to overfit.

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'loss': ['deviance', 'exponential'],
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100],
    'max_depth': [3, 5, 10, 20]
}

grid_cv = GridSearchCV(g_boost, param_grid, scoring='accuracy', cv=cv_kfold)
grid_cv.fit(X_train, y_train)
grid_cv.best_estimator_


# In[ ]:


print(grid_cv.best_score_)
print(grid_cv.best_params_)


# Now that we have the best parameters we could find, it's time to create and train the model on the training data.

# In[ ]:


g_boost = GradientBoostingClassifier(min_samples_split=5, loss='deviance', n_estimators=1000, 
                                     max_depth=3, min_samples_leaf=2)


# In[ ]:


g_boost.fit(X_train, y_train)


# In[ ]:


feature_values = pd.DataFrame({
    'Feature': X_final.columns,
    'Importance': g_boost.feature_importances_
})

print(feature_values)
sns.factorplot('Importance', 'Feature', data=feature_values, size=6)


# ### Prediction on the testing set and output
# Now our model is ready, and we can make a prediction on the testing set and create a .csv output for submission.

# In[ ]:


prediction = g_boost.predict(X_final)


# In[ ]:


submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': prediction
})


# In[ ]:


#submission.to_csv('submission.csv', index=False)

