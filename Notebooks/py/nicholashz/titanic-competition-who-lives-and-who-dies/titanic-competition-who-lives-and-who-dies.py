#!/usr/bin/env python
# coding: utf-8

# # Playing Poseidon: Deciding (or at least Predicting) Who Lives and Who Dies
# 
# ## Nick Heise
# 
# ### 18 October 2018
# 
# <ol>
# <li>Breaking the ice (introduction)</li>
# <li>Getting our feet wet (setting up the data)</li>
#     <ol>
#     <li>Drop the unimportant stuff</li>
#     <li>Observe missing values</li>
#     </ol>
# <li>Taking only what really matters (feature extraction)</li>
#     <ol>
#     <li>Please exit your cabin in an orderly manner</li>
#     <li>At least someone's got a map... (mapping numerical to categorical)</li>
#     <li>"Lady"s first (feature engineering from Names column)</li>
#     <li>Keeping it in the family</li>
#     <li>Age is but a [missing] number</li>
#     <li>Sticking together (Grouping continuous numerical data)</li>
#     </ol>
# <li>Diving in (machine learning)</li>
# <li>Conclusions</li>
# </ol> 

# ## 1. Breaking the ice (introduction)
# 
# <p>I am currently a Masters student studying Data Science, and am using this fun and interesting Kaggle competition for practice. Hopefully this kernel is useful (it's my first), and if you have suggestions for a budding data scientist please leave a comment.</p>
# <p>This kernel is a mixture of my own analysis and what I've learned from other kernels. If you would like me to direct you to other kernels I used to improve my analysis, don't hesitate to ask. Particularly, these two were exceptional and deserve credit for greatly helping me learn:</p>
# 
# [Titanic Top 4% with ensemble modeling](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)
# 
# [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)  
# 
# <p>You can find this project along with my others on Github.</p>
# 
# [Github](https://github.com/nicholashz)

# In[ ]:


# Packages for the data
import pandas as pd
import numpy as np

# Package for visualization
import seaborn as sns

# Packages for machine learning classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


# ## 2. Getting our feet wet (setting up the data)
# 
# <p>For our initial gathering of the data, we'll want to read the data and fill any missing values.</p>

# In[ ]:


# Read the data into dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Put all data together so we can wrangle all the data at the same time
all_data = pd.concat(objs=[train_df, test_df], axis=0, sort=True).reset_index(drop=True)

all_data.head()


# ### A. Drop the unimportant stuff
# 
# <p>Taking a look at our data, we can see that Ticket values are likely useless for our purposes. I've seen some other kernels have tried to extract the prefix and use it, but we won't do that here.</p>
# 
# <p>Let's drop this data and deal with the important values.</p>

# In[ ]:


train_df.drop(labels='Ticket', axis='columns', inplace=True)
test_df.drop(labels='Ticket', axis='columns', inplace=True)


# ### B. Observe missing values
# 
# <p>Let's take a look at what missing values we'll have to handle.</p>

# In[ ]:


# Find out what columns have null values
all_data.isnull().sum()


# <p>We have a few columns with missing values: Age, Cabin, Embarked, and Fare. The missing values from the Survived column are from our test set, so we'll ignore that.</p>
# <p>We can easily deal with the few missing values in Embarked and Fare by filling with the mode and median, respectively. As we'll see in the next section, the missing values in Cabin actually give us information so we'll handle that accordingly. Finally, Age will be slightly more complicated to fill and will be handled later.</p>
# 
# <p>So, for now, we fill the Embarked and Fare values with mode and median.</p>

# In[ ]:


for df in [train_df, test_df]:
    # Fill NaN values for Embarked and Fare
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

train_df.isnull().sum()


# ## 3. Taking only what really matters (feature extraction)
# 
# <p>We've got some work to do to get our data 'machine learning'-ready.</p>
# 
# 
# ### A. Please exit your cabin in an orderly manner
# 
# <p>One thing you may have noticed in the last section is that the Cabin column has many missing values - in fact, more passengers are missing a Cabin value than have one. We assume that this means that passenger doesn't have a cabin. Let's replace Cabin data with categorical data based on whether or not the passenger has a cabin.</p>

# In[ ]:


for df in [train_df, test_df]:
    df.loc[df['Cabin'].isnull(), 'HasCabin'] = 0
    df.loc[df['Cabin'].notnull(), 'HasCabin'] = 1
    
    df.drop(labels='Cabin', axis='columns', inplace=True)

pd.crosstab(train_df['Sex'], train_df['HasCabin'])


# 
# ### B. One-hot encoding the categorical data
# 
# <p>We have two columns which are categorical that should be modified: Sex and Embarked. We'll want to represent the categorical data by numbers instead of strings, so we'll do a one-hot encoding to make new columns with binary values.</p>
# 
# <p>First, let's see how Sex and Embarked are related to the survival of the passengers.</p>

# In[ ]:


sns.catplot(x='Sex', y='Survived', kind='bar', data=train_df)


# In[ ]:


sns.catplot(x='Embarked', y='Survived', kind='bar', data=train_df)


# <p>We can see that females are much more likely to survive than males. This is no surprise given the "women and children first" procedure for evacuating. Further, it looks like there is significant variation in the survival rates of passengers boarding from the different Embarked locations.</p>
# 
# <p>Let's create the new columns for our one-hot encoding.</p>

# In[ ]:


train_dummies = pd.get_dummies(data=train_df[['Sex', 'Embarked']])
train_df = pd.concat([train_df, train_dummies], axis=1)

test_dummies = pd.get_dummies(data=test_df[['Sex', 'Embarked']])
test_df = pd.concat([test_df, test_dummies], axis=1)

for df in [train_df, test_df]:
    df.drop(labels=['Sex', 'Embarked'], axis='columns', inplace=True)

train_df.head()


# ### C. "Lady"s first (feature engineering from Names column)
# 
# <p>Conveniently for us, each passenger's name includes their title. This might be useful for determining social status and, consequently, their likelihood to survive.</p>
# <p>First, let's pull each passenger's title with a simple regex and see what we've got.</p>

# In[ ]:


for df in [train_df, test_df]:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Pclass'], train_df['Title'])


# <p>We have a few titles that are very popular, which we can leave alone. It doesn't do much good to leave passengers in groups of size 1 or 2 though, so we'll convert some of the titles to an 'Other' group. In addition, some titles are essentially repetitions (Mme is French for Mrs, Mlle is French for Miss, and Ms is the same as Miss) so we'll take care of those too.</p>
# <p>Once we've got our titles separated properly, we'll take a look at how the survival rates varies by title.</p>

# In[ ]:


for df in [train_df, test_df]:
    # Combine odd or repeated titles
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr',                                        'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Dona'], 'Other')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Drop the names
    df.drop(labels='Name', axis='columns', inplace=True)

sns.catplot(x='Title', y='Survived', kind='bar', data=train_df)


# <p>We see that those with the Mr title have low survival rate, likely for the same reason that males in general do. Master is a title for children, so it makes sense that this survival rate would be much higher. Finally, the Other category has many titles associated with higher social standing so it also makes sense that we'd see higher survival in this group.</p>
# 
# <p>As with Sex and Embarked, let's do a one-hot encoding for title values.</p>

# In[ ]:


train_dummies = pd.get_dummies(data=train_df['Title'], prefix='Title')
train_df = pd.concat([train_df, train_dummies], axis=1)

test_dummies = pd.get_dummies(data=test_df['Title'], prefix='Title')
test_df = pd.concat([test_df, test_dummies], axis=1)

for df in [train_df, test_df]:
    df.drop(labels='Title', axis='columns', inplace=True)

train_df.head()


# ### D. Keeping it in the family
# 
# <p>The data we were given include two columns related to family information for each passenger: SibSp and Parch. We'll combine these into one column called FamilySize, and then see how that correlates with survival.</p>

# In[ ]:


for df in [train_df, test_df]:
    # Replace SibSp and Parch with a single FamilySize column
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(labels=['SibSp', 'Parch'], axis='columns', inplace=True)

train_df[['FamilySize', 'Survived']].groupby('FamilySize').mean()


# <p>Based on this information, it makes sense to place FamilySize into 4 bins for families of size 1, 2-3, 4, and 5+. Let's do that now.</p>

# In[ ]:


for df in [train_df, test_df]:
    df.loc[df['FamilySize'] == 1, 'FamilySize'] = 0
    df.loc[(df['FamilySize'] == 2) | (df['FamilySize'] == 3), 'FamilySize'] = 1
    df.loc[df['FamilySize'] == 4, 'FamilySize'] = 2
    df.loc[df['FamilySize'] > 4, 'FamilySize'] = 3

sns.catplot(x='FamilySize', y='Survived', kind='bar', data=train_df)


# ### E. Age is but a [missing] number
# 
# <p>For the Embarked and Fare columns, which were missing few values each, we simply filled with the mode and median. Age, however, is missing a few hundred values and is likely an important factor in survival. So, we'll try using the other features that we just cleaned up in order to fill appropriate values for Age.</p>
# <p>In order to fill in appropriate values for our missing age data, we'll look at how the other numerical features of our data correlate to Age.</p>

# In[ ]:


corr = train_df[['Age', 'Pclass', 'HasCabin', 'FamilySize']].corr()
sns.heatmap(corr, cmap=sns.color_palette('coolwarm', 7), center=0)


# In[ ]:


corr


# <p>Since Age is most strongly correlated with Pclass and FamilySize, we'll use those data to fill out the missing Age values. We'll replace eaching missing Age with the median of the ages in that passenger's class/familysize subset.</p>

# In[ ]:


for df in [train_df, test_df]:
    age_medians = np.zeros(shape=(3, 4))
    for pclass in range(age_medians.shape[0]):
        for familysize in range(age_medians.shape[1]):
            age_medians[pclass][familysize] = df.loc[(df['Pclass'] == pclass+1) &                                                      (df['FamilySize'] == familysize),
                                                     'Age'].median()
            df.loc[(df['Age'].isnull()) &                    (df['Pclass'] == pclass+1) &                    (df['FamilySize'] == familysize),
                    'Age'] = age_medians[pclass][familysize]

# Double check that we have no more null values
train_df.isnull().sum()


# ### F. Sticking together (Grouping continuous numerical data)
# 
# <p>There is one final way we will be cleaning our data. We currently have two columns containing continuous numerical data: Age and Fare. It would be helpful to categorize these data into bands and place each passenger in an Age category and a Fare category.</p>
# <p>Beginning with Age, we'll split up the passengers into different groups.</p>

# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 10)
train_df[['AgeBand', 'Survived']].groupby('AgeBand').mean()


# <p>Based on these bands, it makes sense to break the ages up as follows: a group for ages under 8, a group for ages between 8 and 32, a group for ages 32 to 56, and a group for all passengers older than 56.</p>

# In[ ]:


train_df.drop(labels='AgeBand', axis='columns', inplace=True)

for df in [train_df, test_df]:
    df.loc[(df['Age'] > 0) & (df['Age'] <= 8), 'AgeGroup'] = 0
    df.loc[(df['Age'] > 8) & (df['Age'] <= 32), 'AgeGroup'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 56), 'AgeGroup'] = 2
    df.loc[df['Age'] > 56, 'AgeGroup'] = 3

    df.drop(labels='Age', axis='columns', inplace=True)
    
sns.catplot(x='AgeGroup', y='Survived', kind='bar', data=train_df)


# <p>Similary, we'll split up the continuous numerical data in the Fare column into groups. However, unlike with Age, we will split the Fare values based on quartiles.</p>

# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby('FareBand').mean()


# In[ ]:


train_df.drop(labels='FareBand', axis='columns', inplace=True)

for df in [train_df, test_df]:
    df.loc[df['Fare'] <= 7.91, 'FareGroup'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.45), 'FareGroup'] = 1
    df.loc[(df['Fare'] > 14.45) & (df['Fare'] <= 31), 'FareGroup'] = 2
    df.loc[df['Fare'] > 31, 'FareGroup'] = 3
    
    df.drop(labels='Fare', axis='columns', inplace=True)
    
sns.catplot(x='FareGroup', y='Survived', kind='bar', data=train_df)


# <p>Great, we've done all the data wrangling and feature extraction we'll need for machine learning! Check out our shiny new dataset for training:</p>

# In[ ]:


train_df.head()


# ## 4. Diving in (machine learning)
# 
# <p>Now that the data has been properly cleaned, we can try to find a model that works well for making our predictions. We'll first split our training set into the features (Xtrain) and the results (Ytrain). We'll also load in some classifiers which we will compare.</p>

# In[ ]:


Ytrain = train_df['Survived']
Xtrain = train_df.drop(columns=['PassengerId', 'Survived'])

RFC = RandomForestClassifier()
Ada = AdaBoostClassifier()
KNN = KNeighborsClassifier()
classifiers = [RFC, Ada, KNN]
clf_names = ['Random Forest', 'AdaBoost', 'K Nearest Neighbors']


# <p>For this analysis, we'll only be comparing across three classifiers: Random Forest, AdaBoost, and K Nearest Neighbors. For more information on other potential (or more complicated) classifiers I invite you to check out the other kernels posted by those who top the leaderboards for this competition.</p>
# 
# <p>For each of these classifiers, we'll want to make sure we create the models with the optimal parameters. We can do this with a Grid Search. We define the set of parameters we want to scan for each type of classifier, and then run our grid searches.</p>

# In[ ]:


# Use kfold as our cross validation
kfold = StratifiedKFold(n_splits=10)

# Set grid search parameter settings
rfc_param_grid = {'max_depth': [None],
                 'max_features': [1, 4, 8],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 5, 10],
                 'bootstrap': [False],
                 'n_estimators': [100, 300, 500],
                 'criterion': ['gini']}
ada_param_grid = {'n_estimators': [25, 50, 100, 200],
                 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
knn_param_grid = {'n_neighbors': [5, 10, 20, 30, 50, 100],
                  'weights': ['uniform', 'distance'],
                 'leaf_size': [5, 10, 20, 30, 50, 100]}
param_grids = [rfc_param_grid, ada_param_grid, knn_param_grid]

# Perform grid searches to get estimators with the optimal settings
grid_searches = []
for i in range(len(classifiers)):
    grid_searches.append(GridSearchCV(estimator=classifiers[i], param_grid=param_grids[i], 
                                      n_jobs=4, cv=kfold, verbose=1))


# <p>We'll now want to see the training scores for each of our models and determine which one works the best. We'll fit each model to our training set and add the best scores from each to a list.</p>

# In[ ]:


# Train the models
best_scores = []
for i in range(len(grid_searches)):
    grid_searches[i].fit(Xtrain, Ytrain)
    best_scores.append(grid_searches[i].best_score_)


# <p>Let's see the best scores for each classifier.</p>

# In[ ]:


# Best scores
for i in range(len(best_scores)):
    print(clf_names[i] + ": " + str(best_scores[i]))


# <p>Based on these training scores, it makes the most sense to use the Random Forest Classifier to make the predictions. We'll predict on the test set, and then write the predictions to a csv file for submission.</p>

# In[ ]:


# Make predictions
Xtest = test_df.drop(columns='PassengerId', axis='columns')
predictions = grid_searches[0].predict(Xtest)

# Write predictions to output csv
pred_df = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                        'Survived': predictions})
pred_df.to_csv('predictions.csv', index=False)

print("Done writing to csv")


# ## 4. Conclusions
# 
# <p>After submission of these predictions, I received a test score of about 80%.</p>
# 
# <p>Principal potential areas of improvement to our analysis would be outlier extraction, feature removal, and/or the addition of more models. There may be certain outliers (for example, during the analysis I noted that there are just 1 or 2 passengers over age 70, who are both 80 and both survived) which could be removed to make the model more generalizable. One could also experiment with removing some of the features, as there may be overlap (title and sex as an example may be essential the same feature). Finally, more advanced models (such as XGBoost) could be implemented or one could consider a voting classifier to include input from all models that were considered.</p>
# 
# <p>Thank you for reading my kernel! As I mentioned in the introduction, I know there is plenty of room for improvement and any feedback or comments are greatly appreciated. Good luck in your own projects!</p>
