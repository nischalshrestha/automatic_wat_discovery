#!/usr/bin/env python
# coding: utf-8

# # Titanic: Learning from a disaster 
# <br />
# **Elie Kawerk**
# 
# <br />
# I'm a data science and machine learning newbie. Here's a brief overview of my plan to approach for the problem:
# 
# 1. Load the datasets and take a peek at the data
# 2. Exploratory Data Analysis (EDA)
# 3. Data cleaning and feature engineering through EDA
# 4. Preparing the data to feed it into the machine learning models
# 5. Training a set of machine learning models
# 6. Fine tuning the best models
# 7. Diagnosing the fine-tuned models
# 8. Submission
# 
# If you like this notebook, kindly consider voting for it.
# 
# Let's get started!

# # 1. Load the datasets and take a peek at the data

# In[ ]:


import numpy as np  # linear algebra
import pandas as pd # data wrangling
import matplotlib.pyplot as plt # plotting
import seaborn as sns # statistical plots and aesethics
import re # regular expression

######### Preprocessing #######
from sklearn.preprocessing import (LabelEncoder, Imputer, StandardScaler) # data preparation

##### Machine learning models ##############
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier)
from xgboost import XGBClassifier

##### Model evaluation and hyperparameter tuning ##############
from sklearn.model_selection import (cross_val_score, GridSearchCV, StratifiedKFold,                                   RandomizedSearchCV, train_test_split,                                   learning_curve, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, classification_report, roc_auc_score, roc_curve)


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_train.head(3)


# In[ ]:


df_test.head()


# Let's examine the structure of the two datasets.

# In[ ]:


df_train.info()


# The columns Age, Cabin and Embarked present missing values. 

# In[ ]:


df_test.info()


# For the test set, the columns presenting missing values are: Cabin, Fare and Age.

# # 2. Exploratory Data Analysis (EDA)
# 
# 
# * **Shape of the datasets**

# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# The training set consists of 891 observation and the test set consists of 418 observations. 

# In[ ]:


pd.set_option('display.max_rows', 500)
print(df_train.dtypes)


# * **Summary Statistics**
# 
# First, let's examine the distribution of the numerical features.

# In[ ]:


df_train.drop(['PassengerId', 'Survived','Pclass'], axis=1).describe()


# The features are not on the same scale. Later, if these same features are fed into a machine learning algorithm we should keep in mind standarizing them for optimal performance. Let's examine the distribution of the categorical features.

# In[ ]:


categorical_variables = ['Survived', 'Pclass', 'Sex','Embarked']
for cat_var in categorical_variables:
    print("----------------------------")
    print("Distribiton of %s" %(cat_var))
    print(df_train[cat_var].value_counts())
    print("----------------------------")


# About 62 % of the passengers from the training set passed away in the accident.  This is a hint that the target is unbalanced. This should be kept in mind for diagnosing out machine learning algorithms.
# 
# Surprisingly, for the training set, there were more people in the 1st class than the 2nd class. Also, there were more males on the ship than females. 
# 
# Finally, it appears that far more people embarked from Southampton than the other ports of embarkation. Cherbourg is the least common port for passengers of the Titanic.

# * **Data Visualization**
# 
# Let's begin by visualizing the distribution of the Survived target variable by Class.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot('Pclass', data=df_train, hue="Survived")


# Obviously, 3rd class passengers were most likely to pass away in the accident while 1st class passengers were most likely to survive. Let's examine the survival by the Sex feature.

# In[ ]:


sns.countplot('Sex', data=df_train, hue="Survived")


# Females were most likely to survive while males were most likely to pass away in the incident. it would be interesting to examine the age distribution of the several categories.

# In[ ]:


g = sns.FacetGrid(df_train, row = 'Survived', hue='Sex', size=4, aspect=2)
g = (g.map(sns.kdeplot,'Age', shade='True')).add_legend()


# There isn't any discriminatory information here. Let's now plot the same distributions by Class.

# In[ ]:


g = sns.FacetGrid(df_train, row= "Pclass" , col = 'Survived', hue='Sex', size=4, aspect=1)
g = (g.map(sns.kdeplot,'Age', shade='True')).add_legend()


# It appears that from the people who passed away in the first class, females tended to have a lower age than males. In contrast, among the survivors from the 2nd class, females were most likely to be older than males. Let's now examine the distribution of the survivors by port of embarkation.

# In[ ]:


sns.countplot('Embarked', data=df_train, hue='Survived')


#  Passengers who embarked form Cherbourg were most likely to survive the incident while passengers who embarked from Southampton were most likely to pass away. The  port of embarkation could be related to the economic status of the passengers. Let's go ahead and plot the same distributions by Class.

# In[ ]:


g = sns.FacetGrid(df_train, col= "Pclass", hue ='Survived' , size=4, aspect=1)
g = (g.map(sns.countplot,'Embarked')).add_legend()


# Interestingly, while the port of embarkation does not give any discriminatory information about survival of  first class passengers, it appears that 2nd and 3rd class passengers embarking from Southampton were most likely to pass away in Titanic's sinking.  Let's continue by plotting the fare distribution by survival.

# In[ ]:


g = sns.FacetGrid(df_train, hue='Survived', size=4, aspect=2)
g = (g.map(sns.kdeplot, 'Fare', shade=True)).add_legend()
plt.xlim(-10, 125)
plt.show()


# Apparently, we are corroborating our previous observations. Passengers of  lower economic status were most likely to pass away in the disaster. Let's examine the survival distribution by the number of siblings (SibSp) and the number of Parents/Children (Parch).

# In[ ]:


sns.countplot('SibSp', data=df_train, hue='Survived')
plt.show()


# In[ ]:


sns.countplot('Parch', data=df_train, hue='Survived')
plt.legend(loc='center')
plt.show()


# It appears that passengers who had 1 sibling on-board were most likely to survive. The same holds for passengers who had 1 parent/child on-board. Let's now examine the Parch distribution for each survival category by Sex.

# In[ ]:


g = sns.FacetGrid(data=df_train, col='Survived', hue='Sex', size=4, aspect=1)
g = (g.map(sns.countplot, 'Parch')).add_legend()
plt.show()


# # 3. Data cleaning, feature engineering and more EDA
# 
# <br /> 
# 
# * **Heatmap of missing values**
# 
# First, let's plot a heatmap of the missing values by feature for the training and test sets.

# In[ ]:


sns.heatmap(df_train.isnull(),  yticklabels=False, cbar=False, cmap='viridis')
plt.suptitle('Missing values in the training set')
plt.show()


# In[ ]:


sns.heatmap(df_test.isnull(),  yticklabels=False, cbar=False, cmap='viridis')
plt.suptitle('Missing values in the test set')
plt.show()


# As mentioned in the first section, the training set presents missing values in the columns: Age, Cabin and Embarked. The test set presents missing values in the columns:Age, Fare and Cabin.
# The missing data for Age and Embarked should be handled cleverly by imputation since there are less missing values that filled values. For the Cabin feature, we have to decide whether to drop this feature entirely or to perform a clever imputation because the column corresponding to this feature is mostly populated by missing values. 
# 
# We should keep in mind that such imputations should be done based on the features/targets in the training dataset. Information from the test set should not be revealed because this may lead to data snooping.
# 
# <br />
# 
# * **Data cleaning through EDA**
# 
# 
# Let us think of a way to impute the missing values in the Age column by examining the boxplots of distributions corresponding to different categories/numbers in some categorical/numerical features.

# In[ ]:


for feature in ['Pclass', 'Embarked','Sex', 'Survived', 'SibSp', 'Parch']:
    plt.suptitle('Age distribution by %s' %(feature))
    sns.boxplot(x=feature, y='Age', data=df_train)
    plt.show()


# The distributions of the class feature (Pclass), the number of Parents/Children (Parch) and the number of siblings (SibSp) seem to discriminate a passenger's age the most conveniently.  
# 
# For example, for Parch = 2 (most likely a minor passenger with both of his parents), the median age is 16.5 years which is pretty reasonable. To impute a missing value in the Age column, we will examine the Parch class of the observation and fill the value with the median age of the corresponding Parch category.
# 
#  Let's print the median age by Parch and then we can write a function that does the imputation we as described here-above.

# In[ ]:


medians_by_parch = []

for i in df_train['Parch'].unique().tolist():
    medians_by_parch.append(df_train[df_train['Parch'] == i]['Age'].median())

for i, median_age in enumerate(medians_by_parch):
    print('For a number of Parents/Children of %d, the median age is %f' %(i,median_age))


# In[ ]:


def impute_age(cols, medians_by_parch):
    Parch = cols['Parch']
    Age = cols['Age']
    
    if pd.isnull(Age):
        return medians_by_parch[Parch]
    else:
        return Age
    
df_train['Age'] =  df_train.apply(impute_age, args =(medians_by_parch,) , axis=1)
df_test['Age']  =  df_train.apply(impute_age, args =(medians_by_parch,) , axis=1)


# Now that we're done with the Age column, let's proceed with the Embarked column. 

# In[ ]:


df_train[pd.isnull(df_train['Embarked'])]


# The missing values in the Embarked column correspond to female passengers who were in the first class and who survived the incident.

# In[ ]:


cond = (df_train['Sex']=='female') & (df_train['Survived']==1) & (df_train['Pclass']== 1)
sns.countplot(df_train[cond]['Embarked'])


# The most reasonable choice would be to fill the missing value in Embarked by S.

# In[ ]:


cond = pd.isnull(df_train["Embarked"])
df_train.loc[cond,'Embarked'] = 'S'


# Let's now examine the number of missing values in the Fare column of the test set.

# In[ ]:


sum(pd.isnull(df_test['Fare']))


# There is only one missing value.

# In[ ]:


df_test[pd.isnull(df_test['Fare'])]


# Let's fill this missing value by the median of the Fare value of the 3rd class tickets from the training set.

# In[ ]:


df_test[pd.isnull(df_test['Fare'])] = df_train[df_train['Pclass'] == 3]['Fare'].median()


# The majority of the cells in the cabin column have missing values. Let's check the distribution of this column by dropping the NaNs.

# In[ ]:


Cabin_dist = df_train["Cabin"].dropna().apply(lambda x: x[0])

sns.countplot(Cabin_dist, palette='coolwarm')
plt.show()


# Of the available values, cabins C, E, D and B had the most passengers. However, the cabin feature has a lot of missing values which doesn't make it useful. We will proceed by dropping it from both dataframes.

# In[ ]:


del Cabin_dist 

df_train.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)


# Let's now plot a heatmap of the correlation between the different numerical variables.

# In[ ]:


corr = df_train.drop("PassengerId",axis=1).corr()
print(corr)

plt.figure(figsize=(12,12))
sns.heatmap(corr, annot=True, cbar=True, square=True, fmt='.2f', cmap='coolwarm')
plt.show()


# Pclass correlates negatively with Fare. This is expected since a lower class number corresponds to a higher class and a more expensive fare. The number of siblings and the number of parents children correlates positively. Let's do a pairplot to visualize the different scatter plots.

# In[ ]:


plt.figure(figsize=(12,12))
sns.pairplot(df_train[['Age','SibSp','Parch','Fare']])
plt.show()


# *  **Feature engineering**

# Let us begin by creating a new binary feature indicating if a passenger was alone on the ship.

# In[ ]:


def is_alone(passenger):
    var = passenger['SibSp'] + passenger['Parch']
    # if var = 0 then passenger was alone 
    # Otherwise passenger was with siblings or family or both
    if var == 0:
        return 1
    else:
        return 0
    
df_train['Alone'] = df_train.apply(is_alone, axis=1)
df_test["Alone"] = df_test.apply(is_alone, axis=1)


# In[ ]:


sns.countplot('Alone', data=df_train, hue='Survived' )


# Interestingly, most of the people who were alone passed away after the ship's sinking. This indicates that this variable is most likely to be meaningful.  

# We can also create a new binary feature to see if the passenger is a minor. 

# In[ ]:


def is_minor(age):
    if age < 18.0:
        return 1
    else:
        return 0 

df_train['Minor'] = df_train["Age"].apply(is_minor)
df_test['Minor'] = df_test["Age"].apply(is_minor)


# In[ ]:


sns.countplot('Minor', data=df_train, hue='Survived')


# It appears that minors had an equal chance of dying or surviving the ship's sinking. On the other hand adults were most likely to pass away. We can proceed by checking the titles of the passengers. This can be extracted from the Name column with a regular expression.

# In[ ]:


def get_title(name, title_Regex):
    if type(name) == str:
        return title_Regex.search(name).groups()[0]
    else:
        return 'Mr'

title_Regex = re.compile(r',\s(\w+\s?\w*)\.\s', re.I)
    
df_train["Title"] =  df_train["Name"].apply(get_title, args=(title_Regex,))
# There s a floating number in the test set at index 152, I created a function  (get_title) to surpass this
# and replace it with 'Mr'
df_test["Title"] =  df_test["Name"].apply(get_title, args = (title_Regex,))

plt.figure(figsize=(14,7))
g = sns.countplot('Title', data=df_train)
plt.xticks(rotation=50)


# In[ ]:


print(df_train["Title"].unique())


# It appears that there was a nobility class and high socio-economic classes on the ship. These correspond to the titles: 'Dona', 'Lady', 'the Countess', 'Capt', 'Col', 'Don',  'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'. Let's map these with a dictionary.

# In[ ]:


dict_title = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mlle': 'Miss',
    'Mrs': 'Mrs',
    'Mme': 'Mrs',
    'Dona': 'Nobility',
    'Lady': 'Nobility', 
    'the Countess': 'Nobility',
    'Capt': 'Nobility',
    'Col': 'Nobility',
    'Don': 'Nobility',
    'Dr': 'Nobility',
    'Major': 'Nobility',
    'Rev': 'Nobility', 
    'Sir': 'Nobility',
    'Jonkheer': 'Nobility',    
  }

df_train["Title"] =  df_train["Title"].map(dict_title)

plt.figure(figsize=(14,7))
sns.countplot('Title', data=df_train)
plt.show()


# In[ ]:





# In[ ]:





# To be continued ....
