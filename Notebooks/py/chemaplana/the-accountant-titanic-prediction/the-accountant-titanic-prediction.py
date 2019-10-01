#!/usr/bin/env python
# coding: utf-8

# So, what if you are not one of the many brilliant data scientist here in Kaggle (even if there is only one that’s not: me!)? 
# What if you are just an data aficionado trying to have some fun with numbers? Maybe an accountant, we do have fun with numbers (hard to believe, huh?). What would you do if you don't know ML?
# Well, this would be the very simple, plain approach to the Titanic challenge for an unskilled, but dedicated accountant.
# 
# Any feedback and correction, always welcome. I'm learning as I go.

# First thing: the goal. We want to be able to predict if passangers included in a list will/did survive or not the Titanic disaster, by training our machine with some data we know about other passangers, including if they survived or not.
# 
# Let's start loading some modules and having a look at the data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Ok, so the data we know and we will use to train our machine is in the 'train.csv' file and the list of passangers we want to predict in the 'test.csv' file (very intuitive). Let's upload the train file and see inside.

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
print (df_train.info())


# In[ ]:


print (df_train.head())


# So, we have 891 records of passagers split by survided or not and some info about them. How can any of the variables have any influence in the survival of the passanger? My thoughts:  passanger class, sex and age. But, they are text format (object) so I create new columns in 0/1 format so the machine can read it (female = 1, and three age groups).
# 
# Some age information is missing, so I assume that if missing I give them the mid category.

# In[ ]:


df_train['sex_female'] = df_train['Sex'].apply(lambda x: 1 if x=='female' else 0)
df_train['age_snr'] = df_train['Age'].apply(lambda x: 1 if x >= 50 else 0)
df_train['age_mid'] = df_train['Age'].apply(lambda x: 1 if (x > 10 and x < 50) else 0)
df_train['age_jnr'] = df_train['Age'].apply(lambda x: 1 if x <= 10 else 0)
df_train['known_age'] = df_train['Age'].apply(lambda x: 0 if pd.isnull(x) else 1)
df_train.loc[df_train['known_age'] == 0, 'age_mid'] = 1


# In[ ]:


print (df_train.head())


# In[ ]:


print (df_train.describe())


# By looking at other kernels (God gave you eyes, so COPY) I found a heatmap chart for seaborn. I select only some columns and see if there is any correlation between pairs of columns.

# In[ ]:


train = df_train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
fig = plt.subplots(figsize=(20,10))
sns.heatmap(train.astype(float).corr(), annot=True, cmap='plasma') # my daugther's favourite color


# What does this mean (to me)? Just looking at the Survived column, there is a negative correlation to Pclass (the higher the class, lower the survival), and positive to fare and sex_female. About age, 0.12 to age_jnr, it sounds low to me.

# Now, to the model. I found some books about ML (main one, Python Machine Learning, by Sebastian Raschka) and one model of the models we can use to predict Yes/No situations is the Logistic Regression. A very quick summary: the model  gives some weight to certain variables so when the same weight is applied to the test data, it returns the probability of being 0 or 1. When probability of being 1 is equal or higher than 50%, then it returns 1. Otherwise, returns 0.
# 
# As we want to know in advance how good the model is, we use scikit-learn modules to split the train data in 70% for learning and 30% fo validating. Once we have a model with the train subsample, we apply it to the validating subsample and can actually compare to the results we know.
# 

# In[ ]:


from sklearn.cross_validation import train_test_split
X = df_train.loc[:, ['PassengerId', 'Pclass', 'sex_female', 'age_jnr', 'known_age']]
y = df_train['Survived']
X_train, X_valid, y_train, y_valid = train_test_split(
X, y, test_size = 0.3, random_state = 0)


# Same book suggests the standarization of data. Not sure what this is, but I will do anyway. I will investigate later.

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)


# And now the model, also from the same book. There are some values (C, random_state) applied in the model, but I will use just the same as in the book.
# Also, with the accuracy score, we can see how good the model is in the validating subsample.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_valid_std)
print ((y_valid != y_pred).sum())
print (accuracy_score(y_valid, y_pred))


# So, we have a model that successfully predicts about 79% of the subsample. This is no guarantee for success, as the model can perform well with the train information and not with the test data. This is called Overfitting.
# 
# Now, to the test data, the list of passangers we don't know the answer (until we submit it). We have to process it the same way we did for the train data.

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
print (df_test.info())
print (df_test.head())


# In[ ]:


df_test['sex_female'] = df_test['Sex'].apply(lambda x: 1 if x=='female' else 0)
df_test['age_snr'] = df_test['Age'].apply(lambda x: 1 if x >= 50 else 0)
df_test['age_mid'] = df_test['Age'].apply(lambda x: 1 if (x > 10 and x < 50) else 0)
df_test['age_jnr'] = df_test['Age'].apply(lambda x: 1 if x <= 10 else 0)
df_test['known_age'] = df_test['Age'].apply(lambda x: 0 if pd.isnull(x) else 1)
df_test.loc[df_test['known_age'] == 0, 'age_mid'] = 1
print (df_test.head())


# Select our variables, standarize the data and do the prediction based on the lr model we defined above.

# In[ ]:


test = df_test.loc[:, ['PassengerId', 'Pclass', 'sex_female', 'age_jnr', 'known_age']]
test_std = sc.transform(test)
yy_test = lr.predict(test_std)


# In[ ]:


df_submission = pd.DataFrame({'PassengerId': test['PassengerId'],
                                                 'Survived': yy_test})
print (df_submission.head())
df_submission.to_csv('accountant_titanic_01.csv', index=False)


# Score: 0.77033, #6161 in the ranking (as of 11.56h CET Dec 24th). Not too bad for an accountant, I think. Can it be improved with limited ML knowledge? That will be in the next journal.
# Thanks all.

# Next step: what if we use a list of classifiers and evaluate them using the valid subset? Then take the best performer and run the prediction in the test database.

# In[ ]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


classif_list = [SVC(kernel='linear', C=0.025), SVC(gamma=2, C=1), KNeighborsClassifier(10),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(), GaussianNB()]


# In[ ]:


def classif_func(data_train, label_train, data_valid, label_valid, classif):
    classif.fit(data_train, label_train)
    y_pred = classif.predict(data_valid)
    return ((label_valid != y_pred).sum()), accuracy_score(label_valid, y_pred)


# In[ ]:


for classif in classif_list:
    print (classif, classif_func(X_train_std, y_train, X_valid_std, y_valid, classif))


# From the list, I take those that returned the lower number of errors (and higher accuracy rate), SVC(gamma) and KNeighborsClassifier, and run a range of parameters for each. I must admit, I know nothing about them, but will investigate.

# In[ ]:


def classif_Kne(data_train, label_train, data_valid, label_valid, k):
    classif = KNeighborsClassifier(k)
    classif.fit(data_train, label_train)
    y_pred = classif.predict(data_valid)
    return ((label_valid != y_pred).sum()), accuracy_score(label_valid, y_pred)


# In[ ]:


for i in range(1,30):
    print (i, classif_Kne(X_train_std, y_train, X_valid_std, y_valid, i))


# In[ ]:


def classif_SVC(data_train, label_train, data_valid, label_valid, k):
    classif = SVC(gamma=2, C=k)
    classif.fit(data_train, label_train)
    y_pred = classif.predict(data_valid)
    return ((label_valid != y_pred).sum()), accuracy_score(label_valid, y_pred)


# In[ ]:


for i in range(1,30):
    print (i, classif_SVC(X_train_std, y_train, X_valid_std, y_valid, i))


# Best results (not supergreat, but over 80%) are 20 for Kne and 3 for SVC. So let's run the submission with these classifiers and parameters.

# In[ ]:


classif = KNeighborsClassifier(20)
classif.fit(X_train_std, y_train)
yy_test = classif.predict(test_std)
df_submission = pd.DataFrame({'PassengerId': test['PassengerId'],
                                                 'Survived': yy_test})
print (df_submission.head())
df_submission.to_csv('accountant_titanic_02.csv', index=False)


# Result 0.76076, worse than Logistic.

# In[ ]:


classif = SVC(gamma=2, C=3)
classif.fit(X_train_std, y_train)
yy_test = classif.predict(test_std)
df_submission = pd.DataFrame({'PassengerId': test['PassengerId'],
                                                 'Survived': yy_test})
print (df_submission.head())
df_submission.to_csv('accountant_titanic_03.csv', index=False)


# Result 0.68421, even worse!
# 
# What happened? My interpretation, this is my first case of Overfitting, the model is good, not even very good, for the validation subset but does not perform well when out of this subset.
# 
# How do you correct Overfitting? Some responses from Quora [here](https://www.quora.com/How-do-you-correct-overfitting)
# 
# * You do not correct Overfitting but prevent it
# * Use regularization methods
# * Find more data (we don't have any more here)
# * Simplify the model (good idea, not sure what I did myself)
# * Gradient descent
# 
# However, there is something I haven't done yet, improve the data. Let's work on this in my next journal entry.
# 

# First, I run another submission to be used as comparison.

# In[ ]:


classif = SVC(probability=True)
classif.fit(X_train_std, y_train)
yy_test = classif.predict(test_std)
df_submission = pd.DataFrame({'PassengerId': test['PassengerId'],
                                                 'Survived': yy_test})
print (df_submission.head())
df_submission.to_csv('accountant_titanic_04.csv', index=False)


# 0.76555

# **Journal 3**

# So far we have used data with basic transformations and just a few classifiers to prepare our submissions. Best score was 77% with the Logistic Regression, the first classifier. None of the other classifiers (KNeighbors, SVC, in two separated forms) actually worked. Surely, because I don't understand how they work.
# 
# Next step will be to add and improve the data to achieve better results. For this, I will use the brilliant tutorial [a-data-science-framework-to-achieve-99-accuracy](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook) by [LD Freeman](https://www.kaggle.com/ldfreeman3). Reading is a step by step tour on ML projects and a must-read (IMHO).
# 
# 

# Let's start with a clean set of data again.

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_tr1 = df_train.copy()
df_te1 = df_test.copy()
dataset = [df_tr1, df_te1]


# The Completing (NaNs using median, mode) and Create (Title from name, Family Size and IsAlone atributes) steps.
# 
# I'm confused about how title can have any influence in the survived yes/no, I guess it is a proxy for Sex? Same case for Embarked. Family Size and IsAlone is probaby an indicator, as maybe bigger families tried to stay together and were more difficult to save. Fare should be related to Pclass, maybe we can analyse correlation later.
# 

# In[ ]:


for data in dataset:
    data['Title'] = data['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]
    data['Title_count'] = data.groupby('Title')['Title'].transform('count')
    data['Title'].loc[data['Title_count'] <= 10] = 'Misc'
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 1
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    


# Create new variables with grups of age (equal-width) and quantiles for Fare.

# In[ ]:


for data in dataset:
    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)
    data['FareBin'] = pd.qcut(data['Fare'], 4)


# And Convert.
# 
# SK-Learn provides a tool (LabelEncoder) to create new columns assigning a value (0, 1, 2...) per each value in the variable. This way, we convert strings into integers, easier to be processed by algorithms.
# 
# There is another tool called OneHotEncoder, to create yes/no (1/0) values. Something to explore later, for the moment, we continue with LabelEnconder.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for data in dataset:
    data['Sex_code'] = encoder.fit_transform(data['Sex'])
    data['Pclass_code'] = encoder.fit_transform(data['Pclass'])
    data['Title_code'] = encoder.fit_transform(data['Title'])
    data['Age_code'] = encoder.fit_transform(data['AgeBin'])
    data['Fare_code'] = encoder.fit_transform(data['FareBin'])
    data['Embarked_code'] = encoder.fit_transform(data['Embarked'])


# And now we can remove the string columns that will not add any calculation value.

# In[ ]:


for data in dataset:
    data.drop(['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 
               'Embarked', 'Title', 'Title_count', 'AgeBin', 'FareBin'], axis=1, inplace=True)


# In[ ]:


for data in dataset:
    print (data.info())


# Now we can define a function to try a few classifiers and see results in the validation subsample.

# In[ ]:


def model_classif(dataset, columnt_list, classif):
    X = dataset.loc[:, column_list]
    y = dataset['Survived']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_valid_std = sc.transform(X_valid)
    classif.fit(X_train_std, y_train)
    y_pred = classif.predict(X_valid_std)
    return (classif.__class__.__name__, (y_valid != y_pred).sum(), 
            accuracy_score(y_valid, y_pred), classif.get_params())


# In[ ]:


classif_list = [SVC(probability=True), 
                LogisticRegression(C=10.0, random_state=0), 
                KNeighborsClassifier(4), 
               GaussianNB()]
column_list = ['Pclass', 'FamilySize', 'Sex_code', 'Title_code',
                       'Age_code', 'Fare_code', 'Embarked_code']
for classif in classif_list:
    print (model_classif(df_tr1, column_list, classif))
    print ('----------------')


# In[ ]:


X = df_tr1.loc[:, column_list]
y = df_tr1['Survived']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)
X = df_te1.loc[:, column_list]
X_std = sc.transform(X)


# In[ ]:


classif = SVC(probability=True)
classif.fit(X_train_std, y_train)
y_pred = classif.predict(X_std)
y_hat = pd.DataFrame({'PassengerId': df_te1['PassengerId'], 'Survived': y_pred})
print (y_hat.info())
print (classif.__class__.__name__, classif.get_params())
y_hat.to_csv('accountant_titanic_05.csv', index=False)


# Score 0.78947, improvement from previous results!!! Jump to position 3,326 as of 12.00 pm CET (December 31st). Thanks to LD Freeman's data processing and also the SVC he used (SVC(Probability=True), not the one I used earlier).

# In[ ]:


classif = KNeighborsClassifier(4)
classif.fit(X_train_std, y_train)
y_pred = classif.predict(X_std)
yy_hat = pd.DataFrame({'PassengerId': df_te1['PassengerId'], 'Survived': y_pred})
print (yy_hat.info())
print (classif.__class__.__name__, classif.get_params())
yy_hat.to_csv('accountant_titanic_06.csv', index=False)


# Score 0.77511, lower than SVC but also better than the previous, simple data Logistic Regression.
