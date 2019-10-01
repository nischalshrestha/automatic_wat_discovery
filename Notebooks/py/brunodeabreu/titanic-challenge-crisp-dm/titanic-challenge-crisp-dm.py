#!/usr/bin/env python
# coding: utf-8

# # Titanic challenge

# ### Phase #1 Business understanding
# 
# ##### Competition Description
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# ##### Goal
# It is your job to predict if a passenger survived the sinking of the Titanic or not. 
# For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.
# 
# #### Metric
# Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.
# 

# ### Phase #2 Data Understanding
# 

# In[1]:


import numpy as np
import pandas as pd


# for text / string processing
import re


# for plotting
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# for tree binarisation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# to build the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# to evaluate the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('../input/train.csv')
submission = pd.read_csv('../input/test.csv')


# In[214]:


data.head()


# In[215]:


submission.head()


# ##### Variable Types : 
# Identify categorical and numerical variables, also variable that should not be used ( example ID), you can create a list of categotical and numericalvariables. In categorical variables you can check if some variables have mixed values with numerical and non-numerical values. In numerical is important to identify the discrete and continuous variables, also the target and unique variable sucha as ID. Create a summary of analysis, like (“X categorical , that y can be treate as mixed; W numerical, where X discrete, Y continuous , Z ID and 1 binary-targe_t”) to make sure do not lost any information

# In[216]:


data.dtypes


#  5 Categorical :
# - Name            object 
# - Sex             object 
# - Ticket          object 
# - Cabin           object
# - Embarked        object
# 
# 7 Numerical variables:
# - PassengerId      int64
# - Survived         int64
# - Pclass           int64 
# - Age            float64 
# - SibSp            int64 
# - Parch            int64
# - Fare           float64
# 

# PassengerId should not be used since there one for each passager:

# In[217]:



print('Number of passager labels on train : ' , len(data.PassengerId.unique()), 'of ' , len(data) , ' on dataset')
print('Number of passager labels on test : ' , len(submission.PassengerId.unique()))


# Create a list of categorical:

# In[218]:


categorical = [var for var in data.columns if data[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))


# In[219]:


numerical = [var for var in data.columns if data[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))


# Checking the categorical variables to see mixed values, we can see that Ticket and Cabin have numbers and non-numbers values

# In[220]:


data[categorical].head()


# Check the numerical variable to identify the discrete and continuous

# In[221]:


data[numerical].head()


# Discrete: 
#  - SibSp
#  - Parch
#  - Pclass
#  
# Continuous:
#  - Age
#  - Fare
#  
# Target : 
#  - Survided
#  
# Not usefull:
#  - PassengerId
#  

# Check discrete variables:
# 
# Can see that there are few values of discrete variables

# In[222]:


for var in ['SibSp', 'Parch', 'Pclass']:
    print(var, ' :  ', data[var].unique())


# 
# ##### Summary of variable analysis:
# There are 7 Numerical Variables  : where 3 are discrete ,  2 continuous , 1 target and 1 not useful ID
# There are 5 Categorical Variables: where 2 are Mixed 

# ____

# ##### Check Missing Data : 
# At this point to understand the data is important to know if there are lot of missing data, can check the mean of NULL for each feature, can be performed using isnull().mean() function

# In[223]:


data.isnull().mean()


# * 19.86% of Age are missing
# * 77% of Cabin are missing
# * 0.2% of Embarked are missing

# ##### Check Outliners for continuous variables (FARE and AGE) : 
# Based on the list created without target and variable such as ID we can check the outliers for continuous variable using boxplot and also the discribuition in order to see if we have Gaussian or skewed distribuition, for that we can use hist.

# In[224]:


numerical = [var for var in numerical if var not in ['Survived','PassengerId']]
numerical


# In[225]:


plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
data.boxplot(column='Fare')

plt.subplot(1,2,2)
data.boxplot(column='Age')


# Both Continuous variables (Fare and Age) have outliers
# 
# Check the distribuition:
# 
#  - Fare is a skewed and Age is a little Gaussian

# In[226]:


plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
fig = data.Age.hist()
fig.set_ylabel('# Passagers')
fig.set_xlabel('Age')

plt.subplot(1,2,2)
fig = data.Fare.hist()
fig.set_ylabel('# Passagers')
fig.set_xlabel('Fare')


# * Gaussian assumption for Age
# 
# For Gaussian, a suggestion is create two variable, one called Lowerboundary with mean - 3 times standard deviation and another Upperboundary with mean + 3 times standard deviation.

# In[227]:


Lowerboundary = data.Age.mean() - 3 * data.Age.std()
Upperboundary = data.Age.mean() + 3 * data.Age.std()
print('Age outliers < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lowerboundary, upperboundary=Upperboundary))


# 
# 
# * Interquantile range for Fare
# 
# For Skewed, create a IQR that is quantile(0.75) - quantile(0.25), that going to be used to calculate the Lowerfence that is quantile(0.25) - (IQR * 3) and Upperfence quantile(0.75) + (IQR * 3

# In[228]:


IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)
Lowerfence = data.Fare.quantile(0.25) - (IQR * 3)
Upperfence = data.Fare.quantile(0.75) + (IQR * 3)
print('Fare outliers < {lowerfence} or > {upperfence}'.format(lowerfence=Lowerfence,upperfence=Upperfence))


# ##### Summary of variable analysis #2:
#  - There are 7 Numerical Variables : where 3 are discrete , 2 continuous , 1 target and 1 not useful ID 
#  - There are 5 Categorical Variables: where 2 are Mixed
#  - Age outliers are < -13 or > 73    
#      - Maybe use Top-Coding
#  - Fare outliers are < -61 or > 100
#      - Maybe use Discretization
# 
# 

# ##### Check Outliners for discrete variables ('SibSp', 'Parch', 'Pclass') : 
# 
# Outliers in discrete variables : Calculate the % of values(sample data[var].valuecounts() / np.float(len(data)) , can consider outliers those values that are present in less than 1% of the occurrences.

# In[229]:


for var in ['SibSp', 'Parch', 'Pclass']:
    print(var , data[var].value_counts() / np.float(len(data)))
    print()


# ##### Summary of variable analysis #3:
#  - There are 7 Numerical Variables : where 3 are discrete , 2 continuous , 1 target and 1 not useful ID 
#  - There are 5 Categorical Variables: where 2 are Mixed
#  - Age outliers are < -13 or > 73    
#      - Top-Coding
#  - Fare outliers are < -61 or > 100
#      -  Discretization
#  - Pclass do not have outliers
#  - Parch values > 2 are outliers
#      - Top-Coding(2)
#  - SibSp values > 4 are outliers
#      - Top-Coding(4)

# ##### Cardinality of Categofical variables: 
# Check the number of labels, in order to see the cardinality of variables, can do a loop on categorical list and check : len(data[var].unique()), here you going to have an oportunity to see the rare labels and have an idea to group or note that labels.

# In[230]:


for var in categorical:
    print(var, ' :  ', len(data[var].unique()), '  labels')


# ### Phase #3 Data Preparation

# #### 1st Lets work with Mixed values for Cabin and Ticket
# 
# For Cabin lets create two new variables Cabin_numerical and Cab_categorical and delete the Cabin

# In[231]:


data['Cabin_numerical'] = data.Cabin.str.extract('(\d+)')
data['Cabin_numerical'] = data['Cabin_numerical'].astype('float')
data['Cabin_categorical'] = data.Cabin.str[0]

submission['Cabin_numerical'] = submission.Cabin.str.extract('(\d+)')
submission['Cabin_numerical'] = submission['Cabin_numerical'].astype('float')
submission['Cabin_categorical'] = submission.Cabin.str[0]

data[['Cabin', 'Cabin_numerical', 'Cabin_categorical']].tail()


# In[232]:


# Remove Cabin

data.drop(labels='Cabin', inplace=True, axis=1)
submission.drop(labels='Cabin', inplace=True, axis=1)


# For Ticket, extract the last part of ticket as number and first part as categorical

# In[233]:


data['Ticket_numerical'] = data.Ticket.apply(lambda s: s.split()[-1])
data['Ticket_numerical'] = np.where(data.Ticket_numerical.str.isdigit(),data.Ticket_numerical, np.nan )
data['Ticket_numerical'] = data['Ticket_numerical'].astype('float')

data['Ticket_categorical'] = data.Ticket.apply(lambda s: s.split()[0])
data['Ticket_categorical'] = np.where(data.Ticket_categorical.str.isdigit(), np.nan, data.Ticket_categorical )


submission['Ticket_numerical'] = submission.Ticket.apply(lambda s: s.split()[-1])
submission['Ticket_numerical'] = np.where(submission.Ticket_numerical.str.isdigit(),submission.Ticket_numerical, np.nan )
submission['Ticket_numerical'] = submission['Ticket_numerical'].astype('float')

submission['Ticket_categorical'] = submission.Ticket.apply(lambda s: s.split()[0])
submission['Ticket_categorical'] = np.where(submission.Ticket_categorical.str.isdigit(), np.nan, submission.Ticket_categorical)


data[['Ticket', 'Ticket_numerical', 'Ticket_categorical']].head()


# In[234]:


# Remove Ticker

data.drop(labels='Ticket', inplace=True, axis=1)
submission.drop(labels='Ticket', inplace=True, axis=1)


# Exploring the Ticket_categorical
#  * Can be improved removing the non letters.

# In[235]:


data.Ticket_categorical.unique()


# In[236]:


def get_title(passenger):
    # extracts the title from the name variable
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
data['Title'] = data['Name'].apply(get_title)
submission['Title'] = submission['Name'].apply(get_title)

data[['Name', 'Title']].head()

# drop the original variable
data.drop(labels='Name', inplace=True, axis=1)
submission.drop(labels='Name', inplace=True, axis=1)


# Before lead with outliers we can create a new variable Family_size with SibSp + Parch

# In[237]:


data['Family_size'] = data['SibSp'] + data['Parch'] + 1
submission['Family_size'] = submission['SibSp'] + submission['Parch'] + 1

print(data.Family_size.value_counts()/np.float(len(data)))
(data.Family_size.value_counts()/np.float(len(data))).plot.bar()


# #### Check the outliers for numerical new variables : 
# 
# as expected the new numerical variables from cabin and ticket have NAs

# In[238]:


data[['Cabin_numerical', 'Ticket_numerical', 'Family_size']].isnull().mean()


# In[239]:


plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
data.boxplot(column='Cabin_numerical')

plt.subplot(1,2,2)
data.boxplot(column='Ticket_numerical')


# In[240]:


plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
fig = data.Cabin_numerical.hist(bins=20)
fig.set_ylabel('# Passagers')
fig.set_xlabel('Cabin_numerical')

plt.subplot(1,2,2)
fig = data.Ticket_numerical.hist(bins=20)
fig.set_ylabel('# Passagers')
fig.set_xlabel('Ticket_numerical')


# ##### Summary of variable analysis #4:
#  - There are 7 Numerical Variables : where 3 are discrete , 2 continuous , 1 target and 1 not useful ID 
#  - There are 5 Categorical Variables: where 2 are Mixed (DONE)
#      - 4 new variables created :
#          - Cabin_numerical and Cabin_categorical
#          - Ticket_numerical and Ticket_categorical
#  - Ticket_numerical have outliers as we can see above
#  - Age outliers are < -13 or > 73    
#      - Top-Coding
#  - Fare outliers are < -61 or > 100
#      -  Discretization
#  - Pclass do not have outliers
#  - Parch values > 2 are outliers
#      - Top-Coding(2)
#  - SibSp values > 4 are outliers
#      - Top-Coding(4)

# ###### Check the outliers for Ticket_numerical 

# In[241]:


IQR = data.Ticket_numerical.quantile(0.75) - data.Ticket_numerical.quantile(0.25)
Lowerfence = data.Ticket_numerical.quantile(0.25) - (IQR * 3)
Upperfence = data.Ticket_numerical.quantile(0.75) + (IQR * 3)
print('Ticket_numerical outliers < {lowerfence} or > {upperfence}'.format(lowerfence=Lowerfence,upperfence=Upperfence))


# ##### Check new categorical variables: Missing values

# In[242]:


data[['Cabin_categorical', 'Ticket_categorical']].isnull().mean()


# ##### Check cardinality
# 

# In[243]:


for var in ['Cabin_categorical', 'Ticket_categorical']:
    print(var, ' contains ', len(data[var].unique()), ' labels')


# ##### Check the rare labels
# 
# - For Cabin_categorical G and T are in less then 1% are rare
# - For Ticket_categorical several are rare

# In[244]:


# rare / infrequent labels (less than 1% of passengers)
for var in ['Cabin_categorical', 'Ticket_categorical']:
    print(data[var].value_counts() / np.float(len(data)))
    print()


# ##### Summary of variable analysis #5:
#  - There are 7 Numerical Variables : where 3 are discrete , 2 continuous , 1 target and 1 not useful ID 
#  - There are 5 Categorical Variables: where 2 are Mixed (DONE)
#      - 4 new variables created :
#          - Cabin_numerical  : Do not have outliers
#          - Ticket_numerical : have outliers   < -981730.0 or > 1343691.0
#          - Cabin_categorical: G and T are in less then 1% are rare , replace by most frequent 
#          - Ticket_categorical: lot of rare, replace by rare 
#          - Family_size
#  - Age outliers are < -13 or > 73    
#      - Top-Coding
#  - Fare outliers are < -61 or > 100
#      -  Discretization
#  - Pclass do not have outliers
#  - Parch values > 2 are outliers
#      - Top-Coding(2)
#  - SibSp values > 4 are outliers
#      - Top-Coding(4)

# ### Split the dataset:
# 
# 

# In[245]:


# Let's separate into train and test set

X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.3)
X_train.shape, X_test.shape


# ##### Create two list :
# * Categorical
# * Numerical (without ID and Target)

# In[246]:


def find_categorical_and_numerical(df):
    var_cat = [col for col in df.columns if df[col].dtype == 'O']
    var_num = [col for col in df.columns if df[col].dtype != 'O']
    return var_cat,var_num

categorical, numerical = find_categorical_and_numerical(data)       


# In[247]:


print(categorical)
print(numerical)


# In[248]:


numerical = [var for var in numerical if var not in ['PassengerId', 'Survived']]
numerical


# ##### Engineering missing values in numerical variables 

# Check variable with missing data

# In[249]:


for col in numerical:
    if X_train[col].isnull().mean()> 0:
        print(col, X_train[col].isnull().mean())
    


# * Age and Ticket have < 50% of NA : Approach : create a new variable NA + Random sample imputation
# * Cabin have > 50% of NA : Approach : impute NA by value far the distribution

# In[250]:


def impute_na(X_train, df, variable):
    # make temporary df copy
    temp = df.copy()
    
    # extract random from train set to fill the na
    random_sample = X_train[variable].dropna().sample(temp[variable].isnull().sum())
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = temp[temp[variable].isnull()].index
    temp.loc[temp[variable].isnull(), variable] = random_sample
    return temp[variable]


# In[251]:


# Age and ticket
# add variable indicating missingness
for df in [X_train, X_test, submission]:
    for var in ['Age', 'Ticket_numerical']:
        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    
# replace by random sampling
for df in [X_train, X_test, submission]:
    for var in ['Age', 'Ticket_numerical']:
        df[var] = impute_na(X_train, df, var)
    

# Cabin numerical
extreme = X_train.Cabin_numerical.mean() + X_train.Cabin_numerical.std()*3
for df in [X_train, X_test, submission]:
    df.Cabin_numerical.fillna(extreme, inplace=True)


# ##### Engineering Missing Data in categorical variables

# In[252]:


for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# * Embarked NA imputed by most frequent category, because NA is low
# * Cabin_categorical imputed by 'Missing', because NA is high
# * Ticket_categorical imput by 'Missing', because NA is high

# In[253]:


for df in [X_train, X_test, submission]:
    df['Embarked'].fillna(X_train['Embarked'].mode()[0], inplace=True)
    df['Cabin_categorical'].fillna('Missing', inplace=True)
    df['Ticket_categorical'].fillna('Missing', inplace=True)


# In[254]:


submission.isnull().mean()


# * Checking the 3 datasets submission have Fare still with NA value, will replace by median

# In[255]:


submission.Fare.fillna(X_train.Fare.median(), inplace=True)


# #### Outliers in Numerical variables

# As have identified on last summary:
# 
# 
# * Ticket_numerical outliers  are < -981730.0 or > 1343691.0  -> Discretization
# * Age outliers are < -13 or > 73 -> Top-Coding(73)
# * Fare outliers are < -61 or > 100 -> Discretization
# * Parch values > 2 are outliers -> Top-Coding(2)
# * SibSp values > 4 are outliers -> Top-Coding(4)

# In[256]:


def top_coding(df, variable, top):
    return np.where(df[variable] > top, top, df[variable])


for df in [X_train,X_test, submission]:
    df['Age'] = top_coding(df,'Age', 73)
    df['Parch'] = top_coding(df,'Parch', 2)
    df['SibSp'] = top_coding(df,'SibSp', 4)
    df['Family_size'] = top_coding(df, 'Family_size', 7)


# In[257]:


for var in ['Age','Parch','SibSp','Family_size']:
    print(var, 'Max value : ', X_train[var].max() )


# In[258]:


# find quantiles and discretise train set
X_train['Fare'], bins = pd.qcut(x=X_train['Fare'], q=8, retbins=True, precision=3, duplicates='raise')
X_test['Fare'] = pd.cut(x = X_test['Fare'], bins=bins, include_lowest=True)
submission['Fare'] = pd.cut(x = submission['Fare'], bins=bins, include_lowest=True)

t1 = X_train.groupby(['Fare'])['Fare'].count() / np.float(len(X_train))
t2 = X_test.groupby(['Fare'])['Fare'].count() / np.float(len(X_test))
t3 = submission.groupby(['Fare'])['Fare'].count() / np.float(len(submission))

temp = pd.concat([t1,t2,t3], axis=1)
temp.columns = ['train', 'test', 'submission']
temp.plot.bar(figsize=(12,6))


# In[259]:


# find quantiles and discretise train set
X_train['Ticket_numerical'], bins = pd.qcut(x=X_train['Ticket_numerical'], q=8, retbins=True, precision=3, duplicates='raise')
X_test['Ticket_numerical'] = pd.cut(x = X_test['Ticket_numerical'], bins=bins, include_lowest=True)
submission['Ticket_numerical_temp'] = pd.cut(x = submission['Ticket_numerical'], bins=bins, include_lowest=True)


# In[260]:


submission.Ticket_numerical_temp.isnull().sum()


# In[261]:


submission[submission.Ticket_numerical_temp.isnull()][['Ticket_numerical', 'Ticket_numerical_temp']]


# In[262]:


X_train.Ticket_numerical.unique()


# In[263]:


submission.loc[submission.Ticket_numerical_temp.isnull(), 'Ticket_numerical_temp'] = X_train.Ticket_numerical.unique()[0]
submission.Ticket_numerical_temp.isnull().sum()


# In[264]:


submission['Ticket_numerical'] = submission['Ticket_numerical_temp']
submission.drop(labels=['Ticket_numerical_temp'], inplace=True, axis=1)
submission.head()


# #### Engineering rare labels in categorical variables

# Find the rare labels

# In[265]:


for var in categorical:
    print(var, X_train[var].value_counts()/np.float(len(X_train)))
    print()


# As listed on last summary:
# - Cabin contains the rare labels G and T: replace by most frequent 
# - Ticket contains a lot of infrequent labels: replace by rare
# 

# In[266]:


def rare_imputation(variable, which='rare'):    
    # find frequent labels
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]
    
    # create new variables, with Rare labels imputed
    if which=='frequent':
        # find the most frequent category
        mode_label = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], mode_label)
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], mode_label)
        submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], mode_label)
    
    else:
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
        submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')


# In[267]:


rare_imputation('Cabin_categorical', 'frequent')
rare_imputation('Ticket_categorical', 'rare')


# In[268]:


# let's check that it worked
for var in categorical:
    print(var, X_train[var].value_counts()/np.float(len(X_train)))
    print()


# #### Encode categorical variables 

# In[269]:


categorical


# * sex we can use One-Hot Encoding
# * Others we can replace by risk probability 

# In[270]:


for df in [X_train, X_test, submission]:
    df['Sex']  = pd.get_dummies(df.Sex, drop_first=True)


# In[272]:


X_train.Sex.unique()


# In[273]:


def encode_categorical_variables(var, target):
        # make label to risk dictionary
        ordered_labels = X_train.groupby([var])[target].mean().to_dict()
        
        # encode variables
        X_train[var] = X_train[var].map(ordered_labels)
        X_test[var] = X_test[var].map(ordered_labels)
        submission[var] = submission[var].map(ordered_labels)

# enccode labels in categorical vars
for var in categorical:
    encode_categorical_variables(var, 'Survived')
    


# In[275]:


# parse discretised variables to object before encoding
for df in [X_train, X_test, submission]:
    df.Fare = df.Fare.astype('O')
    df.Ticket_numerical = df.Ticket_numerical.astype('O')
    


# In[276]:


# encode labels
for var in ['Fare', 'Ticket_numerical']:
    print(var)
    encode_categorical_variables(var, 'Survived')


# #### Feature scaling

# In[277]:


X_train.describe()


# Separate the variables to traning

# In[279]:


training_vars = [var for var in X_train.columns if var not in ['PassengerId', 'Survived']]
training_vars


# In[280]:


# fit scaler
scaler = MinMaxScaler() # create an instance
scaler.fit(X_train[training_vars]) #  fit  the scaler to the train set and then transform it


# ### Phase #4 Modeling & #5 Evaluation

# ##### Machine Learning algorithm building

# #### xgboost

# In[281]:


xgb_model = xgb.XGBClassifier()

eval_set = [(X_test[training_vars], y_test)]
xgb_model.fit(X_train[training_vars], y_train, eval_metric="auc", eval_set=eval_set, verbose=False)

pred = xgb_model.predict_proba(X_train[training_vars])
print('xgb train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = xgb_model.predict_proba(X_test[training_vars])
print('xgb test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# #### Random Florest

# In[282]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train[training_vars], y_train)

pred = rf_model.predict_proba(X_train[training_vars])
print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = rf_model.predict_proba(X_test[training_vars])
print('RF test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# #### Adaboost

# In[283]:


ada_model = AdaBoostClassifier()
ada_model.fit(X_train[training_vars], y_train)

pred = ada_model.predict_proba(X_train[training_vars])
print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = ada_model.predict_proba(X_test[training_vars])
print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# #### Logistic Regression

# In[4]:


logit_model = LogisticRegression()
logit_model.fit(scaler.transform(X_train[training_vars]), y_train)

pred = logit_model.predict_proba(scaler.transform(X_train[training_vars]))
print('Logit train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = ada_model.predict_proba(scaler.transform(X_test[training_vars]))
print('Logit test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# ### Phase #6 Deployment

# #### Select threshold for maximum accuracy

# In[287]:


pred_ls = []
for model in [xgb_model, rf_model, ada_model, logit_model]:
    pred_ls.append(pd.Series(model.predict_proba(X_test[training_vars])[:,1]))

final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)
print('Ensemble test roc-auc: {}'.format(roc_auc_score(y_test,final_pred)))


# In[288]:


tpr, tpr, thresholds = metrics.roc_curve(y_test, final_pred)
thresholds


# In[289]:


accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(final_pred>thres,1,0)
    accuracy_ls.append(metrics.accuracy_score(y_test, y_pred, normalize=True))
    
accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],
                        axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
accuracy_ls.head()


# In[290]:


pred_ls = []
for model in [xgb_model, rf_model, ada_model, logit_model]:
    pred_ls.append(pd.Series(model.predict_proba(submission[training_vars])[:,1]))

final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)


# In[293]:


final_pred = pd.Series(np.where(final_pred>0.40,1,0))


# In[294]:


temp = pd.concat([submission.PassengerId, final_pred], axis=1)
temp.columns = ['PassengerId', 'Survived']
temp.head()


# In[295]:


temp.to_csv('submission.csv', index=False)

