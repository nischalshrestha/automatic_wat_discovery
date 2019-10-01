#!/usr/bin/env python
# coding: utf-8

# 
# this is a combination of my learning from kaggle tutorials by Jeff Delaney,  Omar El Gabry
# 
# 1.	**Objective:** Predict who survived the sinking of the Titanic
# 2.	**type of machine learning:** Supervised classification
# 3.	**Suitable models**: Random Forest, KNN, SVM (see cheat sheet)**
# 4.	**Do I have the data or is data scraping needed:** Already have data
# 5.	**Does data need to be split into train and test?:** No, already done.
# 6.	**Load data into Pandas using pd.read_csv("")**

# In[ ]:


#6)Answer: Load data into Pandas using pd.read_csv("")
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

from pandas import Series,DataFrame
    
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")


# **7) Determine which column in your data is the target variable using train.head()**

# In[ ]:


#7.Answer) Determine which column in your data is the target variable using train.head()
train.head()


# **7.answer: target variable: survived**

# **8. How many classes are there in the target variable?**

# In[ ]:


#8)Answer: How many classes are there in the target variable?
SurvivedorNot= train['Survived'].value_counts(dropna=False)
print(SurvivedorNot)


# **9)Is the data imbalanced? (ie, the target variable)**

# In[ ]:


testtotal= print (len(train.index)) 
traintotal=print (len(test.index)) 


# **Fixing missing data:**  Do .info() on all data frames to find missing values.  You can tell if there are missing/null/NaN values if their entries are lower (ie in this case, not 891 for train and not 418 for test)

# In[ ]:


test.info()
train.info()


# In[ ]:


testtotal= print (len(train.index)) 
traintotal=print (len(test.index)) 


# In[ ]:


traintotal=(len(train.index))
print("there are", traintotal,"rows in the train data")

testtotal=(len(test.index))
print("there are {} rows in the test data".format(testtotal))


# In[ ]:


Survialcount= train.Survived[train.Survived > 0].sum()
Survialcountpercentage=(Survialcount/traintotal)*100
print(Survialcountpercentage)


# In[ ]:


survivalcountrounded= np.ceil(Survialcountpercentage)


# In[ ]:



print("{}percent survived".format(survivalcountrounded))


# In[ ]:




print(" ",survivalcountrounded,"survived")


# In[ ]:





# In[ ]:


Embarked_classes_count= train['Embarked'].value_counts(dropna=False)


# In[ ]:


Count_of_each_cabin_classes = train['Cabin'].value_counts(dropna=False)


# In[ ]:


print(Embarked_classes_count)


# In[ ]:


print (Count_of_each_cabin_classes)


# In[ ]:


print(len(train.Cabin))


# In[ ]:


print(len(train.Embarked))


# In[ ]:


train.count()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


embarkedNULL= train["Embarked"].isnull().sum()
cabinNULL=train["Cabin"].isnull().sum()
AgeNULL=train["Age"].isnull().sum()

print(embarkedNULL)
print(cabinNULL)
print(AgeNULL)


# In[ ]:





# In[ ]:


#a better version:

embarkedNULL= train["Embarked"].isnull().sum()
cabinNULL=train["Cabin"].isnull().sum()
AgeNULL=train["Age"].isnull().sum()

print("Embarked has"{} "NaN values, which is" .format(embarkedNULL))
print(cabinNULL)
print(AgeNULL)


# In[ ]:


print (embarkedNULL)/len(train)*100
print (cabinNULL)/len(train)*100
print (AgeNULL)/len(train)*100


# **List of variables missing data from TEST.csv:**
# 1)AGE
# 2)FARE
# 3)CABIN
# 
# **List of variables missing data from TRAIN.csv:**
# 1)AGE
# 2)CABIN
# 3)EMBARKED

# FILLING IN MISSING DATA:
# ------------------------

# In[ ]:


#how to do a standard bar chart in seaborn
#call sns.countplot()
#remember to state columnname first, then data=nameofyourpandasdataframe


# In[ ]:


#EMBARKED
#categorical, so have chosen the most frequent occurance which is s. 
train["Embarked"] = train["Embarked"].fillna("S")


# **Bar chart 1: Feature(embarked) by categories**

# In[ ]:


#how to do a standard bar chart in seaborn
#1)If missing only a few values,(NaN) fillna with the most common(best guess) 
#2)call sns.countplot()
#3)remember to state columnname first, then data=nameofyourpandasdataframe
sns.countplot(x='Embarked', data=train)


# **Bar chart2:Feature(embarked) by categories, separated by target variable(survived)**
# 
# This has two variables, as the target variable (survived) is now introduced.

# In[ ]:


sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0])


# **barchart3:** Is based on a newly created column called embark_perc. 
# it is **the mean/average of those who survived, by embarked**. 
# It shows on average how many surived, depending on when the embarked. 
# This way it doesnt matter if more survived from S simply because more from S.

# In[ ]:


embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'])


# In[ ]:


#Can also use fig, (axis1,axis2,axis3) =plt.subplots allows multiple bar charts.





fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Embarked', data=train, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)

embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)


# **Visualising data to determine which variables need to be grouped** 

# In[ ]:


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train)
sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)


# In[ ]:






sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);


# **Perform Pandas.describe to see the quartiles/min/max, to determine how to simplify.**

# **Create functions to simplify data, fill in blanks, drop unnecessary columns**
# , 

# In[ ]:



#these work because df could be anything. could be "cheese". 
#And if more than one arguement, just goes by what order you putin
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

train = transform_features(train)
test = transform_features(test)
train.head()


# In[ ]:





# In[ ]:





# In[ ]:


len(train)


# In[ ]:



#note that  df.Age = df.Age.fillna(-0.5) fills the age column with -0.5 where its missing values
#important to do this so the rest of the Age simplifying function works. 
#bins and group name creates a variable list.
#pd.cut is used to bin this data. Arguements are the column (in this case df.age)
# categories = pd.cut(df.Age, bins, labels=group_names)

#)First you get rid of the null/NaN values (.fillna(-0.5))
#)Important to get rid of null/NaN so can simplify data and later change from categorical. 
#)create a variable called bins and create a list based on what you seen from df.describe
#and where graphs suggest matter
#)create a variable called group names with the same amount.

#)create a variable called catergories and call the command (pd.cut) 
#)then in pd.cut arguements you state a) the column you are simplifying
#)b) bins and c) labels=group_names 
#) then change name of the column to = this new variable, categories.
#) return df.  done! 


# In[ ]:


#test on lamda
#note(lambda x:x[0]) returns the first letter only. 
#note (lambda x:x[1]) returns the second letter only
#note (lambda x:x[0:2]) returns the first 3 letters. 
def simplify_lol(df):
    df.Sex = df.Sex.fillna('N')
    df.Sex = df.Sex.apply(lambda x: x[0])
    return df

def transform_featuress(df):
    df = simplify_lol(df)
    return df


train = transform_featuress(train)
train.head()


# In[ ]:


#def format_name(df):
    #df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    #df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    #return df
    
    #explaination on how this works. creates a column called lname and delimiter is " " (blanks space)
    #[0] grabs everything before it. 
    #then create a column called NamePrefix and using " " again, but this time [1] so 
    #grabs everything in front of it. 


# In[ ]:


sns.barplot(x="Age", y="Survived", hue="Sex", data=train);


# In[ ]:


sns.barplot(x="Cabin", y="Survived", hue="Sex", data=train);


# In[ ]:


sns.barplot(x="Fare", y="Survived", hue="Sex", data=train);


# In[ ]:


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
train, test = encode_features(train,test)
train.head()


# In[ ]:


print (test.info())


# **For x_all can chose target variable by stating the dataframe followed by ["columnname"] For y_all (features/variables) =dataframe.drop["useless column"]**
# ------------------------------------------------------------------------

# **the test_size and train_size parameters to define the amount of data used in the "train" split and the amount used in the "test" split. If the parameters are floats, they represent the proportion of the dataset in the split; if they are ints, the represent the absolute number of samples in the split.**

# In[ ]:


from sklearn.model_selection import train_test_split

X_all = train.drop(['Survived', 'PassengerId'], axis=1)
y_all = train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)


# In[ ]:


predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


# Notes on Random Forests
# -----------------------
# 
# Random Forests/Ensemble learning: multiple learning models, combined to increase accuracy (also known as bagging. 
# 
# Bias: imagine a bullseye. All of the darts miss, but they miss the same amount all the time. 
# Variance: All around. Different amounts.  
# 
# **Traits of Random Forests:**
# few tuning parameters/easier to use than neural networks 
# No need to standardise data all to the standard deviation of 1 etc. 
# Inbuilt cross validation 
# The more trees, the better
# More features, the less bias 
# Dept of trees too much can be overfitting, need some generalisation otherwise cant classify/predict.
# 
# Randomforestclassifer and randomforestregressor. 
# 
# n estimators = NUMBER OF TREES
# max_features= number of features to consider at each split. auto is all. 
# max_depth= none
# min samples_split= eg 3, then if only two left, leaves node as leaf. 
# min samples_leaf= any leaf must have this many samples 
# min_weight= 10% (.1) needs to be 100 samples in each leaf. 
# max_leaf_nodes=None
# n_jobs=1)
# 

# In[ ]:


from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)


# In[ ]:


ids = test['PassengerId']
predictions = clf.predict(test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions2.csv', index = False)
output.head()


# In[ ]:


#the end? 


# In[ ]:




