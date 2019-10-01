#!/usr/bin/env python
# coding: utf-8

# <h3><strong>Objective</strong></h3>
# <p>Predict which passengers survived on the Titanic and which didn't </p>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# First and foremost, our data! Let's import it now
trainingData = pd.read_csv("../input/train.csv")
# We will, for the most part, be ignoring the 'testData' until later in the script.
testData = pd.read_csv("../input/test.csv")


# In[ ]:


# I'm interested in the shape and description of the data. 
# What types of data, are they clean, column names, NaN's? These are but few of the 
# many initial questions you should typically ask yourself when first looking at a dataset. 
trainingData.shape


# In[ ]:


trainingData.head()


# In[ ]:


# a quick method to visualize two variables
sns.barplot(x="Embarked", y="Survived",hue="Sex",data=trainingData)


# In[ ]:


sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=trainingData,
             palette={"male":"blue", "female":"pink"},
             markers=["*","o"],linestyles=["-","--"])


# In[ ]:


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins=(-1,0,5,12,18,25,35,60,120)
    group_names=["Unknown",'Baby','Child',"Teenager","Student","Young Adult","Adult","Senior"]
    categories = pd.cut(df.Age,bins,labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1,0,8,15,31,1000)
    group_names = ['Unknown','1_quartile','2_quartile','3_quartile','4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x:x.split(' ')[1])
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

trainingData = transform_features(trainingData)
testData = transform_features(testData)


# In[ ]:


sns.barplot(x="Age",y="Survived",hue='Sex',data=trainingData)


# In[ ]:


sns.barplot(x="Cabin",y="Survived",hue='Sex',data=trainingData)


# In[ ]:


sns.barplot(x="Fare",y="Survived",hue='Sex',data=trainingData)


# In[ ]:


from sklearn import preprocessing

def encode_features(df_train, df_test):
    features = ['Fare','Cabin','Age','Sex','Lname','NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

trainingData, testData = encode_features(trainingData, testData)


# In[ ]:


from sklearn.model_selection import train_test_split

trainingClean = trainingData.drop(['Survived','PassengerId'], axis = 1)
yClean = trainingData['Survived']

num_test = 0.2
trainingClean, X_test, yClean, Y_test = train_test_split(trainingClean,yClean, test_size = num_test, random_state=23)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier

clf = RandomForestClassifier()

# Choose some combination of parameters to try
parameters = {'n_estimators': [4, 6, 9],
              'max_features':['log2','sqrt','auto'],
              'criterion':['entropy','gini'],
              'max_depth':[2,3,5,10],
              'min_samples_split':[2,3,5],
              'min_samples_leaf':[1,5,8]
              }

# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)

# Run the Grid Search
grid_obj = GridSearchCV(clf,parameters,scoring=acc_scorer)
grid_obj = grid_obj.fit(trainingClean,yClean)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best alorithm to the data
clf.fit(trainingClean,yClean)


# In[ ]:


prediction = clf.predict(X_test)
deciacc = accuracy_score(prediction,Y_test)
print(str(deciacc * 100) + "% accuracy")


# In[ ]:


from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold +=1
        trainingData, X_test = trainingClean.values[train_index], trainingClean.values[test_index]
        yClean, Y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(Y_test,predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracY: {1}".format(fold,accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))

    
run_kfold(clf)


# In[ ]:


ids = testData['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis=1))

output = pd.DataFrame({'PassengerId':ids, 'Survived':predictions })

output.to_csv('Titanic-predictions.csv', index = False)


# In[ ]:





# In[ ]:




