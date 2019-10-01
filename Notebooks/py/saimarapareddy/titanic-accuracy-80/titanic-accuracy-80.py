#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# List of libraries used 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.preprocessing import LabelEncoder   # Used for label encoding the data


# In[ ]:


#Import the data
data_train = '../input/train.csv'
data_train_df = pd.read_csv(data_train)


data_test = '../input/test.csv'
data_test_df = pd.read_csv(data_test)


# In[ ]:


# Find missing data
data_train_df.info()


# In[ ]:


#Using data_train_df.info() we can observe that Age, Cabin, Embarked have missing values
#lets analyze Age 
#reference - http://seaborn.pydata.org/generated/seaborn.FacetGrid.html

sns.set(style="ticks", color_codes=True)
g = sns.FacetGrid(data_train_df, col="Sex", row="Survived")
g = g.map(plt.hist, "Age")


# In[ ]:


#Lets Analyze Embarked missing values
#data_train_df[data_train_df['Embarked'].isnull()]
data_train_df[data_train_df['Fare'] > 70.0]
#&& data_train_df['Cabin']=='B28']


# In[ ]:


# Import the data
data_train = '../input/train.csv'
data_train_df = pd.read_csv(data_train)

data_test = '../input/test.csv'
data_test_df = pd.read_csv(data_test)


# In[ ]:


# Find missing data 
data_train_df.info()
# Print Missing data
print("Age, Cabin, Embarked have missing values")


# In[ ]:


# Analyze missing values in Embarked column

#  Lets check which rows have null Embarked column
#data_train_df[data_train_df['Embarked'].isnull()]

#data_train_df[data_train_df['Name'].str.contains('Martha')]
#data_train_df[(data_train_df['Fare'] > 50) & (data_train_df['Age'] > 37) & (data_train_df['Survived']==1 ) & 
#              (data_train_df['Pclass']== 1 ) &  (data_train_df['Cabin'].str.contains('B')) ]
#data_train_df[data_train_df['Ticket'] == 113572]
data_train_df[data_train_df['Ticket']==111361]


# In[ ]:


#Segregate and trim the data

# Apply Label encoding
data_train_df['Embarked'] = data_train_df['Embarked'].astype(str)
data_train_df['Cabin'] = data_train_df['Cabin'].astype(str)

data_test_df['Embarked'] = data_test_df['Embarked'].astype(str)
data_test_df['Cabin'] = data_test_df['Cabin'].astype(str)

le = LabelEncoder()
data_train_df = data_train_df.apply(LabelEncoder().fit_transform)
#display(data_train_df)

data_test_df = data_test_df.apply(LabelEncoder().fit_transform)
#display(data_test_df)

data_train_df_survived = data_train_df['Survived']

#returns a numpy array
data_train_df_trim = data_train_df.drop(['Survived','Name','PassengerId'], axis=1).values
data_test_df_trim = data_test_df.drop(['Name','PassengerId'], axis=1).values


# In[ ]:


#Normalize data 
min_max_scaler = preprocessing.MinMaxScaler()   # Used for normalized of the data
data_train_df_trim_scaled = min_max_scaler.fit_transform(data_train_df_trim)
data_train_df_trim_scaled = pd.DataFrame(data_train_df_trim_scaled)

data_test_df_trim_scaled = min_max_scaler.fit_transform(data_test_df_trim)
data_test_df_trim_scaled = pd.DataFrame(data_test_df_trim_scaled)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.model_selection import ShuffleSplit

def implement_randomForestClassifier(X_train,y_train,X_test,number_of_estimators=10,max_depth=None, 
                  minimum_samples_split=2,minimum_samples_leaf=1,random_number=42):
    """
    This function fits and transforms data using 
    Random Forest Classifier technique and 
    returns the mean of y_pred value
    """
    clf = RandomForestClassifier(n_estimators=number_of_estimators,min_samples_split=minimum_samples_split,
                                  min_samples_leaf=minimum_samples_leaf,random_state=random_number)
    kf = KFold(n_splits=3, random_state=2)
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    predictions = cross_val_predict(clf, X_train,y_train,cv=kf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    scores = cross_val_score(clf, X_train, y_train,scoring='f1', cv=kf)
    print(scores.mean())
    
    '''
    Plot the features wrt their importance
    '''
    
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
    '''
    Return the mean of predicted scores
    '''
    
    return y_pred


# In[ ]:


#Finding optimum estimator in case of RFC
#reference - https://matthew-nm.github.io/pages/projects/gender04_content.html
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def calculate_optimum_estimator_rfc(X_train,y_train,X_test,y_test,interval=5):
    error_rate = []
    nvals = range(1,800,interval) # test a range of total trees to aggregate

    for i in nvals:
        rfc = RandomForestClassifier(n_estimators=i)
        rfc.fit(X_train,y_train)
        y_pred_i = rfc.predict(X_test)
        error_rate.append(np.mean(y_pred_i != y_test))



    plt.plot(nvals, error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. Number of Predictors')
    plt.xlabel('Number of Predictors')
    plt.ylabel('Error Rate')

    # Determine location of best performance
    nloc = error_rate.index(min(error_rate))
    print('Lowest error of %s occurs at n=%s.' % (error_rate[nloc], nvals[nloc]))
    return nvals[nloc]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data_train_df_trim_scaled,data_train_df_survived,test_size=0.2, random_state=42)
optimum_value = calculate_optimum_estimator_rfc(X_train,y_train,X_test,y_test,5)


# In[ ]:


y_pred = implement_randomForestClassifier(data_train_df_trim_scaled,data_train_df_survived,
                       data_test_df_trim_scaled,101)


# In[ ]:


data_test_v1 = pd.read_csv(data_test)

submission = pd.DataFrame({
        "PassengerId": data_test_v1["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv("titanic_submission.csv", index=False)


# In[ ]:




