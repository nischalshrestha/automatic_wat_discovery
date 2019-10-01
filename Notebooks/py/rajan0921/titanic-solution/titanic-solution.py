#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm


# Import training data and test data from the location
df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')


# In[ ]:


df_train.head(5) 
# .head(n) shows  first n rows of the data frame, df_train. 


# Lets find some informations about our dataframe, df_train:

# In[ ]:


df_train.info()


# total entries: 891, but for Age, Cabin and Embarked we see 714, 204, and 889 entries respectively. It means those features column has some empty (NaN) entries. We will solve that later (feature engineering).
# 
# For now, let's first visualise the information that we have right now:

# In[ ]:


# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set()
sns.countplot(x='Survived', data=df_train);


# In[ ]:


sns.countplot(x='Sex', data=df_train);


# In[ ]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);


# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


# count total passangers and groupby sex, total males and total female passangers in data set
df_train.groupby(['Sex']).count()


# In[ ]:


print('Total no of female survived:',df_train[df_train.Sex == 'female'].Survived.sum())
print('Total no of female passangers:',df_train[df_train.Sex == 'female'].Survived.count())
print('Percentage of female survived',df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female']
      .Survived.count())

print('Total no of male survived:',df_train[df_train.Sex == 'male'].Survived.sum())
print('Total no of male passangers:',df_train[df_train.Sex == 'male'].Survived.count())
print('Percentage of male survived:',df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].
      Survived.count())


# In[ ]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the 
#feature 'Pclass'
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);


# In[ ]:


# Use seaborn to plot a histogram of the 'Age' column of df_train. You'll need to drop null values before doing so
df_train_drop = df_train.dropna()
sns.distplot(df_train_drop.Age, kde=False);


# It seems like the passangers of age group 20-50 were most likely to survive.

# In[ ]:


df_train_drop = df_train.dropna()
sns.distplot(df_train_drop.Age, kde=True);


# In[ ]:


# Plot a strip plot & a swarm plot of 'Fare' with 'Survived' on the x-axis
sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.5, jitter=True);


# From the plot, its seems like there is marginal correlation (or no relation) between fares and survival of the passangers.

# In[ ]:


sns.swarmplot(x='Survived', y='Fare', data=df_train);


# In[ ]:


# Use seaborn to plot a scatter plot of 'Age' against 'Fare', colored by 'Survived'
sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.7});


# Feature Engineering: Remove/ fill NaNs and process dataframe in order to clean and make ready to feed the machine learning algorithm.

# In[ ]:


df_train.describe()


# In[ ]:


df_train.median()


# In[ ]:


df_train['Age'] = df_train.Age.fillna(df_train.Age.median())
df_train['Fare'] = df_train.Fare.fillna(df_train.Fare.median())

# Check out info of data
df_train.info()


# As Age and Fare were numerical values, it was easy to fill NaN values(either with mean or median values of respective feature, however placing median value will introduce less error as its the most repeatitive data in that column).
# 
# Most of the machine learning algorithms takes numerical inputs in their traing sets. Now we cast male and female values of 'Sex' to numerical value by applying get_ dummies() to 'Sex' column of dataset.

# In[ ]:


df_train = pd.get_dummies(df_train, columns=['Sex'], drop_first=True)


# In[ ]:


df_train.head()


# The new column 'Sex_male' will be extented to the existing dataframe while removing the original 'Sex' column.
# 
# drop_first=True, drops the Sex_female of the dummy column of the new data frame. drop_first=False, we will have 'Sex_female' and 'Sex_male' columns extended to our original dataframe removing the original 'Sex column'.

# In[ ]:


df_train = pd.get_dummies(df_train, columns=['Embarked'], drop_first=False)


# In[ ]:


df_train.head(5)


# In[ ]:


# Select columns and view head, those columns will be the features for machine learning algorithm
training_vectors = df_train[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp','Embarked_C','Embarked_Q',
                             'Embarked_S']]
training_vectors.head()


# Now these features (Sex_male, Fare, Age, Pclas, Sibsp, Embarked_C, Embarked_Q, Embarked_S) will be our training_vectors (X) and corresponding target vectors (Y) (Survived) will be used to build our machine learning model.

# In[ ]:


target_vectors=df_train['Survived']


# In[ ]:


target_vectors.head()


# As we see, target vectors is a column matrix where
# 
# 0 -> dead
# 
# 1 -> survived
# 
# Now its time to train a machine learning model with (training_vectors, target_vectors). Lets use Support Vector Machine algorithm first for binary classification (survived/ Dead) in this task and see the result of classification/ prediction and  accuracy of our model well.

# In[ ]:


# Support Vector Classifier
classifier_svm= svm.SVC()
classifier_svm.fit(training_vectors,target_vectors)


# Our SVM classifier is ready to make the prediction. But are we ready with our test data???
# 
# In order to test and get the best result, test data must be in the same format of training_vectors (no of features column and their order).

# In[ ]:


# Again edit NaN of Numerial variables of test data as well, fill Nan with their median value, it 
#introduces less error in system
df_test['Age'] = df_test.Age.fillna(df_test.Age.median())
df_test['Fare'] = df_test.Fare.fillna(df_test.Fare.median())
#fill dummies in 'Sex' and 'Embarked' columnts with method get_dummies()
df_test = pd.get_dummies(df_test, columns=['Sex'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Embarked'], drop_first=False)


# In[ ]:


df_test_test= df_train[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp','Embarked_C','Embarked_Q',
                             'Embarked_S']]


# In[ ]:


df_test_test.head()


# For features preparation , we can implement a separate method that takes datasets (df_train or df_test) as input and returns the formatted dataframe that fits for both training and testing.
# 
# We can now use our classifier model (classifier_svm) to predict the survival of any passenger in test data. lets check for the 1st row of the modified test data (df_test_test).

# In[ ]:


# take 1st row of the df_test_test and pass it to our classifier_svm to predict the survival of that 
#passanger
#df_test_test.iloc[0] -> first row extracted
print('Survived:',classifier_svm.predict([df_test_test.iloc[0]]))


# In the same way, we can implement our classifier model(classifier_svm) to predict the survival of any passanger in test dataframe, df_test_test. Also it will be a good idea to loop over the df_test_test data frame to predict all the passangers in that dataframe at once.

# In[ ]:


# predict the survival of 101 th row of df_test_test
print('Survived:',classifier_svm.predict([df_test_test.iloc[100]]))


# In[ ]:


#we can also calculate the accuracy of your classifier model. 
classifier_svm.score(training_vectors,target_vectors)


# It shows that the SVM classifier model that we have developed here has the accuracy of 88.67 percent.

# In[ ]:


# Lets use another ML algorithm to build our classifier model. RandomForest !!
from sklearn.ensemble import RandomForestClassifier

classifier_randomforest = RandomForestClassifier(n_estimators=100)
classifier_randomforest.fit(training_vectors,target_vectors)


# In[ ]:


#we can also calculate the accuracy of our second classifier model,classifier_randomforest 
classifier_randomforest.score(training_vectors,target_vectors)


# Here, Random Forest classifer seems to be more accurate than the previous SVM classifier.

# Thank you for reading my post. I would appreciate any comments regarding my work. I am trying to dive into the data science and this is my first work of its kind.

# In[ ]:




