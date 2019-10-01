#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival In-depth Analysis.    [ Accuracy: 0.85 ]

# * You can also view the Notebook on the link below.
# *  Github  Link -> **https://github.com/Chinmayrane16/Titanic-Survival-In-Depth-Analysis/blob/master/Titanic-Survival.ipynb**
# * Do Upvote if you like it : )

#  ### Let's Start With Importing Required Libraries -->

# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Handle table-like data and matrices
import numpy as np
import pandas as pd 

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().magic(u'matplotlib inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
params = { 
    'axes.labelsize': "large",
    'xtick.labelsize': 'x-large',
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.figsize': [25, 7]
}
plt.rcParams.update(params)


# In[ ]:


# Center all plots
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""");


# # Extract Train and Test Data
# * Specify the location to the dataset and import them

# In[ ]:


train = pd.read_csv('../input/train.csv' ) # Train
test = pd.read_csv('../input/test.csv' ) # Test
test_df = test.copy()


# In[ ]:


# Explore Train Data
train.head()


# # Features
# * PassengerId : The id given to each traveller on the boat.
# * Pclass : The Passenger class. It has three possible values: 1,2,3 (first, second and third class).
# * Name : The Name of the passeger.
# * Sex : The Gender of the Passenger.
# * Age : The Age of the Passenger.
# * SibSp : The number of siblings and spouses traveling with the passenger.
# * Parch : number of parents and children traveling with the passenger.
# * Ticket : The ticket number of the Passenger.
# * Fare : The ticket Fare of the passenger
# * Cabin : The cabin number.
# * Embarked : This describe three areas of the Titanic from which the people embark. Three possible values S,C,Q (Southampton,     Cherbourg, Queenstown).
# 
# 
# *Qualitative Features (Categorical) : PassengerId , Pclass , Survived , Sex , Ticket , Cabin , Embarked.*
# 
# *Quantitative Features (Numerical) : SibSp , Parch , Age , Fare.*
# 
# ### Survival is the Target Variable.

# In[ ]:


train.shape


# So, We have 891 rows and 12 columns

# In[ ]:


train.describe()


# **Let's look at all the columns and examine Null Values**

# In[ ]:


train.info()


# #### We see that Age, Cabin and Embarked have Null values.

# In[ ]:


train.dtypes


# ### Let's look at Test data

# In[ ]:


test.head()


# In[ ]:


# Test data doesn't have 'Survived' column and that's what we have to predict


# In[ ]:


# Let's look at the figures and Understand the Survival Ratio
train.Survived.value_counts()


# In[ ]:


# Look at the percentage
train.Survived.value_counts(normalize=True)


# ### So, out of 891 examples only 342 (38%) survived and rest all died.

# In[ ]:


sns.factorplot(x='Survived' , kind='count' , data=train , palette=['r','g'] , size=3 , aspect=.6)


# # Examine Features :

# ## Pclass ->
# Let's examine Survival based on Pclass.

# In[ ]:


pd.crosstab(train.Pclass , train.Survived , margins=True)


# In[ ]:


train[['Pclass' , 'Survived']].groupby('Pclass').mean()


# So, there is *62.96%* Survival chance for **1st Class**. This clearly shows us that **First Class People were given priority first**.

# In[ ]:


# Let's Plot graph to better Visualize


# In[ ]:


sns.factorplot(x='Pclass' , data=train , col='Survived' , kind='count' , size=5 , aspect=.8)


# **We can clearly see that 3rd Class People were alloted the least priority and they died in large numbers.**

# ## Sex ->
# Let's examine Survival based on Gender

# In[ ]:


pd.crosstab(train.Sex , train.Survived , margins=True)


# In[ ]:


# To view percentage, add normalize='index'
pd.crosstab(train.Sex , train.Survived , normalize='index')


# It seems that **Female Survival Probability (74%)** is almost *thrice* that of **Men (18%)** . Or we can say Females were more likely to Survive.

# In[ ]:


# Let's plot some graphs to visualize


# In[ ]:


sns.factorplot(x='Sex' , y='Age' , data=train , hue='Survived' , kind='violin' , palette=['r','g'] , split=True)


# We can see that Males have **Surviving Density** less than Females (Bulged) . And majority of those who survived belonged to the category of Age limit 20-30 . Same is true for the Death Scenario.

# In[ ]:


sns.factorplot(x='Sex' , data=train , hue='Survived' , kind='count' , palette=['r','g'])


# ## Age ->
# Let's examine Survival bassed on Age.

# **Remember! Age has Null values , so we need to fill it before proceeding further.**

# In[ ]:


# Age is a continuous variable.. Therefore , let's Analyze the Age variable by plotting Distribution graphs before and after filling Null values


# In[ ]:


sns.kdeplot(train.Age , shade=True , color='r')


# **Fill the Age with it's Median, and that is because, for a dataset with great Outliers, it is advisable to fill the Null values with median.**

# In[ ]:


print('Median : ' + str(train.Age.median()) + '  Mean : ' + str(train.Age.mean()))


# In[ ]:


# For this case mean and median are both close, so we can fill with any of these, but I'll go with median.


# In[ ]:


print(train.Age.count())
# Train
train['Age'].fillna(train.Age.median() , inplace=True)


# In[ ]:


print(train.Age.count())  # Null values filled


# In[ ]:


# Now plot kde plot.
sns.kdeplot(train['Age'] , shade=True , color='r')


# **We can see that the plot has peak close to 30. So, we can infer that majority of people on Titanic had Age close to 30.**

# In[ ]:


sns.factorplot(x='Sex',y='Age' , col='Pclass', data=train , hue='Survived' , kind = 'box', palette=['r','g'])


# In[ ]:


# Understanding Box Plot :

# The bottom line indicates the min value of Age.
# The upper line indicates the max value.
# The middle line of the box is the median or the 50% percentile.
# The side lines of the box are the 25 and 75 percentiles respectively.


# ## Fare ->
# let's examine Survival on the basis of Fare.

# In[ ]:


sns.factorplot(x='Embarked' , y ='Fare' , kind='bar', data=train , hue='Survived' , palette=['r','g'])


# **We can see that those who paid high were likely to Survive.**

# ## Embarked ->
# Let's examine Survival based on point of embarkation.

# **Remember! Embarked has Null values, The best way to fill it would be by most occured value (Mode) .**

# In[ ]:


print(train.Embarked.count())
# Fill NaN values
train['Embarked'].fillna(train['Embarked'].mode()[0] ,inplace=True)


# In[ ]:


train.Embarked.count() # filled the values with Mode.


# In[ ]:


pd.crosstab([train.Sex,train.Survived] , [train.Pclass,train.Embarked] , margins=True)


# In[ ]:


sns.violinplot(x='Embarked' , y='Pclass' , data=train , hue='Survived' , palette=['r','g'])


# **We can see that those who embarked at C with First Class ticket had a good chance of Survival.
# Whereas for S, it seems that all classes had nearly equal probability of Survival.
# And for Q, third Class seems to have Survived and Died with similar probabilities.**

# ## SibSp ->
# Let's Examine Survival on the basis of number of Siblings.

# In[ ]:


train[['SibSp' , 'Survived']].groupby('SibSp').mean()


# **It seems that there individuals having 1 or 2 siblings/spouses had the highest Probability of Survival, followed by individuals who were Alone.**

# In[ ]:


# Similarly let's examine Parch


# ## Parch ->
# Examine Survival on basis of number of Parents/Children.

# In[ ]:


train[['Parch','Survived']].groupby('Parch').mean()


# **It seems that individuals with  1,2 or 3 family members had a greater Probability of Survival, followed by individuals who were Alone.**

# * **Looking at Parch and SibSp, we can see that individuals having Family Members had a slightly greater chance of Survival.**

# ### We have Explored all the Important features , now let's proceed to Feature Engineering.

# # Feature Engineering :

# * Let's create a New Attribute __'Alone'__ , which would be True, if he/she is travelling Alone.

# In[ ]:


# create New Attribute

# Train
train['Alone']=0
train.loc[(train['SibSp']==0) & (train['Parch']==0) , 'Alone'] = 1

# Similarly for test
test['Alone']=0
test.loc[(test['SibSp']==0) & (test['Parch']==0) , 'Alone'] = 1


# In[ ]:


# Check
train.head()
# test.head()


# * **Cabin contains a lot of Null values, so I'm going to drop it.**
# * **Names, PassengerId and Ticket Number doesn't help in finding Probability of Survival, so I'll be Dropping them too.**
# * **Also, we have created Alone feature and therefore I'll be Dropping SibSp and Parch too..**

# In[ ]:


drop_features = ['PassengerId' , 'Name' , 'SibSp' , 'Parch' , 'Ticket' , 'Cabin']
# Train
train.drop(drop_features , axis=1, inplace = True)
# Test
test.drop(drop_features , axis=1 , inplace = True)


# In[ ]:


train.head()


# In[ ]:


test.info()


# **We have a few Null values in Test (Age , Fare) , let's fill it up.**

# In[ ]:


# Fill NaN values in test
test['Age'].fillna(test['Age'].median() , inplace=True)
test['Fare'].fillna(test['Fare'].median() , inplace=True)

# Check
test.info()


# ## Convert the Categorical Variables into Numeric
# 
# This is Done because Modelling Algos cannot process Categorical String Variables.
# 
# * Sex Attribute has (Male/Female) , which will be mapped to 0/1.
# * Divide Age into 5 categories and Map them with 0/1/2/3/4.
# * Divide Fare into 4 categories and Map them to 0/1/2/3.
# * Embarked Attribute has (S/C/Q) , which will be mapped to 0/1/2.
# * Alone Attribute is already mapped.
# * Pclass Attribute is already mapped.
# 

# In[ ]:


# Create a function to Map all values

def map_all(frame):
    # Map Sex
    frame['Sex'] = frame.Sex.map({'female': 0 ,  'male': 1}).astype(int)
    
    # Map Embarked
    frame['Embarked'] = frame.Embarked.map({'S' : 0 , 'C': 1 , 'Q':2}).astype(int)
    
    # Map Age
    # Age varies from 0.42 to 80, therefore 5 categories map to range of 16.
    frame.loc[frame.Age <= 16 , 'Age'] = 0
    frame.loc[(frame.Age >16) & (frame.Age<=32) , 'Age'] = 1
    frame.loc[(frame.Age >32) & (frame.Age<=48) , 'Age'] = 2
    frame.loc[(frame.Age >48) & (frame.Age<=64) , 'Age'] = 3
    frame.loc[(frame.Age >64) & (frame.Age<=80) , 'Age'] = 4
    
    # Map Fare
    # Fare varies from 0 to 512 with and we will map it depending upon the quartile variation.
    # Look at train.describe() above, you will see 25% -> 7.91 , 50% -> 14.453 , 75% -> 31
    frame.loc[(frame.Fare <= 7.91) , 'Fare'] = 0
    frame.loc[(frame.Fare > 7.91) & (frame.Fare <= 14.454) , 'Fare'] = 1
    frame.loc[(frame.Fare > 14.454) & (frame.Fare <= 31) , 'Fare'] = 2
    frame.loc[(frame.Fare > 31) , 'Fare'] = 3


# In[ ]:


train.head()


# In[ ]:


map_all(train)
train.head()


# In[ ]:


# Similarly for test
map_all(test)
test.head()


# # Now, Let's Apply Models :

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(train.drop('Survived',axis=1),train.Survived,test_size=0.20,random_state=66)


# In[ ]:


models = [LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),
        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB() , LinearDiscriminantAnalysis() , 
        QuadraticDiscriminantAnalysis()]

model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',
             'GradientBoostingClassifier','GaussianNB', 'LinearDiscriminantAnalysis','QuadraticDiscriminantAnalysis']

accuracy = []

for model in range(len(models)):
    clf = models[model]
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    accuracy.append(accuracy_score(pred , y_test))
    
compare = pd.DataFrame({'Algorithm' : model_names , 'Accuracy' : accuracy})
compare


# ## Well, DecisionTree did a great job there, with the highest accuracy [85%] .

# In[ ]:


sns.factorplot(x='Accuracy',y='Algorithm' , data=compare , kind='point' , size=3 , aspect = 3.5)


# ## Let's try to tune Parameters :

# In[ ]:


params_dict={'criterion':['gini','entropy'],'max_depth':[5.21,5.22,5.23,5.24,5.25,5.26,5.27,5.28,5.29,5.3]}
clf_dt=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params_dict,scoring='accuracy', cv=5)
clf_dt.fit(x_train,y_train)
pred=clf_dt.predict(x_test)
print(accuracy_score(pred,y_test))
print(clf_dt.best_params_)


# In[ ]:


#now lets try KNN.
#lets try to tune n_neighbors. the default value is 5. so let us vary from say 1 to 50.
no_of_test=[i+1 for i in range(50)]
#no_of_test
params_dict={'n_neighbors':no_of_test}
clf_knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=params_dict,scoring='accuracy')
clf_knn.fit(x_train,y_train)
pred=clf_knn.predict(x_test)
print(accuracy_score(pred,y_test))
print(clf_knn.best_params_)


# ## Generate CSV file based on DecisionTree Classifier.

# In[ ]:


pred = clf_dt.predict(test)

d = {'PassengerId' : test_df.PassengerId , 'Survived' : pred}
answer = pd.DataFrame(d)
answer.to_csv('Prediction.csv' , index=False)


# 
# 
# # THANK YOU :)

# In[ ]:




