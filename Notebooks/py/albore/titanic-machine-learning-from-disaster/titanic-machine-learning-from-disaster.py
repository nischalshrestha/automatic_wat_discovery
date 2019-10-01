#!/usr/bin/env python
# coding: utf-8

# # Titanic - Machine Learning from Disaster
# 
# This is my first Kernel where I will try to predict the survived variable (target variable) of the test dataset provided from the following Kaggle's competition: https://www.kaggle.com/c/titanic
# 
# *Please upvote and share if this helps you. 
# Also, feel free to fork this kernel to play around with the code and test it for yourself. If you plan to use any part of this code, please reference this kernel. 
# I will be glad to answer any questions you may have in the comments. Thank You!*
# 
# This Kernel is divided in the following section:
# 
# 1) Importing Library and Dataset
# 
# 2) Wrangling Missing Values
# 
# 3) Plotting, Visualizing and Analyzing Data
# 
# 4) Model and Prediction

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# ## 1) Importing Library and Dataset
# For starting, I import the libraries that I need for working

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import statistics
import seaborn as sns
import sklearn.metrics as metrics

from numpy import inf
from scipy import stats
from statistics import median
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # For Random Forest Classification
from sklearn.svm import SVC # For SVC algorithm
from sklearn import model_selection # For Power Tuning
from sklearn.model_selection import cross_val_score


# Loading the train dataset and I visualize the content and structure

# In[ ]:


train_dataset = pd.read_csv("../input/train.csv")
train_dataset.head()


# In[ ]:


print ("Shape:", train_dataset.shape)


# Count the number of missing value for each attribute of the dataset

# In[ ]:


train_dataset.isnull().sum()


# ## 2) Wrangling Missing Values
# In this section I will dealing with missing values of Age and Embarked feature.

# In[ ]:


print("The", round(train_dataset.Age.isnull().sum()/891,2)*100, "% of the observations have a missing Age attribute")
print("The", round(train_dataset.Cabin.isnull().sum()/891,2)*100, "% of the observations have a missing Cabin attribute")


# Due to many missing value I decided to drop the Cabin attribute.

# In[ ]:


train_dataset = train_dataset.drop(labels=['Cabin'], axis=1)


# I track the id_row where there isn't the Age of the passenger or the Embarked information

# In[ ]:


# id of row where age is null
PassengerID_toDrop_AgeNull = train_dataset.PassengerId[train_dataset.Age.isnull()].tolist()
PassengerID_toDrop_EmbarkedNull = train_dataset.PassengerId[train_dataset.Embarked.isnull()].tolist()


# Isolating the rows where Age is missed to understand if they are relevant

# In[ ]:


dataset_toDelete = []
for i in range(train_dataset.shape[0]):
    for j in range(pd.DataFrame(PassengerID_toDrop_AgeNull).shape[0]):
        if train_dataset.loc[i, 'PassengerId'] == PassengerID_toDrop_AgeNull[j]:
            dataset_toDelete.append(train_dataset.iloc[i,:])
            
dataset_toDelete = pd.DataFrame(dataset_toDelete)

# Histogram of the client with Age null
get_ipython().magic(u'matplotlib inline')

count = dataset_toDelete['Survived'].value_counts()

fig = plt.figure(figsize=(5,5)) # define plot area
ax = fig.gca() # define axis    

count.plot.bar(ax = ax, color = 'b')


# It isn't possible to delete the clients without Age because they describe the survivor bias, so I will draw a nested barplot to show Age for Class and Sex

# In[ ]:


# Draw a nested barplot to show Age for class and sex
sns.set(style="whitegrid")

g = sns.FacetGrid(train_dataset, row="Pclass", col="Sex")
g.map(plt.hist, "Age", bins=20)


# The Age distribution it is mainly normal between the first and the second class, while in the third class there isn't a perfect normal distribution for the Age attribute of the male and the female.

# For replacing the NaN values of the Age attribute, I count the median's age, partitioning the dataset by Sex, Class and Survived Column.
# 
# I will replace the NaN value of Age with the medians, coherently with each group

# In[ ]:


for sur in train_dataset['Survived'].unique():
    for sex in train_dataset['Sex'].unique():
        for pclass in sorted(train_dataset['Pclass'].unique()):
            median_age = train_dataset[(train_dataset['Survived'] == sur) & (train_dataset['Sex'] == sex) & (train_dataset['Pclass'] == pclass)]['Age'].median()
            if sur == 0:  
                print("Median age for Not Survived", sex, "of the", pclass, "°class =", median_age)
            else:
                print("Median age for Survived", sex, "of the", pclass, "°class =", median_age)
            train_dataset.loc[(train_dataset['Survived'] == sur) & (train_dataset['Sex'] == sex) & (train_dataset['Pclass'] == pclass) & (train_dataset['Age'].isnull()), 'Age'] = median_age 


# In[ ]:


train_dataset.isnull().sum()


# Now there are only two Nan value and they are in Embarked Feature

# In[ ]:


# Code for isolating the client with Embarked Null 

dataset_toDelete = []
for i in range(train_dataset.shape[0]):
    for j in range(pd.DataFrame(PassengerID_toDrop_EmbarkedNull).shape[0]):
        if train_dataset.loc[i, 'PassengerId'] == PassengerID_toDrop_EmbarkedNull[j]:
            dataset_toDelete.append(train_dataset.iloc[i,:])
            
dataset_toDelete = pd.DataFrame(dataset_toDelete)
Embarked_null = dataset_toDelete
Embarked_null


# It is normal that more person have the same Ticket Number, so I can think that the two ladies with Nan Embarked have went up from the same Embarked. For assigning an Embarked Info, I will investigate more in details the dataset.

# In[ ]:


# Bar Plot of Embarked
df = train_dataset

# Grouped boxplot
sns.set(font_scale = 1.50)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(7, 7))
graph = sns.countplot(y="Embarked", data=df, ax = ax, color="b")
#graph.set_xticklabels(graph.get_xticklabels(), rotation='vertical')
graph.set_title('Bar Chart of Embarked')


# It seems that the major person have embarked from Southampton, but what is the major embarkation for First Classes' Women?

# In[ ]:


# Draw a nested barplot to show embarked for class and sex
sns.set(style="whitegrid")

g = sns.catplot(x="Embarked", hue="Pclass", col="Sex", kind="count", data=train_dataset, palette="muted")


# It seems that the girls that get in Titanic and stayed in First Class are get in from S and C

# In[ ]:


FirstClass_Women_S = train_dataset.loc[ (train_dataset.Sex == "female") & (train_dataset.Embarked == "S") & (train_dataset.Pclass == 1), :]
print("% Survived Women in First Class from Southampton:", round((FirstClass_Women_S['Survived'].sum()/FirstClass_Women_S['Survived'].count())*100,2))


# In[ ]:


FirstClass_Women_C = train_dataset.loc[ (train_dataset.Sex == "female") & (train_dataset.Embarked == "C") & (train_dataset.Pclass == 1), :]
print("% Survived Women in First Class from Cherbourg:", round((FirstClass_Women_C['Survived'].sum()/FirstClass_Women_C['Survived'].count())*100,2))


# The two woman with null Embarked place, will be assigned to Cherbourg because they survived and the major women staying in the first class and that survived come from Cherbourg.

# In[ ]:


# Fill na in Embarked with "C"
train_dataset.Embarked = train_dataset.Embarked.fillna('C')


# In[ ]:


train_dataset.isnull().sum()


# Now there aren't missing value in the dataset.

# ## 3) Plotting, Visualizing and Analyzing Data

# In[ ]:


train_dataset.head()


# Following nested histograms of Age by Passengers Class and Survived Boolean

# In[ ]:


g = sns.FacetGrid(train_dataset, row="Survived", col="Pclass")

g.map(plt.hist, "Age", bins = 20)


# The major people that haven't survived came from the third class and they were mainly young with an Age around 20 and 30. 
# In the first class, we can see as there are more young people survived than old ones.

# In[ ]:


g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train_dataset, kind = "bar")
g.set_ylabels("Survival Probability")
g.fig.suptitle("Survival Probability by Sex and Passengers Class")


# From the chart above it is possible to notice as in every Class the Survival Probability is bigger for woman than man. Furthermore, the people in first class (both men and women) have more probability to survive than the other's classes people.

# In[ ]:


# Distribution plot of Fares
g = sns.distplot(train_dataset["Fare"], bins = 20, kde=False)
g.set_title("Distribution plot of Fares")


# In[ ]:


train_dataset.Fare.describe()


# From the chart above, it seems that most people have paid a low rate, while few people have paid a very high price.
# I divide the dataset in two kind of fare: 
# 
# 1) Upper the median = 1;
# 
# 2) Equal or lower the median's Fare = 2;

# In[ ]:


train_dataset.loc[ train_dataset.Fare > train_dataset.Fare.median(), "Fare_Bound" ] = 1 # High Fare type
train_dataset.loc[ train_dataset.Fare <= train_dataset.Fare.median(), "Fare_Bound" ] = 2 # Low Fare type


# In[ ]:


g = sns.catplot(x="Fare_Bound", hue="Pclass", col="Survived", kind="count", data=train_dataset, palette="muted")


# From the chart above, it is possible notice that, from the survived people there is nobody that came from the first class & had paid a fare lower than the median's fare.
# The major people died, stayed in third class & had paid a fare lower than the median.

# In[ ]:


g = sns.catplot(x="Survived", col="Fare_Bound", kind="count", data=train_dataset, palette="muted")


# Among the people that paid a fare more than the median, there isn't a clear trend between survived and no-survived; while among the people that paid a fare equal or lower than the median, there is more died than survived.

# In[ ]:


print("Survived people that had paid more than median Fare:", train_dataset.loc[ (train_dataset.Fare_Bound == 1) & (train_dataset.Survived == 1), "Survived" ].sum())
print("Survived people that had paid equal or less than median Fare:", train_dataset.loc[ (train_dataset.Fare_Bound == 2) & (train_dataset.Survived == 1), "Survived" ].sum())
print("People that had paid more than median Fare:", train_dataset.loc[ (train_dataset.Fare_Bound == 1), "Survived" ].count())
print("People that had paid equal or less than median Fare:", train_dataset.loc[ (train_dataset.Fare_Bound == 2), "Survived" ].count())
print(" % Survived among people that had paid more than median:", round(train_dataset.loc[ (train_dataset.Fare_Bound == 1) & (train_dataset.Survived == 1), "Survived" ].sum()/train_dataset.loc[ (train_dataset.Fare_Bound == 1), "Survived" ].count()*100,2),"%")
print(" % Survived among people that had paid equal or less than median:", round(train_dataset.loc[ (train_dataset.Fare_Bound == 2) & (train_dataset.Survived == 1), "Survived" ].sum()/train_dataset.loc[ (train_dataset.Fare_Bound == 2), "Survived" ].count()*100,2), "%")


# Among the people that had paid equal or less than the median Fare there are more people that died.
# Instead in the cluster "People that had paid more than median" the number of people that didn't survived are similar to the people that survived (51.8%).

# In[ ]:


train_dataset.head()


# I create 4 variables for analyzing the relationship among Siblings/Spouses and Parents/Children with the target variable, Survived.

# In[ ]:


SibSp_Pos = train_dataset.loc[ train_dataset.SibSp > 0, :]
SibSp_Null = train_dataset.loc[ train_dataset.SibSp == 0, :]
Parch_Pos = train_dataset.loc[ train_dataset.Parch > 0, :]
Parch_Null = train_dataset.loc[ train_dataset.Parch == 0, :]


# In[ ]:


print("% Survived with Siblings/Spouses number positive:", round(SibSp_Pos.Survived.sum()/SibSp_Pos.Survived.count()*100), "%")
print("% Survived with Siblings/Spouses number null:", round(SibSp_Null.Survived.sum()/SibSp_Null.Survived.count()*100), "%")
print("% Survived with Parents / Children number positive:", round(Parch_Pos.Survived.sum()/Parch_Pos.Survived.count()*100), "%")
print("% Survived with Parents / Children number null:", round(Parch_Null.Survived.sum()/Parch_Null.Survived.count()*100), "%")


# It seems that the people with Siblings / Spouses or Parents / Children on board had more probability to survive than the people with no family on board.

# In[ ]:


g = sns.catplot(x="SibSp", y="Survived", kind="bar", data=train_dataset)
g.set_ylabels("Survival Probability")


# In[ ]:


g = sns.catplot(x="Parch", y="Survived", kind="bar", data=train_dataset)
g.set_ylabels("Survival Probability")


# ## 4) Model and Prediction

# For this classification problem I am using the RandomForestClassifier.
# 
# 
# Before to use the algorithm I am handling the dataset. I will start to convert Sex column in a dummy variable

# In[ ]:


train_dataset = pd.get_dummies(train_dataset, columns=['Sex'])


# In[ ]:


train_dataset.head()


# I will delete Name and Ticket columns. Next I will convert the column Embarked in a dummy variable.

# In[ ]:


dataset_to_train = train_dataset.drop(labels=['Name', 'Ticket'], axis = 1)
dataset_to_train = pd.get_dummies(dataset_to_train, ['Embarked'])


# I create a vector "label_to_train" where I save the column (target variable) "Survived", then I delete the column from the dataset.

# In[ ]:


label_to_train = dataset_to_train.loc[:, ['Survived']]
dataset_to_train = dataset_to_train.drop(labels=['Survived'], axis = 1)


# In[ ]:


dataset_to_train.head()


# I split the training dataset in two dataset, one for training the algorithm the other one for testing.
# 
# After splitting the dataset, I delete the PassengerID column from the training and testing dataset .

# In[ ]:


data_train, data_test, label_train, label_test = train_test_split(dataset_to_train, label_to_train, test_size = 0.3, random_state=7)
data_train = data_train.drop(labels=['PassengerId'], axis = 1)
data_test = data_test.drop(labels=['PassengerId'], axis = 1)


# I use a Power Tuned Random Forest Classifier where the tuned parameter is 'n_estimators'.
# For making this, I will create a vector "x" for trying different parameters in 'n_estimators'.

# In[ ]:


x = []
for i in range(100):
    x.append(i+1)


# I set the model and the parameters's model to optimize with GridSearchCV

# In[ ]:


model = RandomForestClassifier(oob_score = True)
parameters = {'n_estimators':x}
power_tuning = model_selection.GridSearchCV(model, parameters)
model.fit(data_train, label_train)


# In[ ]:


model_tuned = power_tuning.fit(data_train, label_train.Survived)


# In[ ]:


model_tuned.best_estimator_


# In[ ]:


print("The best parameter for 'n_estimator' is:", model_tuned.best_estimator_.n_estimators)


# In[ ]:


model_Power_tuned = RandomForestClassifier(n_estimators = model_tuned.best_estimator_.n_estimators, oob_score = True)
model_Power_tuned.fit(data_train, label_train.Survived)


# In[ ]:


print("Score of the training dataset obtained using an out-of-bag estimate:" , round(model_Power_tuned.oob_score_,6))


# I make the prediction using the data_test.

# In[ ]:


prediction = model_Power_tuned.predict(data_test)


# I compute performance measures:

# In[ ]:


confusion_mat = metrics.confusion_matrix(label_test, prediction)
columns = ['Not-Survived', 'Survived']
plt.imshow(confusion_mat, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Confusion Matrix')
plt.xticks([0,1], columns, rotation='vertical')
plt.yticks([0,1], columns)
plt.colorbar()
plt.show()


# In[ ]:


print("Accuracy:", round(metrics.accuracy_score(label_test, prediction),6))
print("Recall:", metrics.recall_score(label_test, prediction))
print("Precision:", round(metrics.precision_score(label_test, prediction),6))


# In[ ]:


score_Cross_val = cross_val_score(model_Power_tuned, data_train, label_train.Survived, cv=10).mean()
print ("Cross Validation Score:", round(score_Cross_val,6))


# In[ ]:


print("Accuracy:", round(metrics.accuracy_score(label_test, prediction),6))
print("Recall:", metrics.recall_score(label_test, prediction))
print("Precision:", round(metrics.precision_score(label_test, prediction),6))


# In[ ]:


score_Cross_val = cross_val_score(model, data_train, label_train.Survived, cv=10).mean()
print ("Cross Validation Score:", score_Cross_val)


# ### Now I make the prediction of the Kaggle's test_dataset

# I import the test dataset from the competition.

# In[ ]:


test_dataset = pd.read_csv("../input/test.csv")
test_dataset.head()


# I isolate the passenger_ID for using them after the prediction.

# In[ ]:


Passenger_id_test = test_dataset.loc[:, 'PassengerId']


# I count the Nan in the dataset.

# In[ ]:


test_dataset.isnull().sum()


# I replace the Nan values in Age column with the median age of the cluster defined with Sex and Pclass.

# In[ ]:


test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "female") ]["Age"].median())
test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "female") ]["Age"].median())
test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "female") ]["Age"].median())

test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "male") ]["Age"].median())
test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "male") ]["Age"].median())
test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "male") ]["Age"].median())


# In[ ]:


test_dataset.isnull().sum()


# There is one record with a Nan Fare.

# In[ ]:


test_dataset.loc[test_dataset.Fare.isnull(), :]


# I search the Fare payed from people similar to Mr. Thomas Storey (Pclass = 3, Male & Age > 60)

# In[ ]:


train_dataset.loc[ (train_dataset.Pclass == 3) & (train_dataset.Sex_male == 1) & (train_dataset.Age > 60), :]


# The median Fare of the above table is the following:

# In[ ]:


train_dataset.loc[ (train_dataset.Pclass == 3) & (train_dataset.Sex_male == 1) & (train_dataset.Age > 60), ['Fare']].median()


# I will replace the Nan value of the fare with the median computed before.

# In[ ]:


test_dataset.loc[test_dataset.Fare.isnull(), :] = test_dataset.loc[test_dataset.Fare.isnull(), :].fillna(train_dataset.loc[ (train_dataset.Pclass == 3) & (train_dataset.Sex_male == 1) & (train_dataset.Age > 60), ['Fare']].median())


# In[ ]:


test_dataset.isnull().sum()


# I drop the Cabin column for too Missing values, and then I drop PassengerId, Name and Ticket columns because I suppose that they aren't important for the analysis.

# In[ ]:


test_dataset = test_dataset.drop(labels=['Cabin', 'PassengerId', 'Name', 'Ticket'], axis = 1)


# In[ ]:


test_dataset.head()


# I create the column "Fare_Bound" for disclosing Fare > median and Fare <= Median.
# I will convert Sex and Embarked columns in dummy variables.

# In[ ]:


test_dataset.loc[ test_dataset.Fare > test_dataset.Fare.median(), "Fare_Bound" ] = 1 # High Fare type
test_dataset.loc[ test_dataset.Fare <= test_dataset.Fare.median(), "Fare_Bound" ] = 2 # Low Fare type
test_dataset = pd.get_dummies(test_dataset, columns=['Sex', 'Embarked'])


# In[ ]:


test_dataset.head()


# I make the prediction and I submit.

# In[ ]:


prediction = model.predict(test_dataset)


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': Passenger_id_test, 'Survived': prediction})
my_submission.to_csv('submission.csv', index=False)

