#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset - Prediction of Passenger Survival
# This is my first submission to a competition and first time using the algorithms - Support Vector Machines, K Nearest Neighbour, Random Forest and Decision Tree. Please upvote if you find this useful! Suggestions for improvement welcome!

# # (1). Libraries Necessary
# We will need the important libraries for data analysis like
# 1. Numpy - For matrix manipulations.
# 2. Pandas - For easy to use data structures.
# 3. Matplotlib - For plotting our graphs (2D graphs only).
# 4. Seaborn - A high level interface on Matplotlib - for statistical visualization.
# 5. I also import warnings because they look bad :P (However, they are important for debugging!)

# In[ ]:


#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
#WARNINGS
import warnings
warnings.filterwarnings('ignore') #ALWAYS DO THIS ON COMPLETING THE NOTEBOOK


# # (2). Loading the Dataset
# Pandas library helps us read and store the dataset in an easy to use data structure using the command -> pd.read_csv.
# To help us see the dataset's first few entries, we can use the head() function

# In[ ]:


#LOADING THE DATASET
trainingset=pd.read_csv("../input/train.csv")
testingset=pd.read_csv("../input/test.csv")
#VISUALZING THE DATASET
trainingset.head()


# # (3). Analysis of the Dataset
# In this step, we will analyse the dataset to see the column headings, the completeness of the dataset and to see the data types of the columns.

# In[ ]:


#COLUMN HEADINGS
print(trainingset.columns)
#DATATYPE OF EACH COLUMN
print(trainingset.dtypes)
#DATASET SUMMARY
trainingset.describe(include="all")


# From the above analysis we can see that:
# 1. There are three types of data in the dataset - integer, float and object(string).
# 2. Integer Datatypes - Survived, Pclass, SibSp and Parch.
# 3. Float Datatypes - Age and Fare.
# 4. String Datatypes - Name, Sex, Ticket, Cabin and Embarked. 
# 5. Dataset Summary:
#     * There are 891 passengers.
#     *  Most of the dataset is complete except for the Age, Embarked and Cabin columns.
#     * The age column has 177 missing entries, which need to be filled.
#     * The cabin column has 687 missing entries, which can be filled or dropped.
#     * The embarked column has 2 missing entries, which can be easily filled.

# # (4). Visualizing the Dataset
# We can make some rough prediction by visualizing the dataset. Graphically, patterns are more visible, and hence, I will make some prediction after the visualization.

#    **(1). Class of the Passenger(Pclass)**

# In[ ]:


#BARPLOT OF THE SURVIVAL RATE vs CLASS OF THE PASSENGER
sns.barplot(x='Pclass',y='Survived',color='yellow',data=trainingset)


# We can now confidently predict that:
# * The survival rate of passengers in a higher class is more.

# **(2). Sex of the Passenger(Sex)**

# In[ ]:


#BARPLOT OF SURVIVAL RATE vs SEX OF THE PASSENGER
sns.barplot(x='Sex',y='Survived',color='blue',data=trainingset)


# We can now predict that:
# * If the passenger was a female, she will have a higher chance of survival as compared to a male.
# * Since, there is such a huge contrast between the survival rates, Sex will be an important feature in prediction.    

# **(3). Siblings or Spouse of the Passenger(SibSp)**

# In[ ]:


#BARPLOT OF SURVIVAL RATE vs NUMBER OF SIBLINGS/SPOUSE ON BOARD
sns.barplot(x='SibSp',y='Survived',color='Green',data=trainingset)
#NUMBER OF PEOPLE IN EACH CATEGORY
print(trainingset['SibSp'].value_counts())


# From the graph, we can predict that:
# * The more siblings/spouse you have on board, the less likely you were to survive.
# * The people with 1 or 2 siblings/spouse on board were more likely to survive.
# * However, since the number of people who were alone were more, the number of people who survived the most were singles.

# **(4). Family Members of the Passenger(Parch)**

# In[ ]:


#BARPLOT OF SURVIVAL RATE VS FAMILY MEMBERS ON BOARD
sns.barplot(x='Parch',y='Survived',color='orange',data=trainingset)
#NUMBER OF PEOPLE IN EACH CATEGORY
print(trainingset['Parch'].value_counts())


# We can now predict that:
# * People travelling with less than 4 family members on board are more likely to survive.
# * The percentage of people with 1-3 members are more likely to survive than those travelling alone.
# * However, like SibSp, the number of singles survived more due to their numbers being more.

# **(5). Age of the Passenger(Age)**

# In[ ]:


#SINCE AGE CAN VARY, WE NEED TO PUT THEM INTO BINS.
trainingset['Age'] = trainingset['Age'].fillna(-0.5)
testingset['Age'] = testingset['Age'].fillna(-0.5)
agebins = [-1,2,8,13,19,25,38,55,np.inf]
labels = ['Missing','Babies','Children','Teenagers','Young Adults','Adults','Seniors','Old']
trainingset['AgeBin'] = pd.cut(trainingset['Age'],agebins,labels=labels)
testingset['AgeBin'] = pd.cut(testingset['Age'],agebins,labels=labels)
#BARPLOT OF SURVIVAL RATE vs AGE OF THE PASSENGER
sns.barplot(x='AgeBin',y='Survived',color='red',data=trainingset)
#NUMBER OF PEOPLE PER CATEGORY
print(trainingset['AgeBin'].value_counts())


# From the above graph, we can conclude that:
# * Babies are more likely to survive than the other classes.
# * The people who survived the most were young adults and seniors while children survived the least.

# **(6). Cabin of the Passenger (Cabin)**

# In[ ]:


#MAKE BINS FOR CABIN AND PLACE THEM IN THE BIN
trainingset["CabinBin"] = (trainingset["Cabin"].notnull().astype('int'))
testingset["CabinBin"] = (testingset["Cabin"].notnull().astype('int'))


# Since Cabin has only 204 entries and a lot of the data is missing, I thought I can drop it. However, I felt that the recorded cabins might show a higher class and hence I felt the need to make bins for the recorded and non-recorded cabins.

# # (5). Data Cleaning
# We need to fill up the missing values and also remove the unnecessary columns from the testing dataset.

# In[ ]:


#VISUAL ANALYSIS OF THE TESTING DATASET
testingset.describe(include="all")


# We can see that:
# 1. Most of the columns are complete and won't require filling up.
# 2. The fare column has one missing value.
# 3. The cabin column has 327 missing entries and we can drop this feature.
# 4. To simplify the set, we can drop the ticket featur as well.

# In[ ]:


#DROPPING THE CABIN AND TICKET FEATURES FROM BOTH DATASETS
trainingset = trainingset.drop(['Cabin'],axis=1)
trainingset = trainingset.drop(['Ticket'],axis=1)
testingset = testingset.drop(['Cabin'],axis=1)
testingset = testingset.drop(['Ticket'],axis=1)
testingset.head()


# In[ ]:


#EMBARKED FEATURE
print("Southampton (S):")
s = trainingset[trainingset["Embarked"] == "S"].shape[0]
print(s)

print("Cherbourg (C):")
c = trainingset[trainingset["Embarked"] == "C"].shape[0]
print(c)

print("Queenstown (Q):")
q = trainingset[trainingset["Embarked"] == "Q"].shape[0]
print(q)


# Since most of the passengers boarded the ship at Southhampton, we can fill the missing value with S.

# In[ ]:


trainingset = trainingset.fillna({'Embarked':'S'})


# The Age feature will be a challenge. There is a lot of data missing, so we can't do the same thing as embarked. We will need to predict these values, in correlation with the other values.

# In[ ]:


#COMBINE THE DATASETS
full = [trainingset,testingset]
for i in full:
    i['Title'] = i.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(trainingset['Title'],trainingset['Sex'])


# In[ ]:


#PLACE THE RARE TITLES INTO THE MORE COMMON TITLES FOR SIMPLIFICATION
for i in full:
    i['Title'] = i['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    i['Title'] = i['Title'].replace(['Countess', 'Lady', 'Sir'], 'Honararies')
    i['Title'] = i['Title'].replace('Mlle', 'Miss')
    i['Title'] = i['Title'].replace('Ms', 'Miss')
    i['Title'] = i['Title'].replace('Mme', 'Mrs')

trainingset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


#MAPPING OF EACH GROUP FROM ABOVE INTO NUMERICS FOR EASY MANIPULATION
#map each of the title groups to a numerical value
mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Honararies": 5, "Rare": 6}
for i in full:
    i['Title'] = i['Title'].map(mapping)
    i['Title'] = i['Title'].fillna(0)

trainingset.head()


# In[ ]:


#PREDICTION OF MISSING VALUES IN AGE BASED ON THE TITLE BY BINNING THEM INTO PREVIOUSLY MADE BINS
mr = trainingset[trainingset["Title"] == 1]["AgeBin"].mode() 
miss = trainingset[trainingset["Title"] == 2]["AgeBin"].mode()
mrs = trainingset[trainingset["Title"] == 3]["AgeBin"].mode()
master= trainingset[trainingset["Title"] == 4]["AgeBin"].mode()
honararies = trainingset[trainingset["Title"] == 5]["AgeBin"].mode()
rare= trainingset[trainingset["Title"] == 6]["AgeBin"].mode() 

agemapping = {1: "Adults", 2: "Young Adults", 3: "Seniors", 4: "Babies", 5: "Seniors", 6: "Seniors"}

for x in range(len(trainingset["AgeBin"])):
    if trainingset["AgeBin"][x] == "Missing":
        trainingset["AgeBin"][x] = agemapping[trainingset["Title"][x]]
        
for x in range(len(testingset["AgeBin"])):
    if testingset["AgeBin"][x] == "Missing":
        testingset["AgeBin"][x] = agemapping[testingset["Title"][x]]
testingset.head()


# Now we have to map each feature to a numeric value for easy processing.

# In[ ]:


#MAPPING AGE BINS INTO A NUMERIC VALUE
agemappings = {'Babies': 1, 'Children': 2, 'Teenagers': 3, 'Young Adults': 4, 'Adults': 5, 'Seniors': 6, 'Old': 7}
trainingset['AgeBin'] = trainingset['AgeBin'].map(agemappings)
testingset['AgeBin'] = testingset['AgeBin'].map(agemappings)
trainingset = trainingset.drop(['Age'], axis = 1)
testingset = testingset.drop(['Age'], axis = 1)
testingset.head()


# In[ ]:


#DROP NAMES BECAUSE THEY ARE OF NO USE ANYMORE
trainingset=trainingset.drop(['Name'],axis=1)
testingset=testingset.drop(['Name'],axis=1)
testingset.head()


# In[ ]:


#MAPPING SEX INTO A NUMERIC VALUE
sexmapping = {"male": 0, "female": 1}
trainingset['Sex'] = trainingset['Sex'].map(sexmapping)
testingset['Sex'] = testingset['Sex'].map(sexmapping)
testingset.head()


# In[ ]:


#MAPPING EMBARKED INTO A NUMERIC VALUE
embarkedmapping = {"S": 1, "C": 2, "Q": 3}
trainingset['Embarked'] = trainingset['Embarked'].map(embarkedmapping)
testingset['Embarked'] = testingset['Embarked'].map(embarkedmapping)
testingset.head()


# In[ ]:


#FILLING MISSING FARE VALUES AND MAPPING THEM INTO NUMERIC VALUES
#MISSING VALUE IS BASED ON THE CLASS OF THE PASSENGER
for x in range(len(testingset["Fare"])):
    if pd.isnull(testingset["Fare"][x]):
        pclass = testingset["Pclass"][x]
        testingset["Fare"][x] = round(trainingset[trainingset["Pclass"] == pclass]["Fare"].mean(), 4)
trainingset['FareBin'] = pd.qcut(trainingset['Fare'], 4, labels = [1, 2, 3, 4])
testingset['FareBin'] = pd.qcut(testingset['Fare'], 4, labels = [1, 2, 3, 4])
trainingset = trainingset.drop(['Fare'], axis = 1)
testingset = testingset.drop(['Fare'], axis = 1)
testingset.head()


# # (6). Algorithm Modelling
# We will now use the training set to test the accuracy of the SVM, RF, KNN and DT algorithms.

# In[ ]:


from sklearn.model_selection import train_test_split
p = trainingset.drop(['Survived', 'PassengerId'], axis=1)
targetset = trainingset["Survived"]
x_train, x_val, y_train, y_val = train_test_split(p, targetset, test_size = 0.22, random_state = 0)


# In[ ]:


#USING SUPPORT VECTOR MACHINES
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,fbeta_score,make_scorer
svm = SVC(random_state=59090)
svm.fit(x_train,y_train)
preds=svm.predict(x_val)
accuracysvm = round(accuracy_score(y_val,preds)*100,2)
print(accuracysvm)
#USING DECISION TREE CLASSIFICTION
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_val)
accuracydt = round(accuracy_score(y_pred,y_val)*100,2)
print(accuracydt)
#USING RANDOM FOREST CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
rmfr = RandomForestClassifier()
rmfr.fit(x_train, y_train)
y_pred = rmfr.predict(x_val)
accuracyrf = round(accuracy_score(y_pred, y_val) * 100, 2)
print(accuracyrf)
#USING KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
accuracyknn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(accuracyknn)


# As my first attempt in a competition, I am very happy with my Random Forest Accuracy of 85.28!
# I hope someone can help me increase the accuracy even more!

# # (7). Submission File
# Creating the submission.csv file for upload!

# In[ ]:


ids = testingset['PassengerId']
predrmfr = rmfr.predict(testingset.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predrmfr })
output.to_csv('submission.csv', index=False)

