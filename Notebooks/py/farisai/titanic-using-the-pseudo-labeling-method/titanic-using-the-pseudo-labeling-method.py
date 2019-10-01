#!/usr/bin/env python
# coding: utf-8

# **Import the relevant libraries**
# * numpy for linear algebra and math 
# * pandas for interacting with data and manipulation
# * seaborn for visualization

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# ***Reading the csv files for train and test data***

# In[ ]:


# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pd.read_csv("../input/train.csv")
# Print the first 5 rows of the dataframe.
titanic.head()


# **We look at the details of the values and examine if there is any missing values**

# In[ ]:


#the info method displays the data type and null not null and how many entries there
titanic.info()


# ***As we can see the total number of records is 891 and the followsing attributes showing missing values:***    
#     *     Age            714
#     *     Cabine         204
#     *     Embarked       889
# *Now we have to find out how to handle the missing values, some of the methods are:*
#     *     Eleminating missing values
#     *     Replace missing values with the mean value

# In[ ]:


#This can tell us how many missing values are there in the dataset
titanic.isnull().sum()
#Cabin seems to have the most of the missing values 
#Age has 177, we should know that replacing missing ages by the mean or median will result in
#a less accurate estimations


# **We drop the irrelevant attributes for our task here**
# * Passinger id
# * Passinger Name
# * Ticket
# * Cabin 

# In[ ]:


df = titanic.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis =1)
df.head(5)


# *If we look close into the missing values for the embarked we can see the possibility to estimate the missing values. Because there are only 2 missing values and there are no other missing values for the two records.*

# In[ ]:


df[df['Embarked'].isnull()]


# **Observations:**
# * Same passingers are of class "1"
# * Fare for both is "80"
# *Lets look at the boxplot below to guess where they embarked from*

# In[ ]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=df)


# **We can observe the following:**
# * First class tickets are mostly at embarked "S" and "C" (the blue box)
# * First class median line is crossing at Fare value "80" at Embarked "C" 

# In[ ]:


# we replace the missing values in the embarked to "C"
df["Embarked"] = df["Embarked"].fillna('C')
df.head(5)


# In[ ]:


sns.factorplot(x="Pclass", y="Age", hue="Sex", data=df, size=6, kind="bar", palette="muted")


# In[ ]:


sns.factorplot(x="Embarked", y="Age", hue="Sex", data=df, size=6, kind="bar", palette="muted")


# In[ ]:


sns.boxplot(x="Embarked", y="Age", hue="Pclass", data=df)


# **There are several ways to estimate ages here, but what we can do is considering 
# the median in each Embarked value. This would would mean we have to group Passingers
# by class then by embarked then take the Median age for each class in each embarked
# and fill the missing values with those medians.
# **

# **Embarked S**

# In[ ]:


#Getting Embarked S data into separte frame and work to calculate Median for each class
s = df.loc[df["Embarked"] == "S"]
v = [1, 2, 3]
for i in v:
    ss = s.where(s["Pclass"] == i)
    print ("Median age of class ",i, " = ",ss["Age"].median())


# **Embarked C**

# In[ ]:


#Getting Embarked C data into separte frame and work to calculate Median for each class
c = df.loc[df["Embarked"] == "C"]
v = [1, 2, 3]
for i in v:
    sc = c.where(c["Pclass"] == i)
    print ("Median age of class ",i, " = ",sc["Age"].median())


# **Embarked Q**

# In[ ]:


#Getting Embarked Q data into separte frame and work to calculate Median for each class
q = df.loc[df["Embarked"] == "Q"]
v = [1, 2, 3]
for i in v:
    sq = q.where(q["Pclass"] == i)
    print ("Median age of class ",i, " = ",sq["Age"].median())


# *Now that we know the median age in each class and Embark,
# our estimation can get much better for the missing ages.*
# 
# **Fill missing values for each sub-dataframe then combine all the sub dataframes into one**

# In[ ]:


#Embark S
s1 = df[(df["Pclass"] == 1) & (df['Embarked'] == "S") & (df['Age'].isnull())].fillna(37)
s2 = df[(df["Pclass"] == 2) & (df['Embarked'] == "S") & (df['Age'].isnull())].fillna(30)
s3 = df[(df["Pclass"] == 3) & (df['Embarked'] == "S") & (df['Age'].isnull())].fillna(25)

#Embark C
c1 = df[(df["Pclass"] == 1) & (df['Embarked'] == "C") & (df['Age'].isnull())].fillna(37.5)
c2 = df[(df["Pclass"] == 2) & (df['Embarked'] == "C") & (df['Age'].isnull())].fillna(25)
c3 = df[(df["Pclass"] == 3) & (df['Embarked'] == "C") & (df['Age'].isnull())].fillna(20)

#Embark Q
q1 = df[(df["Pclass"] == 1) & (df['Embarked'] == "Q") & (df['Age'].isnull())].fillna(38.5)
q2 = df[(df["Pclass"] == 2) & (df['Embarked'] == "Q") & (df['Age'].isnull())].fillna(43.5)
q3 = df[(df["Pclass"] == 3) & (df['Embarked'] == "Q") & (df['Age'].isnull())].fillna(21.5)


# In[ ]:


#Concatinating all the sub-frames into one frame with replaced Age values
scq = pd.concat([s1,s2,s3,c1,c2,c3,q1,q2,q3])
len(scq) #177 rows which equals to the missing values in the age column


# In[ ]:


#We drop records of missing vlaues
df = df.dropna(axis = 0, how = 'any')
#we will be left with only records with non-null values


# In[ ]:


#Now we concatinate the scq (replaced values frame) with main frame
data = pd.concat([df, scq])

#Checking the info to make sure we have same number of records as the original one we started with
data.info()


# **Great !!**
# 
# **Now we have complete set of data with no missing values, but if we looked closer into the data types we will see that
# some of the columns have an object data type. Object is a type that is non-numeric and we refer to them as categorical data. In order to pass our data into a Machine Learning model, we have to ensure that our data includes only numeric values.**
# 
# ***So, the next step is to convert our categorical data into numeric values.***

# In[ ]:


from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()

data['Sex'] = number.fit_transform(data['Sex'].astype('str'))
data['Embarked'] = number.fit_transform(data['Embarked'].astype('str'))


# **The label encoder function we used is simply going to replace values like Male/Female into 0/1 or embark S/C/Q into 0/1/2. This way our data is still the same just expressed numerically.**

# In[ ]:


#Check our Dataframe
data.head()


# **So far our training data is in form that we can pass in to a Machine Learning model, but our testing data is still in a raw form. Lets check our testing data here. Note that our testing data will have the same columns except for the labels. We are expected to generate labels from our trained model.**

# In[ ]:


titanictest = pd.read_csv("../input/test.csv")
titanictest.head()


# In[ ]:


titanictest.info()


# **We model our testing data the following the same way we did with the training data. The following steps:**    
# 1. Remove [ PassengerId, Name, Cabin, Ticket]
# 2. Fill missing values in [Age, Fare]
# 3. Convert categorical values into numeric values [Sex, Embarked]

# In[ ]:


#Drop Columns
Ttest = titanictest.drop(['Name', 'Ticket', 'Cabin'], axis =1)
Ttest.head()


# **Fill Missing values in testing data**

# In[ ]:


#Fill Missing values
Ttest[Ttest['Fare'].isnull()]


# In[ ]:


#calculate median fare
m = Ttest["Fare"].median()

#Since its the only one, we can just replace it with the mean Fare
Ttest["Fare"] = Ttest["Fare"].fillna(m)



# **Age column has got a lot of missing values, we use same way to estimate median age based on Passinger class and Embarked. Same like we did before.**

# In[ ]:


#Embarked S
s = Ttest.loc[Ttest["Embarked"] == "S"]
v = [1, 2, 3]
for i in v:
    ss = s.where(s["Pclass"] == i)
    print ("Median age of class ",i, " = ",ss["Age"].median())


# In[ ]:


#Getting Embarked C data into separte frame and work to calculate Median for each class
c = Ttest.loc[Ttest["Embarked"] == "C"]
v = [1, 2, 3]
for i in v:
    sc = c.where(c["Pclass"] == i)
    print ("Median age of class ",i, " = ",sc["Age"].median())


# In[ ]:


#Getting Embarked Q data into separte frame and work to calculate Median for each class
q = Ttest.loc[Ttest["Embarked"] == "Q"]
v = [1, 2, 3]
for i in v:
    sq = q.where(q["Pclass"] == i)
    print ("Median age of class ",i, " = ",sq["Age"].median())


# *Now we know the median age of the passinger classes. We can fill the missing values in the same un-efficient way. Sorry I couldn't make it better though have tried every way I know.'*

# In[ ]:


#Embark S
s1 = Ttest[(Ttest["Pclass"] == 1) & (Ttest['Embarked'] == "S") & (Ttest['Age'].isnull())].fillna(42)
s2 = Ttest[(Ttest["Pclass"] == 2) & (Ttest['Embarked'] == "S") & (Ttest['Age'].isnull())].fillna(26)
s3 = Ttest[(Ttest["Pclass"] == 3) & (Ttest['Embarked'] == "S") & (Ttest['Age'].isnull())].fillna(24)

#Embark C
c1 = Ttest[(Ttest["Pclass"] == 1) & (Ttest['Embarked'] == "C") & (Ttest['Age'].isnull())].fillna(43)
c2 = Ttest[(Ttest["Pclass"] == 2) & (Ttest['Embarked'] == "C") & (Ttest['Age'].isnull())].fillna(27)
c3 = Ttest[(Ttest["Pclass"] == 3) & (Ttest['Embarked'] == "C") & (Ttest['Age'].isnull())].fillna(21)

#Embark Q
q1 = Ttest[(Ttest["Pclass"] == 1) & (Ttest['Embarked'] == "Q") & (Ttest['Age'].isnull())].fillna(37)
q2 = Ttest[(Ttest["Pclass"] == 2) & (Ttest['Embarked'] == "Q") & (Ttest['Age'].isnull())].fillna(61)
q3 = Ttest[(Ttest["Pclass"] == 3) & (Ttest['Embarked'] == "Q") & (Ttest['Age'].isnull())].fillna(24)

scq = pd.concat([s1,s2,s3,c1,c2,c3,q1,q2,q3])

Ttest = Ttest.dropna(axis = 0, how = 'any')

Testdata = pd.concat([Ttest, scq])
TestdataID = pd.DataFrame(Testdata['PassengerId'], columns = ['PassengerId'])

Testdata = Testdata.drop(['PassengerId'], axis =1)
Testdata.info()


# In[ ]:


#Converting our categorical data inot numeric
Testdata['Sex'] = number.fit_transform(Testdata['Sex'].astype('str'))
Testdata['Embarked'] = number.fit_transform(Testdata['Embarked'].astype('str'))


# **Now that our both DataFrames are ready, one more thing left to do. Looking at our training data below we can see that our labels (Survived) are with the frame and it would be better to separate this column from the rest of the frame.**

# In[ ]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
Y_data = data['Survived'].values
X_data = data[list(features)].values
X_test = Testdata[list(features)].values


# **Training Our linear SVM**

# In[ ]:


from sklearn import svm
lin_clf = svm.LinearSVC()
lin_clf.fit(X_data, Y_data) 


# In[ ]:


#we predict the survival of the testdata and save it
pseudoY_test = lin_clf.predict(Testdata)


# **Now we have the following:**
# 
# 1. training data
# 2. labels for training data.
# 3. trained linear svm.
# 4. predicted labels for test data.
# 
# **Implemtation of Pseudo-labelin semi-supervised learning method**
# 
# *What we should do now if to concatinate our training and testing data, training labels and predicted labels.*
# *Then retrain the model in the new data and analyze the performance.*

# In[ ]:



X = np.vstack((X_data, X_test))
Y = np.concatenate((Y_data, pseudoY_test), axis=0)

pseudo_model = svm.LinearSVC()
pseudo_model.fit(X, Y)


# **Now we can compare the performance of the main model *lin_clf *and the *pseudo_model* as follows:**

# In[ ]:


Accuracyclf = lin_clf.score(X_data, Y_data)
print ("Accuracy of the lin_clf model: ", Accuracyclf*100, "%")

Accuracypseudo = lin_clf.score(X, Y)
print ("Accuracy of the lin_clf model: ", Accuracypseudo*100, "%")


# **The experiment above shows that our model trained with the Pseudo-labelin semi-supervised learning method perfmored better than our linear model.  If we tried to test the performance of our method across several classifiers as below :**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier

classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression(),
    LinearSVC()]

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X, Y)
    accurracy = clf.score(X, Y)
    print(accurracy, name)


# **The top 2 are:**
# 
# *** 97.32% DecisionTreeClassifier**
# 
# *** 95.95% RandomForestClassifier**

# **Note: Kaggle grader gives very weak performance for Decision Tree and Random Forest. However, I have tried with AdaBoost classifier and it performs better. The code below will take you through training and testing the AdaBoost classifier with 500 estimators.**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=500)
scores = cross_val_score(clf, X, Y)
scores.mean()
clf.fit(X, Y)

Accuracy = clf.score(X, Y)
print ("Accuracy in the training data: ", Accuracy*100, "%")

prediction = clf.predict(X_test)
dfPrediction = pd.DataFrame(data=prediction,columns=['Survived'])
dfsubmit = pd.concat([TestdataID['PassengerId'], dfPrediction['Survived']], axis = 1, join_axes=[TestdataID['PassengerId'].index])
dfsubmit = dfsubmit.reset_index(drop=True)
TestPredict = dfsubmit.to_csv('TestPredictADABOOST.csv')


# 
