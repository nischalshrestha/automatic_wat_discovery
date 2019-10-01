#!/usr/bin/env python
# coding: utf-8

# # Using Decision Trees for Classification (Titanic Data Set)

# The sinking of the Titanic remains one of the most tragic incidences in ship accidents. 1502 of the total 2224 occupants passed on during the shipwreck. Following the catastrophe, better safety regulations were developed.
# 
# The main task on this dataset is to create a surviver classifier based on a given features such as gender, age and class.

# # Goal
# As such, it is the aim of this notebook to predict if a passenger survived the sinking of the Titanic or not. 
# For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.

# In[ ]:


#importing relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[ ]:


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


#uploading the Titanic dataset
df1 = pd.read_csv('../input/gender_submission.csv')
df2 = pd.read_csv('../input/test.csv')
df3 = pd.read_csv('../input/train.csv')


# # Previewing our Data

# There data has already been split into train and test. Training should be used to develop a model off features notable in the sample.
# 
# The test data will then be used to assess how well the model performs on unseen data. Since these are merely predictions, a score would help gauge the outcome. the gender_submission provides an idea into how the submission file should look like.

# In[ ]:


#previewing our data i.e the gender_submission data
df1.head()


# In[ ]:


#previewing our test data i.e test data
df2.head()


# In[ ]:


#previewing our train data
df3.head()


# # Feature Engineering 

# # Target is stored in 'y'. 

# In[ ]:


#copy of train data
clean_data = df3.copy()
#copy of test data
clean_data2 = df2.copy()
#y_train
y=clean_data[['Survived']].copy()


# In[ ]:


#Dropping unnecessary columns
df2 = df2.drop(['PassengerId','Name','Ticket','Cabin','Embarked','Fare'],axis=1)


# In[ ]:


#Replacing null values with mean
df2 = df2.fillna(df2.mean())


# In[ ]:


#are there any null values?
df3.isnull().any()


# In[ ]:


#Replacing null values with mean
df3 = df3.fillna(df3.mean())


# In[ ]:


df3.columns


# In[ ]:


#Dropping unnecessary columns
df3 = df3.drop(['PassengerId','Name','Ticket','Cabin','Embarked','Fare','Survived'],axis=1)


# # Binary Classification for Sex

# In[ ]:


#creating a copy of gender in both train and test datasets before encoding
gender=df3[['Sex']].copy()

gender2=df2[['Sex']].copy()


# In[ ]:


values = array(gender)
#Integer encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)


# In[ ]:


values2 = array(gender2)
#Integer encoding(df2)
label_encoder2 = LabelEncoder()
integer_encoded2 = label_encoder2.fit_transform(values2)


# In[ ]:


#Binary encoding
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# In[ ]:


#Binary encoding(df2)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded2 = integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded2 = onehot_encoder.fit_transform(integer_encoded2)


# In[ ]:


#Inverting back NOT RUN YET
#inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0:])])


# In[ ]:


#Replacing the coded column in the dataframe for train data
sex = pd.DataFrame(integer_encoded)
df3['Sex'] = sex
#Replacing the coded column in the dataframe for test data
sex = pd.DataFrame(integer_encoded2)
df2['Sex'] = sex


# In[ ]:


#Assessing descriptives on Sex variable in train dataset
df3['Sex'].describe()


# In[ ]:


#basic information about our train data
df3.info()


# In[ ]:


#type of each column values
df3.dtypes


# In[ ]:


df3['Age'].head()


# In[ ]:


df3.index


# In[ ]:


df3.describe()


# In[ ]:


#number of times the unique Survived appear
clean_data['Survived'].value_counts()


# In[ ]:


df3['Sex'].value_counts()


# In[ ]:


df2['Sex'].value_counts()


# # Data Visualization
# 
# Exploring the Variables of interest to get a feel of how they look

# In[ ]:


#Checking the variables in train data
df3.columns


# Applying different styles

# In[ ]:


plt.style.use("classic")
#plt.style.use("fivethirtyeight")
plt.style.use("ggplot")
#plt.style.use("seaborn-whitegrid")
#plt.style.use("seaborn-pastel")
#plt.style.use(["dark_background", "fivethirtyeight"])


# In[ ]:


#histogram for Age

count, bin_edges = np.histogram(df3,15)

df3['Age'].plot(kind = 'hist',
               figsize = (10,6),
               bins = 15,
               xticks = bin_edges
               )

plt.title(' Histogram of Age in Titanic (train) dataset')
plt.xlabel('Age')
plt.show()


# In[ ]:


df3['Sex'].value_counts()


# In[ ]:


import seaborn as sns
plt.figure(figsize=(10,6))
sns.set(style="darkgrid")
ax = sns.countplot(x="Sex", data=clean_data,saturation=.80)


# In[ ]:


import seaborn as sns
plt.figure(figsize=(10,6))
sns.set(style="darkgrid")
#tit = sns.load_dataset("clean_data")
ax = sns.countplot(x="Pclass",hue="Sex", data=clean_data)


# In[ ]:


import seaborn as sns
plt.figure(figsize=(10,6))
sns.set(style="darkgrid")
ax = sns.countplot(x='SibSp', data=clean_data, saturation=.80)


# In[ ]:


import seaborn as sns
plt.figure(figsize=(10,6))

sns.set(style="darkgrid")
ax = sns.countplot(x='Parch', data=clean_data,saturation=.80)


# # Assessing the influence of different variables to the rate of survival

# In[ ]:


clean_data['Survived'].corr(clean_data['Fare'])


# In[ ]:


clean_data['Survived'].corr(df3['Age'])


# In[ ]:


clean_data['Survived'].corr(df3['Sex'])


# In[ ]:


clean_data['Survived'].corr(df3['Pclass'])


# In[ ]:


clean_data['Survived'].corr(df3['SibSp'])


# In[ ]:


clean_data['Survived'].corr(df3['Parch'])


# In[ ]:


#scatterplot of Age and survival in seaborn
plt.figure(figsize=(10,6))

sns.set(font_scale=1.5)
x = df3['Age']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True, color = 'green')


# In[ ]:


#scatterplot of Sex and survival in seaborn
plt.figure(figsize=(10,6))

x = df3['Sex']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True)


# In[ ]:


#scatterplot of Pclass and survival in seaborn
plt.figure(figsize=(10,6))

x = df3['Pclass']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True)


# In[ ]:


#scatterplot of Parch and survival in seaborn
plt.figure(figsize=(10,6))
x = df3['Parch']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True)


# In[ ]:


#scatterplot of SibSp and survival in seaborn
plt.figure(figsize=(10,6))
x = df3['SibSp']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True)


# # Violin Plots on Possible Feautures

# In[ ]:


plt.figure(figsize=(10,6))
sns.violinplot(y = 'Survived', x = 'Parch', data = clean_data, inner = 'quartile')


# In[ ]:


plt.figure(figsize=(10,6))
sns.violinplot(y = 'Survived', x = 'SibSp', data = clean_data, inner = 'quartile')


# In[ ]:


plt.figure(figsize=(10,6))
sns.violinplot(y = 'Survived', x = 'Sex', data = clean_data, inner = 'quartile')


# In[ ]:


plt.figure(figsize=(10,6))
sns.violinplot(y = 'Survived', x = 'Pclass', data = clean_data, inner = 'quartile')


# # Using Features to predict Survival

# In[ ]:


survival_feautures = ['Sex','Pclass','Parch','SibSp', 'Age']


# In[ ]:


X = df3[survival_feautures].copy()


# In[ ]:


X.columns


# In[ ]:


y.head()


# Method 1

# In[ ]:


#Fitting a DecisionTree

survival_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=389)
survival_classifier.fit(X, y)


# In[ ]:


type(survival_classifier)


# Method 2

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#Fitting a RandomForest

rf_model = RandomForestClassifier(n_estimators=1000,max_features=3,oob_score=True,random_state = 389)


# In[ ]:


rf_model.fit(X,y)


# # Predict on Test Set

# In[ ]:


df2.columns


# In[ ]:


df2.head()


# In[ ]:


df2.info()


# In[ ]:


#predicting on method 1
predictions = survival_classifier.predict(df2)


# In[ ]:


#predicting on method 2
predictions2 = rf_model.predict(df2)


# In[ ]:


predictions[:10]


# In[ ]:


predictions2[:10]


# In[ ]:


type(predictions2)


# # Merging predictions array to test dataset

# In[ ]:


#data frame on prediction
pred_survivors = pd.DataFrame(predictions)
pred_survivors.columns = ['survived']

frames = [clean_data2,pred_survivors]

result = pd.concat(frames, axis=1)


# In[ ]:


#data frame on prediction 2
pred_survivors2 = pd.DataFrame(predictions2)
pred_survivors2.columns = ['survived2']

frames2 = [clean_data2, pred_survivors2]

result2 = pd.concat(frames2, axis=1)


# In[ ]:


#Dropping unnecessary columns from the output
output = result.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

output2 = result2.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)


# In[ ]:


#Print out the head of the expected output
output.head()


# In[ ]:


#Print out the head of the expected output
output2.head()


# In[ ]:


pred_survivors2.head()


# # Measure Accuracy of the Classifier and Comparing Methods

# In[ ]:


#dropping passengerId to assess accuracy on survived column
df1 = df1.drop(['PassengerId'],axis=1)

accuracy_score(y_true=df1,y_pred=predictions)


# In[ ]:


print("OOB_accuracy:")
print(rf_model.oob_score_)


# In[ ]:


for feature, imp in zip(X,rf_model.feature_importances_):
    print(feature,imp)


# In[ ]:


#printing output (method 1) in csv

output.to_csv('survived_submission.csv', header=True, index=True, sep=',')


# In[ ]:


#Creating Submission dataframe for Kaggle
mysubmission = pd.DataFrame({"PassengerId":clean_data2["PassengerId"],
                            "Survived":pred_survivors2["survived2"]})

#Saving to CSV
mysubmission.to_csv("survived_mysubmission.csv", index=False)

