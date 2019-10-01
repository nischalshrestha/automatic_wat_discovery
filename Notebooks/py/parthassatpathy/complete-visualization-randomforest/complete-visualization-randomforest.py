#!/usr/bin/env python
# coding: utf-8

# 1. Detailed Visualization of Data
# 2. Imputing Null values
# 3. Random Forest
# 
# A detailed guide to analyze and visualize Titanic Data Set.

# In[ ]:


#Import the Python Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# 
# We will start with reading the Train and Test files. We will then check their different columns and data.
# 

# In[ ]:


#Read the Train file and save it as Pandas DataFrame
train_df = pd.read_csv("../input/train.csv")

#This will show the initial rows of Train Data
train_df.head()


# In[ ]:


#Let's see the Train Data Details
train_df.info()


#  
# Train has total 891 rows, 12 columns. "Survived" is the Label Data.
# Age has 714 rows, Cabin has 204 rows and Embarked has 889 rows, thus missing 177, 687 and 2 rows respectively.
# 

# 
# Let's check the details of Test Data.
# 

# In[ ]:


#Read the Test file and save it as Pandas DataFrame
test_df = pd.read_csv("../input/test.csv")

#This will show the initial rows of Train Data
test_df.head()


# In[ ]:


test_df.info()


# 
# Test has total 418 rows, 11 columns. "Survived" is the Column we need to predict. 
# Age has 332 rows, Cabin has 91 rows and Fare has 417 rows, thus missing 86, 327 and 1 rows respectively.
# 

# Now that we checked the Train and Test Data, we need to do the following tasks:
# 1. Impute the missing Values
# 2. Visualize Every Feature of the Data
# 
# Whatever changes we will make to the Features, we need do it in both Train and Test Data. So, we will combine the Train and Test Data to make it easier.

# In[ ]:


#Add None Data to a new Survived column in Test Data
test_df['Survived'] = [None]*418

##Join Train and Test Data to form Combine Data Frame
combine_df = pd.concat([train_df,test_df], axis = 0)

print(combine_df.info())
combine_df.head()


# **Visualization**

# 
# Let's visualize all the columns one by one. We will impute missing values on the way.
# We will start with the test label data.
# 

# In[ ]:


Survived_count = train_df['Survived'].value_counts()
#print(Survived_count)

plt.figure(figsize=(8,6))
sns.barplot(Survived_count.index,Survived_count.values)
plt.xlabel("Survival 0- Dead 1-Survived")
plt.ylabel("No of Passangers")
plt.title("Analysis of Survival")
plt.show()


# From the Graph :- There is a very high probability of death(Tragic). 

# We will keep a list of columns we will choose for our model. At first, we will select all the columns and then remove the ones we do not need.

# In[ ]:


selected_Columns = combine_df.columns.values.tolist()

## Survived is the column we will predict and Passanger Id is not of much significance. So, we will remove these two.
selected_Columns.remove('Survived')
selected_Columns.remove('PassengerId')
print(selected_Columns)


# **Gender**

# In[ ]:


Gender_count = train_df['Sex'].value_counts()
#print(Gender_count)

plt.figure(figsize=(8,6))
sns.barplot(Gender_count.index,Gender_count.values)
plt.xlabel("Gender")
plt.ylabel("No of Passangers")
plt.title("Analysis of Gender")
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.ylabel('Number of Passengers')
plt.xlabel('Gender')
plt.show()


# 
# As seen from the above graph, 'Sex' is a very important factor. Very few Male have survived(Approx 5/6th died). There is a very good rate of Survival among Female.

# **Pclass**

# In[ ]:


Pclass_count = train_df['Pclass'].value_counts()
plt.figure(figsize=(8,6))
sns.barplot(Pclass_count.index,Pclass_count.values)
plt.xlabel("Pclass")
plt.ylabel("No of Passangers")
plt.title("Analysis of Pclass_count")
plt.show()


# Most of the passengers were of class 3, followed by 1 and 2.

# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.ylabel('Number of Passengers')
plt.xlabel('Pclass')
plt.show()


# There is a high degree of fatality in the case of class 3 passengers. However, a good survival ratio among class 1. Class 2- Almost equal probability.

# **Parch**

# In[ ]:


Parch_count = train_df['Parch'].value_counts()
print(Parch_count)

plt.figure(figsize=(8,6))
sns.barplot(Parch_count.index,Parch_count.values)
plt.xlabel("Parch")
plt.ylabel("No of Passangers")
plt.title("Analysis of Parch")
plt.show()


# Most of the values reside in 0,1,2. Most values 0 shows most of the people came alone.

# We can make the values greater than 2 as 3.

# In[ ]:


train_df['Parch'].ix[train_df['Parch']>2] = 3
combine_df['Parch'].ix[combine_df['Parch']>2] = 3

plt.figure(figsize=(8,6))
sns.countplot(x='Parch', hue='Survived', data=train_df)
plt.ylabel('Number of Passengers')
plt.xlabel('Parch')
plt.show()


# Person who has 1 or 2 have higher survival ratio than the person who is not a parent nor has a child.

# In[ ]:


SibSp_count = train_df['SibSp'].value_counts()
print(SibSp_count)

plt.figure(figsize=(8,6))
sns.barplot(SibSp_count.index,SibSp_count.values)
plt.xlabel("SibSp")
plt.ylabel("No of Passangers")
plt.title("Analysis of SibSp")
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='SibSp', hue='Survived', data=train_df)
plt.ylabel('Number of Passengers')
plt.xlabel('SibSp')
plt.show()


# We can assume that person having Siblings greater than 4 have very less chance of Survival rate. We can make all values greater than 3 as 4. Person with only 1 sibling survived most.

# In[ ]:


train_df['SibSp'].ix[train_df['SibSp']>3] = 4
combine_df['SibSp'].ix[combine_df['SibSp']>3] = 4

plt.figure(figsize=(8,6))
sns.countplot(x='SibSp', hue='Survived', data=train_df)
plt.ylabel('Number of Passengers')
plt.xlabel('SibSp')
plt.show()


# **Fare**

# Fare has one row missing. We need to impute that. But, luckily Train Data has all the data. So, let's analyze the column and then impute it.

# In[ ]:


plt.figure(figsize=(8,6))
print(train_df['Fare'].describe())
#plt.subplot(121)
plt.hist(train_df['Fare'],bins=40,range=(0,train_df['Fare'].describe()[-1]),normed=True,color='blue',alpha=0.8)
plt.show()


# There seems to be some outliers where ticket prices are very high. Let's check the rows where Fare is higher than 200.

# In[ ]:


plt.subplot(1,3,1)
temp_df = train_df.ix[train_df['Fare']>100]
sns.countplot(x='Sex', hue='Survived', data=temp_df)
plt.ylabel('Number of Passengers')
plt.xlabel('Gender')

plt.subplot(1,3,2)
temp_df = train_df.ix[train_df['Fare']>152]
sns.countplot(x='Sex', hue='Survived', data=temp_df)
plt.ylabel('Number of Passengers')
plt.xlabel('Gender')

plt.subplot(1,3,3)
temp_df = train_df.ix[train_df['Fare']>200]
sns.countplot(x='Sex', hue='Survived', data=temp_df)
plt.ylabel('Number of Passengers')
plt.xlabel('Gender')
plt.tight_layout()
plt.show()


# We can see that above Fare price 152, the pattern of the male and Female Survival rate is almost same. So, we can make the values as 152.

# In[ ]:


train_df['Fare'].ix[train_df['Fare']>151] = 152
combine_df['Fare'].ix[combine_df['Fare']>151] = 152

plt.figure(figsize=(8,6))
plt.hist(train_df['Fare'],bins=40,range=(0,train_df['Fare'].describe()[-1]),normed=True,color='blue',alpha=0.8)
plt.show()


# In[ ]:


sns.boxplot(x="Survived",y="Fare",data=train_df)
plt.show()


# Most of the people who died have tickets with less Fair. But, people with high fare ticket have more survival rate.

# Let's check the detail of the person whose Fare is not available.

# In[ ]:


combine_df[combine_df['Fare'].isnull()]


# We can find the Fares for passengers who are aged and are of class 3, male and have Embarked at S.
# We can update the Fare value as the average of the these values.

# In[ ]:


temp_df = combine_df[(combine_df['Pclass']==3) & (combine_df['Sex']=='male') & 
                     (combine_df['Embarked']=='S') & (combine_df['Age'] > 50)]
combine_df['Fare'][combine_df['Fare'].isnull()] = temp_df['Fare'].mean()


# **Embarked**

# Next let's analyze Embarked Features. We will analyze the part where Embarked is not null. Then we will impute the 2 missing values based on Analyzed Data.

# In[ ]:


temp_df = train_df[train_df['Embarked'].notnull()]

Embarked_count = train_df['Embarked'].value_counts()
print(Embarked_count)

plt.figure(figsize=(8,6))
sns.barplot(Embarked_count.index,Embarked_count.values)
plt.xlabel("Embarked")
plt.ylabel("No of Passangers")
plt.title("Analysis of Embarked")
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='Embarked', hue='Survived', data=temp_df)
plt.ylabel('Number of Passengers')
plt.xlabel('Gender')
plt.show()


# Let's try to impute the two missing rows.

# In[ ]:


combine_df[combine_df['Embarked'].isnull()]


# Both these ladies have paid high ticket price and Survived. They are both of Pclass 1. We have such people at all the three cities. So, we will impute their value as city 'S' which is most occuring.

# In[ ]:


combine_df['Embarked'][(combine_df['Cabin']=='B28')] = 'S'


# **Name**

# Let's check what are the common words in Name. 

# In[ ]:


from wordcloud import WordCloud

text_Name = ''

for ind, row in train_df.iterrows():
    text_Name = " ".join([text_Name,"_".join(row['Name'].strip().split(" "))])

text_Name = text_Name.strip()

plt.figure(figsize=(12,6))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text_Name)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.title("Wordcloud for Name", fontsize=30)
plt.axis("off")
plt.show()


# In[ ]:


Name does no seem to bring new features, but it will be very useful in predicting the Age which has lots of empty rows.
We will try to fetch Title from the Name which will be useful.


# In[ ]:


import re

#A function to get the title from a name.
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

#Get all the titles and print how often each one occurs.
titles = combine_df["Name"].apply(get_title)
print(pd.value_counts(titles))


# There are lots of Titles here. Let's try to create 4 main titles- Mr, Miss, Mrs and Master. We will make some assumption here to add the small number of Titles to the main categories. Some consider to make a new category called rare too.

# In[ ]:


combine_df['titles'] = titles
print(combine_df.titles.unique())

MrList = ['Rev','Dr','Col','Major','Sir','Jonkheer','Don','Capt']
MsList = ['Ms','Mlle']
MrsList = ['Lady','Mme','Dona','Countess']

for ind, row in combine_df.iterrows():
    if row['titles'] in MrList:
        #row['titles'] = 'Mr'
        #print(combine_df.at[ind,'titles'])
        combine_df.at[ind,'titles'] = 'Mr'
    if row['titles'] in MsList:
        combine_df.at[ind,'titles'] = 'Miss'
    if row['titles'] in MrsList:
        combine_df.at[ind,'titles'] = 'Mrs'

print(combine_df.titles.unique())
print(pd.value_counts(combine_df["titles"]))


# **Age**

# The last and most important feature is Age. But,we have around 263 rows where Age is missing.
# 
# Let's Analyze the missing rows first.

# In[ ]:


temp_df = combine_df[combine_df['Age'].isnull()]
Titles_count = temp_df['titles'].value_counts()
print(Titles_count)

plt.figure(figsize=(8,6))
sns.barplot(Titles_count.index,Titles_count.values)
plt.xlabel("Titles")
plt.ylabel("No of Passangers")
plt.title("Analysis of Titles for Missing Age")
plt.show()


# There are very few rows missing for passengers with Title 'Master' and 'Mrs'. So, for them we can impute the Age as the mean or median of other Master and Mrs passengers ages respectively.

# In[ ]:


##Missing master Data
temp_dfMaster = temp_df[temp_df['titles'] == 'Master']
temp_dfMaster


# All the empty rows are for Pclass- 3. Let's find the mean and median ages of all Passengers with Pclass 3 and Title Master.

# In[ ]:


print(combine_df['Age'][(combine_df['Pclass'] == 3) & (combine_df['titles'] == 'Master')
                       & (combine_df['Age'].notnull())].mean())
print(combine_df['Age'][(combine_df['Pclass'] == 3) & (combine_df['titles'] == 'Master')
                & (combine_df['Age'].notnull()) ].median())


# In[ ]:


##Replace with the Mean Value
combine_df['Age'][(combine_df['Pclass'] == 3) & (combine_df['titles'] == 'Master')
                & (combine_df['Age'].isnull())] = combine_df['Age'][(combine_df['Pclass'] == 3) & (combine_df['titles'] == 'Master')
                       & (combine_df['Age'].notnull())].mean()


# In[ ]:


##Passenger with Mrs Title
temp_dfMrs = combine_df[(combine_df['titles'] == 'Mrs') & (combine_df['Age'].isnull())]
temp_dfMrs


# We will replace the Age value as per Pclass of Title Mrs.

# In[ ]:


def imputeMrsAge(cls):
    ##Replace with the Mean Value
    combine_df['Age'][(combine_df['Pclass'] == cls) & (combine_df['titles'] == 'Mrs')
                & (combine_df['Age'].isnull())] = combine_df['Age'][(combine_df['Pclass'] == cls)&(combine_df['titles'] == 'Mrs')
                       & (combine_df['Age'].notnull())].mean()

imputeMrsAge(1)
imputeMrsAge(2)
imputeMrsAge(3)


# For these two titles- Mr and Miss- We will use a model to predict the values. But before that we will create another column called Family Size which will be 1+Parch+Sibsp.
# We also need to labelencode the categorical variables so that they can be used in the models.

# In[ ]:


combine_df['FamilySize'] = 1+ combine_df['Parch'] + combine_df['SibSp']


# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc=LabelEncoder()

cat_vars=['Embarked','Parch','Pclass','Sex','SibSp',"titles","FamilySize"]

for col in cat_vars:
    combine_df[col]=labelEnc.fit_transform(combine_df[col])
    
combine_df.head()


# In[ ]:


##Random Forest model to predict the missing Age values.

##Features we will use to predict the Age
ageFeatures = ['Embarked','Pclass','Sex',"titles","FamilySize",'Fare']

##Train Data will be rows where Age values are present
trainAgeF = (combine_df[combine_df['Age'].notnull()])[ageFeatures]
trainAgeL = (combine_df[combine_df['Age'].notnull()])['Age']
##Train Data will be rows where Age values are null
testAgeF = (combine_df[combine_df['Age'].isnull()])[ageFeatures]

#RandomForest Model
from sklearn.ensemble import RandomForestRegressor
rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
rtr.fit(trainAgeF, trainAgeL)

# Use the fitted model to predict the missing values
predictedAges = rtr.predict(testAgeF)

##Replace the NA values with the predicted values
combine_df['Age'][combine_df['Age'].isnull()] = predictedAges


# Instead of using Age and Fare as numerical values we can create groups in them and make them categorical.

# In[ ]:


combine_df.loc[ combine_df['Fare'] <= 7.91, 'Fare'] = 0
combine_df.loc[(combine_df['Fare'] > 7.91) & (combine_df['Fare'] <= 14.454), 'Fare'] = 1
combine_df.loc[(combine_df['Fare'] > 14.454) & (combine_df['Fare'] <= 31), 'Fare']   = 2
combine_df.loc[ combine_df['Fare'] > 31, 'Fare']   = 3
combine_df['Fare'] = combine_df['Fare'].astype(int)

combine_df.loc[ combine_df['Age'] <= 16, 'Age']= 0
combine_df.loc[(combine_df['Age'] > 16) & (combine_df['Age'] <= 32), 'Age'] = 1
combine_df.loc[(combine_df['Age'] > 32) & (combine_df['Age'] <= 48), 'Age'] = 2
combine_df.loc[(combine_df['Age'] > 48) & (combine_df['Age'] <= 64), 'Age'] = 3
combine_df.loc[ combine_df['Age'] > 64, 'Age'] = 4    


# In[ ]:


combine_df.head(2)


# Let's check the Correlation among the features if any

# In[ ]:


Features = ['Embarked','Pclass','Sex',"titles",'Age','Fare','Parch','SibSp','FamilySize']
cor_matrix = combine_df[Features].corr()
sns.heatmap(cor_matrix)
plt.show()


# There are correlation among Fare,Parch, SibSp and FamilySize.

# We have many features available now to consider. Which ones should we use?
# 
# 1. PassengerId, Ticket and Name do not have much significance. They can be removed.
# 
# 2. Cabin has lots of Null values and imputing and using them may cause error. So, removed.
# 
# 3. There are correlation among Fare, Parch, SibSp and FamilySize. So, we will keep only Fare.

# In[ ]:


Features = ['Embarked','Pclass','Sex',"titles",'Age','Fare']


# In[ ]:


Train = combine_df[combine_df['Survived'].notnull()]
Test = combine_df[combine_df['Survived'].isnull()]

train_x = Train[Features]
train_y = pd.DataFrame(Train['Survived'])
test_x = Test[Features]

train_x_arr = np.array(train_x.values) #Use as Features to fit a Regression Model
train_y_arr = np.array(train_y.Survived.values.tolist()) #Use as labels to fit a Regression Model

test_x_arr = np.array(test_x.values) ##Use as Features to predict a Regression Model


# You can use the Train and test data in any model of your choice. Random Forest seems to be the favorite of many. You can also use XGBoost and SVM too. I am giving example of RandomForest Here. 

# In[ ]:


##We wil use GridSearchCV to first find the best parameters for our model 
#and then  we will use the parameters in our model

#Import the packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [700,1000,1500,2000],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_x_arr, train_y_arr)
print (CV_rfc.best_params_)
print(CV_rfc.best_estimator_)


# In[ ]:


clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1500, n_jobs=-1,
            oob_score=True, random_state=None, verbose=0, warm_start=False)
clf.fit(train_x_arr, train_y_arr)
predictedVal = clf.predict(test_x_arr)
print("%.4f" % clf.oob_score_)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": Test["PassengerId"],
        "Survived": predictedVal
    })

submission.to_csv("rf_submission_1.csv", index=False)


# This kernel might not be the best kernel you see here, but it will definitely give you a detailed view of how to analyze and visualize data.
# 
# If you find it useful, please upvote. 
# 
# Thanks.

# In[ ]:




