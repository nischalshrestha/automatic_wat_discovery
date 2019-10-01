#!/usr/bin/env python
# coding: utf-8

# ##Finished but I'm going to keep trying to improve my score! 

# ### Next, I want to try using GridSearchCV on my models

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[3]:


# I/O
train_dir = '../input/train.csv'
test_dir = '../input/test.csv'

testdata = pd.read_csv(test_dir)
trainingdata = pd.read_csv(train_dir)

pass_id = testdata['PassengerId'] #for submission

modified_train = trainingdata.drop('Survived', axis=1)
combined_df = pd.concat([modified_train, testdata])
combined_df.info()


# # What I know so far about the combined dataset:
# 
#  1. There are 1309 entries/passengers across 11 features
#  2. Age, Cabin, Embarked and Fare are lacking input.
#  3. Next it would make most sense to fix the categorical data and any missing data

# # Analysis

# ### 1. Gender

# In[4]:


#Basic Visualizations

## 1. Survival rate vs Gender
plt.figure()
sns.countplot(x='Sex', hue='Survived', data=trainingdata)
plt.title('Survival Rate of Males and Females')
plt.xticks(np.arange(2), ('Male', 'Female'))


# From the graph above, we see that there were more men on board the ship,  and also that more men died than survied whereas the opposite is true for women. Does passenger class have anything to do with this?

# In[5]:


## 2. Survival rate vs Class and Gender
plt.figure()
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=trainingdata)
plt.title('Survival Rate of Males and Females of different Class')


# Both genders show a similar trend, through there is a difference in class 2 of passengers where more women survived. Overall, the lower class (3) had a pretty bad time. And men from all classes had a pretty bad time too. From this we can see that socioeconomic class did have an impact on the survival rate of passengers.
# 

# ### 2. Class and Origin

# In[6]:


## 2. Origin (Embarked) against survival rate of each gender in trainingdata
plt.figure()
sns.barplot(x='Sex', y='Survived', hue='Embarked', data=trainingdata)
plt.title('Survival Rate of Males and Females of different Origins')

## 3. General Survival rate vs Origin
plt.figure()
sns.pointplot(x='Embarked', y='Survived', data=trainingdata)
plt.title('Survival Rate by Origin')

## 4. Passenger Class count of each Origin
plt.figure()
sns.countplot(x='Embarked', hue='Pclass', data=combined_df)
plt.title('Number of Passengers from Each Origin (Combined Data)')

## 5. Origin and Gender
plt.figure()
sns.countplot(x='Embarked', hue='Sex', data=combined_df)
plt.title('Gender distribution of each Origin')


# The three points of origin are as follows: C = Cherbourg, Q = Queenstown, S = Southampton.

# From the graphs above, we can better understand why it seems as though passengers from Cherbourg survived most, whereas those from Southampton survived the least. 

# Graph 2 depicts the survival rate of each gender across the three social classes. Females in all classes had more survivors than the men. And strangely enough a larger number of class 2 (Middle?) passengers survived relative to the other classes. However, There is not an even distribution of people across the classes and priority also dictated survival rate to an extent. 

# Graph 3 displays the relative survival rates of passengers of all three origins. The error bars are relatively large but nonetheless, this gives a decent idea of how many people survived from each origin. This graph alone serves little purpose, so Graph 4 helps add value to it. Graph 4 shows that there were a lot of passengers from Southampton, and less from Cherbourg and Queenstown. Also the class distribution is uneven, with a lot of class 3 passengers in Queenstown and Southampton by ratio. Cherbourg is the only location with more class 1 passengers than class 3. And with the knowledge that priority was given to Class 1 before Class 2 and Class 3, it only makes sense that a larger number of people from Cherbourg survived. 

# Finally in Graph 5, while all points of origin had more male than female passengers, Southampton had over twice as many males than females, and males do did not survive very much, so relative to the population of Southampton passengers, trends show that there would have been significant death given gender and class distributions. Additionally Queenstown passengers were all 3rd class but show a generally higher survival rate because the gender distribution is equal. More can be said but it would be redundant at this point.

# To conclude on the point of origin - I think that it would be best to remove it as a feature since the slight trends do not seem significant enough to help predict survival in the testing set. Though it would be interesting to test a dataset that includes 'Embarked' as a feature against one that does not. 
# 

# Next, I will consider Age as a feature since surely, it must have mattered. Or so I think.

# ### 3. Age

# In[7]:


#Cleaning up the Age Column
#We are missing some values as seen
na_age_train = trainingdata['Age'].isnull().sum()
na_age_test = testdata['Age'].isnull().sum()
na_age_c = combined_df['Age'].isnull().sum()
#There should be 1309 values
# 177 values missing from the training data
# 86 values missing from the test data


# In[8]:


#Still Cleaning up the Age Column

#This is for the next visualization
maop = combined_df['Age'].mean() #maop = mean age of passengers
stap = combined_df['Age'].std()  #stap = std of age of passengers

max_range = (maop+stap)
min_range = (maop-stap)
rand_age_train = np.random.randint(min_range, max_range, size = na_age_train)
rand_age_test = np.random.randint(min_range, max_range, size = na_age_test)
rand_age_c = np.random.randint(min_range, max_range, size = na_age_c)

#I'm not sure what the right way of doing this is though, can I used the combined data mean to fill
# training and testing data sets separately? If there's a protocol, please let me know. I'll try searching for one..
## UPDATE - yeah that failed, decided to use a random set of numbers from a range around the mean instead
#Credit: Omar El Gabry - I used his general code for this:

trainingdata['Age'].dropna().astype(int)
trainingdata["Age"][np.isnan(trainingdata["Age"])] = rand_age_train

testdata['Age'].dropna().astype(int)
testdata["Age"][np.isnan(testdata["Age"])] = rand_age_test

combined_df['Age'].dropna().astype(int)
combined_df["Age"][np.isnan(combined_df["Age"])] = rand_age_c



# In[9]:


## 6. Mean Survival by Age
plt.figure()
sns.barplot(x='Age', y='Survived', data=trainingdata, ci=None) #ci=None removes error bars
plt.title('Mean Survival by Age (0-80)')
#the axis is messed up


# In[10]:


## 7.1 - General Age Distribution (Train)
x= trainingdata['Age']
plt.figure()
sns.distplot(x, rug=True, hist=False)
plt.ylabel('Density')
plt.title('Age Distribution in Training Data')

## 7.2 - General Age Distribution (Test)
x= testdata['Age']
plt.figure()
sns.distplot(x, rug=True, hist=False)
plt.ylabel('Density')
plt.title('Age Distribution in Test Data')


# In[11]:


## 8. KDE of Age and Survival
#Credit: https://github.com/mwaskom/seaborn/issues/595
plt.figure()
sns.FacetGrid(data=trainingdata, hue="Survived", aspect=4).map(sns.kdeplot, "Age", shade=True)
plt.ylabel('Passenger Density')
plt.title('KDE of Age against Survival')
plt.legend()


# From graph 6 we can see a high survival for children until the age of 10, and then there is a decline in survival rate until we get to more aged people. 

# Graph 7 shows the distribution of ages in training and testing datasets. Both distributions are very similar, and show a large volume of passengers between ages 20 and 30.

# Graph 8  shows the age distribution of passengers who survived and died. 
# Slightly more children survived than died as seen in the 0 - 15 Age range. However more people in the 16-25 range died than survived. When considering this alongside the distribution of gender and class, it makes sense. 

# Next, and possibly the final feature that needs exploration and visualization, is the presence of family on board . "Sibsp" and "Parch"

# ### 4. Parch and SibSp

# In[12]:


## 9 - Survival rate Given Amount of Family on Board

trainingdata['Related'] = trainingdata['SibSp'] + trainingdata['Parch']
testdata['Related'] = testdata['SibSp'] + testdata['Parch']

combined_df['Related'] = combined_df['SibSp'] + combined_df['Parch']

plt.figure()
sns.barplot(data=trainingdata, x='Related', y="Survived")
plt.title('Survival rate Given Amount of Family on Board')


# It would make more sense to dichotomize the data, though from this alone, we can see that people with 1-3 family members had a higher survival than lone passengers, whereas those with 4-6 family members had a lower survival rate. 

# In[13]:


#https://pandas.pydata.org/pandas-docs/stable/indexing.html
#----->     df.loc[row_indexer,column_indexer]
trainingdata['Related'].loc[trainingdata['Related'] > 0] = 1
trainingdata['Related'].loc[trainingdata['Related'] == 0] = 0

## 10 - Dichotomized Survival Given Family on Board 
plt.figure
sns.barplot(data=trainingdata, x='Related', y='Survived', hue='Pclass', ci=None)
plt.xticks(np.arange(2),("No Family", "Family"))
plt.title('Dichotomized Survival Given Family on Board')


# As seen from Graph 10, for all passenger classes, there was a higher survival rate for passengers with relatives on board. The relative difference is highest in the 2nd passenger class.

# #Feature Engineering: Cabin Letter, Ticket Heads, Titles, and  Name Lengths

# In[14]:


def tit(a):
    for x in [a]:
        x['Titles'] = x['Name'].str.split(',').str.get(1).str.split('.').str.get(0)
    return a

def nlen(train, test):
    for x in [train, test]:
        x["n_len"] = x["Name"].apply(lambda x: len(x))
    return train, test

def cabin(a):
    for x in [a]:
        x['c_let'] = x['Cabin'].apply(lambda x: str(x)[0])
    return a

def ticket(a):
    for x in [a]:
        x['tic_h'] = x['Ticket'].apply(lambda x: str(x)[0])
    return a


# # Machine Learning

# In[15]:


#Fixing Fare since its missing one value
fare_fill = combined_df['Fare'].median()
combined_df['Fare'].fillna(fare_fill, inplace=True)

#Embarked is missing 2 values
combined_df['Embarked'].fillna('S', inplace=True)


# #Dummies

# In[16]:


tit(combined_df);
nlen(trainingdata, testdata);
cabin(combined_df);
ticket(combined_df);


#Thanks Kevin Lin for pointing out my error in v.49
combined_df = pd.get_dummies(combined_df, columns=['Pclass', 'Embarked', 'Titles', 'Sex', 'tic_h', 'c_let' ])

train_df = combined_df[0:891]
train_df['Survived'] = trainingdata['Survived']

test_df = combined_df[891:]



# In[17]:


X_train = train_df.drop(['Survived', 'Cabin','Ticket', 'Name', 'Parch', 'SibSp'],axis=1)
Y_train = train_df['Survived']
Y_train  = Y_train.values.reshape(-1, 1)

X_test = test_df.drop(['Cabin', 'Ticket', 'Name', 'Parch', 'SibSp'], axis=1) 


# ### Correlation

# In[18]:


corr = X_train.corr()
f, ax = plt.subplots(figsize=(32,18))
sns.heatmap(corr, linewidths=0.1,vmax=1.0, square=True)


# ###Random Forest Classifier

# In[19]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, min_samples_split = 5, min_samples_leaf = 2, n_jobs = 50)
Y_train = np.ravel(Y_train)
rf.fit(X_train, Y_train)


# ###Gaussian Naive Bayes

# In[20]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
Y_train = np.ravel(Y_train)
gnb.fit(X_train, Y_train)


# ###Extra Trees Classifier

# In[21]:


from sklearn.ensemble import ExtraTreesClassifier
ext = ExtraTreesClassifier()
Y_train = np.ravel(Y_train)
ext.fit(X_train, Y_train)


# ### Logistic Regression

# In[22]:


from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
Y_train = np.ravel(Y_train)
lreg.fit(X_train, Y_train)


# ### General Scoring and Voting

# In[23]:


from sklearn.ensemble import VotingClassifier


vclf = VotingClassifier(estimators=[('rf', rf), ('ext', ext), ("gnb", gnb), ("lreg", lreg)], voting='hard')
Y_train = np.ravel(Y_train)
vclf = vclf.fit(X_train, Y_train)


# In[24]:


final = vclf.predict(X_test)
output = pd.DataFrame({'PassengerId': pass_id, "Survived": final})
output.to_csv('prediction.csv', index=False)

