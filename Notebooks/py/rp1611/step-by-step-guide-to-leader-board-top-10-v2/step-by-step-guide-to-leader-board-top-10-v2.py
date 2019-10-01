#!/usr/bin/env python
# coding: utf-8

# **This is my 2nd Kernel and is kind of version 2 to my previous kernel which can be found here**
# 
# (http://www.kaggle.com/rp1611/step-by-step-guide-to-leader-board-top-10) 
# 
# Along with all usual visualization, I have used Keras to build a simple NN model. I am just providing you a platform which you can finetune to get better results with NN.
# 
# All the existing variables along with feature engineered once are available in combined data. Use the notebook to finetune your model by adding/dropping/modifying the existing features. 
# 
# **If you like this notebook or find this notebook helpful, Please upvote and/or leave a comment**

# **What would be the workflow?**
# 
# I will keep it simple & crisp rather than using buzz words & useless data science frameworks. Frankly speaking no one cares. 
# 
# This will help you to stay on track. So here is the workflow.
# 
# **1. Problem Identification**
# 
# **2. What data do we have?**
# 
# **3. Exploratory data analysis**
# 
# **4. Data preparation including feature engineering**
# 
# **5. Developing a Neural Network model**
# 
# **6. Predict using Neural Network**
# 
# **7. Conclusions**
# 
# That's all you need to solve a data science problem.

# **Problem Identification** 
# 
# **Best Practice -** The most important part of any project is correct problem identification. Before you jump to "How to do this" part like typical Data Scientists, understand "What/Why" part.  
# Understand the problem first and draft a rough strategy on a piece of paper to start with. Write down things like what are you expected to do & what data you might need or let's say what all algorithms you plan to use. 
# 
# Now the <a href="https://www.kaggle.com/c/titanic/"> Titanic challenge</a>  hosted by Kaggle is a competition in which the goal is to predict the survival or the death of a given passenger based on a set of variables describing  age, sex, or passenger's class on the boat.
# 
# ![](http://www.tyro.com/content/uploads/2016/04/blog-twenty-one-business-icebergs-sink-business-280416.jpg)
# 
# So it is a classification problem and you are expected to predict Survived as 1 and Died as 0.

# **2. What data do we have?**
# 
# 
# Let's import necessary libraries & bring in the datasets in Python environment first. Once we have the datasets in Python environment we can slice & dice the data to understand what we have and what is missing.

# In[ ]:


# Import the python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep')
import warnings
warnings.filterwarnings('ignore')

# Read the datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
IDtest = pd.DataFrame(test["PassengerId"])


# In[ ]:


train.info() # We have 891 observations & 12 columns. See the mix of variable types.


# In[ ]:


train.head(5) # Ticket & Cabin variables are alphanumeric


# In[ ]:


test.info() # We have 417 observations & 11 columns (no response 'Survived' column).


# **3. Exploratory data analysis **
# 
# One important aspect of machine learning is to ensure that the variables show almost the same trend across train & test data. If not, it would lead to overfitting because model is representing a relationship which is not applicable in the test dataset. 
# 
# I will give you one example here. As we do variable analysis, try to replicate (wherever applicable) the code for test data and see if there is any major difference in data distribution. 
# 
# **Example** - Let's start with finding the number of missing values. If you compare the output you will see that missing value percentages do not vary much across train & test datasets.
# 
# Use the groupby/univariate/bivariate analysis method to compare the distribution across Train & Test data

# In[ ]:


train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
miss_train = pd.DataFrame({'Train Missing Ratio' :train_na})
miss_train.head()


# In[ ]:


test_na = (test.isnull().sum() / len(test)) * 100
test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)[:30]
miss_test = pd.DataFrame({'Test Missing Ratio' :test_na})
miss_test.head()


# In[ ]:


# Fill empty and NaNs values with NaN
train = train.fillna(np.nan)
test = test.fillna(np.nan)


# **PassengerId**
# 
# Not relevant from modeling perspective so we will drop this variable later

# **Pclass**
# 
# Pclass is categorical variable. Let's look at the distribution.

# In[ ]:


# Analyze the count of survivors by Pclass

ax = sns.countplot(x="Pclass", hue="Survived", data=train)
train[['Pclass', 'Survived']].groupby(['Pclass']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze the Survival Probability by Pclass

g = sns.barplot(x="Pclass",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)


# Approximately 62% of Pclass = 1 passenger survived followed by 47% of Pclass2.

# **Name**
# 
# Not relevant from analysis & modeling perspective. We will drop this feature later after creating a new variable as Title.

# **Sex**
# 
# Based on analysis below, female had better chances of survival. 
# 
# ![](https://www.ajc.com/rf/image_large/Pub/p9/AJC/2018/07/12/Images/newsEngin.22048809_071418-titanic_Titanic-Image-7--2-.jpg)

# In[ ]:


# Count the number of passengers by gender
ax = sns.countplot(x="Sex", hue="Survived", data=train)

# Analyze survival count by gender
train[["Sex", "Survived"]].groupby(['Sex']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze the Survival Probability by Gender

g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[["Sex", "Survived"]].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)


# Based on data above, female passengers had better chances of survival than male passengers

# **Age**
# 
# The insight below connects back to "Ladies and Kids First" scene of the movie. It shows that a good number of babies & young kids survived.

# In[ ]:


# Let's explore the distribution of age by response variable (Survived)
fig = plt.figure(figsize=(10,8),)
axis = sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='Survived')
axis = sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='b',shade=True,label='Did Not Survived')
plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 20)
plt.xlabel("Passenger Age", fontsize = 12)
plt.ylabel('Frequency', fontsize = 12);


# In[ ]:


sns.lmplot('Age','Survived',data=train)

# We can also say that the older the passenger the lesser the chance of survival


# **SibSP**
# 
# This variable refers to number of siblings/spouse onboard. SibSP = 1 and SibSP = 2 shows higher chances of survival.

# In[ ]:


# Analyze the count of survivors by SibSP

ax = sns.countplot(x="SibSp", hue="Survived", data=train)
train[['SibSp', 'Survived']].groupby(['SibSp']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze probability of survival by SibSP

g  = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train[["SibSp", "Survived"]].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)


# **Parch**
# 
# Parch indicates number of parents / children aboard the Titanic. Note that Parch = 3 and Parch = 1 shows higher survival probabilities. 

# In[ ]:


# Analyze the count of survivors by Parch

ax = sns.countplot(x="Parch", hue="Survived", data=train)
train[['Parch', 'Survived']].groupby(['Parch']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze the Survival Probability by Parch

g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Survival Probability")
train[["Parch", "Survived"]].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)


# **Ticket**
# 
# This variable has alphanumeric value which might not be related to Survival directly but we can use this variable to create some additional features.

# In[ ]:


train['Ticket'].head()


# **Fare**
# 
# Let's check the distribution first.

# In[ ]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics
sns.distplot(train['Fare'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['Fare'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('Fare distribution')


# The Fare variable is right skewed. We need to transform this variable using log function and make it more normally distributed. We will do this during feature engineering process.

# **Cabin**
# 
# Alphanumeric variable. 
# 
# 687 missing values in train & 327 missing values in test data - which needs to be treated. We can create more features using this Cabin variable. 

# In[ ]:


# Let's check the unique values
train['Cabin'].unique()


# **Embarked**
# 
# C = Cherbourg, Q = Queenstown, S = Southampton
# 
# Let's explore the variable with Survival rate. Embarked represents port of embarkation. As the analysis output below suggests Emabrked C shows high probabilities of survival.

# In[ ]:


# Analyze the count of survivors by Embarked variable

ax = sns.countplot(x="Embarked", hue="Survived", data=train)
train[['Embarked', 'Survived']].groupby(['Embarked']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze the Survival Probability by Embarked

g  = sns.factorplot(x="Embarked",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train[["Embarked", "Survived"]].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False)


# **Additional analysis**
# 
# Let's create few additional charts to see how different variables are related.

# In[ ]:


# Age, Pclass & Survival
sns.lmplot('Age','Survived',data=train,hue='Pclass')


# In[ ]:


# Age, Embarked, Sex, Pclass
g = sns.catplot(x="Age", y="Embarked",  hue="Sex", row="Pclass",   data=train[train.Embarked.notnull()], 
orient="h", height=2, aspect=3, palette="Set3",  kind="violin", dodge=True, cut=0, bw=.2)


# In[ ]:


# Relation among Pclass, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="Pclass", data=train, saturation=.5, 
                kind="bar", ci=None, aspect=.6)


# In[ ]:


# Relation among SibSP, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="SibSp", data=train, saturation=.5, 
                kind="bar", ci=None, aspect=.6)


# In[ ]:


# Relation among Parch, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="Parch", data=train, saturation=.5, 
                kind="bar", ci=None, aspect=.6)


# **Data preparation including feature engineering**
# 
# What we need to do to process following variables  - 
# 
# **PassengerID** - No action required
# 
# **PClass** - Have only 3 numerical values. We will use it as it is.
# 
# **Name** - Can be used to create new variable Title by extracting the salutation from name.
# 
# **Sex** - Create dummy variables
# 
# **Age** - Missing value treatment, followed by creating dummy variables
# 
# **SibSP** - Create dummy variables
# 
# **Parch** - Create dummy variables
# 
# **Ticket** - Create dummy variables post feature engineering
# 
# **Fare** - Missing value treatment followed by log normalization
# 
# **Cabin** - Create dummy variables post feature engineering
# 
# **Embarked** - Create dummy variables

# In[ ]:


# Let's combining train & test for quick feature engineering. 
# Variable source is a kind of tag which indicates data source in combined data
train['source']='train'
test['source']='test'
combdata = pd.concat([train, test],ignore_index=True)
print (train.shape, test.shape, combdata.shape)


# In[ ]:


# Let's check the data
combdata.head()


# **PassengerID**

# In[ ]:


# PassengerID - Drop PassengerID
combdata.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# **Pclass**

# In[ ]:


# Pclass - Use as it is
combdata['Pclass'].unique()


# **Name**

# In[ ]:


# Name - Extract Salutation from Name variable

salutation = [i.split(",")[1].split(".")[0].strip() for i in combdata["Name"]]
combdata["Title"] = pd.Series(salutation)
combdata["Title"].unique()


# **Title**

# In[ ]:


# Convert other salutations to fixed Title 
combdata["Title"] = combdata["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combdata["Title"] = combdata["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
combdata["Title"] = combdata["Title"].astype(int)
combdata["Title"].unique()


# In[ ]:


# Drop Name variable
combdata.drop(labels = ["Name"], axis = 1, inplace = True)
combdata.head()


# **Age**

# In[ ]:


# Age

## Fill Age with the median age of similar rows according to Sex, Pclass, Parch and SibSp
# Index of NaN age rows
missing_index = list(combdata["Age"][combdata["Age"].isnull()].index)

for i in missing_index :
    median_age = combdata["Age"].median()
    filled_age = combdata["Age"][((combdata['Sex'] == combdata.iloc[i]["Sex"]) & 
                                (combdata['SibSp'] == combdata.iloc[i]["SibSp"]) & 
                                (combdata['Parch'] == combdata.iloc[i]["Parch"]) & 
                                (combdata['Pclass'] == combdata.iloc[i]["Pclass"]))].median()
    if not np.isnan(filled_age) :
        combdata['Age'].iloc[i] = filled_age
    else :
        combdata['Age'].iloc[i] = median_age


# **Sex**

# In[ ]:


# Sex - Create dummy variables
#combdata["Sex"] = combdata["Sex"].map({"male": 0, "female":1}) or
combdata = pd.get_dummies(combdata, columns = ["Sex"])


# **Creating Family Size variable using SibSp & Parch**

# In[ ]:


# Create a variable representing family size from SibSp and Parch
combdata["Fsize"] = combdata["SibSp"] + combdata["Parch"] + 1

# Create new feature of family size
combdata['Single'] = combdata['Fsize'].map(lambda s: 1 if s == 1 else 0)
combdata['SmallF'] = combdata['Fsize'].map(lambda s: 1 if  s == 2  else 0)
combdata['MedF'] = combdata['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
combdata['LargeF'] = combdata['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


# Analyze the Survival Probability by Fsize

g  = sns.factorplot(x="Fsize",y="Survived",data=combdata,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Survival Probability")
combdata[["Fsize", "Survived"]].groupby(['Fsize']).mean().sort_values(by='Survived', ascending=False)


# **SibSp**

# In[ ]:


# SibSp - Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["SibSp"])


# **Parch**

# In[ ]:


# Parch - Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["Parch"])


# **Ticket**

# In[ ]:


# Ticket - Extracting the ticket prefix. This might be a representation of class/compartment.
# If there is no prefix replace with U (Unknown). 

Ticket = []
for i in list(combdata.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("U")
        
combdata["Ticket"] = Ticket
combdata["Ticket"].unique()

combdata = pd.get_dummies(combdata, columns = ["Ticket"], prefix="T")


# **Fare**

# In[ ]:


# Fare - Check the number of missing value
combdata["Fare"].isnull().sum()


# In[ ]:


# Only 1 value is missing so we will fill the same with median
combdata["Fare"] = combdata["Fare"].fillna(combdata["Fare"].median())


# In[ ]:


# Use the numpy fuction log1p which  applies log(1+x) to all elements of the column
combdata["Fare"] = np.log1p(combdata["Fare"])

#Check the new distribution 
# sns.distplot(combdata['Fare'] , fit=norm);


# In[ ]:


# Cabin - Replace the missing Cabin number by the type of cabin unknown 'U'
combdata["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'U' for i in combdata['Cabin'] ])


# In[ ]:


# Let's plot the survival probability by Cabin
g  = sns.factorplot(x="Cabin",y="Survived",data=combdata,kind="bar", size = 7 ,
                    palette = "muted",order=['A','B','C','D','E','F','G','T','U'])
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[ ]:


# Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["Cabin"], prefix="Cabin")


# **Embarked**

# In[ ]:


# Embarked - Find the number of missing values
combdata["Embarked"].isnull().sum()

# Fill Embarked missing values of dataset set with mode 'S'
combdata["Embarked"] = combdata["Embarked"].fillna("S")

# Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["Embarked"], prefix="Emb")


# **Developing a Neural Network model**

# In[ ]:


## Separate train dataset and test dataset using the index variable 'source'

train_df = combdata.loc[combdata['source']=="train"]
test_df = combdata.loc[combdata['source']=="test"]
test_df.drop(labels=["Survived"],axis = 1,inplace=True)

train_df.drop(labels=["source"],axis = 1,inplace=True)
test_df.drop(labels=["source"],axis = 1,inplace=True)

test_df.info()


# In[ ]:


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train_df.drop(labels = ["Survived"],axis = 1)

X_train.info()


# In[ ]:


import keras 
from keras.models import Sequential # intitialize the ANN
from keras.layers import Dense      # create layers

# Initialising the NN
model = Sequential()

# layers
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 75))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_train, Y_train, batch_size = 32, epochs = 200)


# **Predict using Neural Network**

# In[ ]:


y_pred = model.predict(test_df)
y_final = (y_pred > 0.5).astype(int).reshape(test_df.shape[0])
output = pd.DataFrame({'PassengerId': IDtest['PassengerId'], 'Survived': y_final})
output.to_csv('Neural Network Prediction.csv', index=False)


# **Conclusion**
# 
# Title, Sex_Female, Fare & PClass seems to be common features preferred for classification.
# 
# While Title & Age feature represents the Age category of passengers the features like Fare, PClass, Cabin etc. represents the economic status. 
# 
# Based on our findings we can conclude that Age, Gender & features representing social/economic status were primary factors affecting the survival of passenger.
# 
# **If you like this notebook or find this notebook helpful, Please upvote and/or leave a comment**
