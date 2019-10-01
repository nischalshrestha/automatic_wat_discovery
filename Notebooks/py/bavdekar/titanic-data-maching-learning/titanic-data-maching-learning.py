#!/usr/bin/env python
# coding: utf-8

# # **This is my introduction to Kaggle. In this work, I will try out different ML and data analysis algorithms, and explain my thought process.**

# # **Table of Contents**
# 1. **Introduction** 
# 1. ** Exploratory data analysis **
#     1. Data Cleaning    
# 1. ** Machine Learning Algorithms **
#     1.     Fisher Discriminant Analysis or Linear Discriminant Analysis

# # **Introduction**
# The Titanic is probably the world's most famous (and notorious) passenger ship. Almost no one remembers Christopher Columbus' ship's name or Vasco-da-Gama's ship's name, but a huge majority of the world knows the Titanic. The meticulous record keeping of the British and American shipyards has given us this opportunity to practice our data analysis skills, and to a certain extent our Markdown language skills too

# Import the libraries

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # generally required for plotting, as well as seaborn
import seaborn as sns # seaborn for data visualization


# # Read the training data as a dataframe
# 1. Check the head of the dataframe
# 2. Check the information of the dataframe
# It is from this training data, that we will reconstruct the probabilities of survival of passengers
# 

# In[2]:


titanic_data = pd.read_csv('../input/train.csv')


# In[3]:


titanic_data[titanic_data['Parch']!=0][:5]


# In[4]:


titanic_data.head(n=7)


# In[5]:


titanic_data.info()


# The above information shows us that there are possibly missing data in the Age, Cabin, and Embarked columns. We now check whether this is true or not. For that we use seaborn's heatmap

# In[6]:


sns.heatmap(titanic_data.isnull(),cbar=False,yticklabels=False,cmap='plasma')


# From the above figure, we can confirm that there is some data missing from the 'Age' column and a lot of data missing from the 'Cabin' column. Now, for purposes of any kind of modeling or machine learning, the 'Cabin' column isn't of any use since it points to the specific cabin number of the passenger. It can probably be safely dropped from the dataframe, which we will do at a later stage.

# # Data Cleaning
# 
# **Missing Data**
# 
# We now try to fill the missing data in the 'Age' column. We can either drop all those data points (which will result in loss of information) or substitute the missing values with a statistical value. We chose the latter, since we don't want to lose information. We have the following options of substitution:
# 1. Fill up all missing data with either the mean or median of the 'Age' column. However, this would mask quite a bit of demographic information, since the mean or median will get distributed across all classes equally.
# 1. Fill in the missing data based on 'Sex' of the passenger. So, female passengers' age is substituted using the mean or median age of females, and male passengers' age is substituted with the mean or median of mens' age.
# 1. Fill in the missing data with the mean or median age of passengers in each class. This puts each passenger close to the mean or median age of their fellow passengers.

# We now look at the distributions of ages of males and females to determine if mean or median is a better metric. Note that I will be making a choice based on heuristics and visual inspection, i.e. if the distribution appears fairly symmetric, I will choose mean.

# In[7]:


fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(12,8))
sns.distplot(titanic_data[(titanic_data['Sex']=='female') & (titanic_data['Age'].isnull()==0)]['Age'],bins=30,color='red',ax=axes[0])
axes[0].set_title('Distribution of age for females')
sns.distplot(titanic_data[(titanic_data['Sex']=='male') & (titanic_data['Age'].isnull()==0)]['Age'],bins=30,color='blue',ax=axes[1])
axes[1].set_title('Distribution of age for males')


# The above two graphs look fairly symmetric, and hence using the mean of the ages of each sex to fill up the missing data, wouldn't unusually skew the results.

# The mean of ages of males and females are

# In[8]:


print("Mean age of males: %d"%np.round(np.nanmean(titanic_data[titanic_data['Sex']=='male']['Age'])))
print("Mean age of females: %d" %np.round(np.nanmean(titanic_data[titanic_data['Sex']=='female']['Age'])))      


# np.nanmean() computes the mean of the column entries by ingorning the NaN values.

# We can indeed go one step further, and look at the distribution of the ages of males and females in each class

# In[9]:


fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12,12))
for i in range(3):
    aax = axes[0,i]
    sns.distplot(titanic_data[(titanic_data['Sex']=='female') & (titanic_data['Age'].isnull()==0) & (titanic_data['Pclass']==i+1)]['Age'],color='red',bins=20, ax=aax)
    aax.set_title('Female age in class %d'%(i+1))
    aax = axes[1,i]
    sns.distplot(titanic_data[(titanic_data['Sex']=='male') & (titanic_data['Age'].isnull()==0) & (titanic_data['Pclass']==i+1)]['Age'],color='blue',bins=20, ax=aax)
    aax.set_title('Male age in class %d'%(i+1))


# The mean ages of males and females in each class are

# In[10]:


male_mean_age_class = []
female_mean_age_class = []
for i in range(3):
    mage = np.nanmean(titanic_data[(titanic_data['Sex']=='male') & (titanic_data['Pclass']==(i+1))]['Age'])
    male_mean_age_class.append(np.round(mage))
    print("The mean age of males in Passenger class %d"%(i+1)+" is %d"%np.round(mage))
for i in range(3):
    fage = np.nanmean(titanic_data[(titanic_data['Sex']=='female') & (titanic_data['Pclass']==(i+1))]['Age'])
    female_mean_age_class.append(np.round(fage))
    print("The mean age of females in Passenger class %d"%(i+1)+" is %d"%np.round(fage))


# So, the mean age computed for each sex across the entire ship, regardless of passenger class, is quite different from the mean ages of men and women for each class. Thus, it is more beneficial to substitute the missing values of age with the mean age of the passengers' gender and the class they were traveling in.

# We now write a function that will take in the passenger age, sex, and class. If the passenger age column has a null value (i.e. empty or NaN) it will substitute that with the mean age corresponding to the passenger's sex and class. We will then apply this function to the Titanic age data and eliminate the missing data

# In[11]:


def sub_age(data,male_age,female_age):
    age = data[0] # age of passenger
    s = data[1] # sex of passenger
    pclass = data[2] # Travel class of passenger
    if pd.isnull(age)==1:
        if s == 'male':
            age_subs = male_age[pclass-1]
        else:
            age_subs = female_age[pclass-1]
    else:
        age_subs = age
    return age_subs


# In[12]:


titanic_data['Age'] = titanic_data[['Age','Sex','Pclass']].apply(sub_age,axis=1,args=(male_mean_age_class,female_mean_age_class))


# We can now see if there are any missing data in the Age column, by using the same heat map

# In[13]:


sns.heatmap(titanic_data.isnull(),cbar=False,yticklabels=False,cmap='plasma')


# In[14]:


titanic_data.drop('Cabin',axis=1,inplace=True)


# Machine learning and analysis algorithms require its data to be in numerical format. Now, since Sex is a binary variable, i.e. since a passenger is either male or female, and is a likely contributing factor to the surviviability of the passenger, we need to include it in the data analysis. I will now convert the Sex column to a binary variable. I will maintain the original Sex column just in case it is needed later

# In[15]:


SexBin = pd.get_dummies(titanic_data['Sex'],drop_first=True)


# In[16]:


titanic_data = pd.concat([titanic_data,SexBin],axis=1)


# In[17]:


titanic_data.head()


# We will now determine which variables are important to predict whether a passenger has survived or not. This is done using a pairplot

# In[18]:


sns.heatmap(titanic_data.corr(),annot=True,cmap='YlGnBu')


# The above graph shows a strong positive correlation between survived and "fare", and a strong negative correlation between survived and "male", "Pclass". There is a weak positive correlation between survived and "Parch", and a weak positive correlation between survived and "SibSp", "Age". This is mostly along expected behaviour. It is well known that the rescue favoured Class 1 passengers over Classes 2 and 3. Hence, the higher the fare, the more you had a chance of surviving. The negative correlation between Pclass and Survived is due to the fact that Pclass 1 is higher than Pclass 2, which is higher than Pclass 3. Similarly due to the "Women and children first" approach of rescue, meant that males are less likely to survive than females.

# **Fisher Discriminant Analysis (FDA) OR Linear Discriminant Analysis (LDA)**
# 
# FDA/LDA is a classification approach  tries to minimize the within class discrimination and maximize the inter-class discrimination. FDA/LDA is a supervised algorithm, i.e. during training it needs to be explicitly told, which data belongs to which class. We will use the FDA here to first build a classification model, and then use that model to predict if the passenger in the test data survived or not.

# In[19]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
titanic_lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True)


# We now split the data in to its components. For FDA, the output is the class, while the inputs are the variables or attributes. In this case, there are two classes: 'Survived', denoted by 1, and 'Not Survived', denoted by 0.

# In[20]:


y_train = titanic_data['Survived']
X_train = titanic_data[['Fare','Pclass','male','Parch','SibSp','Age']]


# In[21]:


titanic_lda.fit(X_train,y_train)


# We now load the test data

# In[22]:


titanic_test_data = pd.read_csv("../input/test.csv")
titanic_test_data.head()


# In[23]:


sns.heatmap(titanic_test_data.isnull(),cbar=False,yticklabels=False,cmap='plasma')


# In[24]:


titanic_test_data['Age'] = titanic_test_data[['Age','Sex','Pclass']].apply(sub_age,axis=1,args=(male_mean_age_class,female_mean_age_class))


# In[25]:


sns.heatmap(titanic_test_data.isnull(),cbar=False,yticklabels=False,cmap='plasma')


# In[26]:


SexBint = pd.get_dummies(titanic_test_data['Sex'],drop_first=True)
titanic_test_data = pd.concat([titanic_test_data,SexBint],axis=1)
titanic_test_data.info()


# The above information shows that there is one missing data point in the 'Fare' cloumn, and quite a few missing data points in the 'Cabin' column. Now, we aren't using  Cabin in our machine learning, so we will ignore that. The missing data in 'Fare' is replaced by the mean of the fare paid for that particular class.

# In[27]:


titanic_test_data.head()


# In[28]:


fare_mean_per_class = []
for i in range(3):
    mfare = np.nanmean(titanic_test_data[titanic_test_data['Pclass']==i+1]['Fare'])
    mfare= (np.around(mfare,decimals=4))
    fare_mean_per_class.append(mfare)


# In[29]:


def sub_faref(data,fare_mean,i):
    farep = data[0]
    pclass = int(data[1])
    if pd.isnull(farep)==True:        
        subs_fare = fare_mean[pclass-1]        
    else:
        subs_fare = farep        
    return subs_fare


# In[30]:


titanic_test_data['Fare'] = titanic_test_data[['Fare','Pclass']].apply(sub_faref,axis=1,args=(fare_mean_per_class,1))


# In[31]:


X_test = titanic_test_data[['Fare','Pclass','male','Parch','SibSp','Age']]


# In[32]:


X_test.head()


# In[33]:


y_pred = titanic_lda.predict(X_test)
y_pred = np.array(y_pred)


# In[34]:


survivor_pred = pd.DataFrame({'Survived':y_pred,
                             'PassengerId':titanic_test_data['PassengerId']})
survivor_pred.head(n=10)


# We now load the example submission. Now, I know that this may not be the exact submission, but it allows me to compare my model predictions, with that of the example submission. So, if this example is an accurate submission, I'll have an accurate assessment of my model. Nevertheless, this allows me to practice my skills of model evaluations.

# In[35]:


true_survive = pd.read_csv("../input/gender_submission.csv")
true_survive.head(n=10)


# In[36]:


from sklearn.metrics import classification_report,confusion_matrix


# First, we look at the classification report, which is indicative of the following. Precision denotes the ratio of the true positives (tp) to the total number of positives (tp + fp, where fp=false positives). Recall denotes the ratio of tp to total number of actual positives, i.e. true positives + false negatives (fn). Thus recall ratio is an indicator of the accuracy of the model.

# In[37]:


print(classification_report(true_survive['Survived'],survivor_pred['Survived'],target_names=['Not Survived','Survived']))


# From the classification report, we can see that the LDA model is pretty accurate in classifying the passengers in to whether they survived or not.
# 
# We now look at the confusion matrix, which gives the number of true negatives in (0,0), true positives in (1,1), false positives in (0,1), and false negatives in (1,0). We want the numbers in (0,0) and (1,1) to be as close to the true values as possible.

# In[38]:


fig,ax=plt.subplots(figsize=(12,8))
sns.heatmap(confusion_matrix(true_survive['Survived'],survivor_pred['Survived']),annot=True)
ax.set_title('Confusion Matrix for LDA')


# The heatmap of the confusion matrix reinforces the accuracy of the LDA model in classification of the passengers in to whether they survived or not.
# 
