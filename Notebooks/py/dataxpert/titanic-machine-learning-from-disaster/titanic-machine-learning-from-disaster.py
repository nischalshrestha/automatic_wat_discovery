#!/usr/bin/env python
# coding: utf-8

# # Titanic- Machine Learning from Disaster using Python

# ## Introduction
# 
# This predictive analysis, using Python data analysis and machine learning libraries, is for Kaggle's competition for **"Titanic - Machine Learning from Disaster"**. This is my first kernel for Kaggle.
# 
# 
# ## Background
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# ## Problem Statement
# 
# Analyse what sorts of people were likely to survive the Titanic disaster. In particular, apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# 
# ## Notebook Content
# 
# This Notebook contains the following key elements of Titanic data analysis.
# 
#   - **Exploratory Data Analysis**
#   - **Data Wranggling / Data Munging**
#   - **Data Imputation & Data Cleanup**
#   - **Feature Engineering**
#   - **Predictive Analysis / Machine Learnining**
#   - **Evaluation of Machine Learning Models**
#   - **Data Preparation for Kaggle Submission**
#  
# ## Titanic Data
# 
# ### Overview
# 
# The data has been split into two groups:
# 
# training set (train.csv)
# test set (test.csv)
# The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
# 
# The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 
# ### Data Dictionary - Variables, Definitions and Keys
# 
#    - **survival**
#       - Survival (0 = No, 1 = Yes)
#    - **pclass**
#       - Passengers' class of travel	(1 = 1st, 2 = 2nd, 3 = 3rd)
#    - **sex**
#       - Gender
#    - **Age**
#       - Age in years	
#    - **sibsp**
#       - Number of siblings / spouses aboard the Titanic	
#    - **parch**
#       - Number of parents / children aboard the Titanic	
#    - **ticket**
#       - Ticket number	
#    - **fare**
#       - Fare paid by passenger	
#    - **cabin**
#       - Cabin number
#    - **embarked**
#       - Port of Embarkation	(C = Cherbourg, Q = Queenstown, S = Southampton)
# 
# ### Variable Notes
# 
#    - **pclass**: A proxy for socio-economic status (SES)
#      - 1st = Upper
#      - 2nd = Middle
#      - 3rd = Lower
# 
#    - **age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
#    - **sibsp**: The dataset defines family relations in this way
#    - **Sibling** = brother, sister, stepbrother, stepsister
#    - **Spouse** = husband, wife (mistresses and fiancés were ignored)
#    - **parch** : The dataset defines family relations in this way
#       - *Parent*= mother, father
#       - *Child* = daughter, son, stepdaughter, stepson
#       - Some children travelled only with a nanny, therefore parch=0 for them.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# ## 1. Loading Python Libraries

# In[ ]:


# Basic / Essential Set of Python Library

import pandas as pd                         # Dataframe / data analysis
import numpy as np                          # Linear Algebra functionality
import seaborn as sns                       # Data visualization
import matplotlib.pyplot as plt             # Data visualization  

sns.set_style('whitegrid')                  # Default sytle for Seaborn / Matplotlib Visualization libraries

get_ipython().magic(u'matplotlib inline')

#Machine Learning libararies and other modules will be loaded and displayed within their respective
#sections for ease of understanding and continuity 


# ## 2. Data Loading

# In[ ]:


titanic = pd.read_csv("../input/train.csv")    # Loading of 'train.csv' data file in 'titanic" dataframe


# ## 3. Initial Data Exploration

# ### Dataframe Shape

# In[ ]:


print("'titanic' dataframe shape:  ", titanic.shape)  # To get a quick count of number of columns and rows in the datasets


# It shows there are **891** rows and **12** columns in **"titanic"** dataset.

# ### Dataframe Features / Columns

# In[ ]:


titanic.info()   # Concise summary of combined 'titanic' dataset containing column names and data types


# ### Data Overview

# In[ ]:


titanic.head()          # This provides an overview of first five rows of titanic dataset


# ### Descriptive Statistics

# In[ ]:


titanic.describe(include='all')     # To generate descriptive statitics of combined titanic dataset.


# ## 4. Exploratory Data Analysis, Data Munging and Feature Engineering

# Before we perform detailed analysis, let's perform Data Munging on *titanic* dataset.
# 
# 
# ### Missing Data Identification

# In[ ]:


titanic.isnull().sum()          # To get the count of missing data in each column


# This clearly shows that:
#   - **"Age"** column has **177** missing values 
#   - **"Cabin"** columns has significant large number of missing data, i.e. **687** 
#   - **"Embarked"** column has only **2** mising records.
# 
# Following graphical representation of the missing data in each column. Missing data is repsentated by the short horizontal lines.

# In[ ]:


plt.figure(figsize=(8,5))
sns.set_style('whitegrid')
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False)   # Visualization of missing data
plt.title('Graphical Representations of Null Values in Titanic Dataframe')


# After identifying the missing data, the next step is missing data imputation, detailed data analysis and feature engineering. 
# 
# 
# ### Embarked
# 
# Let's take the easiest column, **Embarked** first to populate the missing values. 
# 
# As shown in the above-mentioned missing data stats that there are only two values missing in the *Embarked* column. These two instances are for two ladies, who travelled in "Class 1", together on the same ticket and even shared the Cabin (B28).  
# 
# As they were travelling in "Class 1", we can use "groupby" statement to get the port from where most "Class 1" travellers boarded, which is **"S"** or Southampton Port (where **127** passengers from Class-1 travellers were boarded). 
# 
# This further confirms by the *"Describe"* table shown above that **"S"** is the most common value in "Embarked" column. Therefore, the missing *Embarked* value will be populated with **"S"**.

# In[ ]:


titanic[titanic['Embarked'].isnull()]    # This will give us record where there are missing values in "Embarked" column


# In[ ]:


titanic[(titanic['Pclass']==1)].groupby('Embarked')['Pclass'].count() # Shows the number of 1st Class passengers by Embarked


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(data=titanic, x='Pclass', hue='Embarked')
plt.title('Number of Passengers by Pclass from Port of Emabarkation')


# In[ ]:


titanic['Embarked'].fillna('S',inplace=True) # To impute missing values in "Embarked" column with "S"


# In[ ]:


# This will give us the number of missing values within "Embarked" column AFTER fillin

titanic[(titanic['PassengerId'] == 62) | (titanic['PassengerId'] == 830)]   


# Now that all missing values in *Embarked* column have been populated, let's analyse it in detail. 

# In[ ]:


plt.figure(figsize=(10,5))
sns.pointplot(x='Embarked', y='Survived',data=titanic, color='g')  # Overall survival based on port of embark
plt.title('Survival vs. Port of Embarkation')


# In[ ]:


sns.factorplot('Embarked','Survived', col='Pclass',data=titanic) # Survival based on Embarked vs. Pclass


# In[ ]:


sns.factorplot('Embarked','Survived', hue='Sex', data=titanic, palette='RdBu_r') # Surivial based on port of Embark and gender
plt.title('Survival vs. Port of Embarkation based on Gender')


# In[ ]:


plt.figure(figsize=(10,5))
sns.violinplot(data=titanic, x='Embarked', y= 'Age', hue='Sex',palette='RdBu_r',split=True)
plt.title('Port of Embarkation vs. Age & Gender')


# ### Age
# 
# "Age" column has **177** missing values. But before we populate the missing values, let's first draw a histogram of **Age** with null values. The following histogram shows the mean age is approxiatemly 30 years (taking into account the missing values). The graphs shows that most of the passengers were between **18** and **40** years old.
# 

# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(titanic['Age'].dropna(),bins=50,color='blue',kde=True)
plt.title('Age Distribution')


# In[ ]:


print ("Mean Age:   ", titanic.Age.dropna().mean(), "years")   # Calculates the mean age
print ("Median Age: ", titanic.Age.dropna().median(), "years") # Calculates the mediam age


# There are various ways these missing values can be populated. One common method is to use the average age (mean / median) of passengers, which is approximately 30 and 28 years respectively. However, for this analysis, we cannot not merely use the overall average age because the **"Age"** feature is dependent on many other aspects such as passengers' class **"Pclass"** and **"Sex"**. Let's draw some boxplots to determine average age per passenger class. 
# 
# 
# The first graph (on left-side) shows simple boxplot for **Age** against **Pclass** without taking into account **Sex** aspect. The second graph shows age averages based on passengers' gender. 
# 

# In[ ]:


fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(15,7))
sns.violinplot(x = 'Pclass', y = 'Age', data=titanic,palette='rainbow', ax=ax1)
ax1.set(title='Age vs. Class of Travel')

sns.boxplot(x = 'Pclass', y = 'Age', data=titanic, hue='Sex',palette='RdBu_r', ax=ax2)  
ax2.set(title='Age vs. Class of Travel - Based on Gender')
#sns.despine()


# We can improve these averages based on passengers' class of travel (**Pclass**) and gender (**Sex**) and a combination of both. 

# In[ ]:


# Overall average Age per Class
print("Class 1, Overall average age: ",(titanic[titanic['Pclass']==1])['Age'].mean())
print("Class 2, Overall average age: ",(titanic[titanic['Pclass']==2])['Age'].mean())
print("Class 3, Overall average age: ",(titanic[titanic['Pclass']==3])['Age'].mean())


# In[ ]:


# Average Age per Class based on Gender
print("Class 1, Male average age  : ",(titanic[(titanic['Pclass']==1) & (titanic['Sex']== 'male')])['Age'].mean())
print("Class 1, Female average age: ",(titanic[(titanic['Pclass']==1) & (titanic['Sex']== 'female')])['Age'].mean())
print("Class 2, Male average age  : ",(titanic[(titanic['Pclass']==2) & (titanic['Sex']== 'male')])['Age'].mean())
print("Class 2, Female average age: ",(titanic[(titanic['Pclass']==2) & (titanic['Sex']== 'female')])['Age'].mean())
print("Class 3, Male average age  : ",(titanic[(titanic['Pclass']==3) & (titanic['Sex']== 'male')])['Age'].mean())
print("Class 3, Female average age: ",(titanic[(titanic['Pclass']==3) & (titanic['Sex']== 'female')])['Age'].mean())


# ### Name

# However, these averages cannot be used as is to impute missing values in *Age* column. 
# 
# Firstly, there were good number of children onboard and we cannot assign above-mentioned average age (e.g. 30 or 28 years) to a child!
# 
# But in the given data, there is no single column/feature that can differenciate child from adult passengers, except **Age**, which is what we are trying to impute. Hence, we need to perform **feature engineering** to extract new feature(s). If we closely have a look, one good candidate is passengers' "**Name**". We can get a count of onboard children based on the *titles/salutation* in passengers' names and their ages. This will also help us in further analysis and missing data imputation apart from child passengers.
# 
# We can use **Regex(Regular Expressions)** on **"Name"** column to create a new feature/column "**Title**".
# 

# In[ ]:


titanic['Title'] = titanic.Name.str.extract('([A-Za-z]+)\.',expand=True)


# Let's have a quick look at the top three rows to confirm that new feature is added with correct values.

# This all looks good. Let's reconfirm this by running a 'value_counts' statement on "**Title**" to check all possible values that we have.

# In[ ]:


titanic['Title'].value_counts()


# The problem with above-mentioned stats for titles is that the only indicator for a child is the title, "**Master**" (for a male child). As adult females a well as girl child will have the same title "**Miss**", "**Ms.**", "**the Countess**", "**Mme**","**Mlle**", its obvious that *Title* alone cannot be used to determine female child.
# 
# Therefore, we rely only on title is "**Master**" to use to determine a Child or adult.

# We can calculate the mean age of all missing values for *Age* based on all titles. However, we can combine most of these titles into "Male", "Female", "Child", and "Others" which can then be used to apply average age into missing values.

# In[ ]:


pd.crosstab(titanic['Title'],titanic['Sex'],margins=True)


# In[ ]:


# Replacing titles to reduce overall times to Child, Mr, Mrs, Miss, and Other

titanic['Title'].replace(['Master','Ms','Mlle','Mme','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],            ['Child','Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:


titanic.Title.value_counts()  # Rechecking whether the changes are made correctly


# Now we will calculate passengers' mean age based on the new feature **Title**.

# In[ ]:


titanic.groupby('Title')['Age'].mean()   # Mean age based on the engineered feature "Title"


# Although we can use the above-mentioned average age to impute missing values. However, passengers' travel class also plays an important role in determining the survival. Hence, let's recalculate the mean age based on "Title" and "Pclass".

# In[ ]:


titanic.groupby(['Title','Pclass'])['Age'].mean()   # Mean age based on "Title" and "Pclass"


# In[ ]:


# Let's round the above mentioned values to the nearest whole number before imputing into missing/null values of Age column
round(titanic.groupby(['Title','Pclass'])['Age'].mean())


# In[ ]:


sns.factorplot(data=titanic, x='Title', col = 'Pclass',kind='count',hue='Survived')


# ### Data Imputation for Age

# In[ ]:


"""
The following function will be used to populate missing / null values of Age column that are calculated above.

"""

def age_fix(cols):
    
    Age = cols[0]
    Pclass = cols[1]
    Title = cols[2]
    
    if pd.isnull(Age):
        
        if Pclass == 1 and Title == 'Child':
            return 5
        elif Pclass == 2 and Title == 'Child':
            return 2
        elif Pclass == 3 and Title == 'Child':
            return 5
        
        elif Pclass == 1 and Title == 'Miss':
            return 30
        elif Pclass == 2 and Title == 'Miss':
            return 23
        elif Pclass == 3 and Title == 'Miss':
            return 16
        
        elif Pclass == 1 and Title == 'Mr':
            return 42
        elif Pclass == 2 and Title == 'Mr':
            return 33
        elif Pclass == 3 and Title == 'Mr':
            return 29
        
        elif Pclass == 1 and Title == 'Mrs':
            return 41
        elif Pclass == 2 and Title == 'Mrs':
            return 34
        elif Pclass == 3 and Title == 'Mrs':
            return 34
        
        elif Pclass == 1 and Title == 'Other':
            return 51
        elif Pclass == 2 and Title == 'Other':
            return 43
              
        else:
            return Age
    else:
        return Age
    


# Finally, let's use the **age_fix** function to impute missing values in *Age* column

# In[ ]:


titanic['Age'] = titanic[['Age','Pclass','Title']].apply(age_fix,axis=1) #The "age_fix" function is applied to "titanic" dataset


# Let's reconfirm that there are no more missing / null values in *Age* column.

# In[ ]:


titanic.isnull().sum()


# In[ ]:


sns.factorplot(x='Pclass',y='Survived',col='Title',data=titanic)


# ### 'Age' Feature Engineering
# 
# Now that we have populated missing values in *Age* feature and have conducted some preliminary analysis, we need to convert *'Age'* to a categorical feature. To achieve this, we need to peform feature engineering to create new feature that can be used for predictive analysis (i.e. machine learning). 
# 
# Let's start by getting the describive stats and drawing some tables and graphs.

# In[ ]:


titanic['Age'].describe(include=all)


# In[ ]:


titanic['Age'].plot(kind='hist',bins=30,xlim=(0,75),figsize=(12,4))


# Let's create a new feature **'AgeBins'** which will have 5 equal numbers of passengers based on their *Age*.

# In[ ]:


titanic['AgeBins'] = 0  # New feature "AgeBins" created and an initial value '0' is assigned to it


# In[ ]:


titanic['AgeBins']=pd.qcut(titanic['Age'],5)  # Divides data into five equal bins


# In[ ]:


titanic.groupby('AgeBins')['AgeBins'].count()    # Confirms the values in each bin


# In[ ]:


pd.crosstab(titanic['AgeBins'],titanic['Survived'],margins=True)


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='AgeBins',hue='Survived',data=titanic,palette='rainbow')  # AgeBins vs. Survival
plt.title('Survival vs. Age Bins')


# In[ ]:


pd.crosstab(titanic['AgeBins'],titanic['Survived'],margins=True)


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='AgeBins',hue='Sex',data=titanic,palette='RdBu_r')   # AgeBins vs. Gender (Sex)
plt.title('Age Bins vs. Gender')


# In[ ]:


plt.figure(figsize=(10,5))
sns.stripplot(data=titanic, x='Pclass', y= 'Age',size=7)        # Age vs. Pclass
plt.title('Age vs. Class of Travel')


# Now, lets create the utimate feature, **'NAge'** for *Age* that we will use in predictive analysis (machine learning).

# In[ ]:


titanic['NAge'] = 0  # Create a new feature 'NAge' and assign initial value '0'


# In[ ]:


titanic.loc[titanic['Age']<=19.00,'NAge']=0
titanic.loc[(titanic['Age']>19.00)&(titanic['Age']<=26.00),'NAge']=1
titanic.loc[(titanic['Age']>26.00)&(titanic['Age']<=30.00),'NAge']=2
titanic.loc[(titanic['Age']>30.00)&(titanic['Age']<=40.00),'NAge']=3
titanic.loc[(titanic['Age']>40.00)&(titanic['Age']<=81.00),'NAge']=4


# In[ ]:


titanic.groupby('NAge')['NAge'].count()   # Confirm values in 'NAge" feature after imputation


# #### Inshights
# 
#   - The youngest passenger was approximately **5 month** old (*0.42 years*)
#   - The eldest one was **80 years** old and was saved!
#   - Overall average age (mean) was **29.46** years.
#  - Being a Child will give the best chance of survival, especially if you are in Class 1 or Class 2.
#   - Female passengers (regardless of married or single) are also showing a similar pattern, i.e. best chances of survival in Class 1 and Class 2 whrease less in Class 3 but still better than Male passengers.
#   - Male passengers' had least chances of survival regardless of their age. 
#   

# ### Child
# 
# Now that **Age** column is completed populated with missing values, let's create another feature **"Child"** based on the criteria that all passengers with Age less than or equal to 16 years and title is not "Mrs". There were **"135"** passengers that we can termed as "Child" based on this criteria.
# 
# The reason for exclusion based on title "Mrs" is because there were two female passengers aged 14 and 15 and their titles were 'Mrs' (as shown below).
# 

# In[ ]:


titanic[(titanic['Age'] <= 16) & (titanic['Title'] == 'Mrs')]   # Child brides?


# In[ ]:


titanic[(titanic['Age'] <= 16) & (titanic['Title'] !='Mrs')]['Age'].count()   # Count of Child Passengers


# In[ ]:


titanic['Child'] = 0    # Creates a new feature "Child" and assigns initial value '0'


# In[ ]:


# Assigns value '1' to all Children based on the above-mentioned criteria
titanic.loc[(titanic['Age'] <= 16) & (titanic['Title'] !='Mrs'),'Child'] = 1 


# In[ ]:


titanic.Child.value_counts()   # Reconfirms that values have been successfully put


# In[ ]:


pd.crosstab(titanic['Child'],titanic['Survived'],margins=True)    # Survived vs. Child


# In[ ]:


sns.factorplot('Child','Survived',data=titanic)       # Children surived vs. died
plt.title('Children Survival')


# ### Cabin
# 
# The **Cabin** column has significantly large number of missing values, i.e. **687** which is approximately **77%** of the total data. There is a possibility to peform some **feature engineering** to extract meaniningful data from the available *Cabin* values. 
# 
# It looks like that the first character in the Cabin numbers is indicating the Deck, which might be useful in our analysis. Let's create a new feature/column **"Deck"** using the first character of Cabin Number.
# 

# In[ ]:


titanic['Deck'] = titanic['Cabin'].astype(str).str[0]  # Extracting first character in "Cabin" to create a new column "Deck"


# Let's quickly check first few rows in the *titanic* dataset to see how the new feature looks like. Please note that the "NaN' values will appear with a small **"n"**.

# In[ ]:


titanic.head(3)


# In[ ]:


titanic.Deck.value_counts()  # Gives the count for each value in the "Deck" column


# In[ ]:


pd.crosstab(titanic['Deck'],titanic['Survived'],margins=True)


# The above-mentioned value count in Deck column shows that although we can use the cabin initials **(A to G and T)** to create dummy variables. However, determining the Deck for missing **687** values will be very arbitrary. The crosstab table shows that only **30%** passengers with **"NaN"** values survived. However, we can perform **feature engineering** to create a new feature **"IsCabin"** to indicate whether or not the cabin is available.

# In[ ]:


titanic['IsCabin'] = 1 # Create a new feature "IsCabin" and assign a default value "1"


# In[ ]:


titanic.loc[titanic['Cabin'].isnull(),'IsCabin'] = 0  # Populate "IsCabin" with value '0' where "Cabin" is Null/NaN


# In[ ]:


titanic.loc[titanic['Cabin'].isnull(),'IsCabin'] = 0  # Populate "IsCabin" with value '0' where "Cabin" is Null/NaN


# In[ ]:


titanic['IsCabin'].value_counts()  # Calculate values in 'IsCabin' feature


# In[ ]:


sns.factorplot(x='IsCabin',y='Survived',col='Pclass',hue='Sex',data=titanic, palette='RdBu_r')


# ### Pclass
# 
# 
# Passengers' class of travel (**Pclass**) had significant impact on survival chances. Some of these have already been mentioned in the combination of other features, such as *Title*. Let's plot some more graphs to dipict the analysis on *Pclass*.
# 
# Let's first count the passengers travelled in each class and the number of passengers survived and died per class of travel.

# In[ ]:


pd.crosstab(titanic['Pclass'],titanic['Survived'],margins=True,) # Passengers survived vs. died based on Pclass feature


# In[ ]:


# Percentage of passengers travelled per class of travel
titanic['Pclass'].value_counts().plot.pie(explode=[0.02,0.02,0.02],autopct='%1.1f%%',figsize=(7,7))
plt.title('Passengers per Class of Travel (%age)')


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='Pclass',hue='Survived',data=titanic,palette='rainbow')    # Survived vs. Died per class of travel
plt.title('Survival vs. Class of Travel')


# In[ ]:


plt.figure(figsize=(8,5))
sns.factorplot(x='Sex',y='Survived',col='Pclass',data=titanic)


# This clearly shows that social status (procured through wealth) does play important role. Passengers in Class-1 had the highest survival rate, i.e. **63%**. Class-2 had almost equal ratio of survival vs. died (**47%** to be precise). Whrease mortality rate was very high in Class-3, i.e. only **24%** survived.

# ### SibSp
# 
# *SibSp* feature shows the number of siblings travelled together. Let's plot some graphs and analyse the data to determine it's impact on passengers' survival.

# In[ ]:


pd.crosstab(titanic['SibSp'],titanic['Survived'],margins=True) # Passengers survived vs. died based on SibSp feature


# In[ ]:


# Graphical representation of passengers survived vs. die based on Siblings and Spouses  
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
sns.pointplot('SibSp','Survived',hue='Pclass',data=titanic,ax=ax1)
ax1.set(title='Siblings & Spouses Survival based on Class of Travel')
sns.countplot(x='SibSp',data=titanic,hue='Survived',ax=ax2) 
ax2.set(title='Siblings & Spouses Survival')


# In[ ]:


pd.crosstab(titanic['SibSp'],titanic['Pclass'])


# Although it might be true that there is safety in numbers, Titanic passengers' data showed a strage pattern. 68% passengers were travelling alone, out of which only 35% survived! There is 54% survival rate of passengers with only one sibling or spouse followed for 46% survival rate for passengers with three siblings / spouse. From that point onwards, the survival rate diminishes as the SibSp number increases. 
# 
# But the most important phenomenon is that all large familities (more than 3 siblings / spouses) were travelleing in Class-3, which has very high mortality rate.
# 
# 

# ### Parch
# 
# *Parch* feature shows the number of parents and children travelled together. Let's plot some graphs and analyse the data to determine it's impact on passengers' survival.

# In[ ]:


pd.crosstab(titanic['Parch'],titanic['Survived'],margins=True) # Passengers survived vs. died based on Parch feature


# In[ ]:


# Graphical representation of passengers survived vs. die based on Siblings and Parch  
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
sns.pointplot('Parch','Survived',hue='Pclass',data=titanic,ax=ax1)
ax1.set(title='Parent/Child Survival - Based on Class of Travel')
sns.countplot(x='Parch',data=titanic,hue='Survived',ax=ax2, palette='rainbow') 
ax2.set(title='Parents & Children Survival')


# In[ ]:


pd.crosstab(titanic['Parch'],titanic['Pclass'])


# Parent-Child (*Parch*) feature shows a similar trend as we have observed in *SibSp* above. Overall, 76% passengers were travelling without any parent or child, out which 34% survived. There was 55% survival rate for the passengers who were travelleing with at leaset on child or parent and 50% survival rate is observed for those who were travelling with two children or both parents. From that point onwards, the number of passengers become very low and therefore insignificant who were travelling with more than three children or parents.
# 
# Moreover, exactly like SibSp, all large familities (more than 2 parents/child) were travelleing in Class-3, which has very high mortality rate.

# ### Feature Engineering on 'SibSp' and 'Parch' features
# 
# As observed in **SibSp** and **Parch** analysis, we can conclude that:
#   
#   - Most of the large families travelled in Class-3
#   - Most large familities did not surivive
#   - Most people that were travelling alone also did not survive
# 
# Hence, we can perform **feature engineering** to create two new features, i.e. **FamSize** and **Alone**. 
# 
# ### FamSize (Family Size)
# 
# **FamSize** will have the combination of *Parch* and *SibSp* values.
# 
# 

# In[ ]:


#Creating new feature "FamSize" by adding values in "SibSp" and "Parch"

titanic['FamSize'] = titanic['SibSp'] + titanic['Parch'] 


# Let's reconfirm the changes have been done successfully by running the *value_counts()* statement.

# In[ ]:


titanic.FamSize.value_counts()


# In[ ]:


pd.crosstab(titanic['Survived'],titanic['FamSize'],margins=True)  # Survival vs. family size feature


# ### Alone
# 
# **Alone** will have only two values:
# 
#   - '0' means not alone
#   - '1' mean alone

# In[ ]:


titanic['Alone'] = 0  # Creating a new feature "Alone" with default value = 0


# In[ ]:


titanic.loc[titanic['FamSize']== 0,'Alone'] = 1  # Populate "Alone" with value '1' where family size is '0'


# Let's reconfirm the changes have been done successfully by running the *value_counts()* statement.

# In[ ]:


titanic.Alone.value_counts()


# In[ ]:


pd.crosstab(titanic['Alone'],titanic['Survived'],margins=True)  # Survival vs. Alone feature


# In[ ]:


f,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
sns.pointplot('FamSize','Survived',data=titanic,ax=ax1)
ax1.set(title='Survival Based on Family Size')
sns.pointplot('Alone','Survived',hue='Sex',data=titanic,palette='RdBu_r',ax=ax2)
ax2.set(title='Survival vs. Alone Based on Gender')


# ### Sex
# 
# As we have observed in the above-mentioned analysis for various features that passengers' *gender* (**Sex**) played a key role in determining the chances of survival.
# 
# Overall, there were 35% females and 65% males onboard (based on 891 records) out of which 74% females and only 19% males survived the disaster! 
# 
# It will be interesting to draw some tabels and plot some graphs to see the relationship of *Sex* feature with other features.
# 

# In[ ]:


pd.crosstab(titanic['Sex'],titanic['Survived'],margins=True)  # Survival vs. gender (Sex)


# In[ ]:


f, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='Sex',data=titanic,hue='Survived',palette='rainbow',ax=ax1)
ax1.set(title='Survival Based on Gender')
sns.pointplot(x='Embarked',y='Survived',hue='Sex',data=titanic, palette='RdBu_r',ax=ax2)
ax2.set(title='Survival Based on Port of Embarkation & Gender')


# In[ ]:


sns.factorplot('Sex','Survived', col='Pclass',data=titanic,kind='bar',palette='RdBu_r') 


# ### Fare
# 
# Fare is a continuous feature that has verious diverse amounts regardless of class of travel. The highest fare paid by passenger was '**512.32**' and lowest fare was **zero**. There are three counts of the highest *512.32* fare, paid on the same ticket "PC 17755" and therefore it's most likely a typo. 
# 
# As it's obvious that higher class passengers had paid highest fare and because Class-1 passengers had highest survival rate, followed by Class-2, we can easily deduced that *Fare* has significant impact on determining chances of survival.
# 

# In[ ]:


titanic['Fare'].describe(include=all) # Descriptive stats for "Fare"


# In[ ]:


titanic[titanic['Fare'] >= 300]  # Passengers paid more than 300 


# Let' draw distribution charts for *'Fare'* to see the dispersion across the dataset.

# In[ ]:


titanic['Fare'].plot(kind='hist',bins=50,xlim=(-1,100),figsize=(12,4))
plt.title('Fare Distrubution')


# The historgram shows that *'Fare'* is right-skewed, i.e. it has a long tail on the right side. Most passengers paid between 5 to 15, which aligns with the higher number of Class-3 passengers. The values diminishes as we move towards right with occassional values in the higher bins. 
# 
# As *'Fare'* is a continuous variable, we have to convert it into categorical variable for our machine learning phase. Therefore, we have to perform yet another **feature engineering** to create a new feature **FareBins** that will have different ranges of Fare. 
# 
# 

# In[ ]:


titanic['FareBins']=pd.qcut(titanic['Fare'],4)  # Divides data into equal bins


# In[ ]:


titanic.groupby('FareBins')['FareBins'].count()  # Confirms the values in each bin


# In[ ]:


titanic['NFare'] = 0  # Creates a feature 'NFare' and assign an initial value '0'


# Now, let's assign a value (from 0 to 3) based on the *'FareBins'*

# In[ ]:


titanic.loc[titanic['Fare']<=7.91,'NFare']=0
titanic.loc[(titanic['Fare']>7.91)&(titanic['Fare']<=14.454),'NFare']=1
titanic.loc[(titanic['Fare']>14.454)&(titanic['Fare']<=31),'NFare']=2
titanic.loc[(titanic['Fare']>31)&(titanic['Fare']<=513),'NFare']=3


# In[ ]:


pd.crosstab(titanic['NFare'],titanic['Survived'],margins=True)  # Survived vs. NFare


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(x='FareBins',hue='Survived',data=titanic,palette='rainbow')
plt.title('Survival Based on Fare Bins')


# In[ ]:


sns.factorplot('Pclass','NFare',data=titanic, hue='Survived')
plt.title('Survival Based on Fare & Class of Travel')


# To conclude, after feature engineering, it's obvious that as the *NFare* value increases, the chances of survival also increase. This nicely complements with the "Pclass" vs. "NFare" that also proves that as the 'Pclass' increases 'NFare' also increases.

# ### Ticket
# 
# *Ticket* feature contains the ticket number of passengers. In some cases, it's a mix of strings and numbers whrease in most of the cases only numbers are given. Let's analyse it in detail to determine whether or not it has any impact on passengers' survival.

# In[ ]:


titanic['Ticket'].value_counts().head(20)   # A quick value count of "Ticket"


# Well, it looks like several passengers travelled on single ticket, which can be used to do extract a new feature "**SharedTicket**" to show whether or not a family travelled on a single. The values will be as follows:
#   - '0' means Individual Passenger / ticket not shared
#   - '1' means Shared Ticket (used by a group of passengers)

# In[ ]:


titanic['SharedTicket']= 0 # A new feature "FanTicket" created with initial value "0"


# Precisely, **547** passengers' travelled alone (*which aligned to FamSize=0 calculated above*) and remaining **344** passengers travelled on **134** shared tickets as calculated below.  

# In[ ]:


ticketV = titanic['Ticket'].value_counts()  #Calculates passengers groups on each tickets and assign it to a variable 'ticketV'
ticketV.head(2)


# In[ ]:


single = ticketV.loc[ticketV ==1].index.tolist()  # Creates a list of tickets used by individual(single) passemgers
multi  = ticketV.loc[ticketV > 1].index.tolist()  # Creates a list of tickets shared by group of passemgers


# In[ ]:


print("Number of Individual Tickets: ", len(multi))     # Prints individual tickets count
print("Number of Shared Tickets    : ", len(single))    # Prints shared tickets count


# It's time to plugin values (**'0'** *or* **'1'**) in **'SharedTicket'** feature based on **'single'** or **'multi'**. As we have already assigned the initial value *'0'* to "*SharedTicket*" feature, we will only plugin '*1*' for '*multi*' variable as calculated above.

# In[ ]:


# Compares the ticket number in the "multi" list that was created above with titanic dataset "Ticket" feature and plugin '1'
for ticket in multi:
    titanic.loc[titanic['Ticket'] == ticket, 'SharedTicket'] = 1
    


# In[ ]:


titanic['SharedTicket'].value_counts() # Checks the values in "SharedTicket" column to confirm the accuracy of imputation


# In[ ]:


pd.crosstab(titanic['SharedTicket'],titanic['Survived'],margins=True) # Survived vs. SharedTicket


# In[ ]:


f, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='SharedTicket',data=titanic,hue='Survived',palette='rainbow',ax=ax1)
ax1.set(title='Survival Based on Shared Ticket')
sns.pointplot(x='SharedTicket',y='Survived',hue='Sex',data=titanic, palette='RdBu_r',ax=ax2)
ax2.set(title='Survival Based on Shared Ticket & Gender')


# In[ ]:


pd.crosstab(titanic['SharedTicket'],titanic['Pclass'],margins=True) # Pclass vs. SharedTicket


# In[ ]:


# Survival based on SharedTicket vs. Pclass taking into account the gender (Sex)
sns.factorplot('SharedTicket','Survived', col='Pclass',hue='Sex',data=titanic, palette='RdBu_r') 


# Overall, **39%** passengers travelled on shared tickets out of which **52%** survived. Whrease **61%** passengers travelled on invididual tickets (alone) and only **30%** were survived. However, we still cannot deduce that travelleing together had more chances of survival. There are many other facts that played the role in determine survival. For example, class of travel (*'PClass'*). As most passengers in Class-1 and Class-2 survived, we can see in the above table (*PClass vs. SharedTicket*) that proportionately most shared ticket passengers (**49%**) were from Class-1 and Class-2 as compared to Class-3 **30%** passengers who were travelleing on the same ticket.

# ## 6. Data Cleaning & Dummy Variables

# Quite a lot has been done in Section-5. It's time to get our data cleansed and ready for predictive modeling. This will include creation of Dummy Variables, removing unwanted features and convertion of those features that contain non-numeric values to numerica values. But first let start with preliminary sanity checks.  

# In[ ]:


titanic.shape   # Our dataset now contains 23 features, most of which are not required for predictive modeling.


# In[ ]:


titanic.info()  # Overview of dataset features


# In[ ]:


titanic.head()   # First five rows of titanic dataset


# ### Creation of Dummy Variables
# 
# Let's first create Dummy Variable for **"Embarked"** and **"Sex"** features.
# 

# In[ ]:


int1 = titanic.copy()


# In[ ]:


int1.isnull().sum()


# In[ ]:


emb  = pd.get_dummies(titanic['Embarked'],drop_first=True) #Creates two Dummy Varable "Q" and "C" and drops the values for "S"   
nsex = pd.get_dummies(titanic['Sex'],drop_first=True)     #Creates Dummy Varable "male" and drops the values for Female


# Now, lets Concatenate the newly created Dummy Variables with titanic dataframe

# In[ ]:


titanic = pd.concat([titanic,emb],axis=1)  # Concatenate titanic dataset with emb
titanic = pd.concat([titanic,nsex],axis=1)  # Concatenate titanic dataset with nsex


# Let's check the shape of the titanic dataset after creating dummy variables, which shows that three new features have been added.

# In[ ]:


titanic.shape


# ### Convertion to Numeric Values

# **"Title"** feature contains strings that cannot be used for machine learning. We need to convert it to numeric values.

# In[ ]:


titanic['Title'].replace(['Mr','Mrs','Miss','Child','Other'],[0,1,2,3,4],inplace=True)


# ### Removal of Unwanted Features
# 
# We are not ready to cleanse data and remove all those features that we do not required. But, before we remove them, let's quickly have a look at all of them.

#     -'PassengerId'  : It's not required for our predictive modelling and hence will be removed.  
#     -'Survived'     : This is the "Target Variable" and therefore we will retain it.
#     -'Pclass'       : We will retain passenger's class of travel (Pclass).
#     -'Name'         : We have already extract other key features out of it. Hence, this will be removed.
#     -'Sex'          : We have created dummy variable "male" from it and hence, it will be removed.
#     -'Age'          : We have extracted "NAge" from it and hence it will be removed.
#     -'SibSp'        : We have combined it with "Parch" to create "FamSize" and "Alone" features. Hence, it will be removed. 
#     -'Parch'        : We have combined it with "SibSp" to create "FamSize" and "Alone" features. Hence, it will be removed.
#     -'Ticket'       : This will be removed as we extract "SharedTicket" from this feature.
#     -'Fare'         : This will be removed as we have extracted new feature "NFare" from it.
#     -'Cabin'        : This will be removed as we have extracted new feature "IsCabin" from it.
#     -'Embarked'     : This will be removed as we have created two dummy features ("Q' and "S") from it.
#     -'Title'        : This is our extracted feature and will be kept.
#     -'Deck'         : This was an interim feature that we extracted from 'Cabin' to create "IsCabin". Hence will be removed.
#     -'FamSize'      : This is our extracted feature and will be kept.
#     -'Alone'        : This is our extracted feature and will be kept.
#     -'IsCabin'      : This is our extracted feature and will be kept.
#     -'SharedTicket' : This is our extracted feature and will be kept.
#     -'FareBins'     : This is our extracted feature and will be kept.
#     -'NFare'        : This is our extracted feature and will be kept.
#     -'AgeBins'      : This was an interim feature that we extracted from Age to create "NAge". Hence will be removed.
#     -'Child'        : This is our extracted feature and will be kept.
#     -'Q'            : This is our extracted feature and will be kept.
#     -'S'            : This is our extracted feature and will be kept.
#     -'male'         : This is our extracted feature and will be kept.  

# In[ ]:


# Removes unwanted features
titanic.drop(['PassengerId','Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked',              'AgeBins','Deck', 'FareBins', ],inplace=True,axis=1)      


# ### Correlation Chart
# Now that we have the clean dataset, ready for predictive modelling, let's plot a final graph to see the correlation among all remaining features.

# In[ ]:


plt.figure(figsize=(15,7))
sns.heatmap(titanic.corr(),cmap='RdYlGn_r',annot=True)
plt.title('Titanic Correlation Chart')


# ## 7. Predictive Modelling / Machine Learning
# 
# Finally, our data is all set and we are eventually ready for Predictive Modelling / Machine Learning. We will use the following machine algorithms for Predictive Modelling:
# 
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. Support Vector Machine (Linear & rbc)
# 5. K-Nearest Neightbors
# 
# However, let's start to have a quick look at our final dataset.

# In[ ]:


titanic.shape


# In[ ]:


titanic.head(2)


# ### Creation of Predictor & Target Variables
# It all looks good. Let's now spilit the data into X (predictors) and y (target) variables.

# In[ ]:


y = titanic['Survived']                             # Target variable
X = titanic.drop('Survived',inplace=False,axis=1)   # Predictors


# In[ ]:


X.head(2)         # Header of the Predictor Variable (X)


# In[ ]:


y.head(2)       # Header of the Target Variable (y)


# ### Data Split - Training & Test Data
# Let's now split the data into Test and Train datasets (within *titanic* dataset).
# Let's import Scikit-Learn libarary and perform the split.
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)   # A 25-70 split of Test and Train data


# Let's confirm that count of train and test data split.

# In[ ]:


print ("'X_train', train data count : ", X_train.shape)
print ("'X_test', test data count   : ", X_test.shape)


# And, finally, let's apply different machine learning algorithms to determine the accuracy of our predictions.

# ### Logistic Regression

# #### Model Creation, Training and Prediction

# In[ ]:


from sklearn.linear_model import LogisticRegression      # Importing of Logistic Regression Library from Scikit-Learn
lr = LogisticRegression()                                # Creation of Logistic Regression Model    
lr.fit(X_train,y_train)                                  # Model Training
lr_pred = lr.predict(X_test)                             # Prediction based on X_test


# #### Validation - Logistic Regression

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report   # Importing of different metrics
print('\nAccuracy Score = ',accuracy_score(y_test,lr_pred))                         # Accuracy Score of the Model
print('\nConfusion Matrix:','\n',confusion_matrix(y_test,lr_pred))                  # Confusion Matrix
print('\nClassification Report:','\n',classification_report(y_test,lr_pred))        # Classification Report 


# ### Decision Tree Classifier

# #### Model Creation, Training and Prediction

# In[ ]:


from sklearn.tree import DecisionTreeClassifier          # Importing of Logistic Regression Library from Scikit-Learn
dt = DecisionTreeClassifier()                            # Creation of Decision Tree Classifier Model   
dt.fit(X_train,y_train)                                  # Model Training
dt_pred = dt.predict(X_test)                             # Prediction based on X_test


# #### Validation - Decision Tree

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report   # Importing of different metrics
print('\nAccuracy Score = ',accuracy_score(y_test,dt_pred))                         # Accuracy Score of the Model
print('\nConfusion Matrix:','\n',confusion_matrix(y_test,dt_pred))                  # Confusion Matrix
print('\nClassification Report:','\n',classification_report(y_test,dt_pred))        # Classification Report 


# ### Random Forest Classifier 

# #### Model Creation, Training and Prediction

# In[ ]:


from sklearn.ensemble import RandomForestClassifier      # Importing of Logistic Regression Library from Scikit-Learn
rfc = RandomForestClassifier(n_estimators=100)           # Creation of Random Forest Classifier Model  
rfc.fit(X_train,y_train)                                 # Model Training
rfc_pred = rfc.predict(X_test)                           # Prediction based on X_test


# #### Validation - Random Forest

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report    # Importing of different metrics
print('\nAccuracy Score = ',accuracy_score(y_test,rfc_pred))                         # Accuracy Score of the Model
print('\nConfusion Matrix:','\n',confusion_matrix(y_test,rfc_pred))                  # Confusion Matrix
print('\nClassification Report:','\n',classification_report(y_test,rfc_pred))        # Classification Report 


# ### Support Vector Machine (rbf)

# #### Model Creation, Training and Prediction

# In[ ]:


from sklearn.svm import SVC                              # Importing of Logistic Regression Library from Scikit-Learn
svr = SVC(kernel='rbf')                                  # Creation of Support Vector Machine (Radial) Model  
svr.fit(X_train,y_train)                                 # Model Training
svr_pred = svr.predict(X_test)                           # Prediction based on X_test


# #### Validation - SVM (Kernel= rbf)

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report    # Importing of different metrics
print('\nAccuracy Score = ',accuracy_score(y_test,svr_pred))                         # Accuracy Score of the Model
print('\nConfusion Matrix:','\n',confusion_matrix(y_test,svr_pred))                  # Confusion Matrix
print('\nClassification Report:','\n',classification_report(y_test,svr_pred))        # Classification Report 


# ### Support Vector Machine - Linear

# #### Model Creation, Training and Prediction

# In[ ]:


from sklearn.svm import SVC                              # Importing of Logistic Regression Library from Scikit-Learn
sv = SVC(kernel='linear')                                # Creation of Support Vector Machine (Linear) Model  
sv.fit(X_train,y_train)                                  # Model Training
sv_pred = sv.predict(X_test)                             # Prediction based on X_test


# #### Validation - SVM Linear

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report   # Importing of different metrics
print('\nAccuracy Score = ',accuracy_score(y_test,sv_pred))                         # Accuracy Score of the Model
print('\nConfusion Matrix:','\n',confusion_matrix(y_test,sv_pred))                  # Confusion Matrix
print('\nClassification Report:','\n',classification_report(y_test,sv_pred))        # Classification Report 


# ### KNearst Neighbors Classifier

# #### Model Creation, Training and Prediction

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier      # Importing of Logistic Regression Library from Scikit-Learn
knn = KNeighborsClassifier()                            # Creation of K-Nearest Neighbors Classifier Model 
knn.fit(X_train,y_train)                                # Model Training
knn_pred = knn.predict(X_test)                          # Prediction Based on X_train 


# #### Validation - KNN

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report    # Importing of different metrics
print('\nAccuracy Score = ',accuracy_score(y_test,knn_pred))                         # Accuracy Score of the Model
print('\nConfusion Matrix:','\n',confusion_matrix(y_test,knn_pred))                  # Confusion Matrix
print('\nClassification Report:','\n',classification_report(y_test,knn_pred))        # Classification Report 


# #### Finding Best Score in KNN (based on number of neighbors)
# 
# The default K-value (neighbors) is 5. Now, let's apply a range of K-values (from 1 to 20) and see how this impacts the overall accuracy score.

# In[ ]:


cnt = list(range(1,21))       # Range for the count of K-values
knn_score = []                # List to populate accurracy scores based on the K-value  

for i in cnt:                 # Loop to iterate through K-values, apply KNN Model and append the accuracy score to the list    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    kpred = knn.predict(X_test)
    score = accuracy_score(y_test,kpred)
    knn_score.append(score)

# Plotting the graph to dipict Accuracy Score vs. K-values

plt.figure(figsize=(12,5))
plt.plot(cnt,knn_score,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy Score vs. K Values')
plt.xlabel('K-Value')
plt.ylabel('Accuracy Score')


# ### Predictive Modelling Summary

# Finally, we are done! Let's summarize the accuarcy values as been calculated above after applying different Machine Learning algorithms to determine the best algorithm / model for our Predictive Modelling.

# In[ ]:


# Summarizing accuracy scores into a dictionery and plotting the graph
summary = {'Logistic Reg':accuracy_score(y_test,lr_pred),'DecisionTree':accuracy_score(y_test,dt_pred),           'RandomForest':accuracy_score(y_test,rfc_pred),'SVM-Linear':accuracy_score(y_test,sv_pred),           'SVM-rbc':accuracy_score(y_test,svr_pred),'KNN':accuracy_score(y_test,knn_pred)}
print(summary)
plt.figure(figsize=(12,5))
sns.pointplot(x=list(summary.keys()),y=list(summary.values()))
plt.title('Accuracy Scores vs. Predictive Models')
plt.xlabel('Predictive Models')
plt.ylabel('Accuracy Scores')


# By running mutiple times we have either **"Random Forest Classifier (RFC)"** or **"Support Vector Machines with rbc"**. I decided to use SVM rbc for my submission of Titanic's Kaggle data, i.e. **('test.csv')**.

# ## 8. Data Preparation and Submission for Kaggle Competition
# 
# Now that we have completed our analysis of the main data and have finalized the Machine Learning model, let prepare and submit the data for Kaggle competition ***"Titanic- Machine Learning from Disaster"***.
# 
# 
# ### Data Loading
# 
# Let's first load the ***test.csv*** data into **"test"** dataframe.

# In[ ]:


test = pd.read_csv("../input/test.csv")    # Loading of 'test.csv' data file in 'test" dataframe


# ### Data Imputation, Feature Engineering & Data Cleanup
# I have summarized all the codes that was used for missing data population, feature engineering and data clean-up of main data ***"train.csv"*** (that we used as **titanic dataset** above) all in one place.

# In[ ]:


# Missing data imputation, feature engineering and data cleanup

test.loc[test['Fare'].isnull(),'Fare'] = 35.63
test['Title'] = test.Name.str.extract('([A-Za-z]+)\.',expand=True)  # Extract 'Title' from 'Name"
# Replacing titles to reduce overall times to Child, Mr, Mrs, Miss, and Other
test['Title'].replace(['Dona','Master','Rev','Col','Dr','Ms'],['Miss','Child','Other','Other','Other','Miss'],inplace=True)
test.groupby(['Title','Pclass'])['Age'].mean()   # Mean age based on "Title" and "Pclass"

#Function to populate missing values in test dataset
def test_age_fix(cols):
    
    Age = cols[0]
    Pclass = cols[1]
    Title = cols[2]
    
    if pd.isnull(Age):
        
        if Pclass == 1 and Title == 'Child':
            return 10
        elif Pclass == 2 and Title == 'Child':
            return 5
        elif Pclass == 3 and Title == 'Child':
            return 7
        
        elif Pclass == 1 and Title == 'Miss':
            return 32
        elif Pclass == 2 and Title == 'Miss':
            return 17
        elif Pclass == 3 and Title == 'Miss':
            return 20
        
        elif Pclass == 1 and Title == 'Mr':
            return 41
        elif Pclass == 2 and Title == 'Mr':
            return 32
        elif Pclass == 3 and Title == 'Mr':
            return 27
        
        elif Pclass == 1 and Title == 'Mrs':
            return 46
        elif Pclass == 2 and Title == 'Mrs':
            return 33
        elif Pclass == 3 and Title == 'Mrs':
            return 30
        
        elif Pclass == 1 and Title == 'Other':
            return 51
        elif Pclass == 2 and Title == 'Other':
            return 36
              
        else:
            return Age
    else:
        return Age
    

test['Age'] = test[['Age','Pclass','Title']].apply(test_age_fix,axis=1) #The "test_age_fix" function is applied to "test" dataset

test['NAge'] = 0  # Create a new feature 'NAge' and assign initial value '0'

test.loc[test['Age']<=19.00,'NAge']=0
test.loc[(test['Age']>19.00)&(test['Age']<=26.00),'NAge']=1
test.loc[(test['Age']>26.00)&(test['Age']<=30.00),'NAge']=2
test.loc[(test['Age']>30.00)&(test['Age']<=40.00),'NAge']=3
test.loc[(test['Age']>40.00)&(test['Age']<=81.00),'NAge']=4

test['Child'] = 0    # Creates a new feature "Child" and assigns initial value '0'

# Assigns value '1' to all Children based on the above-mentioned criteria
test.loc[(test['Age'] <= 16) & (test['Title'] !='Mrs'),'Child'] = 1 

test['Deck'] = test['Cabin'].astype(str).str[0]  # Extracting first character in "Cabin" to create a new column "Deck"

test['IsCabin'] = 1 # Create a new feature "IsCabin" and assign a default value "1"

test.loc[test['Cabin'].isnull(),'IsCabin'] = 0  # Populate "IsCabin" with value '0' where "Cabin" is Null/NaN

#Creating new feature "FamSize" by adding values in "SibSp" and "Parch"

test['FamSize'] = test['SibSp'] + test['Parch'] 

test['Alone'] = 0  # Creating a new feature "Alone" with default value = 0

test.loc[test['FamSize']== 0,'Alone'] = 1  # Populate "Alone" with value '1' where family size is '0'

test['FareBins']=pd.qcut(test['Fare'],4)  # Divides data into equal bins

test['NFare'] = 0  # Creates a feature 'NFare' and assign an initial value '0'

# Now, let's assign a value (from 0 to 3) based on the *'FareBins'*

test.loc[test['Fare']<=7.91,'NFare']=0
test.loc[(test['Fare']>7.91)&(test['Fare']<=14.454),'NFare']=1
test.loc[(test['Fare']>14.454)&(test['Fare']<=31),'NFare']=2
test.loc[(test['Fare']>31)&(test['Fare']<=513),'NFare']=3

test['SharedTicket']= 0 # A new feature "FanTicket" created with initial value "0"

ticketV = test['Ticket'].value_counts()  #Calculates passengers groups on each tickets and assign it to a variable 'ticketV'

single = ticketV.loc[ticketV ==1].index.tolist()  # Creates a list of tickets used by individual(single) passemgers
multi  = ticketV.loc[ticketV > 1].index.tolist()  # Creates a list of tickets shared by group of passemgers

# Compares the ticket number in the "multi" list that was created above with test dataset "Ticket" feature and plugin '1'
for ticket in multi:
    test.loc[test['Ticket'] == ticket, 'SharedTicket'] = 1

emb  = pd.get_dummies(test['Embarked'],drop_first=True) #Creates two Dummy Varable "Q" and "C" and drops the values for "S"   
nsex = pd.get_dummies(test['Sex'],drop_first=True)     #Creates Dummy Varable "male" and drops the values for Female

test = pd.concat([test,emb],axis=1)  # Concatenate test dataset with emb
test = pd.concat([test,nsex],axis=1)  # Concatenate test dataset with nsex

test['Title'].replace(['Mr','Mrs','Miss','Child','Other'],[0,1,2,3,4],inplace=True)

test1 = test.copy()
# Removes unwanted features
test.drop(['Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','Deck', 'FareBins']          ,inplace=True,axis=1) 


# In[ ]:


dsubmit = test.copy()    # Make a copy of test dataset as dsubmit that will be used for further processing
dsubmit.drop('PassengerId',inplace=True,axis=1)


# Let's check the header of the **dsubmit** dataset

# In[ ]:


dsubmit.head(2)


# ### Predictive Modelling
# 
# As mentioned above, we will be using **"Support Vector Machines with kernel rbf"** for predictive modelling. Moeover, we will apply full **titanic** dataset for model building, training and prediction.

# In[ ]:


# Creating trainX and trainY datasets using full titanic dataset
trainX = titanic.drop('Survived',axis=1)
trainY = titanic['Survived']


# In[ ]:


trainX.head(2)            # Header of trainX


# In[ ]:


trainY.head(2)           # Header of trainY


# ### Model Training and Prediction

# In[ ]:


from sklearn.svm import SVC                 # Importing Support Vector Machine library from Scikit-Learn  
model = SVC(kernel='rbf')                   # Model building
model.fit(trainX,trainY)                    # Model training
kpred = model.predict(dsubmit)               # Prediction  


# ### Kaggle Submission
# And finally, lets generate the csv file for Kaggle submission.

# In[ ]:


submit = pd.read_csv("../input/gender_submission.csv")
submit.set_index('PassengerId',inplace=True)

submit['Survived'] = kpred
submit['Survived'] = submit['Survived'].apply(int)
submit.to_csv('submit_titanic.csv')


# This submission got a score of **78.947%** which is not exactly what I was expecting. I expected to gain at least 80%. I will keep improving this Notebook. Hence, more to come!
# 
# Thanks so much for your time in getting this far and I would really appreciate your valuable feedback and suggestions.

# ### Acknowledgements
# 
# Some of the concepts in this Notebook are inspired by the great work done by the following Kaggle submitters. 
# 
#    - https://www.kaggle.com/ash316/eda-to-prediction-dietanic
#    - https://www.kaggle.com/headsortails/pytanic
#    - https://www.kaggle.com/omarelgabry/a-journey-through-titanic
# 
# Moreover, special thanks to Jose Portilla from whom I have learned most of Machine Learning.
# 
#    - https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview
# 
