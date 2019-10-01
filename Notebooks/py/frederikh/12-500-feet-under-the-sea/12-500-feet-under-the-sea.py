#!/usr/bin/env python
# coding: utf-8

# # Table of Content
# 1. [Disclaimer](#disclaimer)<br>
# 2. [Introduction](#introduction)<br>
#     2.1 [Run(away) Data, run!](#runaway) <br>
#     2.2 [Mind the Gap](#gap)<br>
# 
# # 1. Disclaimer<a id='disclaimer'></a>
# 
# This is my first try at data analysis. Most code is taken from various stackexchange posts or other sources. I tried to include all references to other notebooks when I borrowed ideas. If you recognize an idea or part of code you think is taken from another notebook, please leave a comment with the corresponding notebook and I will add the reference if I did indeed take it from there.
# 
# # 2. Introduction<a id='introduction'></a>
# 
# The first part of this notebook deals with the visualization of existing files. Through a visual representation one can get a first impression of which parts of the data are important, which are less important and which are only partially present.
# For this reason, the training data are loaded together with the imported libraries.

# In[ ]:


# Standard Libraries
import numpy as np
import pandas as pd

# Visualzation
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns


# The training data is the important file. The entries of this file will be used for the visualisation. The following cell reads the CSV-file "train.csv" and will display the first 4 rows of the data together with the explanatory collumn titles. 

# In[ ]:


train_set = pd.read_csv("../input/train.csv")

#Display first 5 rows (4+description)
train_set.head(5)


# In[ ]:


test_set = pd.read_csv("../input/test.csv")

#Display first 5 rows (4+description)
test_set.head(5)


# The individual entries of the table can be addressed in the following way.

# In[ ]:


# First entry Name of the collumn, second entry number of row (excluding the title row)
print(train_set['Name'][3])


# ## 2.1 Run(away) Data, run!<a id='runaway'></a>
# 
# In statistics, an outlier is an observation point that is distant from other observationshttps://en.wikipedia.org/wiki/Outlier. In this part I will check any non-binary features for outliers or "runaway data".

# In[ ]:


# Check Age for runaway data
sns.regplot(x=train_set["PassengerId"], y=train_set["Age"])
plt.show()

sns.regplot(x=test_set["PassengerId"], y=test_set["Age"])
plt.show()


# In[ ]:


# Check Fare for runaway data

# use the function regplot to make a scatterplot
sns.regplot(x=train_set["PassengerId"], y=train_set["Fare"])
plt.show()

sns.regplot(x=test_set["PassengerId"], y=test_set["Fare"])
plt.show()


# In[ ]:


train_set.loc[train_set['Fare'] > 300]


# In[ ]:


test_set.loc[test_set['Fare'] > 300]


# Some research reveals, that Mrs. Charlotte Drake Cardeza and her son Mr. Thomas Cardenza traveled with their servants, Anna Ward and Gustave Lesurer.[https://en.wikipedia.org/wiki/Charlotte_Drake_Cardeza] They are clear outliers. Noone else paid a price as high as these passengers. Yet it paid of for them, they all survived the sinking of the Titanic in Lifeboat #3[https://www.encyclopedia-titanica.org/titanic-survivor/gustave-lesueur.html].
# 

# In[ ]:


# Show with outliers
PClass_palette = {1:"b", 2:"y", 3:"r"}
sns.boxplot( x=train_set["Pclass"], y=train_set["Fare"], palette=PClass_palette, showfliers=True)
plt.show()

# Show without outliers
sns.boxplot( x=train_set["Pclass"], y=train_set["Fare"], palette=PClass_palette, showfliers=False)
plt.show()


# In[ ]:


# Show with outliers
PClass_palette = {1:"b", 2:"y", 3:"r"}
sns.boxplot( x=test_set["Pclass"], y=test_set["Fare"], palette=PClass_palette,  showfliers=True)
plt.show()

# Show without outliers
sns.boxplot( x=test_set["Pclass"], y=test_set["Fare"], palette=PClass_palette, showfliers=False)
plt.show()


# # 2.2 Mind the Gap<a id='gap'></a>
# 
# Now lets take a look at the previously mentioned missing values. Both the training and the test data has gaps in some collumns.

# In[ ]:


train_set.isnull().sum()


# In[ ]:


test_set.isnull().sum()


# There are four collumns with missing entries in the training and test set. I will try to fill these gaps with values corresponding to the present data. The first collumn I will look at is the Port of Embarkment in the training set. There is one correlation useful in filling the gaps. The Fare can be used as a method to find the port. I took this method from [Megan Risdals Kernel "Exploring Survival on the Titanic"](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic) and take no credit for it.

# In[ ]:


sns.boxplot(x=train_set["Embarked"], y=train_set["Fare"], hue=train_set["Pclass"],
            palette=PClass_palette,showfliers=False)
plt.plot([80, -80], [80, 80], linewidth=3, linestyle='--')
plt.show()


# In[ ]:


train_set[pd.isnull(train_set['Embarked'])]


# Both passengers with missing ports of embarkment paid £80 and traveled first class. Using all training data one can see this fits perfectly with the mean of the first class passengers which boarded the Titanic in Cherbourg. This value will be used.

# In[ ]:


# HIER WEITER MACHEN. 
# Wert Embarked muss auf 80.0 geändert werden
train_set.loc[train_set["Embarked"].isnull()]


# In[ ]:


# Make default histogram of sepal length
sns.distplot( train_set.dropna()["Age"], bins=12)
plt.show()


# ## Women and Children First
# The data provide a clear but not as strong as expected presentation of the tradition of "women and children first". Most women survived the collision with the iceberg and the following sinking of the titanic.

# In[ ]:


# Total number of male and female passengers
#male_total = (data['Sex'] == 'male').sum()
#female_total = (data['Sex'] == 'female').sum()

# Survivors per gender
#male_survived = (data[data['Survived']==1]['Sex']=='male').sum()
#female_survived = (data[data['Survived']==1]['Sex']=='female').sum()

# Deaths per gender
#male_dead = (data[data['Survived']==0]['Sex']=='male').sum()
#female_dead = (data[data['Survived']==0]['Sex']=='female').sum()

# Plotting
#Gender_data = data.groupby(['Sex', 'Survived'])['Sex'].count().unstack('Survived')
#plot1 = Gender_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])
#plot1.legend(["Dead","Survived"])

# Percentage survived by gender
#print(100/male_total*male_survived,"% of the male passengers survived.")
#print(100/female_total*female_survived,"% of the the female passengers survived.")


# But this is only the difference between male and female passengers. What about the second part of "women *and children* first"? For this first visualisation the age of 14 will be used as the limit between beeing a child and considering someone an 'adult'.

# In[ ]:


# Plotting
#Age_data = data.groupby(['Age', 'Survived'])['Age'].count().unstack('Survived')
#plot2 = Age_data.plot(kind='area', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])
#plot2.legend(["Dead","Survived"])


# Total number of children and adults
#children_total = (data['Age'].dropna(axis=0, how='any') <= 14).sum()
#adult_total = (data['Age'].dropna(axis=0, how='any') > 14).sum()


# Survivors per agegroup
#children_survived = (data[data['Survived']==1]['Age']<=14).sum()
#adult_survived = (data[data['Survived']==1]['Age']>14).sum()

#percentage of survivors by age above and below 14
#print(100/adult_total*adult_survived,"% of the adult passengers survived.")
#print(100/children_total*children_survived,"% of the the passengers below the age of 14 survived.")


# In[ ]:


# Numpy array that holds unique age values
#age_arr = data.sort_values("Age")['Age'].unique()
# Remove NaN
#age_arr = age_arr[~np.isnan(age_arr)]
# Make int
#age_arr = age_arr.astype(int)

# Numpy array that holds mean values of survival for each age
#surv_arr_mean = data.groupby('Age')['Survived'].mean().values
# Numpy array that holds median values of survival for each age
#surv_arr_median = data.groupby('Age')['Survived'].median().values
# Numpy array that holds sum of survival for each age
#surv_arr_sum = data.groupby('Age')['Survived'].sum().values

#sum_all = data.groupby('Age')['Age'].value_counts()

#from scipy import signal
#sum_all_smooth = signal.savgol_filter(sum_all, 11, 2)
#surv_arr_sum_smooth = signal.savgol_filter(surv_arr_sum, 11, 2)

#plt.figure()
#plt.figure(figsize=(16,8))
#plt.title('Age vs Survival rate', fontsize=20, fontweight='bold', y=1.05,)
#plt.xlabel('Age', fontsize=15)
#plt.ylabel('Survival', fontsize=15)
#sns.set_style("whitegrid")
#plt.plot(age_arr, surv_arr_mean, label="Mean")
#plt.plot(age_arr, surv_arr_median, label="Median")
#plt.plot(age_arr, surv_arr_sum_smooth, label="Sum")
#plt.plot(age_arr, sum_all_smooth, label="Sum")

#plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
#plt.show()


# In[ ]:


#surv_arr_mean_smooth = signal.savgol_filter(surv_arr_mean, 7, 4)


#plt.figure()
#plt.figure(figsize=(16,8))
#plt.title('Age vs Survival rate', fontsize=20, fontweight='bold', y=1.05,)
#plt.xlabel('Age', fontsize=15)
#plt.ylabel('Survival', fontsize=15)
#sns.set_style("whitegrid")
#plt.plot(age_arr, surv_arr_mean_smooth, label="Mean")
#plt.plot(age_arr, surv_arr_median, label="Median")
#plt.plot(age_arr, surv_arr_sum_smooth, label="Sum")
#plt.plot(age_arr, sum_all_smooth, label="Sum")

#plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
#plt.show()


# There is a clear sign, that the saying had an influence on the survivors of the disaster. Both of these factors, gender and adulthood, have a clear influence on the survivability of an individual on the titanic. The age itself will be used later on for more discrimination between various age groups. A clear cut as seen above around the 14-16 Year old mark should not distort the influence of the ages. Therefore a new collumn will be added to the training data which indicates if a person is treated as an adult or as a child. This new feature should make the age a better feature itself.
# 
# For this we create a copy of the original dataset. We will work on this new dataset in the future and compare the results to the original, unaltered data.

# In[ ]:


#The Data will be adjusted and therefore we will create a new dataset which we will manipulate.
#data_adj = data.copy()

# Add an empty collumn with the title 'ChildAdult'
#data_adj['ChildAdult'] = np.nan

#print(data_adj.isnull().sum(axis=0)['ChildAdult'])

# If Age is known, add the right tag
#for i in data_adj['Age'].iteritems():
#    if i[1]>14:
#        data_adj.loc[i[0],'ChildAdult'] = 'Adult'
#    elif i[1]<=14:
#        data_adj.loc[i[0],'ChildAdult'] = 'Child'
        
#print(data_adj.isnull().sum(axis=0)['ChildAdult'])

# If Age is unknown...

# First Method: Children are (mostly) not married. Any 'Mrs.' will be
#               treated as an adult.
#for i in data_adj['ChildAdult'].iteritems():
#    if data_adj['ChildAdult'][i[0]] != ('Child' or 'Adult'):
#        tmp = data_adj['Name'][i[0]]
#        tmp = tmp.split()
#        if 'Mrs.' in tmp:
#            data_adj.loc[i[0],'ChildAdult'] = 'Adult'
            
#print(data_adj.isnull().sum(axis=0)['ChildAdult'])

# Second Method: It is unlikely that a child is traveling alone,
#               therefore any passenger with an unknown age, no siblings, sposes,
#               parents or children will be consideren as an adult
#for i in data_adj['ChildAdult'].iteritems():
#    if data_adj['SibSp'][i[0]] == 0 and data_adj['Parch'][i[0]] == 0 and data_adj['ChildAdult'][i[0]] != ('Child' or 'Adult'):
#        data_adj.loc[i[0],'ChildAdult'] = 'Adult'
        
#print(data_adj.isnull().sum(axis=0)['ChildAdult'])
        
# Third Method: If there are more than 3 Spouses/Siblings it is likely,
#                that it is an underage passenger with many siblings.
#                Adult passengers are mostly alone, with one sibling/spouse
#                or with many children rather than siblings.
#for i in data_adj['ChildAdult'].iteritems():
#    if data_adj['SibSp'][i[0]] > 3 and data_adj['ChildAdult'][i[0]] != ('Child' or 'Adult'):
#        data_adj.loc[i[0],'ChildAdult'] = 'Child'
        
#print(data_adj.isnull().sum(axis=0)['ChildAdult'])


# Using these methods, the number of unassigned passengers could be reduced to 24. This should be enough to be useful in the following calculations.

# ## Money Floats
# 
# Anyone who saw James Camerons movie Titanic 'knows' that being wealthy probably played a key factor in the survival rate of the passengers. Lets take a look at the data for the classes. Once divided by class alone and once by gender and age.

# In[ ]:


# Plotting total
#Money_data = data_adj.groupby(['Pclass', 'Survived'])['Pclass'].count().unstack('Survived')
#plot1 = Money_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])
#plot1.legend(["Dead","Survived"])


# In[ ]:


# Plotting gender and class
#Money_data = data_adj.groupby(['Pclass', 'Survived', 'Sex'])['Pclass'].count().unstack('Survived')
#plot1 = Money_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])
#plot1.legend(["Dead","Survived"])


# In[ ]:


# Plotting adulthood and class
#Money_data = data_adj.groupby(['Pclass', 'Survived', 'ChildAdult'])['Pclass'].count().unstack('Survived')
#plot1 = Money_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])
#plot1.legend(["Dead","Survived"])


# In[ ]:


# Total number of children in third class
#children_poor_total = (data_adj[data_adj['Pclass']==3]['Age']<=14).sum()

# Survivors of third class children
#children_poor_survived = ((data_adj['Survived']==1) & (data_adj['Age']<=14) & (data_adj['Pclass']==3)).sum()

#percentage of survivors by age above and below 14
#print(100/children_poor_total*children_poor_survived,"% of the the passengers below the age of 14 in the third class survived.")


# 
# 
# The results are very meaningful. If a passenger on the titanic is wealthy and a child or woman he/she almost certainly survived the sinking of the ship. Zero children of the first and second class died, while more than half of the children of the third class did not survive. It is noteworthy that there are only very few children in the first and second class, therefore the data might be biased. Extrapolating from the gender data there is still the same trend
# 

# ## Road To Overfitting
# 
# I want to create new, hopefully meaningful features from the present data. I am aware that this will probably lead to overfitting, therefore each new feature will be tested alone (with the original data) and in combination with various other features.
# 
# The first new feature was already created in the previous chapter. The "ChildAdult" feature implements various original data entries and kategorises most passengers into the "Adult" or "Child" section.
