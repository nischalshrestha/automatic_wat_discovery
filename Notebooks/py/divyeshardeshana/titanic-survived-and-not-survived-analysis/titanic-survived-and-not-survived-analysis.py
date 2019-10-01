#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[ ]:


import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sn
import os
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20,.2f}'.format
import colorsys


# # Import Gender CSV

# In[ ]:


DataSetGender = pd.read_csv('../input/gender_submission.csv')


# # DataSetGender Shape

# In[ ]:


DataSetGender.shape


# # Total Null Value for DataSetGender

# In[ ]:


DataSetGender.isnull().sum()


# # Total Number of Data Count for TotalDataSetGender

# In[ ]:


TotalDataSetGender = len(DataSetGender)
print("Total Number of Data Count :", TotalDataSetGender)


# # Fill Null value with 0 in DataSetGender

# In[ ]:


DataSetGender.fillna(0)


# # DataSetGender Top 10 Records

# In[ ]:


DataSetGender.head(10)


# # Import Test CSV

# In[ ]:


DataSetTest = pd.read_csv('../input/test.csv')


# # DataSetTest Shape

# In[ ]:


DataSetTest.shape


# # Total Null Value for DataSetTest

# In[ ]:


DataSetTest.isnull().sum()


# # Total Number of Data Count for TotalDataSetTest

# In[ ]:


TotalDataSetTest = len(DataSetTest)
print("Total Number of Data Count :", TotalDataSetTest)


# # Fill Null value with 0 in DatSetTest

# In[ ]:


DataSetTest.fillna(0)


# # DataSetTest Top 10 Records

# In[ ]:


DataSetTest.head(10)


# # Import Train CSV

# In[ ]:


DataSetTrain = pd.read_csv('../input/train.csv')


# # DataSetTrain Shape

# In[ ]:


DataSetTrain.shape


# # Total Null Value for DataSetTrain

# In[ ]:


DataSetTrain.isnull().sum()


# # Total Number of Data Count for TotalDataSetGender

# In[ ]:


TotalDataSetGender = len(DataSetGender)
print("Total Number of Data Count :", TotalDataSetGender)


# # Fill Null value with 0 in DataSetTrain

# In[ ]:


DataSetTrain.fillna(0)


# # DataSetTrain Top 10 Records

# In[ ]:


DataSetTrain.head(10)


# # Sex wise total with Graph

# In[ ]:


SexCount = DataSetTrain['Sex'].value_counts()
print(SexCount)
pivot = DataSetTrain["Sex"].value_counts().nlargest(10)
pivot.plot.bar()
plt.show()


# # Survived Graph

# In[ ]:


palette = 'cividis'
plt.figure(figsize=(8,5))
sn.heatmap(DataSetTrain.drop('Survived', axis=1).corr(), annot=True, cmap=palette+'_r')


# # Sex wise Mean Total Survived

# In[ ]:


DataSetTrain[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# # Embarked, Survived, Pclass wise count

# In[ ]:


DataSetTrain.groupby(['Embarked','Survived', 'Pclass'])['Survived'].count()


# # Passengers Survived by Class

# In[ ]:


sn.set(font_scale=1)
g = sn.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=DataSetTrain, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.9)
g.fig.suptitle('How Passengers Survived by Class');


# # Total Survived Count

# In[ ]:


SurvivedCount = DataSetTrain["Survived"].value_counts().nlargest(10)
print("Survived Count \n")
print(SurvivedCount)
pivot = DataSetTrain["Survived"].value_counts().nlargest(10)
pivot.plot.bar()
plt.show()


# # List of All Survived Passenders

# In[ ]:


DataSetTrain[DataSetTrain["Survived"]==1][["Name", "Sex" , "Age", "Pclass", "Cabin"]]


# # List of All Not Survived Passenders

# In[ ]:


DataSetTrain[DataSetTrain["Survived"]==0][["Name", "Sex" , "Age", "Pclass", "Cabin"]]


# # Survived and Dead with Sex

# In[ ]:


f, ax = plt.subplots()
sn.countplot('Sex', hue='Survived', data=DataSetTrain, ax=ax)
ax.set_title('Survived and Dead with Sex')
plt.show()


# # Find All Unique Cabin Names

# In[ ]:


UniqueCabin = DataSetTrain['Cabin'].unique()
print("All Unique Cabin Name \n")
print(UniqueCabin)


# # Total Survived Fare

# In[ ]:


TotalSurvived = DataSetTrain.groupby(['Survived'])['Fare'].sum().nlargest(15)
print("Total Fare for Survived and Not Survived\n")
print(TotalSurvived)


# # All Fare Max, Min and Mode Details

# In[ ]:


FareMax = DataSetTrain['Fare'].max()
print ("Max Fare Mode is :", round(FareMax,2))
FareMin = DataSetTrain['Fare'].min()
print ("Min Fare Mode is :", round(FareMin,2))
FareMean = DataSetTrain['Fare'].mean()
print ("Mean Fare Mode is :", round(FareMean,2))


# # Survived Fare Max, Min and Mode Details

# In[ ]:


SurvivedFareData=DataSetTrain[DataSetTrain['Survived']==1]
SurvivedFareMax = SurvivedFareData['Fare'].max()
print ("Survived Max Fare Mode is :", round(SurvivedFareMax,2))
SurvivedFareMin = SurvivedFareData['Fare'].min()
print ("Survived Min Fare Mode is :", round(SurvivedFareMin,2))
SurvivedFareMean = SurvivedFareData['Fare'].mean()
print ("Survived Mean Medal Mode is :", round(SurvivedFareMean,2))


# # Not Survived Fare Max, Min and Mode Details

# In[ ]:


NotSurvivedFareData=DataSetTrain[DataSetTrain['Survived']==0]
NotSurvivedFareMax = NotSurvivedFareData['Fare'].max()
print ("Not Survived Max Fare Mode is :", round(NotSurvivedFareMax,2))
NotSurvivedFareMin = NotSurvivedFareData['Fare'].min()
print ("Not Survived Min Fare Mode is :", round(NotSurvivedFareMin,2))
NotSurvivedFareMean = NotSurvivedFareData['Fare'].mean()
print ("Not Survived Mean Medal Mode is :", round(NotSurvivedFareMean,2))

