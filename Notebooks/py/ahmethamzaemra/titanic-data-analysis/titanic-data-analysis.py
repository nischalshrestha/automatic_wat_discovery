#!/usr/bin/env python
# coding: utf-8

# # Titanic Data Analysis
# in this project i work on Titanic data and try to figure out questions like what factor made people more likely to survive and others 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import unicodecsv
import seaborn as sns
import matplotlib.pyplot as plt
import os
path="../input"
os.chdir(path)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic_df=pd.read_csv('train.csv')
titanic_df.head()


# In[ ]:


titanic_df.describe()


# ***
# some datas from age column are missing.
# 
# we will work on them later.
# ***

# In[ ]:


numeric_variables=list(titanic_df.dtypes[titanic_df.dtypes!='object'].index)
titanic_df[numeric_variables].head()


# ## What factor made people more likely to survive?
# 
# this is very general question, but we will work on the questions that will also answer this question but more spesific questions. Such as:
# 
#     * is gender a factor that effects geting the lifeboat?
#     * is there any relationship between fare and age?
#     * is there any relationship between age, sex and surviving?

# In[ ]:


#Standarilizng the data Fare
def standardize_colum(column):
    return (column-column.mean())/column.std()


# In[ ]:


standardize_colum(titanic_df['Fare']).plot()
plt.title("Standardized Fare Chart")
plt.xlabel("Passenger Id")
plt.ylabel("standardized fare value")
plt.show()


# 
# ### is Age and Gender effect on Survive?

# In[ ]:


average_age_titanic    =titanic_df['Age'].mean()
std_age_titanic        =titanic_df['Age'].std()
count_nan_age_titanic  =titanic_df['Age'].isnull().sum()

rand_1=np.random.randint(average_age_titanic-std_age_titanic,average_age_titanic+std_age_titanic, size=count_nan_age_titanic)
# plot original Age values
# fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
#convert them to int
titanic_df['Age']=titanic_df['Age'].astype(int)

titanic_df['Age'].hist(bins=70)
plt.title('Ages of peoples in Titanic')
plt.xlabel('Ages')
plt.ylabel('Number of people')
plt.show()


# ***
# In this step:
# there are 177 data on Age column missing. So that this missing values are narrowing our reduces statistical power. So that we fill the missing values with the random values that one std above and belowe the mean of age column. 
# ***

# In[ ]:


facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()
fig, axis1 = plt.subplots(1,1,figsize=(18,6))
average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
plt.show()


# ***
# we make this chars to see if the childs resqued first. First chart show is standarilized age and survived chart. according to this we can say that there are more people who survive that ones who died under 15 years olds. also on chart 2 mean of people per age is closer than one on under 15 years olds than ones age more than 15.
# ***

# ## Sex and Survive realation

# In[ ]:


df1=(titanic_df.groupby(['Survived', 'Sex'])).count().unstack('Sex')['PassengerId']
df1[['male', 'female']].plot(kind='bar', stacked=True)
labels=['Died', 'Survived']

plt.title("Survived and Gender Relation")
plt.ylabel("number of people")
plt.show()


# ***
# 
# as we can see most of the people who survive is females but this is not enough to make comment on this part of data
# 
# ***
# 
# 

# In[ ]:


total_gender=titanic_df.groupby('Sex').size()
port_class_groups=titanic_df.groupby(['Sex'], as_index=False).get_group('female')
famele_survive=port_class_groups.groupby('Survived').count()*100/port_class_groups.count()


# In[ ]:


labels='famele died','famele survived'
values=famele_survive["Age"]
plt.pie(values, labels=labels,autopct='%1.1f%%', shadow=True)
plt.show()


# ***
# I calculate the persentage of surviving of olverall fameles. 74.2% of famele survived from the titanic disester. this chart support the point that "women more likely to survive" 
# ***

# In[ ]:


total_gender=titanic_df.groupby('Sex').size()
port_class_groups=titanic_df.groupby(['Sex'], as_index=False).get_group('male')
famele_survive=port_class_groups.groupby('Survived').count()*100/port_class_groups.count()
labels='male died','male survived'
values=famele_survive["Age"]
plt.pie(values, labels=labels,autopct='%1.1f%%', shadow=True)
plt.show()


# ***
# form this data we can say that more than %75 of the female population has survived but %79.5 of males are died so;
# usualy womans and childs are being resqued first in this kind a stuation.
# ***

# ###     
# ### Was Class differences another factor to Survived?

# In[ ]:


df2 =titanic_df.groupby(['Survived', 'Pclass'])['PassengerId'].count().unstack('Survived').fillna(0)
df2


# In[ ]:


df2[[0, 1]].plot(kind='bar', stacked=False)
plt.title('Embarked and Classes effect on surviving')
plt.ylabel('Number of People')
plt.xlabel("Passenger classes")
plt.legend(['Survived', 'Died'])
plt.show()


# ***
# this chart show us in first class people more likely to be alive.
# ***

# ### I also want to check fare and age

# In[ ]:


def correlation(x,y):
    std_x=(x-x.mean())/x.std(ddof=0)
    std_y=(y-y.mean())/y.std(ddof=0)
    return (std_x*std_y).mean()


# In[ ]:


tdf=titanic_df.dropna(subset=['Age'])


# In[ ]:


correlation(tdf['Age'],tdf['Fare'])


# # Summary
# 
# 

# as a sumary we have been working on a titanic disaster. for this data set i prepere some questions. Such as :
#     * is gender a factor that effects geting the lifeboat?
#     * is there any relationship between fare and age?
#     * is there any relationship between age, sex and surviving?
# 
# we can come up with many other question. Because our dataset is capabile of. but we have some mising values on our data. for example there is a lot of data is mising in that column. it reduces statistical power. but there is always way, we cannot find the exect values but we can simly fill them with random values. I basicaly fill them random values that 1 std above and 1 std belove the mean. so they will be in %65 of data area. 
# In kagels web page, it says 'some groups of people were more likely to survive than others, such as women, children, and the upper-class.' so I check if this is true. if ones looks at in a gender aproch they one can say women more likely to survive. So next steps should be age effects on surviving, according to our diagram, people who are under 15, there are more alive people that dead. on the other hand, upper classes again according to diagrams, more likely to survive. I also chekc the correlation of fare and age. but correlation doesn't imply causation. so there might not be a relationship. 
