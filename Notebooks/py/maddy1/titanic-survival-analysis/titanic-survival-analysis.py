#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.dtypes


# In[ ]:


df['Age'].count()


# In[ ]:


df['PassengerId'].count()


# In[ ]:


missing_age = np.where(df["Age"].isnull() == True)

print(len(missing_age[0]))


# In[ ]:


age_isnumber=df[df['Age']>=0]
print(len(age_isnumber))


# In[ ]:


not_survived=df[df['Survived']==0]
print(len(not_survived))


# In[ ]:


survived=df[df['Survived']==1]
print(len(survived))


# In[ ]:


get_ipython().magic(u'pylab inline')
df.hist(column='Age',    # Column to plot
                   figsize=(10,6),   # Plot size
                   bins=20)         # Number of histogram bins


# The ages has 177 values which is too huge to drop hence we can use interpolate() function to add values by interpolation.We need the ages to be in ineteger rather than float values which makes convient to use so we can convert the datatype as integer.¶
# 

# In[ ]:


df['filled_ages'] = df['Age'].interpolate()

df['filled_ages']=df['filled_ages'].astype(int)
df['filled_ages'],df['Age']


# The graph doe not change too much by interpolaton.Hence we will continue to use these values.¶
# 

# In[ ]:


ax=df['filled_ages'].hist( figsize=(10,6),   # Plot size
                   bins=20)         # Number of histogram bins
ax.set_xlabel('Age')
ax.set_ylabel('Number of Passengers')


# In[ ]:


#### Lets check the values of fares.How it is different and highes and lowest values.
df.sort_values(['Fare'], ascending=[True])


# In[ ]:


fig = plt.figure(figsize=(12,4))




ax=df.plot(x='Fare',y='filled_ages',kind='scatter')

fig = plt.figure(figsize=(12,4))


# In[ ]:


###The features ticket and cabin have many missing values and so can’t add much value to our analysis
df = df.drop(['Ticket','Cabin'], axis=1) 


# In[ ]:


survived_df=df[df.Survived ==1]
len(survived_df)
#Since we are intersted in the survival factors made a df of the survivors.


# In[ ]:


# So 342 of 891 survived and hence the survival percentage would be
float(len(survived_df))/len(df)


# Let us Box plot to check fare based on class¶
# 

# In[ ]:


df.boxplot(column='Fare', by = 'Pclass')


# The fares have for first class is higher and second class is high and third class is
# low which makes sense.The first class has some high values above 500.But you dont kow why it
# is high.Maybe they have super luxurious tickets.We can keep that data as such and
# continue.

# In[ ]:


df[df['Fare'] >500] 


# ####All the 3 survived.
# 
# #### I googled for the name.
# #####http://www.encyclopedia-titanica.org/titanic-survivor/thomas-cardeza.html

# Survivors based on Sex.It seems more female survivors than male.
# 

# In[ ]:


temp1 = survived_df.groupby('Sex').Survived.count()
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(122)
ax1.set_xlabel('Sex')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers Survived by Sex")
temp1.plot(kind='bar')


temp2 = df.groupby('Sex').Survived.count()

ax2 = fig.add_subplot(121)
ax2.set_xlabel('Sex')
ax2.set_ylabel('Count of Passengers')
ax2.set_title("Total Passengers by Sex")
temp2.plot(kind='bar')


# Though the male passengers were more than female passengers,The survival rate of female passengers is more than male passengers.

# In[ ]:


df.pivot_table('Survived', index='Sex', columns='Pclass')


# In[ ]:


temp1 = df.groupby('Pclass').Survived.count()
temp2 = df.groupby('Pclass').Survived.sum()/df.groupby('Pclass').Survived.count()
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers by Pclass")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by Pclass")


# Even though third class passengers were high in number than first and second class passengers,the first and second class passengers had a higher probability survival rate than the third class passengers,

# In[ ]:


#Some more plots
#Specifying Plot Parameters
# figsize = (x inches, y inches), dpi = n dots per inches
fig = plt.figure(figsize = (11, 8), dpi = 1600)


# Plot: 1
ax4 = fig.add_subplot(221) # .add_subplot(rcp): r = row, c = col, p = position
female_firstclass = df['Survived'][df['Sex'] == 'female'][df['Pclass'] == 1].value_counts()
female_firstclass.plot(kind = 'bar', label = 'Female First Class', color = 'deeppink', alpha = 0.5)
ax4.set_xticklabels(['Survived', 'Dead'], rotation = 0)
ax4.set_xlim(-1, len(female_firstclass))
ax4.set_ylim(0, 400)
ax4.set_title("Female Passengers by First class")
plt.legend(loc = 'best')
#Plot: 2
ax5 = fig.add_subplot(222) # .add_subplot(rcp): r = row, c = col, p = position
female_secondclass = df['Survived'][df['Sex'] == 'female'][df['Pclass'] == 2].value_counts()
female_secondclass.plot(kind = 'bar', label = 'Female Second Class', color = 'deeppink', alpha = 0.5)
ax5.set_xticklabels(['Survived', 'Dead'], rotation = 0)
ax5.set_xlim(-1, len(female_secondclass))
ax5.set_ylim(0, 400)
ax5.set_title("Female Passengers by Second class")
plt.legend(loc = 'best')           
#Plot:3
ax6 = fig.add_subplot(223) # .add_subplot(rcp): r = row, c = col, p = position
female_thirdclass = df['Survived'][df['Sex'] == 'female'][df['Pclass'] == 3].value_counts()
female_thirdclass.plot(kind = 'bar', label = 'Female third Class', color = 'deeppink', alpha = 0.5)
ax6.set_xticklabels(['Survived', 'Dead'], rotation = 0)
ax6.set_xlim(-1, len(female_thirdclass))
ax6.set_ylim(0, 400)
ax6.set_title("Female Passengers by Third class")
plt.legend(loc = 'best')            


# In[ ]:


print(female_firstclass)
print(female_secondclass)
print(female_thirdclass)


# That is 96.8% of female first class passengers survived.92.1% of female second class passengers survived.50% of female third class passengers survived.Chances for female first class passengers to survive was more.

# **Lets check how the class affects the male passengers.
# **

# In[ ]:


fig = plt.figure(figsize = (15, 12), dpi = 1600)

# Plot: 1
ax6 = fig.add_subplot(321) # .add_subplot(rcp): r = row, c = col, p = position
male_firstclass = df['Survived'][df['Sex'] == 'male'][df['Pclass'] == 1].value_counts()
male_firstclass.plot(kind = 'bar', label = 'male First Class', color = 'green', alpha = 0.5)

ax6.set_xlim(-1, len(male_firstclass))
ax6.set_ylim(0, 400)
ax6.set_title("Male Passengers by First class")

ax7 = fig.add_subplot(322) # .add_subplot(rcp): r = row, c = col, p = position
male_secondclass = df['Survived'][df['Sex'] == 'male'][df['Pclass'] == 2].value_counts()
male_secondclass.plot(kind = 'bar', label = 'male second Class', color = 'green', alpha = 0.5)

ax7.set_xlim(-1, len(male_secondclass))
ax7.set_ylim(0, 400)
ax7.set_title("Male Passengers by Second class")

ax8 = fig.add_subplot(323) # .add_subplot(rcp): r = row, c = col, p = position
male_thirdclass = df['Survived'][df['Sex'] == 'male'][df['Pclass'] == 3].value_counts()
male_thirdclass.plot(kind = 'bar', label = 'male third Class', color = 'green', alpha = 0.5)

ax8.set_xlim(-1, len(male_firstclass))
ax8.set_ylim(0, 400)
ax8.set_title("Male Passengers by Third class")

ax6 = fig.add_subplot(324) # .add_subplot(rcp): r = row, c = col, p = position
kidsmale_firstclass = df['Survived'][df['Sex'] == 'male'][df['filled_ages'] < 18][df['Pclass'] == 1].value_counts()
kidsmale_firstclass.plot(kind = 'bar', label = 'kids male First Class', color = 'blue', alpha = 0.5)

ax6.set_xlim(-1, len(male_firstclass))
ax6.set_ylim(0, 400)
ax6.set_title("Male kids Passengers by First class")

ax6 = fig.add_subplot(325) # .add_subplot(rcp): r = row, c = col, p = position
kidsmale_secondclass =df['Survived'][df['Sex'] == 'male'][df['filled_ages'] < 18][df['Pclass'] == 2].value_counts()
kidsmale_secondclass.plot(kind = 'bar', label = 'kids second Class', color = 'blue', alpha = 0.5)

ax6.set_xlim(-1, len(male_firstclass))
ax6.set_ylim(0, 400)
ax6.set_title("Male kids Passengers by second class")

ax9 = fig.add_subplot(326) # .add_subplot(rcp): r = row, c = col, p = position
kidsmale_thirdclass = df['Survived'][df['Sex'] == 'male'][df['filled_ages'] < 18][df['Pclass'] == 3].value_counts()
kidsmale_thirdclass.plot(kind = 'bar', label = 'kids male third Class', color = 'blue', alpha = 0.5)

ax9.set_xlim(-1, len(male_firstclass))
ax9.set_ylim(0, 400)
ax9.set_title("Male kids Passengers by third class")


# In[ ]:


print(kidsmale_firstclass)
print(kidsmale_secondclass)
print(kidsmale_thirdclass)


# In[ ]:


print(male_firstclass)
print(male_secondclass)
print(male_thirdclass)


# The survival rate of first class men are 36.8 and survival rate kids among those men is 67.3%.Similarly for second class 15.7% men and 77 % male kids .In third class 13.5% male passengers and 22.5% kids among male survived.Survival rate for first class male passengers are high.

# Out of 557 male,77 were kids and of 109 male survived,27 were kids.27/77 male kids survived.
# Cannot predict that male adults or kids had a higher rate of survival.

# In[ ]:


kidsfmale_firstclass = df['Survived'][df['Sex'] == 'female'][df['Age'] < 18][df['Pclass'] == 1].value_counts()
kidsfmale_secondclass = df['Survived'][df['Sex'] == 'female'][df['Age'] < 18][df['Pclass'] == 2].value_counts()
kidsfmale_thirdclass = df['Survived'][df['Sex'] == 'female'][df['Age'] < 18][df['Pclass'] == 3].value_counts()
print(kidsfmale_firstclass)
print(kidsfmale_secondclass)
print(kidsfmale_thirdclass)


# In[ ]:


print(female_firstclass)
print(female_secondclass)
print(female_thirdclass)


# 65/132 kids survived.38 female kids of 55 female kids survived.Out of 233 survived female 38 were kids.69% of female kids survived.89.9% female adults survived.

# In[ ]:


Class1=df['Survived'][df['Pclass'] == 1].value_counts()
print(Class1)
print(342-136)
print(891-216)


# In the previous plots we see the Pclasss 1 passengers had higher chance of survival.
# Let conduct a hypothesis test to see if there is a significant difference between proportions.
# P1= proportion of Pclass 1 survivors
# P2=proportion of other than Pclass 1 survivors
#     
#   The test procedure, called the two-proportion z-test,
#  is appropriate when the following conditions are met:
#     
# 1)The sampling method for each population is simple random sampling.
# 2)The samples are independent.
# 3)Each sample includes at least 10 successes and 10 failures.
# 4)Each population is at least 20 times as big as its sample.
# 
# All theses conditions are met and lets assume that 4 th condition is true.Lets do a one tailed test.
# 
# Null hypothesis: P1-P2 =0
# Alternative hypothesis: P1 > P2
#     
#     
#     Formulate an analysis plan. 
#     For this analysis, the significance level is 0.01. The test method is a two-proportion z-test.
#     
#     Analyze sample data.
#     Using sample data, we calculate the pooled sample proportion (p) and the standard error (SE). 
#     Using those measures, we compute the z-score test statistic (z).
#     
#     Total survived=342(found above)
#     Pclass 1 survivors=136
#     Total Pclass=216
#     Others survivors=206
#     Total others=675
#     
# where p1 is the sample proportion in sample of Pclass1, where p2 is the sample proportion in other than Pclass1,
# n1 is the size of Pclass1, and n2 is the size of others.    

# In[ ]:


import math
p1=float(136)/216
p2=float(206)/675
n1=216
n2=675
p = float((p1 * n1 + p2 * n2)) / (n1 + n2) 
SE = math.sqrt( p * ( 1 - p ) * ((float(1)/n1) + (float(1)/n2) ))
z = float(p1 - p2) / SE 
print("p:",p)
print("SE:",SE)
print("z:",z)


# By the z value it can be said Pvalue is going to be lesser.Since the P-value is lesser than the significance level (0.01), 
# we can reject the null hypothesis.And say that there is significant difference in survival of first class passengers.

# ######                                            Findings                     ######
# 
# 
# *********************************************************
# 
# The Survival rate prediction by class,Sex and age factors show that the adult Female first class 
# passengers had higher chances of survival.Though the female passngers had higher chances of survival from the graph.
# The cabin can also be a good example to know the survival factors but most of the cabin values are missing.
# Few values like how many men related to the female survived also survived would be nice way to know 
# the details of survival.And moreover the data given is just for 891 customers and not the whole titanic data
# it is hard to say what factors would exactly affect the survival.
