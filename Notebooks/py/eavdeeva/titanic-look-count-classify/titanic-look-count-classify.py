#!/usr/bin/env python
# coding: utf-8

# - - -
# ## 1. Idea
# - - -
# The objective of the Titanic exercise is to predict which passengers will survive based on the information provided in the dataset: passenger name, sex, age, number of sibling/spouses, number of parents/children, ticket class, the ticket number, fare, cabin, and embarkation port.
# 
# I offer to carefully explore the data by dividing it into groups and searching for groups where chances to survive statistically significantly exceed chances to drown.
# - - -
# ## 2. Initial exploration
# * - - -
# #### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# #### Reading the DataFrame from the .csv file

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df.info()


# #### Looking at counts of survived/dead males in different ticket classes

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Pclass',hue='Survived',data=df[df['Sex']=='male'])


# Almost all males in the 2nd and 3rd class drowned as most males in the 1st class. The survival rate among the 1st class men is the highest. 
# 
# However, due to the larger total number of 3rd class passengers, the total numbers of male survivors in the 1st and 3rd class are about the same.
# 
# - - -
# #### Looking at counts of survived/dead females in different ticket classes.

# In[ ]:


sns.countplot(x='Pclass',hue='Survived',data=df[df['Sex']=='female'])


# Almost all females in the 1st and 2nd class survived while females in the 3rd class had 50/50 chance. 
# - - -
# #### Looking at the age distribution 

# In[ ]:


g = sns.FacetGrid(data=df,col='Survived',row='Sex')
g.map(sns.distplot,'Age')


# **Age distribution for males has much higher children's rate in the 'Survived' sample while the age distribution for females is not as obvious.

# - - -
# ## 3. Introducing new features, gender/class subframes, and function Difference
# 
# As chances to survive depend significantly on gender and class, new gender/class subframes will be introduced and studies separately. If I want to introduce new features, I should do that before I split the original DataFrame.
# - - -
# ### 3.1. New features
# - - -
# #### Introducing 'CabType' feature as the first letter of 'Cabin'

# In[ ]:


df['CabType']=df['Cabin'].apply(lambda s: str(s)[0])
df['CabType'].unique()


# #### Introducing 'Title' feature as the second word in the 'Name' column. 
# - - -
# All second words other than Mr., Mrs., Miss., Master., Dr., and Rev. are combined into a single value 'U.'

# In[ ]:


def CorrectTitle(s):
    if ((s=='Mr.') or (s=='Miss.') or (s=='Mrs.') or (s=='Master.') or (s=='Dr.') or (s=='Rev.')): return s
    return 'U.'

df['Title']=df['Name'].apply(lambda s: s.split()[1])
df['Title']=df['Title'].apply(lambda s: CorrectTitle(s))
df['Title'].value_counts()


# ### 3.2. Subframes
# - - -
# **Introducing subframes** 
#  
# df_F and df_M accumulate all males/females while
# 
# df_F_clX and df_M_clX split males and females into classes

# In[ ]:


df_F = df[(df['Sex']=='female')] # All females
df_F_cl1 = df[(df['Sex']=='female') & (df['Pclass']==1)] # females, class 1
df_F_cl2 = df[(df['Sex']=='female') & (df['Pclass']==2)] # females, class 2
df_F_cl3 = df[(df['Sex']=='female') & (df['Pclass']==3)] # females, class 3

df_M = df[(df['Sex']=='male')] # All males
df_M_cl1 = df[(df['Sex']=='male') & (df['Pclass']==1)] # males, class 1
df_M_cl2 = df[(df['Sex']=='male') & (df['Pclass']==2)] # males, class 2
df_M_cl3 = df[(df['Sex']=='male') & (df['Pclass']==3)] # males, class 3


# ### 3.3. Function 'Difference'
# - - -
# 
# Function 'Difference' counts a number of survived and died people in a DataFrame, computes their difference and rate, and propagates errors.
# 
# Poisson uncertainties on the numbers of survived and dead people:
# 
# $\Delta N_{surv} = \sqrt{N_{surv}},~\Delta N_{dead} = \sqrt{N_{dead}}$
# 
# The difference between the number of survived and dead people: 
# 
# $N_{diff} \pm \Delta N_{diff} = (N_{surv} - N_{dead})\pm \sqrt{N_{surv}+N_{dead}}$
# 
# The survival rate:
# 
# $ R \pm \Delta R = \frac{N_{surv}}{N_{surv}+N_{dead}} \pm \frac{\sqrt{(N_{surv}+N_{dead})^2 \cdot N_{surv} + N_{surv}^2 \cdot N_{dead}}}{(N_{surv}+N_{dead})^2}$
# 

# In[ ]:


def Difference(df):
    Nsurv = df[df['Survived']==1]['Survived'].count()
    Ndied = df[df['Survived']==0]['Survived'].count()
    Ndiff = Nsurv-Ndied
    survRate = Nsurv/(Nsurv+Ndied)
    dNsurv=Nsurv**0.5
    dNdied=Ndied**0.5
    dNdiff=(Nsurv+Ndied)**0.5
    dRateNum = (((Nsurv+Ndied)**2)*Nsurv + (Nsurv**2)*Ndied)**0.5
    dRateDen = (Nsurv+Ndied)**2
    dRate=dRateNum/dRateDen
    print('Nsurv={}+-{:.0f}, Ndied={}+-{:.0f}'.format(Nsurv,dNsurv,Ndied,dNdied))
    print('Nsurv-Ndied={}+-{:.0f}, survRate={:.2f}+-{:.2f}'.format(Ndiff,dNdiff,survRate,dRate))


# A DataFrame is more likely to survive if $(N_{diff}-\Delta N_{diff}>0)$ and $(R-\Delta R>0.5)$
# 
# A DataFrame is more likely to die if $(N_{diff}+\Delta N_{diff}<0)$ and $(R+\Delta R<0.5)$
# - - -
# ** Applying 'Difference' on the whole DataFrame**

# In[ ]:


Difference(df)


# Here $(N_{diff}+\Delta N_{diff}=-207+30=-177<0)$, $(R + \Delta R = 0.38+0.02=0.40<0.5)$
# 
# Thus, for an average Titanic passenger the most probable outcome is to die. If we had no other information in our data besides the column 'Survived', the prediction would be all zeroes, and the accuracy of this model would be 0.60-0.64. Luckily, we have several more columns in our dataset.
# 
# The goal now is to search for groups of people that are more likely to survive than die to improve the base 'all dead' model.
# - - -
# ### 3.4. Applying 'Difference' on female subframes 

# In[ ]:


print('females, class 1:')
Difference(df_F_cl1)
print(' ')
print('females, class 2:')
Difference(df_F_cl2)
print(' ')
print('females, class 3:')
Difference(df_F_cl3)


# Females in the 1st and 2nd class have about 90% of survival chance. 
# 
# Although, there are three females in the 1st class and six females in the 2nd class that died. Can we tell why? Let's take a look at them.
# - - -
# #### Looking at dead 1st class females

# In[ ]:


df_F_cl1[df_F_cl1['Survived']==0]


# What do you see here? Do these females have something in common?
# 
# I cannot tell you anything special about 50-year-old Miss. Ann.
# 
# However, I see that Miss. Helen and Mrs. Hudson share the same ticket and the same last name. 25-year-old Hudson looks very much like a mother of 2-year-old Helen, and they probably were somewhere together when it was time to leave the ship and board to a boat. Maybe Hudson was searching for Helen. Maybe Hudson fell on the floor, hit her head and lost her mind, and couldn't take care of herself nor Helen. 
# - - -
# Now
# #### Looking at dead 2nd class females

# In[ ]:


df_F_cl2[df_F_cl2['Survived']==0]


# 
# They are all adult females embarked at 'S'. 
# - - -
# #### Let us check whether these factors kick them out of a 'likely to survive group' or are these deathes simply a random fluctuation.

# In[ ]:


print('females, class 2, adult:')
Difference(df_F_cl2[df_F_cl2['Age']>20])
print(' ')
print('females, class 2, embarked S:')
Difference(df_F_cl2[df_F_cl2['Embarked']=='S'])
print(' ')
print('females, class 2, adult & embarked S:')
Difference(df_F_cl2[(df_F_cl2['Age']>20) & (df_F_cl2['Embarked']=='S')])


# #### Thus, 2nd class females in these groups, where six dead females belong, still have 90% probability to survive, and these six deathes possibly could not be predicted.
# - - -
# ### 3.5. Applying 'Difference' on male subframes

# In[ ]:


print('males, class 1:')
Difference(df_M_cl1)
print(' ')
print('males, class 2:')
Difference(df_M_cl2)
print(' ')
print('males, class 3:')
Difference(df_M_cl3)


# #### The dead males statistically significantly outnumber alive males in any ticket class. 

# - - -
# ### 3.6. Conclusion of Section 3
# 
# It is found that females in the 1st and 2nd class are more likely to survive.
# 
# The model is improved from
# 
# **'all died' **
# 
# to
# 
# **'if the 1st or 2nd class female, then survived, otherwise died'**
# - - -
# The next steps are to look at other features for the 3rd class females and all classes males to search for groups that are more likely to survive than to die. 
# - - -
# ## 4. Searching for survivors among Females in class 3
# - - -
# ### 4.1. Explore Age

# In[ ]:


g = sns.FacetGrid(data=df_F_cl3,col='Survived')
g.map(sns.distplot,'Age')


# In[ ]:


print('females, 3rd class, Age<14')
Difference(df_F_cl3[df_F_cl3['Age']<14])
print(' ')
print('females, 3rd class, Age<6')
Difference(df_F_cl3[df_F_cl3['Age']<6])
print(' ')
print('females, 3rd class, Age>40')
Difference(df_F_cl3[df_F_cl3['Age']>40])
print(' ')
print('females, 3rd class, Age<40')
Difference(df_F_cl3[df_F_cl3['Age']<40])


# **Survival rates in groups younger than 14 and younger than 6 are still indistinguishable from 0.5. I may or may not assign 3rd class girls as survivors; it statistically should not affect the performance of the model.**
# 
# While females of Age>40 are more likely to die, it still doesn't significantly change a 50/50 situation for females of Age<40. Thus, even though I know from the study above that 3rd class females older than 40 are more likely to die than survive, it doesn't help me to improve my model because I already predict all 3rd class females to die and am searching for groups that has >50% chance to survive.

# - - -
# ### 4.2. Explore Parch and SibSp
# 
# ** Looking at SibSp **

# In[ ]:


sns.countplot(x='SibSp',hue='Survived',data=df_F_cl3)


# There are more survived 3rd class females among those travelling without a sibling or spouse than among those travelling with one (or more). 
# 
# **Is that also the case for the entire DataFrame?**

# In[ ]:


sns.countplot(x='SibSp',hue='Survived',data=df)


# No, for the entire DataFrame the situation is exactly the opposite.
# 
# Now
# 
# ** Looking at Parch**

# In[ ]:


sns.countplot(x='Parch',hue='Survived',data=df_F_cl3)


# And the same for the entire DataFrame again

# In[ ]:


sns.countplot(x='Parch',hue='Survived',data=df)


# Ok, but are the found differences statistically significant?

# In[ ]:


print('females, 3rd class, SibSp==0:')
Difference(df_F_cl3[df_F_cl3['SibSp']==0])
print(' ')
print('females, 3rd class, Parch==0:')
Difference(df_F_cl3[df_F_cl3['Parch']==0])
print(' ')
print('females, 3rd class, SibSp==0 and Parch==0:')
cond1=(df_F_cl3['SibSp']==0)
cond2=(df_F_cl3['Parch']==0)
Difference(df_F_cl3[cond1 & cond2])


# Not really significant. The situation when a 3rd class female has neither a sibling/spouse nor a child/parent is slightly better. I will add these females into the model as survivors, but I don't have to.  
# - - -
# ### 4.3. Explore embarkation port

# In[ ]:


sns.countplot(x='Embarked',hue='Survived',data=df_F_cl3)


# Females in the 3rd class embarked at 'Q' have much higher survival rate than those embarked at other ports.
# 
# **Is that also the case for all passengers? **

# In[ ]:


sns.countplot(x='Embarked',hue='Survived',data=df)


# No, an average passenger has the highest chance to survive if embarked at 'C'.
# - - -
# Ok, going back to the 3rd class females.
# 
# ** Is the difference for those embarked at 'Q' statistically significant?**
# 
# And what about 'C'?

# In[ ]:


print('females, 3rd class, embarked at Q:')
Difference(df_F_cl3[df_F_cl3['Embarked']=='Q'])
print(' ')
print('females, 3rd class, embarked at C:')
Difference(df_F_cl3[df_F_cl3['Embarked']=='C'])


# ** Indeed, the higher chance for survival for the 3rd females embarked at 'Q' IS statistically significant!**
# 
# I'm adding this to the model
# - - -
# ### 4.4. Explore CabType

# In[ ]:


sns.countplot(x='CabType',hue='Survived',data=df_F_cl3)


# There are very few 3rd females in cabins 'F' and 'E' but they all survived.
# 
# Is it statistically significant?

# In[ ]:


Difference(df_F_cl3[(df_F_cl3['CabType']=='F')|(df_F_cl3['CabType']=='E')])


# It is not.
# - - -
# ### 4.5. Explore Fare
# 
# Looking at the fare distribution for the 3rd class females who survived and who did not

# In[ ]:


g = sns.FacetGrid(data=df_F_cl3,col='Survived')
g.map(sns.distplot,'Fare')


# It seems that the 3rd class females who survived tend to have lower fare 
# - - -
# Looking closer

# In[ ]:


g = sns.FacetGrid(data=df_F_cl3,col='Survived')
g.map(sns.distplot,'Fare',bins=7)
plt.xlim(5,40)


# Yes, survived 3rd females tend to have lower fares. 
# 
# Are you surprised? Do not be. Do you remeber that females without companions have a little higher survival chances? Here they are. The fare is the price for the whole party, so if you traveling alone, your fare is smaller.
# 
# **Let us see if we can cut on fare to find more survivors.**

# In[ ]:


print('females, 3rd class, fare<8:')
Difference(df_F_cl3[df_F_cl3['Fare']<8])
print(' ')
print('females, 3rd class, fare<10:')
Difference(df_F_cl3[df_F_cl3['Fare']<10])
print(' ')
print('females, 3rd class, fare<15:')
Difference(df_F_cl3[df_F_cl3['Fare']<15])
print(' ')
print('females, 3rd class, fare<20:')
Difference(df_F_cl3[df_F_cl3['Fare']<20])


# Yes, a cut on fare at 8 ('Fare'<8) creates a sample of the 3rd class females with sirvivors outnumbering dead with a significance. 
# - - -
# ### 4.6. Explore Title

# In[ ]:


sns.countplot(x='Title',hue='Survived',data=df_F_cl3)


# No, we cannot find survivors among the 3rd class females based on their title.
# - - -
# ### 4.7. Conclusion of the Section 4
# 
# Based on the careful exploration of the 3rd class females subframe, two changes were introduced to the model:
# 
# 1) The 3rd class females that were embarked at 'Q' are survived
# 
# 2) The 3rd class females that have neither a sibling/spouse nor a parent/child are survived
# 
# 3) The 3rd class females that have fare<8 are survived
# - - -
# ## 5. Searching for Survivors among Males
# - - -
# ### 5.1. Explore Age

# In[ ]:


g = sns.FacetGrid(data=df_M,col='Pclass',row='Survived')
g.map(sns.distplot,'Age')


# ![](http://)** Are you amazed to see that ALL little boys in the 1st and 2nd class have survived?**
# 
# The third class boys also had higher chance to survive than grown-up males but require more careful exploration.
# - - -
# Searching for age cut-off for the 1st and 2nd class boys

# In[ ]:


df_M_cl12 = df_M[(df_M['Pclass']==1)|(df_M['Pclass']==2)]
print('males, 1st and 2nd class, Age<14:')
Difference(df_M_cl12[df_M_cl12['Age']<14])
print(' ')
print('males, 1st and 2nd class, Age<16:')
Difference(df_M_cl12[df_M_cl12['Age']<16])
print(' ')
print('males, 1st and 2nd class, Age<17:')
Difference(df_M_cl12[df_M_cl12['Age']<17])
print(' ')
print('males, 1st and 2nd class, Age<18:')
Difference(df_M_cl12[df_M_cl12['Age']<18])


# 100% of boys under 14 and 16 are survived, and starting from 16 years old we start to see dead people. Since there is nobody between 14 and 16 years old, from this sample, we can't identify where exactly to introduce cutoff.
# - - -
# Now
# 
# **Looking at boys in the 3rd class**

# In[ ]:


print('males, 3rd class, Age<14')
Difference(df_M_cl3[df_M_cl3['Age']<14])
print(' ')
print('males, 3rd class, Age<5')
Difference(df_M_cl3[df_M_cl3['Age']<5])


# No, it's not possible to find a survivors group among the 3rd class males based solely on the age
# - - - 
# ** Now, when we know that all males in the 1st and 2nd class under 14 years old survived, we will exclude them from the further consideration**

# In[ ]:


df_M_cl12_Adult = df_M_cl12[df_M_cl12['Age']>14]# adult males, 1st and 2nd class
df_M_cl1_Adult = df_M_cl1[df_M_cl1['Age']>14]# adult males, 1st class
df_M_cl2_Adult = df_M_cl2[df_M_cl2['Age']>14]# adult males, 2nd class

# adult males from 1st and 2nd class and all males from 3rd class:
df_M_Further = pd.concat([df_M_cl12_Adult, df_M_cl3])


# - - -
# ### 5.2. Explore Parch and Sibsp

# In[ ]:


sns.countplot(x='SibSp',hue='Survived',data=df_M_Further)


# In[ ]:


sns.countplot(x='SibSp',hue='Survived',data=df_M_cl1_Adult)


# In[ ]:


sns.countplot(x='Parch',hue='Survived',data=df_M_cl1_Adult)


# No survival groups found based on Parch or SibSp
# - - -
# ### 5.3. Explore Embarkation Port

# In[ ]:


sns.countplot(x='Embarked',hue='Survived',data=df_M_cl1_Adult)


# **Checking 3rd class boys.** Since we found 3rd class females had higher chance to survive if they were embarked at 'Q' maybe the same applies for children

# In[ ]:


sns.countplot(x='Embarked',hue='Survived',data=df_M_cl3[df_M_cl3['Age']<14])


# No, there were four 3rd class boys embarked at 'Q', and none of them survived. With the embarkation port 'C', two boys survived while one boy died. This excess is not statistically significant.
# 
# We didn't find any survival groups among males based on the embarkation port.

# - - -
# ### 5.4. Explore CabType
# - - -
# Looking at 1st class adult males

# In[ ]:


sns.countplot(x='CabType',hue='Survived',data=df_M_cl1_Adult)


# In[ ]:


Difference(df_M_cl1_Adult[(df_M_cl1_Adult['CabType']=='E')])


# Looking at all males except 1st and 2nd class children

# In[ ]:


sns.countplot(x='CabType',hue='Survived',data=df_M_Further)


# While some cabin types increase one's chances to survive, none of make an adult male survival the most probable outcome.
# - - - 
# ### 5.5. Explore Fare
# - - -
# Looking at the 1st class adult males fares

# In[ ]:


g = sns.FacetGrid(data=df_M_cl1_Adult,col='Survived')
g.map(sns.distplot,'Fare')


# There is a very expensive outlier in the survivor's distribution.
# 
# Is it statistically significant?

# In[ ]:


Difference(df_M_cl1_Adult[df_M_cl1_Adult['Fare']>400])


# Let's look at this two males

# In[ ]:


df_M_cl1_Adult[df_M_cl1_Adult['Fare']>400]


# Ok, they are from different parties, one is travelling alone, the other one is travelling with a child. Barely two instances are enough to introduce a change into the model.
# 
# What about fare distribution for the 1st class adult males travelling alone?

# In[ ]:


g = sns.FacetGrid(data=df_M_cl1_Adult[(df_M_cl1_Adult['SibSp']==0)&(df_M_cl1_Adult['Parch']==0)],col='Survived')
g.map(sns.distplot,'Fare')


# In[ ]:


g = sns.FacetGrid(data=df_M_cl1_Adult[(df_M_cl1_Adult['SibSp']==0)&(df_M_cl1_Adult['Parch']==0)],col='Survived')
g.map(sns.distplot,'Fare')
plt.xlim(0,100)


# In[ ]:


Difference(df_M_cl1[(df_M_cl1['SibSp']==0)&(df_M_cl1['Parch']==0)&(df_M_cl1['Fare']<40)&(df_M_cl1['Fare']>20)])


# - - -
# ### 5.6. Explore Title

# In[ ]:


sns.countplot(x='Title',hue='Survived',data=df_M_cl1_Adult)


# In[ ]:


sns.countplot(x='Title',hue='Survived',data=df_M_Further)


# In[ ]:


Difference(df_M_cl1_Adult[df_M_cl1_Adult['Title']=='Dr.'])


# ### 5.7. Conclusions of the Section 5
# - - -
# Based on the exploration of males subframes, we introduced a change into the model.
# 
# Instead of 
# 
# ** 'all males died' **
# 
# it is
# 
# ** 'if the 1st or 2nd class and under 14 years old, than survived, othervise died' **
# 
# No other groups of males that are more likely to survive than not were found
# - - -
# ## 6. Prepare the submition file

# In[ ]:


dfNew=pd.DataFrame(columns=('PassengerId', 'Survived'))
dfTest = pd.read_csv('../input/test.csv')

for i in range(418):
    surv=0       
    PassId=dfTest.loc[i]['PassengerId']
    
    # females
    if (dfTest.loc[i]['Sex']=='female'): 
        if (dfTest.loc[i]['Pclass']==1): surv=1
        if (dfTest.loc[i]['Pclass']==2): surv=1
        if (dfTest.loc[i]['Pclass']==3): 
            if (dfTest.loc[i]['Embarked']=='Q'): surv=1
            if ((dfTest.loc[i]['Parch']==0) & (dfTest.loc[i]['SibSp']==0)): surv=1
            if (dfTest.loc[i]['Fare']<8): surv=1
    
    #males
    if (dfTest.loc[i]['Sex']=='male'):
        if (dfTest.loc[i]['Pclass']==1): 
            if (dfTest.loc[i]['Age']<14): surv=1
        if (dfTest.loc[i]['Pclass']==2):
            if (dfTest.loc[i]['Age']<14): surv=1
            
    dfNew.loc[i] = pd.Series({'PassengerId':PassId,'Survived':surv})

dfNew.to_csv('submitClass.csv',index=False)


# The model described in this kernel gives a score of 0.77512

# In[ ]:




