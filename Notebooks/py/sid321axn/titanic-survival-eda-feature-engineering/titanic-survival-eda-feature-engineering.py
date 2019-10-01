#!/usr/bin/env python
# coding: utf-8

# # Welcome to my Kernel
# 
# In this dataset we will do some **Feature Engineering** and we have to predict the survival status of each of the passenger by building our machine learning model <br>
# 
# But the question is What actually Feature engineering is ?<br>
# 
# **Feature engineering** is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive. The need for manual feature engineering can be obviated by automated feature learning.
# 
# Feature Engineering can simply be defined as the process of creating new features from the existing features in a dataset. Let’s consider a sample data that has details about a few items, such as their weight and price.
# 
# <img src="https://image.ibb.co/dHgyR9/feat-engg-example.png" alt="text">****
# 
# 

# Now, to create a new feature we can use Item_Weight and Item_Price. So, let’s create a feature called Price_per_Weight. It is nothing but the price of the item divided by the weight of the item. This process is called **feature engineering.**
# 
# <img src="https://image.ibb.co/n3wyR9/feat-engg-example-2.png">

# But apart from Model building and Feature Engineering we will answer some of the very interesting questions through EDA and Data Visualization which are given below : <br>
# **Q1. Who were the passengers on the Titanic ? (i.e.,  Age, Gender, Class etc.)**<br>
# **Q2. What deck were the passengers on and how does that relate to their class ?**<br>
# **Q3. Where did the passengers come from ?**<br>
# **Q4. Who was alone and who was with family ? ** <br>
# **Q5. What factors helped someone survived the sinking ?**

# **Lets start by first importing essential libraries required**

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic_df=pd.read_csv('../input/train.csv')


# # DATA PREPROCESSING

# In[ ]:


titanic_df.head()


# In[ ]:


titanic_df.info()


# In[ ]:


titanic_df.isnull().sum()


# There are columns which have null values Age, Cabin and Embarked as Age is the numerical column we will fill it with the mean value of Age.

# In[ ]:


titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)


# In[ ]:


sns.countplot(x='Sex', data=titanic_df)


# From above figure, It seems that  there are  more males than females on titanic ship

# In[ ]:


sns.countplot(x='Pclass', data=titanic_df, hue='Sex')


# From above figure it is inferred that there are more **males** in **Pclass 3**  and same in case of **females** too

# # Feature Engineering

# In[ ]:


def male_female_child(passenger):
    age,sex = passenger
    
    if age<10:
       return 'child'
    else :
       return sex


# In[ ]:


titanic_df['person']=titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[ ]:


titanic_df.head(10)


# That's great now we have a separate column that will tell whether the person is male, female or child

# In[ ]:


sns.countplot(x='Pclass', data=titanic_df, hue='person')


# It seems from the above figure that in Pclass3 male,female and child all have higher population

# In[ ]:


titanic_df['Age'].hist(bins=40)


# It is inferred from the above figure that more young people having age between 20 to 40 was on board

# In[ ]:


titanic_df['Age'].mean()


# It means the average age of people on titanic ship is around 30 years

# Now lets see the person wise count 

# In[ ]:


titanic_df['person'].value_counts()


# In[ ]:


fig=sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[ ]:


fig=sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[ ]:


fig=sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[ ]:


deck=titanic_df['Cabin'].dropna()


# In[ ]:


deck.head()


# In[ ]:


levels=[]

for level in deck:
    levels.append(level[0])
    
cabin_df=DataFrame(levels)
cabin_df.columns=['Cabin']


# In[ ]:


sns.countplot('Cabin',data=cabin_df,palette='summer')


# It seems that more people belongs to Cabin C

# In[ ]:


sns.countplot('Embarked',data=titanic_df,hue='Pclass')


# In[ ]:


#who was alone and who was with family
titanic_df['Alone']=titanic_df.SibSp + titanic_df.Parch


# In[ ]:


titanic_df['Alone'].loc[titanic_df['Alone']>0]='With family'
titanic_df['Alone'].loc[titanic_df['Alone']==0]='Alone'


# In[ ]:


titanic_df.head()


# In[ ]:


sns.countplot('Alone',data=titanic_df,palette='Blues')


# In[ ]:


titanic_df['Survivor']=titanic_df.Survived.map({0:'no',1:'yes'})


# In[ ]:


sns.countplot('Survivor', data=titanic_df, palette='Set1')


# It seems from the above figure that maximum passsengers not survived.

# In[ ]:


sns.factorplot('Pclass','Survived',data=titanic_df)


# It seems that passengers who are in Pclass 1 have higher survival rate and Pclass3 have very less survival rate.

# In[ ]:


sns.factorplot('Pclass','Survived',hue='person',data=titanic_df,palette='Set2')


# From the above graph it is clear that males who are in Pclass3 had very less survival rate in comparison to females and child.
# In Pclass 1 most of the females had survived then child and at the last males who had a very less survival rate.
# From the above figure it is inferred that , in overall very less males are survived and most of the females are survived in this mishap.

# In[ ]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[ ]:


sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df)


# In[ ]:


generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter', x_bins=generations)


# In[ ]:


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter', x_bins=generations)


# It seems that older females have better survival rate in comparison to older males.

# In[ ]:


sns.countplot('Survivor',hue='Alone', data=titanic_df, palette='summer')


# It is inferred from the above figure that those people were survived who were with their  family than alone. 

# In[ ]:


sns.factorplot('Sex','Survived',hue='Alone', data=titanic_df, palette='winter')


# It is inferred from the above figure that females who were Alone  have better survival rate than with family but in case of males who were with family have better survival rate than Alone.

# In[ ]:


sns.countplot('Survivor',hue='Embarked', data=titanic_df, palette='Blues')


# One interesting thing inferred from the above graph that those passengers who came from Southampton have high survival rate and higher mortality rate as well . But this might be because more people belongs to Southampton in comparison to other areas.

# In[ ]:


titanic_ndf = titanic_df.dropna()

def Deck(Cabin):

        if Cabin[0] == 'A':

          return 'A'

        elif Cabin[0] == 'B':

         return 'B'

        elif Cabin[0] == 'C':

         return 'C'

        elif Cabin[0] == 'D':

         return 'D'

        elif Cabin[0] == 'E':

         return 'E'

        elif Cabin[0] == 'F':

         return 'F'

        elif Cabin[0] == 'G':

         return 'G'

        else:

         return np.NaN

titanic_ndf['Deck'] = titanic_ndf['Cabin'].apply(Deck)


# In[ ]:


titanic_ndf.head()


# In[ ]:


sns.factorplot('Deck','Survived',hue='person',data=titanic_ndf,palette='Blues')


# It is inferred from the above figure that Childrens who were in Cabin E have higher survival rate but it gradually decreases in Cabins G and C and as far as females are concerned they have higher survival rate in Cabin C, E ,D and B.
# Males who were in Cabin E and D have better survival rate than other Cabins.

# Now, I just want to engineer one more feature named class according to the cost of fare I am just classifying the passengers having less than 25 $ fare to Economy class (E) more than 25 and less than 50$ to Middle class (M) and more than 50$ to Premium class.

# In[ ]:


def Class(Fare):

        if Fare <=25:

          return 'E'

        elif Fare >25 and Fare<=50:

         return 'M'

        elif Fare > 50:

         return 'P'

        

titanic_df['Class'] = titanic_df['Fare'].apply(Class)


# In[ ]:


titanic_df.head(5)


# In[ ]:


sns.factorplot('Class','Survived',hue='person',data=titanic_df,palette='winter')


# From the above figure it is inferred that those females,males and child who have premium class fare have higher survival rate than other classes of fare where as in Economy class more childs are survived but in  premium and middle class more females were survived.

# **Hii guys I hope u liked my kernel if like then kindly upvote and please do comment if any further additions required.**

# In[ ]:




