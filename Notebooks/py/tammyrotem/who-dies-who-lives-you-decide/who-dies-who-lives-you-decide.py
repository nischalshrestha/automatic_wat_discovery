#!/usr/bin/env python
# coding: utf-8

# # Who Lives? Who Dies? You Decide! #
# ## (Actually no, a model will, but you get my drift) ##
# ### Tammy Rotem #### 
# 
# Thanking [Megan Risdal](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic) for her great work with Titanic Survival, which inspired this notebook.
# 
# 1. [Introduction](#introduction) <br/>
#     1.1 [Load packages and check data](#loading) <br/>
# 2. [Feature Engineering](#featureeng)<br/> 
#     2.1 [Passenger Name](#featurename) <br/>
#     2.2 [Reduce Title to Mr., Mrs., Miss. and Master](#title_reduce)<br/>
#     2.3 [How many relatives a passenger has on board?](#famsize)<br/>
# 3. [Handle Missing Values](#handle_missing)<br/>
# 
# 
# 

#  # 1 Introduction <a class="anchor" id="introduction"></a>
#  This is my first competition submission and I'm super excited about it. I really loved what Megan did with this dataset using R, so I thought I should try doing the same with Python. The reason for this is I've been working with R quite a lot in the past few years and I want to be as proficient in Python (First Python notebook!).
#  
# > In addition, I will try to do things the way I think they should be done in a near production environment - meaning, all manipulations on training data will be designed in functions which in reality will be applied to any new data awaiting prediction.
#  
#  So here I go!

#  ## 1.1 Load packages and check data <a class="anchor" id="section1"></a>
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from sklearn import tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Now that our packages our loaded, let's get the data. I'll load the training and test sets each on its own.
# 
# In addition, I will combine both datasets so I can perform feature engineering properly. 
# What I mean is: If I'm tranforming a Nominal column (like Embarked or Ticket), and only looking at the training set - there may be values in the testing set that I didn't know about or seen before - so I won't know how to transform them before executing a predictive model.
# 
# In reality - this is a tricky question - how to transform previously unknown values correctly? When maintaining a model - a lot of attention should be given to all the data that passes through it so It will be handled in the best way. If new values are found in the future - they should be added to the transformation process. **This is the reason why I combine the training and testing sets for feature engineering.**
# 
# 

# In[2]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
#Function to append sets with an idetifying partition field for later seperation or other uses
def append_train_test(train,test): 
    train["Partition"]="train"
    test["Partition"]="test"
    df=train.append(test)
    return df;

full=append_train_test(train,test)

train.info()
print ("------------------------")
test.info()
print ("------------------------")
full.info()


# We notice we have some missing values we should fill (Age & Cabin). Let's look at some examples of the records (the full dataset is a good place to look). At this point we know our data pretty well, the variables we have and their types - so we can move on to feature engineering.

# In[ ]:


full.head()


# # 2 Feature Engineering <a class="anchor" id="featureeng"></a>
# This part is dedicated to data enrichment by creating some new variables. This step is important becuase the set of variables we got with the dataset may not be the best seperators of survival. Deriving new variables from the given set may prove to be better seperators.
# 
# ## 2.1 Get the honorary title from Name column <a class="anchor" id="namefeature"></a>
# What we intend to achieve here is to get more variables out of passenger name. Why? Using the  name in our model will be a lot like using a passenger's id - each passenger has an almost unique value. Instead, we want to look at passeger similarity (or difference) at a higher level. In this case, the honorary **title**  helps group similar passengers, and this title is present in the **name** variable. 
# 
# I'll use regular expressions to extract substrings which end with a dot (e.g. Miss., Master., etc.) from the Name column. I'll put this result into a new column named "Title". The regex I use aims to match a group of words which can be any alphanumeric combination which ends with a dot. I'll be writing this as a custom function, becuase in reality this would be applied to any new data in need of prediction.
# 
# After I create the new variable **Title** I look at all available values and their counts.
# 
# 

# In[3]:


#remember that \w is equivalent to [A-Za-z] all alphanumeric chars, dot must be escaped with "\" 
#and a match group (like the word) is enclosed in "()", the + is a quantifier meaning one or more of \w.
def get_title(df,name_column,new_column): #call function example: get_title(train,"Name","Title")
    df[new_column] = df[name_column].str.extract('(\w+)\.', expand=True)
    return df;
#I then pass the entire dataset to the function to generate the Title column for all the records
get_title(full,"Name","Title")
full.Title.value_counts() #Looks at all available values and their counts


# Once we look at the values we can see that several values are incredibly rare, such as "Don" or "Mme". Leaving this column the way it is will result in a possbile loss of information. Even though there are very little records with these "rare" titles - we don't know how many will be in the test set. So, let's get these in order.
# 
# Time to find equivalence in these titles! According to Wikipedia:<br/>
# Mme = "Medame" = Mrs <br/>
# Mlle = "Mademoiselle" = Miss <br/>
# Ms = "Miz" = Miss <br/>
# 
# In addition, wikipedia teaches us that other titles in this list refer to nobility (from different nationalities) such as: Sir, Lady, Countess and more. Other titles, are military titles such as Col, Major and Capt. There are also other proffesionals like Dr and Reverend.
# 
# Finally, I want the **Title** variable to be reduced to only Mr.,Mrs.,Miss. and Master. In many cases I would like to indicate for certain cases if they have special titles (like nobility or military) but these are so low in frequency - it is redundant.
# 
# ## 2.2 Reduce title to Mr., Mrs., Miss. and Master <a class="anchor" id="title_reduce"></a>
# Some assumptions: Reverends are all men right? How about Doctors? Colonels, Majors, Captains? Let's check this out:

# In[4]:


full[full.Title.isin(["Rev","Dr","Col","Capt","Major"]) & (full.Sex !="male")]


# Well, looks like we have a female doctor on board - and in fact she was a survivor! So, now when we re-encode these titles we will encode with respect to sex as well. If we think about passengers in this day and age - we could have military officials who are females.
# 
# We'll start with a new column with null values, called Reduced_Title, into which we can re-encode all titles to just Mr, Mrs, Miss and Master.

# In[5]:


full.loc[((full.Title.isin(["Rev","Capt","Major","Col","Jonkheer","Don","Sir","Dr"])) & (full.Sex=="male")),"Title"] = "Mr"
full.loc[(full.Title.isin(["Countess","Lady","Dona","Mme"])),"Title"]="Mrs"
full.loc[((full.Title.isin(["Mlle","Ms","Dr"])) & (full.Sex=="female")),"Title"]="Miss"

full.Title.value_counts(normalize=True)


# Now we have a new nominal variable **Title** with a pretty good distribution of values (~59%, 20%,15%,4%). All categories are represented pretty well. This is a good time to see the relationship between this new variable and sruvival.

# In[6]:


sns.countplot('Title',hue='Survived',data=full.iloc[:891])
plt.title("Honorary Title vs Survived")
plt.show()


# Now we know for sure that the "women and children first" policy was indeed enacted on board the Titanic. Women (Especially married women, maybe mothers) have a better chance to survive, so do children.

# ## 2.3 How many relatives a passenger has on board? <a class="anchor" id="famsize"></a>
# We want to use **SibSp** (number of siblings/spouses on board) plus **Parch** (number of parents/children on board) to tell the size of the passenger's family. It is logical to think there is a relationship between family size and survival - having people to look after you, and make sure you get onboard a rescue boat.

# In[7]:


full["FamilySize"]= full["SibSp"] + full["Parch"] + 1


# In[8]:


f,ax=plt.subplots(1,2,figsize=(18,8))
#Let's look at how family size is connected with Title
sns.countplot('FamilySize',hue='Title',data=full.iloc[:891],ax=ax[0])
ax[0].set_title('Family Sizs vs Title')
#Let's look at how survival is distributed along family sizes
sns.countplot('FamilySize',hue='Survived',data=full.iloc[:891],ax=ax[1])
ax[1].set_title('Family size vs Survived')
plt.show()


# The left plot tells us that most lone travellers on the Titanic were adult men, after them young/unmarried women. The right plot tells us this group of people has the slightest chance to survive (Travelling alone never sounded worst!) ->These will be our lone travellers.
# Furthermore, only in families of sizes 2,3 and 4 there is a greater chance to survive, when family size is over 4 again there is a smaller chance to survive. Let's describe that in a new **discrete family size variable**
# 

# In[9]:


full["FamilySizeBand"]=np.nan
full.loc[full["FamilySize"]==1,"FamilySizeBand"]="Loner"
full.loc[(full["FamilySize"]<5) & (full["FamilySize"]>1),"FamilySizeBand"]="SmallFam"
full.loc[full["FamilySize"]>4,"FamilySizeBand"]="BigFam"
#Let's look a survival rate within classes and family sizes
sns.factorplot('FamilySizeBand','Survived',hue='Pclass',col="Sex",data=full.iloc[:891])
plt.show()


# Here is an interesting picture - We already know women are way more likely to survive than men, In addition - women are more likely to survive even when they are part of a big family on board, as long as they are from 1st or 2nd class.
# Men have a better chance to survive if they are part of small families, regardless of class, whilst women would rather be loners (especially from 3rd class) or from big families (and not 3rd class).
# 
# # 3 Handle Missing Values <a class="anchor" id="handle_missing"> </a>
# We know we have some missing values we have to take care of before we go further and move to modeling. I would like to start with the "trivial" imputation (few missing values which can be handled manually, and then examine ways to impute many missing values)
# 
# ## 3.1 How many records need imputing? <a class="anchor" id="count_nulls"></a>

# In[35]:


full.isnull().sum()


# Looks like we have ** 263 missing Age values **,
# ** 1014 missing Cabin values **,
# ** 2 missing Embarked values **,
# ** 1 missing Fare value **
# 
# The line that says 418 null "Survived" values are'nt really mising - that's the test set.

# ## 3.2 Missing Fare Value <a class="anchor" id="missing_fare"></a>
# The best approach I see to impute a missing continuous value like this one is to see if we can infer the required value from other fetures.
# Let's start by finding this row in the data:

# In[36]:


full[full.Fare.isnull()]


# We are looking at a loner, who embarked from Southampton, aged 60.5 and is a man from 3rd class. Let's look at the fare distribution of this group of people.

# In[74]:


fare=full[(full.Embarked=="S") & (full.Title=="Mr") & (full.FamilySizeBand=="Loner") & (full.Pclass==3) & (full.Fare.notnull())]
sns.distplot(fare.Fare)
fare.Fare.describe()


# Up to 75% of 277 passengers (i.e 170 and up) paid less or equal to 8 dollars per ticket. I think it is safe to put this missing fare at the 75% quartile, meaning at 8 dollars. It really makes me wonder about these 3rd class passengers who paid over 10$ per ticket! Why? this is extreme behavior! we might want to consider removing extreme values from the Fare variable later on.

# In[77]:


# imputing the missing fare value
full.loc[(full.Embarked=="S") & (full.Title=="Mr") & (full.FamilySizeBand=="Loner") & (full.Pclass==3) & (full.Fare.isnull()),"Fare"]=8


# ## 3.3 Missing Embarked Values <a class="anchor" id="missing_emb"></a>
# Now we need to take a look at out missing embarkment ports. These are two records we're looking for:

# In[79]:


full[full.Embarked.isnull()]


# Very interesting, there are common factors to these passengers with missing embarkment:
# Both are women (although one is married, the other not neccesarily), both travelling alone, and paid equal fairs for the ticket. In addition, both are from 1st class.
# It is a good time to look at relationships between embarkment and fare:

# In[95]:


sns.boxplot(x="Embarked", y="Fare",data=full[full.Fare.notnull()])
medians = full[full.Pclass==1].groupby(['Embarked'])['Fare'].median().values


# In[ ]:


import seaborn as sns, numpy as np

sns.set_style("whitegrid")
tips = sns.load_dataset("tips")
ax = sns.boxplot(x="day", y="total_bill", data=tips)

medians = tips.groupby(['day'])['total_bill'].median().values
median_labels = [str(np.round(s, 2)) for s in medians]

pos = range(len(medians))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], medians[tick] + 0.5, median_labels[tick], 
            horizontalalignment='center', size='x-small', color='w', weight='semibold')


# Most people embarking from Queenstown paid very few dollars for ambarking

# In[78]:


sns.boxplot(x="Pclass", y="Fare", data=full[full.Fare.notnull()])


# In[ ]:





#Data Preparation

#Var 1: Passenger Class (Pclass)
y=train.Pclass.value_counts()


#The majority of passengers are in 3rd class, #Partition data
features=train.drop("Survived",axis=1)
target=train[['Survived','PassengerId']]
features.head()
target.head()


# In[ ]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(features.drop("PassengerId",axis=1), target.drop("PassengerId",axis=1))

