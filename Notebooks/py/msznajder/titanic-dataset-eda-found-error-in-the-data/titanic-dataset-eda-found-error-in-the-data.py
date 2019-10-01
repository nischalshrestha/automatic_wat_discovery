#!/usr/bin/env python
# coding: utf-8

# # Titanic passengers survival analysis Kaggle competition
# 
# In this notebook we will conduct data analysis of Titanic passengers survival dataset for Kaggle [Titanic dataset](https://www.kaggle.com/c/titanic).
# 
# Titanic passengers survival dataset is one of the most canonical data analysis and machine learning datasets. Let's see how this data looks like, investigate main trends in it and try to predict survival chances of passengers based on numerous data attributes.
# 
# The main steps of the analysis process we will conduct are:
# 1. [Framing the problem](#framing)
# 2. [Data wrangling](#wrangling)
# 3. [Data exploration](#exploring)
# 4. [Summary](#summary)
# 

# In[1]:


import numpy as np

from scipy import stats
from statsmodels.formula.api import ols

import pandas as pd
from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')


# ## 1. Framing the problem 
# 
# Titanic sinking was one of the biggest shipwreck tragedies in history. It killed 1502 out of 2224 passengers. This tragedy led to better safety regulations for ships. The main reason for such a great death toll was not enough number of lifeboats to fit all passengers. This can lead to the conclusion that some groups of people were more likely to survive than others - woman, children and upper-class passeners for example.
# 
# The goal of this analysis is to analyse the data set, explore it answering related questions using data visualization and statistical methods. There are also some questions we would like to answer with Titanic dataset analysis.
# 
# * What is Titanic passengers demographic structure analyzed in terms of attributes?
# 
# * What is the overall Titanic passengers survival ratio?
# 
# * What is the survival ratio for different demographic passengers groups? Which groups have biggest chances for survival and which smallest? Is the difference between survived group statistically significant?
# 
# 

# ## 2. Data wrangling
# 
# Let's get going by loading the dataset.
# 

# In[2]:


titanic_train = pd.read_csv("../input/train.csv")


# Let's see how our data looks like.

# In[3]:


titanic_train.head()


# Assuming that `PassengerId` attribute values are unique (and they are) let's first make `PassengerId` our index column.

# In[4]:


titanic_train = titanic_train.set_index("PassengerId")
titanic_train.head()


# As we can see the dataset contains not that many attributes: we have 11 of them. Six of them are numeric: `Survived` (informing whether passenger survived or not), `Pclass` (passenger class), `Age`, `SibSp` (number of siblings/spouses aboard), `Parch` (number of parents/children aboard), `Fare`. Out of these six `Survived` and `Pclass` attributes are numerically encoded categorical values meaning that calculating descriptive statistics for them is meaningless.
# 
# Five attributes are categorical textual values: `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked` (port of embarkation C=Cherbourg, Q=Queenstown, S=Southampton). 
# 
# Our dataset target attribute is `Survived` marking whether passenger survived or not with 1 or 0 respectively.
# 
# Let's have a closer look at attributes structure.

# In[5]:


titanic_train.info()


# All in all the dataset contains 891 passengers data out of all 2224 passengers.
# 
# We deal here with some missing values. The most missing values are in `Cabin` attribute. That column seems not so much informative and the data is rather scarse which is suggesting droping this attribute when preparing training dataset for machine learning. We have also a lot of missing values in `Age` column, however this column seems crucial for our analysis. That is why later on we will have to take care of these missing values either by filling them in (e.g. median value) or by dropping examples with missing values. In `Embarked` attribute values we miss only two values so we can easily drop these two cases later on when preparing the dataset for modelling algorithms.
# 
# We again see that five attributes have categorical/text values - we will also deal with that later when preparing data for modelling.

# Let's now see some details about numerical values we have in the dataset.

# In[6]:


titanic_train.describe()


# The only attributes for which calculating statistical values make sense are `Age`, `SibSp`, `Parch` and `Fare`  attributes. We will analyse demographical passengers structure a bit later. All in all the values look correct and it seems data needs no more corrections at this stage of the analysis.
# 
# We can also see here that attributes values ranges are very different. We will have to be standardized later on when preparing data for modeling.
# 
# We are ready to move to the data exploration phase.

#  ## 3. Data exploration
# 
# Let's now dig deeper into data internal structure and values.

# ## 3.1 Attributes analysis
# 
# Let's first look at the general numeric data distributions including the target value `Survived` attribute.

# In[7]:


titanic_train.describe()


# In[8]:


titanic_train.hist(bins=20, figsize=(18, 16), color="#f1b7b0");


# We notice a few things in these histograms: 
# - `Age` distribution is centered around 20-30 years ranging to 80 with quite large number of children aged 0-5. The minimal age value is 0.42 and the maximal is 80. The mean value is 29.6991 with quite large standard deviation of 14.5264. The median value is 28.
# - `Fare` attribute values distribution is strongly positevely skewed with mean 32.2042 and large stadard deviation of 49.6934. Because the distribution is skewed median is much smaller than the mean with value of 14.4542. The range of data is large ranging from 0 to 512. At the same time 75% of the data is lower than 31. 
# - `Parch` (the number of parents/children aboard) distribution mode is 0 by far meaning that most of the passengers traveled without any parents/children and also there are some outlier values with 3 and more parents/children aboard.
# - `Pclass` distribution shows that almost 500 out of all 891 passengers in the dataset were travelling the 3rd (lowest) class and almost 200 passengers were travelling both in 1st and 2nd class.
# - `SibSp` (number of siblings/spouses aboard) is similar to `Parch` distribution but with larger number of passengers travelling with one sibling/spouse.
# - `Survived` obviously shows only two values but we can also see that number of survivors is much smaller than the number of passengers who died in the disaster.

# Let's also look at catgorical attributes values: `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`.

# In[9]:


titanic_train["Name"].value_counts()


# `Names`, by definition, are rather unique so its distribution would be uniform. It is not categorical attribute then - just textual. This tells us that in modelling phase this value will not be of too much help since there are no group of values that model can identify.

# In[10]:


titanic_train["Ticket"].value_counts()


# `Ticket` attribute is also not a classical categorical attribute with multiple unique or close to unique values. There some values repetition but rather tickets identifiers (as we assume they are) are unique. This again can lead to the conclusion that this attribute will not be helpful in survival prediction.

# In[11]:


titanic_train["Cabin"].value_counts()


# `Cabin` is similar in structure to the `Name` and `Ticket` attributes. It is again rather textual attribute and not categorical. As we saw earlier this is the attribute with the biggest number of missing values: we have only 204 values out of all 891 passengers and the rest is missing. All this suggest that also this attribute will not be of much help during the modeling phase of this project.

# In[12]:


titanic_train["Sex"].value_counts()


# `Sex` attribute has only two possible values: `male` and `female`. It is a categorical attribute. Let's plot it.

# In[13]:


titanic_train["Sex"].value_counts().plot(kind='bar', figsize=(6, 4), grid=True, color="#f1b7b0", title="Sex")


# We see that on Titanic (as measured by analysed dataset) there were almost twice as much males as females.

# In[14]:


titanic_train["Embarked"].value_counts()


# `Embarked` attribute has three possible values: `S`, `C` and `Q` (standing for Southampton, Cherbourg or Queenstown city of embarkation) meaning that it is categorical attribute. Let's plot it's values.

# In[15]:


titanic_train["Embarked"].value_counts().plot(kind='bar', figsize=(6, 4), grid=True, color="#f1b7b0", title="Embarked")


# We can see that vast majority of passengers embarked in Southampton port, less than 200 passengers embarked in Cherbourg and less then 100 embarked in Queenstown. For now we can tell nothing more out of it but later on we will try to see how this attribute values affected the survival chances of passengers.
# 
# It looks like we have 7 attributes that we can explore for how their values affect pasengers survival chances. These are the five numerical attributes: `Pclass` (passenger class), `Age` (passenger age), `SibSp` (number of siblings/spouses aboard), `Parch` (number of parents/children aboard), `Fare` (fare passenger paid). There are also two categorical attributes that looks like factors that can be analysed for influencing passengers survival chances: `Sex` (passenger sex) and `Embarked` (port of embarkation). 
# 
# In further analysis and modelling we will concentrate on these attributes when analysing and modeling passengers survival chances. `Name`, `Ticket` and `Cabin` attributes are textual non-categorical and rather unique values and as such are hard to analyze in terms of finding some patterns or relations between them and survival chances.
# 
# As a sidenote `Sex` and `Embarked` attributes, as categorical values, can be numerically encoded. We will perform this when preparing the data for modeling stage.
# 
# We can finish now studying individual attributes properties and start looking at relations between them.

# ## 3.2 Relations between attributes analysis
# 
# Let's now dig deeper into data internal relations. Since our dataset is not very large we can create scatter plot between each of the numerical attributes.

# In[16]:


scatter_matrix(titanic_train, figsize=(18, 16), c="#f1b7b0", hist_kwds={'color':['#f1b7b0']});


# It is quite difficult to read something informative from scatter plots since some of the data is not continous.
# 
# Let's then compute the correlation matrix showing correlation coefficient between each numerical attribute.

# In[17]:


corr_matrix = titanic_train.corr()
corr_matrix


# We can see some meaningful correlations here. To get even better intuition let's visualize the correlation matrix.

# In[18]:


fig, axes = plt.subplots(figsize=(8, 8))
cax = axes.matshow(corr_matrix, vmin=-1, vmax=1, cmap=plt.cm.pink)
fig.colorbar(cax)
ticks = np.arange(0, len(corr_matrix), 1)
axes.set_xticks(ticks)
axes.set_yticks(ticks)
axes.set_xticklabels(corr_matrix)
axes.set_yticklabels(corr_matrix)
plt.show()


# Analyzing correlation let's remember that our target value is `Survived` attribute. 
# 
# Let's concentrate for now on correlation between target value and other attributes. We see quite strong positive correlation between `Fare` and `Survived` attributes values (0.2573) meaning that who payed more for the ticked could have for some reason higher chances for survival. There is also quite strong negative correlation between `Pclass` and `Survived` attributes values (-0.3385) similarly meaning that the higher class passenger was travelling the lower was risk for not surviving. The rest of the attributes seems not correlated strongly with `Survived`. 
# 
# Looking at relations between other attributes an obvious intution is strong negative correlation between `Pclass` and `Fare` (-0.5495): the more expensive ticket usually means better standard and lower class number (1st class is the most luxourious).
# 
# Other interesting insight is quite strong positive correlation between `SibSp` and `Parch` attributes (0.41)meaning that someone who travels with siblings or spouse tends to also travel with parents or children.
# 
# Surprising is positive correlation between `Parch` and `Fare` attributes (0.2162) meaning that person traveling with parents or children tends to pay more for the ticket. Similar but with lower correlation in case of `SibSp` attribute.
# 
# Other more distinct relation we can see in the dataset is negative correlation between `Pclass` and `Age` (-0.3692). We can interpret it as the lower class number (and high standard) the higher age which seems reasonable.
# 
# Also interesting is the high negative correlation between `Age` and `SibSp` (-0.3083). It conveys interesting fact that the older passenger was the lower was the number of siblings travelling with.

# We now have quite deep insight in what is going on in the dataset. We will now proceed to answering question posed at the begining of this analysis.

# ## 3.3 What was Titanic passengers demographic structure analyzed in terms of attributes?
# 
# In this part we will analyze attributes we found meaningful in terms of survival analysis: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare` and `Embarked`. We will not analyze `Name`, `Ticket` and `Cabin` attributes since we found that they do not convey information in terms of population survival analysis.
# 
# Let's start with ticket class analysis - how many passengers travelled in each of the ticket classes. On Titanic there were three tickets classes: first, second and third. They are represented in the dataset using values 1, 2 and 3.  

# In[19]:


titanic_class_counts = titanic_train["Pclass"].value_counts(sort=False)
titanic_class_counts.index = ["First class", "Second class", "Third class"]
titanic_class_counts


# We see that vast majority (491) passengers travelled in lowest third class and 216 of passengers travelled in first class. 

# In[20]:


titanic_sex_counts = titanic_train["Sex"].value_counts()
titanic_sex_counts


# Majority of all passengers were males with only 314 females.

# To analyze age we need first to bin the data into meaningful age groups.

# In[21]:


titanic_age_groups_counts = pd.cut(titanic_train["Age"], bins=[0, 14, 24, 34, 44, 54, 64, 80]).value_counts().sort_index()
titanic_age_groups_counts


# 77 passengers were children in age 0-14. A little over 200 of passengers were both between 14-24 and 201 od them were 24-34 years old. Almost 121 of all passengers were 34-44 years old. Only 115 of all passengers were above 44 years old. That tells us that Titanic passengers population was quite young.
# 
# Just out of interest let's dig a bit deeper into passengers age distribution properties.

# In[22]:


titanic_train["Age"].describe()


# We can see that the youngest passenger was 5 months old (0.42 year old). Let's see who she or he was.

# In[23]:


titanic_train.loc[titanic_train["Age"].argmin()]


# It was a boy: Assed Alexander Thomas born in Hardīn, Lebanon on 8 November 1911. He survived Titanic disaster - you can read his story [here](https://www.encyclopedia-titanica.org/titanic-survivor/assad-alexander-thomas-tannous.html).
# 
# On the other side of passengers dataset age distribution is the oldest passenger who was 80. Let's see who she or he was.

# In[24]:


titanic_train.loc[titanic_train["Age"].argmax()]


# **By accident we found error in the Titanic dataset. When you read Algernon Henry Wilson Barkworth biography (you can find it [here](https://www.encyclopedia-titanica.org/titanic-survivor/algernon-barkworth.html)), we can read that he was born in 1864. That means that in 1912, when Titanic disaster happened, he was 48 years old and not 80 as stated in the dataset. **
# 
# **So why 80 in the data set? We can further read that this person died in 1945 - aged 80. This means that dataset contains for this passenger age value of post Titanic death and not day of disaster age as it does for other survived passengers age attribute values. This inconsistency makes the data invalid and confusing when it comes to factual or historical value.**
# 
# For this reason we decide to discard this passenger data from further analysis or modelling.
# 

# In[25]:


titanic_train = titanic_train.drop(titanic_train["Age"].argmax(), axis=0)


# In[26]:


titanic_train["Age"].describe()


# And now we need to bin passengers ages once again.

# In[27]:


titanic_age_groups_counts = pd.cut(titanic_train["Age"], bins=[0, 14, 24, 34, 44, 54, 64, 80]).value_counts().sort_index()
titanic_age_groups_counts


# Let's now check again who is the oldest passenger in the data set - after removing the incorect data record.

# In[28]:


titanic_train.loc[titanic_train["Age"].argmax()]


# So now we have for sure the oldest known Titanic passenger. His name was Johan Svensson, aged 74 and unfortunatelly died in the disaster (read more [here](https://www.encyclopedia-titanica.org/titanic-victim/johan-svensson.html)).

# In[29]:


titanic_sibsp_counts = titanic_train["SibSp"].value_counts()
titanic_sibsp_counts


# Let's move with the demographic analysis to next attribute. Vast majority of passengers travelled without any siblings or spouse - 607 of them. 209 passengers travelled with one child or spouse. Only 74 passengers traveled with more than one sibling or spouse.

# In[30]:


titanic_parch_counts = titanic_train["Parch"].value_counts()
titanic_parch_counts


# The situation is very similar in case of passengers travelling with parents or children. Most of them (677) travelled alone and 118 travelled with just one parent or children. 95 passengers travelled with two and more parents or children.

# In[31]:


titanic_fare_groups_counts = pd.cut(titanic_train["Fare"], bins=[0, 20, 40, 60, 80, 100, 300, 600]).value_counts().sort_index()
titanic_fare_groups_counts


# In terms ticket fare over the half of passengers (500) paid the lowest fare ranging from 0 to 20. Let's remember that we saw that 491 passengers travelled in third class. That follows the result seen here: third class was the cheapest way to travel and the number of passengers in the third class in almost the same as number of passengers who were in the lowest fare group. Again 199 passengers paid between 20 and 40 fare. That approximately corresponds to 184 passengers travelling in the second class. We see that the rest of the ticket prices - which most likely are first class tickets - varies very much probably meaning some additional luxuries and services. And also there is relatively small number of such tickets.

# In[32]:


titanic_embarked_counts = titanic_train["Embarked"].value_counts()
titanic_embarked_counts.index = ["Southampton", "Cherbourg", "Queenstown"]
titanic_embarked_counts


# We can see that vast majority of passengers, 643, embarked in Southampton), 168 embarked in Cherbourg and only 77 of passengers embarked Titanic in Queenstown.

# ## 3.4 What is the overall Titanic passengers survival ratio? 
# 
# We will now try to answer what is the overall Titanic passengers survival ratio. To to that we will create survival ratio metric. We will define it as the ratio between the number of survived passengers and the overall number of passengers. We will use this value for later reference and comparison in passengers subgroups survival analysis.
# 
# Our dataset contains 891 passengers data out of all 2224 passengers. This is partly due to the reason that we work with labeled train dataset with separate unlabeled test dataset containing 418 passengers data set aside. In this analysis we will treat these 891 passengers not as a sample but rather as our whole population of passengers. The same assumption will apply to subsequent parts of this analysis.
# 
# Let's now see what is the total number of survived passengers and and those who did not survive. 

# In[33]:


titanic_survived_counts = titanic_train["Survived"].value_counts(sort=False)
titanic_survived_counts.index = ["Not survived", "Survived"]
titanic_survived_counts


# Let's also create overall survival ratio metric.

# In[34]:


def get_survival_ratio(passengers_df):
    return passengers_df["Survived"].sum() / passengers_df["Survived"].count()


# In[35]:


overall_survival_ratio = get_survival_ratio(titanic_train)
overall_survival_ratio


# We see that out of Titanic 891 passengers only 342 survived and 549 died. The Titanic survival ratio is approximately 0.3838, meaning that only 38.38% of all passengers (again, we are talking here about population of passengers in the dataset) survived the disaster. That is really small percentage. 
# 
# We can see that majority of Titanic passengers died in the disaster. Let's keep these values - especially the overal survival ratio metric among all Titanic passengers of 0.3838 - as a reference point for further analysis.
# 

# ## 3.5 What is the survival ratio for different demographic passengers groups? Which groups have biggest chances for survival and which smallest?
# 
# Let's now finally see how survival ratio changes for passengers belonging to attributes demographic sub-categories. In this analysis we will process the same attributes as in previous parts: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare` and `Embarked`. 
# 
# We will start by grouping dataset using each of the attributes unique values besides `Age` and `Fare` attributes which we will group using values ranges. Later, based on these grouped values, we will calculate survival ratio for each of the attributes subgroups.

# In[36]:


titanic_pclass_group = titanic_train.groupby("Pclass")
titanic_pclass_group.groups


# In[37]:


titanic_sex_group = titanic_train.groupby("Sex")
titanic_sex_group.groups


# In case of `Age` attribute we will group its values into age groups as we did previously in demographic analysis.

# In[38]:


titanic_age_group = titanic_train.groupby(pd.cut(titanic_train["Age"], bins=[0, 14, 24, 34, 44, 54, 64, 80]))
titanic_age_group.groups


# In[39]:


titanic_sibsp_group = titanic_train.groupby("SibSp")
titanic_sibsp_group.groups


# In[40]:


titanic_parch_group = titanic_train.groupby("Parch")
titanic_parch_group.groups


# And also like in demographic analysis we will binnarize the `Fare` attribute values.

# In[41]:


titanic_fare_group = titanic_train.groupby(pd.cut(titanic_train["Fare"], bins=[0, 20, 40, 60, 80, 100, 300, 600]))
titanic_fare_group.groups


# In[42]:


titanic_embarked_group = titanic_train.groupby("Embarked")
titanic_embarked_group.groups


# Let's add this age group information as a separate attribute for later use.

# In[43]:


def group_age(age):
    if age <= 14:
        return "(0, 14]"
    elif age > 14 and age <= 24:
        return "(14, 24]"
    elif age > 24 and age <= 34:
        return "(24, 34]"
    elif age > 34 and age <= 44:
        return "(34, 44]"
    elif age > 44 and age <= 54:
        return "(44, 54]"
    elif age > 54 and age <= 64:
        return "(54, 64]"
    elif age > 64:
        return "(64, 80]"
    else:
        return np.nan
    
def group_fare(fare):
    if fare <= 20:
        return "(0, 20]"
    elif fare > 20 and fare <= 40:
        return "(20, 40]"
    elif fare > 40 and fare <= 60:
        return "(40, 60]"
    elif fare > 60 and fare <= 80:
        return "(60, 80]"
    elif fare > 80 and fare <= 100:
        return "(80, 100]"
    elif fare > 100 and fare <= 300:
        return "(100, 300]"
    elif fare > 300 and fare <= 600:
        return "(300, 600]"
    else:
        return np.nan


# In[44]:


titanic_train["AgeGroup"] = titanic_train.apply(lambda row: group_age(row["Age"]), axis=1)
titanic_train["FareGroup"] = titanic_train.apply(lambda row: group_fare(row["Fare"]), axis=1)


# Let's see it.

# In[45]:


titanic_train.head()


# With passengers data grouped we can now proceed with calculating survival ratio for each of the group. We will also prepend each group name (index value) with the attribute name for later analysis. This will help us to distinct groups with similar names created from separate attributes.

# In[46]:


titanic_pclass_survival_ratio = titanic_pclass_group.apply(get_survival_ratio)
titanic_pclass_survival_ratio.index = ["Pclass: " + str(idx) for idx in titanic_pclass_survival_ratio.index]
titanic_pclass_survival_ratio


# In[47]:


titanic_sex_survival_ratio = titanic_sex_group.apply(get_survival_ratio)
titanic_sex_survival_ratio.index = ["Sex: " + str(idx) for idx in titanic_sex_survival_ratio.index]
titanic_sex_survival_ratio


# In[48]:


titanic_age_survival_ratio = titanic_age_group.apply(get_survival_ratio)
titanic_age_survival_ratio.index = ["Age: " + str(idx) for idx in titanic_age_survival_ratio.index]
titanic_age_survival_ratio


# In[49]:


titanic_sibsp_survival_ratio = titanic_sibsp_group.apply(get_survival_ratio)
titanic_sibsp_survival_ratio.index = ["Sibsp: " + str(idx) for idx in titanic_sibsp_survival_ratio.index]
titanic_sibsp_survival_ratio


# In[50]:


titanic_parch_survival_ratio = titanic_parch_group.apply(get_survival_ratio)
titanic_parch_survival_ratio.index = ["Parch: " + str(idx) for idx in titanic_parch_survival_ratio.index]
titanic_parch_survival_ratio


# In[51]:


titanic_fare_survival_ratio = titanic_fare_group.apply(get_survival_ratio)
titanic_fare_survival_ratio.index = ["Fare: " + str(idx) for idx in titanic_fare_survival_ratio.index]
titanic_fare_survival_ratio


# In[52]:


titanic_embarked_survival_ratio = titanic_embarked_group.apply(get_survival_ratio)
titanic_embarked_survival_ratio.index = ["Embarked: " + str(idx) for idx in titanic_embarked_survival_ratio.index]
titanic_embarked_survival_ratio


# We will now combine all the survival ratios into one sequence. As a reference we will also add an overall titanic survival ratio.

# In[53]:


survival_ratios = pd.concat([titanic_pclass_survival_ratio,
                             titanic_sex_survival_ratio,
                             titanic_age_survival_ratio,
                             titanic_sibsp_survival_ratio,
                             titanic_parch_survival_ratio,
                             titanic_fare_survival_ratio,
                             titanic_embarked_survival_ratio], axis=0)
survival_ratios["Overall"] = overall_survival_ratio
survival_ratios


# We will also prepare passengers count for each of the above groups to visualize how big each of the analysed group was.

# In[54]:


titanic_pclass_survival_count = titanic_pclass_group.apply(len)
titanic_sex_survival_count = titanic_sex_group.apply(len)
titanic_age_survival_count = titanic_age_group.apply(len)
titanic_sibsp_survival_count = titanic_sibsp_group.apply(len)
titanic_parch_survival_count = titanic_parch_group.apply(len)
titanic_fare_survival_count = titanic_fare_group.apply(len)
titanic_embarked_survival_count = titanic_embarked_group.apply(len)

groups_counts = pd.concat([titanic_pclass_survival_count,
                           titanic_sex_survival_count,
                           titanic_age_survival_count,
                           titanic_sibsp_survival_count,
                           titanic_parch_survival_count,
                           titanic_fare_survival_count,
                           titanic_embarked_survival_count
])
groups_counts["Overall"] = len(titanic_train)
groups_counts


# Let's now visualize these results.

# In[55]:


def get_groups_survival_ratio_plot(survival_ratios, groups_counts, labels, figsize=(20, 10)):
    idx = np.arange(len(survival_ratios))
    width = len(survival_ratios) / 50

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
    axes[0].bar(idx, groups_counts, width, color='#f9d9ac', label="Passengers number")
    axes[0].set_title("Atributes groups passengers counts")
    axes[0].set_ylabel("Number of passengers")
    axes[0].legend(loc=2)
    
    axes[1].bar(idx, survival_ratios, width, color='#f1b7b0', label="Survived")
    axes[1].bar(idx, 1 - survival_ratios, width, bottom=survival_ratios, color='#f0f0f0', label="Not survived")
    axes[1].set_title("Survival ratios for atributes groups")
    axes[1].set_ylabel("Survival ratio")
    axes[1].legend(loc=2)
    
    plt.xticks(idx, labels, rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[56]:


get_groups_survival_ratio_plot(survival_ratios.values, groups_counts.values, survival_ratios.index)


# Let's now finally analyse the survival ratio for different attributes subgroups. As a reference on the right hand side of the graph we placed the overal Titanic passengers survival ratio - it is equal to 0.38. To test whether attribute's values affected the passengers survival ratio significantly (in inferential statistics meaning) we will use .05 significance level.

# ### 3.5.1 Pclass attribute
# 
# Let's start with `Pclass` survival ratio levels. As we could expect, based on popular knowledge, the first (and even second) class passengers had a lot greater chances of survival than third class passengers. The survival ratio was 0.63 in the first class meaning that 63% of passengers travelling in this class survived. Second class had a bit lower survival ratio of 0.47. But the third class passengers survival ratio of 0.24 is shocking. Only 24% of third class passengers survived the disaster. That plus the fact that third class was the most populated (third bar in the top plot) with 491 passengers (out of all 891 in this dataset) gives the image of dramatic situation these people were in. Looking for the reasons of this high mortality level is beyond the scope of this analysis.
# 

# In[57]:


model = ols("Survived ~ C(Pclass)", titanic_train).fit()
model.summary()


# All three passengers classes effects are significant with p < .05. 

# ### 3.5.2 Sex attribute
# 
# Looking at `Sex` attribute groups values we also see clear relationship in terms of survival ratio. Clearly women were more likely to survive than men with 0.74 survival ration for women and 0.18 for men. Again this is what we could expect from popular knowledge. This could be expected following "Women and children first" marine code of conduct (more about it [here](https://en.wikipedia.org/wiki/Women_and_children_first)). This fact is clearly visible in the data.

# In[58]:


model = ols("Survived ~ C(Sex)", titanic_train).fit()
model.summary()


# Both sexes effects are significant with p < .05. 

# ### 3.5.3 Age attribute
# 
# Going step forward, was the same rescue law applied to children or maybe other age groups? Analysing `Age` attribute survival ratios it appears that the answer is positive. Passengers in the age group of 0-14 have much higher survival ratio of 0.58 than the rest of passengers. Other passengers age groups oscilate around the overall survival ratio of 0.38. With one exception however: for age group of 64-80 passengers have very small survival ratio of 0.09. Again this analysis is not a place to look for specific reasons for this fact.

# In[59]:


model = ols("Survived ~ AgeGroup", titanic_train).fit()
model.summary()


# All ages groups besides `(54, 64`] group effects are significant with p < .05. 

# ### 3.5.4 Sibsp attribute
# 
# Let's now analyze `Sibsp` and groups of passengers travelling with different number of siblings or spouse. The data set specification defines siblings as: brother, sister, stepbrother, stepsister. It also defines spouse as: husband, wife. There is no other information in the data set that could help us to separate these two groups (whether passenger traveled with sibling, spouse or alone) so we will analyse as is in the data set.
# 
# It is interesting whether travelling with siblings or spouse affected chances for survival. 608 passengers, meaning vast majority, traveled without any with 0.35 survival ratio. Passengers travelling with one and two siblings or spouse had higher survival chances of 0.54 and 0.46. There were passengers travelling with three, four, five and eight siblings or spouse on board. Their survival ratio was lower and reaching 0 for the last two groups. Again analysing these groups we need to remeber that very few passengers belonged to these groups. For example, only 7 passengers belonged to 8 siblings or spouse group. Let's see who they were.

# In[60]:


titanic_train[titanic_train["SibSp"] == 8]


# As we can see it is a case of a tragic story of one big familly, all travelling in the third class. Unfortunatelly all died in the disaster. In fact it was two parents travelling with 8 children but three of them are not included in this dataset. That is why `SibSp` is 8 even though we have only 7 individuals here. You can read more about their story [here](https://www.encyclopedia-titanica.org/titanic-victim/thomas-henry-sage.html).

# In[61]:


model = ols("Survived ~ C(SibSp)", titanic_train).fit()
model.summary()


# Only passengers travelling without any and with one sibling/spouse effects are significant with p < .05. 

# ### 3.5.5 Parch attribute
# 
# Moving on to analysis of `Parch` attribute survival ratio. Again vast majority of passengers (678) traveled without any children or parents with survival ratio of 0.34. We can see that passengers travelling with one, two or three children or parents had bigger chances of survival (0.55, 0.50 and 0.60 accordingly), but again let's remember that these groups had much less passengers (118, 80, 5). There were only 10 passenger travelling with four, five or six parents or children and their survival ratio was small but these groups are to small to draw any conclusions.
# 

# In[62]:


model = ols("Survived ~ C(Parch)", titanic_train).fit()
model.summary()


# Only passengers travelling without any and with one or two children/parents effects are significant with p < .05. 

# ### 3.5.6 Fare attribute
# 
# When we look at `Fare` attribute survival ratio values we can see clearly almost linear relation between the price of the ticket and the survival ratio. It looks like the more passenger paid for the ticket the bigger were the chances of survival. The survival ratio for passengers who bought the tickets for 0-20 was 0.28, for passengers with tickers between 20-40 the survival ration was 0.43. And for further groups: 40-60 0.57, 60-80 0.520833, 80-100 0.86, 100-300, 0.72, and for tickets with price between 300-600 the survival ratio was 1.00 meaning that all of these passengers survived. There are two facts we need to mention. First, the more expesive ticket the higher class in which passenger traveled meaning of course the higher chances for survival. Second, the number of passengers in groups of ticket prices above 40 is about 50 passengers making them small with only three passengers in the most expensive 300-600 ticket price range price group.
# 

# In[63]:


model = ols("Survived ~ C(FareGroup)", titanic_train).fit()
model.summary()


# However all passengers tickets fares groups effects are significant with p < .05. 

# ### 3.5.7 Embarked attribute
# 
# Finally let's look at the `Embarked` atribute passengers groups. As a reminder: `C` stands for Cherbourg port of departure, `Q` stands for Queenstown and `S` stands for Southampton. Majority of Titanic passengers (644) embarked in Southhampton. Also this group of passengers have the lowest survival ratio out of all three ports of embarkation (0.34). Similarly passengers who embarked in Queenstown had survival ratio equal to 0.39. However those who embarked in Cherbourg had survival ratio of 0.55. First of all there were much less passengers embarking in the last two ports: only 77 passengers in Queenstown and 168 passengers in Cherbourg. We could guess, considering large number of passengers, that the third class passengers embarked mostly in Southhampton and Queenstown and that could be the reason for such small, as compared to Cherbourg port, survival ratio for passengers embarking here. Let's check it with data.

# In[64]:


titanic_embarked_group.get_group("S")["Pclass"].value_counts(sort=False)


# In[65]:


titanic_embarked_group.get_group("Q")["Pclass"].value_counts(sort=False)


# In[66]:


titanic_embarked_group.get_group("C")["Pclass"].value_counts(sort=False)


# That confirms our hypothesis: both in Southampton and Queenstown third class passengers were the majority of embarking there. In Cherbourg port the situation was the opposite: first class passengers were the majority and that is the reason survival ratio is so high for that passengers subgroup.

# In[67]:


model = ols("Survived ~ Embarked", titanic_train).fit()
model.summary()


# All passengers embarkation ports effects are significant with p < .05. 

# ## 4. Summary <a class="anchor" id="summary"></a>
# 
# In this analysis we worked with Titanic survival dataset for 891 passengers data out of all 2224 Titanic passengers.
# 
# We first answered what was Titanic passengers demographic structure analyzed in terms of attributes. We saw that vast majority (55.11%) of passengers travelled in lowest third class, almost 25% of passengers travelled in first class and 20.65% traveled in second class. Majority of all passengers were males: 64.76% and only 35.24% of females. 10.78% of passengers were children in age of 0-14. A little over 28% of passengers were both between 14-24 and 24-34 years old. Almost 17% of all passengers were 34-44 years old. Only 16% of all passengers were above 44 years old. That tells us that Titanic passengers population was quite young.
# 
# Vast majority of passengers travelled without any siblings or spouse - 68.24% of them. 23.46% of passengers travelled with one child or spouse. Less than 9% of all passengers traveled with more than one sibling or spouse. The situation is very similar in case of passengers travelling with parents or children. Most of them, 76.09%, travelled alone and 13.24% travelled with just one parent or children. 8.89% of passengers travelled with two parents or children. In terms of ticket fare over the half, 57.08%, of passengers paid the lowest fare ranging from 0 to 20. 22.83% passengers paid between 20 and 40 fare. We saw that the rest of the ticket prices, 20.11%, varies very much ranging from 40 to 600. Finally we can see that vast majority of passengers, 72.44%, embarked in Southampton, 18.9% embarked in Cherbourg and only 8.66% of passengers embarked Titanic in Queenstown.
# 
# In this part of analysis by accident we found error in the Titanic dataset. We found out that the age of the person appearing in the dataset as the oldest (80) is the age of actual death many years after person disaster survival. This means that dataset age information for this passenger, by error, contains age value of the post Titanic death and not the day of disaster age as it does for other survived passengers age attribute. This inconsistency makes the data invalid and confusing when it comes to factual or historical value.
# 
# We answered what is the overall Titanic passengers survival ratio. We found out that from Titanic 891 passengers only 342 survived and 549 died. The Titanic survival ratio is approximately 0.3838, meaning that only 38.38% of all passengers survived the disaster.
# 
# Finally we turned to the analysis of survival ratio for different demographic passengers groups. As for passengers classes survival ratio, the survival ratio for the first class passengers was 0.63. Second class had a bit lower survival ratio of 0.47. But comparing that to the third class passengers survival ratio of 0.24 is shocking. All three passengers classes effects are significant with p < .05. Looking at gender based survival differences we also see clear relationship in terms of survival ratio values. Clearly women were more likely to survive than men with 0.74 survival ratio for women and 0.18 for men. Both gender effects are significant with p < .05. As for age groups survival ratio passengers in the age group of 0-14 have much higher survival ratio of 0.58 than the rest of passengers. Other passengers age groups oscillate around the overall survival ratio of 0.38. All ages groups effects besides (54, 64] group are significant with p < .05.
# 
# As for passengers travelling with different number of siblings or spouse, passengers travelling without any had survival ratio of 0.35. Passengers travelling with one and two sibling or spouse had higher survival chances of 0.54 and 0.46. Only passengers travelling without any and with one sibling/spouse effects are significant with p < .05. Considering groups of passengers depending on the number of parents and children they travelled with, again passengers travelling alone had survival ratio of 0.34. We can see that passengers travelling with one, two or three children or parents had bigger chances of survival, 0.55, 0.50 and 0.60 accordingly, but again let's remember that these groups had much less passengers. Only passengers travelling without any and with one or two children/parents effects are significant with p < .05.
# 
# When we look at ticket fare attribute survival ratio values we can see clearly almost linear relation between the price of the ticket and the survival ratio. It looks like the more passenger paid for the ticket the bigger were chances of survival. The survival ratio for passengers who bought the tickets for 0-20 was 0.28, for passengers with tickers between 20-40 the survival ratio was 0.43. And for further groups: 40-60 0.57, 60-80 0.520833, 80-100 0.86, 100-300 0.72. For tickets with price between 300-600 the survival ratio was 1.00 meaning that all of these passengers survived. All passengers tickets fares groups effects are significant with p < .05. Finally we looked at the Embarked attribute passengers groups. Passengers who embarked in Southampton had the lowest survival ratio out of all three ports of embarkation (0.34). Similarly passengers who embarked in Queenstown had survival ratio equal to 0.39. However those who embarked in Cherbourg had survival ratio of 0.55. We confirmed it with data that the third class passengers embarked mostly in Southampton and Queenstown and that could be the reason for such small, as compared to Cherbourg port, survival ratio for passengers embarking in Southampton. All passengers embarkation ports effects are significant with p < .05.

# In[ ]:





# In[ ]:





# In[ ]:




