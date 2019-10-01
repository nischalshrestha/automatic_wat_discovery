#!/usr/bin/env python
# coding: utf-8

# ![Titanic](https://img00.deviantart.net/af19/i/2014/307/c/f/r_m_s__titanic_class_system_by_monroegerman-d787jna.png)

# * Total On Board (Pax and Creq) = 2224 
# * Total Survivors ~ 700

# # Table of Contents
# * [0. Importing Libraries](#sec0)
# * [1. Loading Data](#sec1)
# * 2. Data Cleaning (missing, outliers, categorial variables)
# * 3. Exploratory Data Analysis
#     * Univariate Analyses (target variable)
#     * Multivariate Analyses (relation between independent and dependent variable)
#         * scatter plots
#         * correlation matrix
#     * hypotheses
# * 4. Feature Engineering and Selection
#     * factorize/one-hot encoding (categorical -> numerical)
#     * binning (numerical -> categorical)
#     * scaling (center data around 0)
#     * feature importance (high/low variance, missing values ratio, correlation, RandomClassifier, shuffling ...)
# * Model Building & Visualization
# 

# <a class="anchor" id='sec0'></a>
# # 0. Importing Libraries
# Importing the most commonly used libs.

# In[1]:


#fundamental package for scientific computing with Python, arrays, matrices
import numpy as np 
# data analysis and wrangling
import pandas as pd
# regular expressions
import re

# machine learning


# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)
get_ipython().magic(u'matplotlib inline')

# util functions
from subprocess import check_output # for ls

# Set jupyter's max row display
pd.set_option('display.max_row', 1000)
# Set jupyter's max column width to 50
pd.set_option('display.max_columns', 50)

# allow interactivity not only with last line in cell
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# check available libraries (versions)
# !pip list
# !pip list | grep "pandas"


# <a class="anchor" id='sec1'></a>
# # 1. Loading Data
# Check contents in ../input/ and then load the data files into pandas data frames.

# In[ ]:


#!pwd
#!ls ../


# In[ ]:


# check input directory
#print(check_output(["ls", "../input"]).decode())


# In[2]:


# load train and test datasets into data frames
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# # 2. Examining the train data (Data Exploration and Visualization)

# Descriptions contained in Kaggle data dictionary:
# *     PassengerID —  A column added by Kaggle to identify each row and make submissions easier
# *     Survived —  Whether the passenger survived or not (0=No, 1=Yes) ==> target value
# *     Pclass — The class of the ticket the passenger purchased (1=1st, 2=2nd, 3=3rd)
# *     Sex — The passenger's sex (male, female)
# *     Age — The passenger's age in years
# *     SibSp — The number of siblings or spouses the passenger had aboard the Titanic
# *     Parch — The number of parents or children the passenger had aboard the Titanic
# *     Ticket — The passenger's ticket number
# *     Fare — The fare the passenger paid
# *     Cabin — The passenger's cabin number
# *     Embarked — The port where the passenger embarked (C=Cherbourg, Q=Queenstown, S=Southampton)

# ## 2.1. Shape, Types, Info

# In[ ]:


train_df.head(10)


# In[ ]:


train_df.tail(10)


# In[ ]:


# 891 Zeilen x 12 Spalten
print('train shape', train_df.shape)
print('test shape', test_df.shape)

# get set difference on column level
train_df.columns.difference(test_df.columns) 


# In[ ]:


# check data types 
pd.DataFrame(train_df.dtypes)


# Check out data types, missing values and number of lines ...

# In[ ]:


train_df.info()
print("_"*40)
test_df.info()


# Typically, you distinguish between the following feature types:
# 
# ![Kinds of Data](https://i.ytimg.com/vi/7bsNWq2A5gI/hqdefault.jpg)

# ### Metadata
# 
# Store meta-information about the variables in a DataFrame. This will be helpful later for  selecting specific variables for analysis, visualization, modeling, ...
# 
# * **role**: input, ID, target
# * **level**: nominal, ordinal, discrete, continuous
# * **keep**: True or False
# * **dtype**: int, float, str
# 

# In[ ]:


data = [] # blank initial list
for f in train_df.columns:
    # Defining the data type 
    dtype = train_df[f].dtype
        
    # Defining the level
    if train_df[f].dtype == float:
        level = 'numerical' #continuous
    elif train_df[f].dtype == object:
        level = 'categorical' #nominal
    else:
        level = 'N.A.'
        
    # Initialize keep to True for all variables 
    keep = True
    role = 'input'
    
    # Creating a Dict that contains all the metadata for the current variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
summary_df = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
summary_df.set_index('varname', inplace=True)

# set special roles manually
summary_df.loc['PassengerId','role']='ID'
summary_df.loc['Survived','role']='target'

# set levels individually
summary_df.loc['PassengerId','level']='categorical' #nominal
summary_df.loc['Survived','level']='categorical' #nominal
# 1st class > 2nd class > 3rd class
summary_df.loc['Pclass','level']='categorical' #ordinal
summary_df.loc['SibSp','level']='numerical' #discrete
summary_df.loc['Parch','level']='numerical' #discrete

summary_df


# number of variables per role and level:

# In[ ]:


summary_df.groupby(['role', 'level']).size().to_frame(name = 'count').reset_index()
#pd.DataFrame({'count' : summary_df.groupby( ['role', 'level'] ).size()}).reset_index()


# **Categorical**: Survived, Sex, and Embarked. **Ordinal**: Pclass.
# 
# **Continous**: Age, Fare. **Discrete**: SibSp, Parch

# ## 2.2. Descriptive Statistics

# **Distribution of numerical feature values across the samples?**

# In[ ]:


# all stats
#train_df.describe(include='all')

# basic stats for numerical features
# train_df.describe(include=['number'])
#train_df[["Survived", "Age", "Fare", "SibSp", "Parch"]].describe()


# or use our newly designed metadata
#s = summary_df[(summary_df['level'].isin(['continuous', 'discrete'])) & (summary_df.keep)].index
s = summary_df[(summary_df['level'].isin(['numerical'])) & (summary_df.keep)].index
# use obtained index for describe and add survived as well
train_df[s.union(['Survived'])].describe()


# * Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
# * Around 38% samples **survived** representative of the actual survival rate at 32%.
# * passenger **ages** range from **0.4 to 80**

# **Distribution of categorical feature values across the samples?**

# In[ ]:


# only features of type Object
#train_df.describe(include=['O'])
s = summary_df[(summary_df['level'].isin(['categorical'])) & (summary_df.keep)].index

# use obtained index for describe, all to ensure type object as well
train_df[s].describe(include='all')


# * **Names** are unique across the dataset (count=unique=891)
# * **Sex** variable has two possible values with 65% male (top=male, freq=577/count=891).
# * **Ticket** feature has high ratio (22%) of duplicate values (unique=681).
# * **Cabin** values have several dupicates across samples. Alternatively several passengers shared a cabin.
# * **Embarked** takes three possible values. S port used by most passengers (top=S)
# * **Pclass** has 3 values, mean is 2.3 so it is more biased to 3rd class passengers
# 

# ## 2.2. Missing Values

# In[ ]:


def Print_Missing_Values_Overview(input_df):
    # put all variables with missing values in a list
    vars_with_missing = []

    for f in input_df.columns:
        missings = input_df[f].isnull().sum()
        if missings > 0:
            vars_with_missing.append(f)
            missings_perc = missings/input_df.shape[0]

            print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))

    print('==> In total, there are {} variables with missing values'.format(len(vars_with_missing)))


# In[ ]:


print("missings for train_df")
Print_Missing_Values_Overview(train_df)

print("\nmissings for test_df")
Print_Missing_Values_Overview(test_df)


# In[ ]:


def Get_Missing_Values(input_df):
    null_cnt = input_df.isnull().sum().sort_values(ascending=False)
    tot_cnt = input_df.isnull().count()
    pct_tmp = input_df.isnull().sum()/tot_cnt*100
    pct = (round(pct_tmp, 1)).sort_values(ascending=False)
    missing_data = pd.concat([null_cnt, tot_cnt, pct], axis=1, keys=['#null', '#Tot','%'])        .reindex(pct.index)
    return missing_data


# ## train

# In[ ]:


missing_vals = Get_Missing_Values(train_df)
display(missing_vals.head(5))

summary_df['#null'] = missing_vals['#null']
summary_df['%null'] = missing_vals['%']


# * **Cabin** has around 77% missing values and could be dropped
# * **Age**: replace with mean?
# * **Embarked**: only two missings. Use mode?

# ## test

# In[ ]:


Get_Missing_Values(test_df).head(5)


# We can see, that in the test dataset there is 1 **missing value for Fare** as well ==> replace with mean?

# ## 2.3. Unique Values

# In[ ]:


train_df["Cabin"].value_counts().shape[0]


# In[ ]:


# get list of unique values
#pd.unique(train_df["Cabin"])


# In[ ]:


# get overview of uniques
train_df.nunique()


# In[ ]:


# get number of unique values
#train_df.apply(lambda x: len(x.unique()))

train_df.apply(pd.Series.nunique).sort_values(ascending=False)

summary_df['#unique'] = train_df.nunique()


# In[ ]:


# list all unique values
# pd.unique(train_df["Ticket"])


# In[ ]:


# check for duplicate ticket numbers
train_df.groupby('Ticket').size().sort_values(ascending=False).head()


# * family has same ticket number? Or infant has same ticket number as parent? 
# * Passenger could have bought one ticket for whole family, or for friend

# In[ ]:


train_df[(train_df['Ticket']=='CA. 2343')]


# In[ ]:


# is there a duplicate ticket number with differing prices?
train_df.groupby(['Ticket', 'Fare']).size().groupby('Ticket').size().sort_values(ascending=False).head()


# In[ ]:


train_df[(train_df['Ticket']=='7534')]


# **Take-Away**: There are patterns in Ticket column not described in data dictionay. Anyways, the column will not be used as feature, due to its high variance.

# ## 2.4. Summary Stats & MetaData

# In[ ]:


summary_df['#non_null'] = train_df.count()
# reorder columns
summary_df = summary_df[['role', 'level', 'keep', 'dtype', '#null', '%null', '#non_null', '#unique']]
summary_df


# ## 2.5. Value Counts

# In[ ]:


# group by each value and get counts
# train_df["Age"].value_counts()


# In[ ]:


# get value counts for all columns
#pieces = []
#for col in train_df.columns:
#    tmp_series = train_df[col].value_counts()
#    tmp_series.name = col
#    pieces.append(tmp_series)
#pd.DataFrame(pieces).T


# ## 2.6. First Observations
# * train = 891 passengers (= datasets)
# * binary (boolean) target variable = "Survived"
# * 9 basic feature variables (12 - 3 excl. target variable and PaxID, TicketNumber due to high variance)
# * "Cabin" does not seem that useful, due to 78% missing in training set
# 
# ## 2.7. First Questions/Ideas
# 
# **Cleansing**
# * NaN treatment (Age, Cabin, Embarked)
# 
# **Analyis**
# * correlate each feature to Survival
# * group survival by class, by gender, by age
# * correlation of class to fare
# 
# **Feature Engineering**
# * features to be dropped:
#     * ticket (duplicates, high variance)
#     * cabin (many missings)
#     * PassengerId (ID feature)
#     * name (highest variance)
#     * Embarked (does not seem to be relevant at first glance)
# * define age range/categories (e.g. infant, child, adult, senior) =>  turn continous numerical feature into an ordinal categorical feature
# * within Cabin, there seems to be the deck included?!
# * "Name" seems to have deep semantics
#     * pax travelling on same ticket (brackets)
#     * nicknames in quotes
#     * -> maybe split name into sub-components (by Space, Coma and Point)
# 
# **Classification**
# * assumptions on strong predictors:
#     * Women (Sex=female) were more likely to have survived
#     * children (Age<?) were more likely to have survived
#     *  upper-class passengers (Pclass=1) were more likely to have survived

# ## 2.8. Pandas Profiling (experimental)
# 
# For each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:
# 
#  *  Essentials: type, unique values, missing values
#  *  Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
#  *  Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
#  *  Most frequent values
#  *  Histogram
#  *  Correlations highlighting of highly correlated variables, Spearman and Pearson matrixes
# 

# In[ ]:


# !pip list | grep profiling
import pandas_profiling

# double click left next to cell to collapse/expand output
profile = pandas_profiling.ProfileReport(train_df)
profile
# rejected_variables = profile.get_rejected_variables(threshold=0.9)
# profile.to_file(outputfile="/tmp/myoutputfile.html")


# # 3. Visual Analyses

# ## 3.1. Univariate Analyses

# In[ ]:


train_df.Survived.value_counts(normalize=True)


# **Take-Away**:  only 38% of the passengers were survived, where as a majority 61% the passenger did not survive the disaster

# In[ ]:


sns.countplot(x='Survived', data=train_df);


# In[ ]:


sns.distplot(train_df.Fare, kde=False);
print(train_df.Fare.mean())


# **Take-away:** Most passengers paid less than 100 for travelling with the Titanic.

# In[ ]:


# delete n.a./missing values beforehand
sns.distplot(train_df.Age.dropna())
plt.title('Age Distribution of Passengers', fontdict={'fontsize': 16})
plt.show()


# In[ ]:


# show box plots for numerical featues fare and age
fig, axes = plt.subplots(2, 1)
sns.boxplot(x="Fare", data=train_df, ax=axes[0])
sns.boxplot(x="Age", data=train_df, ax=axes[1])


# In[ ]:


train_df['Age'].hist(bins=50)


# In[ ]:


facet_grid = sns.FacetGrid(train_df, col='Sex', size=5, aspect=1)
# using histogram (distplot)
facet_grid.map(sns.distplot, "Age")
# move to ensure enough space for title
plt.subplots_adjust(top=0.9)
facet_grid.fig.suptitle('Age Distribution (Males vs Females)', fontsize=16)


# In[ ]:


#use FacetGrid to plot multiple kdeplots on one plot
fig = sns.FacetGrid(train_df,hue='Sex',aspect=4)
#call FacetGrid.map() to use sns.kdeplot() to show age distribution
fig.map(sns.kdeplot,'Age',shade=True)
#set the x max limit by the oldest passenger
oldest = train_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[ ]:


fig = sns.FacetGrid(train_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade='True')
oldest = train_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[ ]:


# use facet grid for a box plot of age distribution
fg = sns.FacetGrid(train_df, col="Pclass")
# using boxplots, order parameter has to be set to prevent wrong output !!
fg.map(sns.boxplot, "Sex", "Age", order=["male", "female"])


# **Take-Away**: passengers travelling in the 1st class were older compared to passengers travelling in the 2nd and 3rd class.

# In[ ]:


sns.swarmplot(x="Pclass", y="Age", hue="Sex", data=train_df)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.title("Age distribution vs Class", fontsize=15)


# In[ ]:


sns.factorplot('Pclass',data=train_df,hue='Sex',kind='count')


# In[ ]:


# 2 rows, 4 columns
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

sns.countplot('Survived',data=train_df,ax=axes[0,0])
sns.countplot('Pclass',data=train_df,ax=axes[0,1])
sns.countplot('Sex',data=train_df,ax=axes[0,2])
sns.countplot('SibSp',data=train_df,ax=axes[0,3])

sns.countplot('Parch',data=train_df,ax=axes[1,0])
sns.countplot('Embarked',data=train_df,ax=axes[1,1])

# numeric/continuous features
sns.distplot(train_df['Fare'], ax=axes[1,2])
# remove null values/rows beforehand
sns.distplot(train_df['Age'].dropna(),ax=axes[1,3])


# ## 3.2. Multivariate Analyses

# ### Survival
# * how well does each feature correlate with Survival 
# * match these quick correlations with modelled correlations later 

# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(train_df["Age"][train_df.Survived == 1], color="green", shade=True)
sns.kdeplot(train_df["Age"][train_df.Survived == 0], color="red", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
plt.show()


# **Take-Away:** The age distribution for survivors and deceased is actually very similar. One notable difference is that, of the survivors, a larger proportion were children. The passengers evidently made an attempt to save children by giving them a place on the life rafts.

# In[ ]:


#s = summary_df[(summary_df['level'].isin(['categorical'])) & (summary_df['#unique'] <= 10)].index
#sl = s.tolist()
cat_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

fig, axs = plt.subplots(nrows=len(cat_vars), figsize=(8,20), sharex=False)
for i in range(len(cat_vars)):
    sns.countplot(x=cat_vars[i], data=train_df, hue='Survived', ax=axs[i])


# In[ ]:


# count plot for exact numbers 
sns.countplot(x="Sex", hue="Survived", data=train_df)

train_df.groupby(["Sex", "Survived"]).size()


# **Take-away:** Women were more likely to survive than men.

# In[ ]:


# figure out survival proportions
print(train_df[train_df.Sex == 'female'].Survived.sum()/train_df[train_df.Sex == 'female'].Survived.count())
print(train_df[train_df.Sex == 'male'].Survived.sum()/train_df[train_df.Sex == 'male'].Survived.count())


# In[ ]:


sns.factorplot(x='Survived', col='Pclass', kind='count', data=train_df);


# **Take-Away**: Passengers that travelled in first class were more likely to survive. On the other hand, passengers travelling in third class were more unlikely to survive. 

# In[ ]:


# Passenger class wise distribution of counts of survival statistics for men and women
sns.factorplot("Sex", col="Pclass", data=train_df, kind="count", hue="Survived")
train_df.groupby(["Sex", "Pclass", "Survived"]).size()


# **Take-Away**: Women also had a better chance of survival in 3rd class.

# In[ ]:


sns.pointplot(x="Pclass", y="Survived", hue="Pclass", data=train_df)


# In[ ]:


# surviving ratio of different classes
sns.factorplot('Pclass','Survived',data=train_df)


# ** Take-Away:**  Being a first class passenger was safest.
# 

# In[ ]:


sns.barplot(x='Sex', y='Survived', hue="Pclass", data=train_df)


# In[ ]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=train_df);


# **Take-Away**: Passengers that embarked in Southampton were less likely to survive. Coincidence?
# 
# ** Take-Away**: Passengers who boarded in Cherbourg, France, appear to have the highest survival rate. Passengers who boarded in Southhampton were marginally less likely to survive than those who boarded in Queenstown. This is probably related to passenger class, or maybe even the order of room assignments (e.g. maybe earlier passengers were more likely to have rooms closer to deck). 

# In[ ]:


sns.factorplot("Pclass", col="Embarked", data=train_df, kind="count", hue="Survived")


# **Take-Away**: It seems, that most of those embarked in Southampton are 3rd class (male) passengers, which per se had a low chance of survival.

# In[ ]:


# check relation to target attribute

# Set up the matplotlib figure
figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))

train_df.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
train_df.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
train_df.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
train_df.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
train_df.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])

sns.boxplot(x="Survived", y="Age", data=train_df,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=train_df,ax=axesbi[1,2])


# In[ ]:


train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# In[ ]:


fig, axs = plt.subplots(ncols=2, figsize=(15, 3))
sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=train_df, ax=axs[0]);
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_df, ax=axs[1]);


# **Take-Away**: We can already see some (strong) correlation between sex, age, Pclass, embarked and survival rate

# **Take-Away Bivariate EDA**
# * male survial rates is around 20%,  female survial rate is about 75% --> gender strong relationship with survival rate
# * clear relationship between Pclass and the survival 
#     * Passengers on Pclass1 had a better survial rate of approx 60% 
#     * Passengers on pclass3 had the worst survial rate of approx 22%
# * There is also a marginal relationship between the fare and survial rate
# * naturally, there is a strong correlation between Fare and Pclass

# In[ ]:


train_df.groupby('Survived').Fare.hist(alpha=0.6);


# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(train_df["Fare"][train_df.Survived == 1], color="green", shade=True)
sns.kdeplot(train_df["Fare"][train_df.Survived == 0], color="red", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(-20,200)
plt.show()


# **Take-Away**: clearly different distributions for the fares of survivors vs. deceased ==> likely that this would be a significant predictor in  final model. Passengers who paid lower fare appear to have been less likely to survive. This is probably strongly correlated with Passenger Class.

# In[ ]:


# although there appears to be a small tendency upwards shown by the regression, 
# there appears to be almost no correlation between the variables “age” and “fare”, 
# as shown by the Pearson correlation coefficient. 
sns.jointplot(x="Age", y="Fare", data=train_df, kind='reg');


# In[ ]:


sns.lmplot(x='Age', y='Fare', hue='Survived', data=train_df, fit_reg=False, scatter_kws={'alpha':0.5});


# **Take-away:** It looks like those who survived either paid quite a bit for their ticket or they were young.

# In[ ]:


sns.factorplot(x="Pclass", y="Age", hue="Survived", data=train_df, kind="box")


# In[ ]:


# display most of the information in a single grid of plots.
# drop nulls before
sns.pairplot(train_df.dropna(), hue='Survived');


# In[ ]:


# correlation matrix for int64 and float64 types
# use pearsons R, alternatively Spearman or Kendal-Tau could be used for categorical features

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 8))
corr = train_df.corr()
#display(corr)

sns.heatmap(corr,
            mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, linewidths=.5, annot=True)


# **Take-Away** Correlation Matrix:
# * There is a positve correlation between Fare and Survived and a negative coorelation between Pclass and Surived
# * There is a negative correlation between Fare and Pclass, Age and Plcass
# 
# 

# In[ ]:


import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objs as go
pyo.init_notebook_mode(connected=False)

corr = train_df.corr().abs().Survived.sort_values(ascending=False)[1:]
data = [go.Bar(
            x=corr.index.values,
            y=corr.values
    )]

pyo.iplot(data, filename='basic-bar')


# # 4. Feature Engineering
# <font color='red'>Should be done on train and test set.<br><br></font>
# 
# * Feature Selection (importance)
#     * manually
#     * automated
# * Feature Generation
#     * create new features
#     * convert categorical features (one-hot-encoding etc.)
#     
# *Converting categorical features to numerical values is required by most model algorithms.*
# 
# 
# In the movie Titanic women and children were given preference to lifeboats (as they were in real life). Also it seemed thath higher class passengers had been prefered as well. 
# 
# This indicates that **Age, Sex, and PClass** may be good predictors of survival. 

# ## 4.1. Missing Values
# <font color='red'>Having missing values in a dataset can cause errors with some machine learning algorithms and either the rows/values that havw missing values should be removed or imputed.<br><br></font>
# 
# methods to treat missings:
# * **deletion** (of column or row)
# * **constant** value: that has meaning within the domain, such as 0 or -1, distinct from all other values. Algorithms treat those values then differently.
# * **Mean/ Mode/ Median** Imputation: fill in the missing values with estimated ones. Replacing the missing data for a given attribute by the mean or median (quantitative attribute) or mode (qualitative attribute) of all known values of that variable. Median is typically chosen over mean in case of many outliers.
# * **prediction model**: create a predictive model to estimate values that will substitute the missing data.  In this case, we divide our data set into two sets: One set with no missing values for the variable and another one with missing values. First data set become training data set of the model while second data set with missing values is test data set and variable with missing values is treated as target variable. We can use regression, ANOVA, Logistic regression and various modeling technique.
# * **KNN imputation**: missing values of an attribute are imputed using the given number of attributes that are most similar to the attribute whose values are missing. The similarity of two attributes is determined using a distance function.

# ### Fare

# In[3]:


# check fare = 0
# every ticket should have a value greater than 0
print((train_df.Fare == 0).sum())
print((test_df.Fare == 0).sum())

for df in train_df, test_df:
    # mark zero values as missing or NaN
    df.Fare = df.Fare.replace(0, np.NaN)
    
    # impute the missing Fare values with the mean Fare value    
    df.Fare.fillna(df.Fare.mean(),inplace=True)


# ### Age

# In[4]:


# we see that there are no Zero values
print((train_df.Age == 0).sum())
print((test_df.Age == 0).sum())

# impute the missing Age values with the mean Fare value
for df in train_df, test_df:
    df.Age.fillna(df.Age.mean(),inplace=True)
    
# use median, as it deals better with outliers ??
# train_df['Age'] = train_df.Age.fillna(train_df.Age.median())


# ### Cabin

# In[5]:


# We see that a majority 77% of the Cabin variable has missing values.
# Hence will drop the column from training a machine learnign algorithem
train_df.Cabin.isnull().mean()


# ### Embarked

# In[6]:


# fill with most frequent value
# train_df.Embarked.mode()
for df in train_df, test_df:
    df["Embarked"].fillna("S", inplace=True)


# ## 4.3. Feature Generation
# * create new features
#     * semantical
#     * interaction variables
#     * Factorize
#     * Binning
#     * Scaling/z-transformation
# 

# ### 4.3.1. Feature Generation (attribute-wise)

# #### PassengerId
# PassengerId may be dropped from training dataset as it does not contribute to survival. Only for identification purposes.

# #### Pclass
# nothing to do here

# #### Name
# * name alone is useless on its own due to high variance
# * Generate new features:
#     * title
#     * number of names (family size)

# In[7]:


for df in train_df, test_df:
    df['Name_len'] = df.Name.str.len()
    df['Name_parts'] = df.Name.str.count(" ") + 1

train_df.shape, test_df.shape


# In[8]:


sns.distplot(train_df.Name_len, kde=False);
print(train_df.Name_len.mean())


# In[9]:


train_df[(train_df.Name_len > 50)]


# In[10]:


# Regular expression to get the title of the Name
for df in train_df, test_df:
    # exlude point
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

print(train_df.shape, test_df.shape)
train_df.Title.value_counts().reset_index()


# In[ ]:


# Show title counts by sex
pd.crosstab(train_df.Sex, train_df.Title)


# **Take-Away**: We can see that Dr. relates to men and women.

# In[11]:


# clean up titles
for df in train_df, test_df:
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major',                                        'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')


# In[12]:


# show new distribution
train_df['Title'].value_counts()


# In[13]:


# Finally, grab surname from passenger name
# just for information/analysis purposes => no relevant predictor
def substrSurname(name):
    end = name.find(', ')
    return name[0:end].strip()

for df in train_df, test_df:
    df["Surname"] = df.Name.map(substrSurname)
# short version with lamda function
#train_df["Surname"] = train_df.Name.map(lambda x: x[0:x.find(', ')].strip())

print('We have', len(np.unique(train_df.Surname)), 'unique surnames.')

train_df.sort_values('Surname').head(20)


# #### Sex
# nothing to do here

# #### Age

# In[ ]:


# show age values that are approximated
train_df[(train_df['Age'] % 1 == 0.5)]


# In[14]:


#interaction variable: since age and class are both numbers we can just multiply them
for df in train_df, test_df:
    df['Age*Class'] = df['Age'] * train_df['Pclass']


# #### SibSp
# 
# Perhaps people traveling alone did better? Or on the other hand perhaps if you had a family, you might have risked your life looking for them, or even giving up a space up to them in a lifeboat.

# In[15]:


# linear combination of features
for df in train_df, test_df:
    # including the passenger themselves
    df['Family_Size'] = df.SibSp + df.Parch + 1


# In[16]:


print("Travelling with family: ", train_df[train_df['Family_Size'] > 1].PassengerId.count())
print("Travelling alone: ",train_df[train_df['Family_Size'] == 1].PassengerId.count())

sns.factorplot('Family_Size', hue='Survived', data=train_df, kind='count')


# In[17]:


# further group familiy size into new column, if traveled alone or with family
for df in train_df, test_df:
    df['Travel_Alone'] = df['Family_Size'].map(lambda x: True if x == 1 else False)
sns.factorplot('Travel_Alone', hue='Survived', data=train_df, kind='count')


# In[18]:


train_df[['Travel_Alone', 'Survived']].groupby(['Travel_Alone'], as_index=False).mean()


# In[19]:


sns.barplot('Travel_Alone', 'Survived', data=train_df, color="mediumturquoise")
plt.show()


# **Take-Away:** Individuals traveling without family were more likely to die in the disaster than those with family aboard. Given the era, it's likely that individuals traveling alone were likely male.
# 

# #### Parch
# has been used in SibSp

# #### Ticket
# Ticket feature may be dropped 
# * high ratio of duplicates (22%) 
# * high variance

# In[ ]:


# tickets have high variance but there seems to be an indication for something
for df in train_df, test_df:
    df['Ticket_First'] = df.Ticket.str[0]

train_df['Ticket_First'].value_counts()


# #### Fare

# In[20]:


#Here we divide the fare by the number of family members traveling together
for df in train_df, test_df:
    df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'])


# #### Cabin
# * Cabin feature may be dropped as highly incomplete with many null values both in training and test dataset. 
# * Look for deck encoded in cabin.

# In[21]:


# first letter of the cabin denotes the cabin level (e.g. A,B,C,D,E,F,G).
# Create a Deck variable. Get passenger deck A - F:
def getCabinDeck(cabin):
    if not cabin or pd.isnull(cabin):
        # use string to make clear it is a category on its own
        return 'None'
    else:
        return cabin[0]

for df in train_df, test_df: 
    df["Deck"] = df.Cabin.map(getCabinDeck)


# In[ ]:


sns.factorplot('Deck',data=train_df,kind='count', hue='Survived')


# ![Titanic Deck Overview](https://www.drdiagram.com/wp-content/uploads/2017/02/template-diagram-of-titanic-ship-diagram-of-titanic-ship-diagram-of-titanic.jpg)

# ** Take-Away**: Deck Survival does not seem to reflect our assumption. Does not seem that relevant for a predictor.

# #### Embarked
# nothing to do here.

# #### <font color=blue> Summary </font>

# In[ ]:


print(train_df.shape, test_df.shape)
train_df.columns.difference(test_df.columns)


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# ### 4.3.2. Factorize (categorical -> numerical)
# encode categorical variables into numerical ones. Having more values than only 0,1 implies an implicit ordinal relationship and may lead to wron DT splits.

# In[ ]:


# Factorize the values 
labels,levels = pd.factorize(train_df.Sex)

train_df['Sex_Class'] = labels

# print(levels)
train_df.head()

#drop again
train_df.drop('Sex_Class', axis=1, inplace=True)


# In[22]:


for df in train_df, test_df:
    df['Sex_Class'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[23]:


for df in train_df, test_df:
    df['Embarked_Class'] = df["Embarked"].map(dict(zip(("S", "C", "Q"), (0, 1, 2))))


# In[24]:


# convert categorical to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for df in train_df, test_df:
    df['Title_Class'] = df['Title'].map(title_mapping)


# In[25]:


# convert from bool to int (0,1)
for df in train_df, test_df:
    df['Travel_Alone'] = df['Travel_Alone'].astype(int)


# ### 4.3.3. Dummy variables / One-hot encoding
# In theory, decision tree and forests work fine with categorical values, but sciki-learn does not accept categorical/string variables.

# In[ ]:


dummy_df = pd.get_dummies(train_df, columns=["Pclass", "Embarked", "Sex"])
dummy_df.drop('Sex_female', axis=1, inplace=True)
dummy_df.head()


# ### 4.3.4. Binning (numerical -> categorical)
# Bin continuous variables in groups

# #### age

# In[ ]:


# define bins automatically
mybins = range(0, int(train_df.Age.max()+10), 10)

# Cut the data with the help of the bins
train_df['age_bucket'] = pd.cut(train_df.Age, bins=mybins)

# Count the number of values per bucket
train_df['age_bucket'].value_counts()


# In[ ]:


## create bins for age
def get_age_group(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 11:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 65:
        a = 'adult'
    else:
        a = 'senior'
    return a


# In[ ]:


for df in train_df, test_df:
    df['Age_Group'] = df['Age'].map(get_age_group)

    # Factorize the values 
    labels,levels = pd.factorize(df.Age_Group)
    df['Age_Group'] = labels

# levels are the same for train and test
print(levels)
train_df.head(20)


# In[ ]:


dummy_df2 = pd.get_dummies(train_df, columns=["Age_Group"])
dummy_df2.head()


# #### Name_len

# In[ ]:


def get_name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'normal'
    else:
        a = 'long'
    return a


# In[ ]:


train_df['Name_len_Class'] = train_df['Name_len'].map(get_name_length_group)
# this should then be factorized again


# In[ ]:


## cuts the column by given bins based on the range of name_length
group_names = ['short', 'medium', 'normal', 'long']
train_df['Name_len_Class2'] = pd.cut(train_df['Name_len'], bins = 4, labels=group_names)
train_df['Name_len_Class2'].value_counts()


# In[ ]:


train_df.drop(['Name_len_Class2','Name_len_Class'] , axis=1, inplace=True, errors='ignore')


# ### Family_Size

# In[ ]:


def get_family_group(size):
    a = ''
    if (size <= 1):
        a = 'alone'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a


# In[ ]:


train_df['Family_Size_Class'] = train_df['Family_Size'].map(get_family_group)


# ### 4.3.5. Scaling/Z-transformation
# Scale features: center your data around 0.

# # 4.5. Feature Selection/Elimination
# * manually
# * dimensionality reduction techniques
#     * Missing Values Ratio
#     * Low Variance Filter
#     * High Correlation Filter
#     * predictive models (e.g. RandomForest)
#     * PCA
#     * Backward/Forward Feature Elimination

# In[ ]:


train_df.head()


# In[26]:


# manually set candidates for deletion
# typically those features are being determined automatically

## # PassengerId has too high variance
## summary_df.loc['PassengerId','keep']=False
## # Ticket column has a lot of various values. It will have no significant impact
## summary_df.loc['Ticket','keep']=False
## # Cabin has too many missings
## summary_df.loc['Cabin','keep']=False
## # name as too many distinct values as well
## summary_df.loc['Name','keep']=False
## 
## s = summary_df[(summary_df.keep == False)].index
## 
## for df in train_df, test_df:
##     for i in s:
##         df.drop(i, axis=1, inplace=True)


# PassengerId has too high variance,  IDs are unnecessary for classification
# drop only in training set, as ID is needed for submission file
train_df.drop('PassengerId', axis=1, inplace=True, errors='ignore')

for df in train_df, test_df:
    # Ticket column has a lot of various values. It will have no significant impact
    df.drop('Ticket', axis=1, inplace=True, errors='ignore')
    # Cabin has too many missings
    df.drop('Cabin', axis=1, inplace=True, errors='ignore')
    # name has too many distinct values as well
    df.drop('Name', axis=1, inplace=True, errors='ignore')
    # drop due to factorization/encoding
    df.drop('Sex', axis=1, inplace=True, errors='ignore')
    df.drop('Embarked', axis=1, inplace=True, errors='ignore')
    df.drop('Title', axis=1, inplace=True, errors='ignore')
    df.drop('Surname', axis=1, inplace=True, errors='ignore')
    df.drop('Ticket_First', axis=1, inplace=True, errors='ignore')
    df.drop('Deck', axis=1, inplace=True, errors='ignore')
    

# drop a list of colu
# drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
# train_df = train_df.drop(drop_elements, axis = 1)


# In[27]:


train_df.head()


# In[ ]:


train_df.columns


# In[ ]:


#train_df.columns[1:]


# In[28]:


# obtain feature importances

# Import `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = list(train_df.drop(['Survived'], axis=1).columns.values)

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['Survived'], axis=1), train_df["Survived"])

# Build the model
# randomly generates thousands of decision trees and takes turns leaving out each variable in fitting the model
rfc = RandomForestClassifier()

# Fit the model
rfc.fit(X_train, y_train)

# Print the results
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), features), reverse=True))

# Isolate feature importances 
#importance = rfc.feature_importances_

# Sort the feature importances 
#sorted_importances = np.argsort(importance)


# # 5. Model Building & Visualization
# 
# * The target variable is the variable you are trying to predict: here it is "survived" --> **binary classification**
# * Other variables are known as "features" (or "predictor variables", the features that you're using to predict the target variable).
# 
# min: presuming that every passenger died 
# max: maximum of around 82%. 
# 

# In this case, understanding the Titanic disaster and specifically what variables might affect the outcome of survival is important. Anyone who has watched the movie Titanic would remember that women and children were given preference to lifeboats (as they were in real life). You would also remember the vast class disparity of the passengers.
# 
# This indicates that **Age, Sex, and PClass** may be good predictors of survival. 

# ## 5.0. Separating dependent and independent variables, splitting

# In[29]:


# imports
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix


# In[30]:


X = train_df.drop(['Survived'], axis=1)
y = train_df["Survived"]


# In[31]:


# splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

# default split is with 25%
X_train.shape, X_test.shape


# In[ ]:


# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# ## 5.1. Simple Model 1
# In the training set, less people survived than didn't. Let's then build a first model that predicts that nobody survived.
# This is a bad model as you know that people survived. But it gives us a baseline: any model that we build later needs to do better than this one.

# In[32]:


test_df['Survived'] = 0
test_df[['PassengerId', 'Survived']].to_csv('output_model1.csv', index=False)


# ## 5.2. Simple Model 2
# As EDA has shown, women are more likely to survive.

# In[34]:


test_df.head()
test_df.drop('Survived', axis=1, inplace=True, errors='ignore')


# In[35]:


test_df['Survived'] = test_df.Sex_Class == 1
# convert bool to int (0,1)
test_df['Survived'] = test_df.Survived.apply(lambda x: int(x))
test_df[['PassengerId', 'Survived']].to_csv('output_model2.csv', index=False)


# ## Decision Tree Classifier

# In[36]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier()
dectree.fit(X_train, y_train)
y_pred = dectree.predict(X_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)


# ## Random Forest Classifier
# randomly select a subset of features and samples, fit one decision tree on each draw and average their predictions. decision trees are created so that rather than selecting optimal split points, suboptimal splits are made by introducing randomness

# In[37]:


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=100,max_depth=9,min_samples_split=6, min_samples_leaf=4)
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
random_accy = round(accuracy_score(y_pred, y_test), 3)
print (random_accy)


# In[38]:


test_df.drop('Survived', axis=1, inplace=True, errors='ignore')
test_df.shape


# In[41]:


# ignore PaxID
test_prediction = randomforest.predict(test_df.iloc[:,1:])
submission = pd.DataFrame({
        "PassengerId": test_df.PassengerId,
        "Survived": test_prediction
    })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv('output_model_RF.csv', index=False)


# ## XGBoost Classifier
# Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models. XGBoost (eXtreme Gradient Boosting) is an implementation of gradient boosted decision trees designed for speed and performance.

# In[ ]:


from xgboost import XGBClassifier
XGBClassifier = XGBClassifier()
XGBClassifier.fit(X_train, y_train)

y_pred = XGBClassifier.predict(X_test)

XGBClassifier_accy = round(accuracy_score(y_pred, y_test), 3)

print(XGBClassifier_accy)


# # 6. Outlook
# * Grid Search for Parameters
# * Cross Validation
# * Voting Classifier

# In[ ]:




