#!/usr/bin/env python
# coding: utf-8

# # Exploring and Processing Data

# * ## Import datasets

# ####  import requires modules

# In[17]:


import numpy as np
import pandas as pd
import os


# #### set raw data file path

# In[21]:


train_file_path ='../input/train.csv'
test_file_path = '../input/test.csv'


# In[22]:


#### import dataset as pandas DataFrames


# In[24]:


train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col='PassengerId')


# In[25]:


#get data frame information


# In[26]:


train_df.info()


# In[ ]:


test_df.info()


# #### survived column is missing in test_df 
# 
# #### add survied column in test dataframe

# In[27]:


test_df['Survived'] = -222 #enter any values


# In[34]:


test_df.info()


# #### concate both train and test data 
# ##### axis=0 means it will append vertically and axis=1 means it will append horizontally

# In[29]:


df = pd.concat((train_df,test_df), axis=0)


# In[36]:


df.info()


# ## selection, indexing and filtering

# In[37]:


#head function returns top n row. default no. of rows is 5.
df.head()


# In[38]:


#tail returns last n rows. default no of rows is 5.
df.tail(6)


# #### selection methods

# In[40]:


#select column from DataFrame
df['Name']


# In[41]:


#selecting colums as list of columns

df[['Name', 'Age']]


# In[317]:


#select specified rows using slicing
df.loc[1:10, 'Age': 'Name']


# In[318]:


#select discrete columns 
df.loc[1:5, ['Name', 'Age', 'Sex']]


# In[319]:


#use iloc for location based indexing
df.iloc[1:5, 2:6]


# In[320]:


#filter rows based on some condition on columns
female_passengers = df.loc[(df.Sex=='female') & (df.Pclass==1)]
len(female_passengers)


# ## summary statitics

# #### numerical fetures

# #### describe function is use for details of all statistics

# In[42]:


df.describe()


# In[43]:


#we can also get mean and other statistics diretly
fare_mean = df.Fare.mean()


# In[44]:


df.Fare.min()


# In[45]:


df.Fare.max()


# In[48]:


df.Fare.quantile(.25)


# #### box-whisker plot using plot method in pandas
# #### % is a magic function. 

# In[49]:


get_ipython().magic(u'matplotlib inline')

df.Fare.plot(kind='box')


# #### categorical features

# In[50]:


#with include parameter we can filter out 
df.describe(include='all')


# In[51]:


#counts the values on basis of unique categories
df.Sex.value_counts()


# In[52]:


#categorical column: #proportion
df.Sex.value_counts(normalize='True')


# In[53]:


df[df.Survived != -888].Survived.value_counts()


# In[54]:


df.Pclass.value_counts()


# In[55]:


df.Pclass.value_counts().plot(kind='bar')


# In[56]:


df.Pclass.value_counts().plot(kind='bar', title= "Passenger counts class wise", rot=0)


# ## Distributuons

# ### Univariate Distribution

# #### histogram and KDA plots

# In[57]:


#plot a histogram for Age 
df.Age.plot(kind='hist', title="histogram for Age")


# In[58]:


#histogram with specified bins
df.Age.plot(kind='hist', title="histogram for Age", bins=20)


# In[59]:


#KDE plot for age
df.Age.plot(kind='kde', title="Kernel density plot for Age")


# In[60]:


df.Fare.plot(kind='hist', title='histogram for fare', bins=20)


# In[61]:


#skewness for Age column
df.Age.skew()


# In[62]:


#skewness for Fare column
df.Fare.skew()


# ### Bivariate Distribution

# #### scatter plot

# In[63]:


#scatter plot for age and fare
df.plot.scatter(x='Age', y='Fare', title="scatter plot: Age vs Fare")


# In[64]:


#we can also use alpha for opacity
df.plot.scatter(x='Age', y='Fare', title="scatter plot: Age vs Fare", alpha=0.1)


# In[65]:


#scatter plot between Pclass and Fare: pclass is categorical feature
df.plot.scatter(x='Pclass', y='Fare', title="scatter plot: Pclass vs Fare", alpha=0.12)


# ## Groping and Aggregation

# In[66]:


#groupby
df.groupby('Sex').Fare.mean()


# In[67]:


df.groupby(['Pclass']).Fare.mean()


# In[68]:


df.groupby(['Pclass']).Age.mean()


# In[69]:


df.groupby(['Pclass'])['Age', 'Fare'].mean()


# In[70]:


#using agg
df.groupby(['Pclass']).agg({'Fare': 'mean', 'Age': 'median'})


# In[71]:


#aggregate dictionary
aggregate = {
    'Fare': {
        'fare_mean': 'mean',
        'fare_median': 'median',
        'fare_max': max,
        'fare_min': np.min
    },
    'Age': {
        'age_mean': 'mean',
        'age_median': 'median',
        'age_max': max,
        'age_min': min,
        'age_range': lambda x : max(x)-min(x)
    }
}


# In[72]:


df.groupby(['Pclass']).agg(aggregate)


# In[73]:


#group based on two and more variables
df.groupby(['Pclass', 'Sex']).Fare.mean()


# In[74]:


df.groupby(['Pclass', 'Sex', 'Embarked']).Fare.mean()


# ## crosstabs

# In[75]:


#crosstabs
pd.crosstab(df.Sex, df.Pclass)


# In[76]:


#crosstabs using bars
pd.crosstab(df.Sex, df.Pclass).plot(kind='bar',title='class vs sex', rot=0)


# ## Pivot Tables

# In[77]:


#pivot table
df.pivot_table(index='Sex', columns='Pclass', values='Age', aggfunc='mean')


# In[78]:


df.groupby(['Pclass', 'Sex']).Age.mean()


# In[79]:


#same result we can get from groupby 
df.groupby(['Pclass', 'Sex']).Age.mean().unstack()


# ## Data Munging

# ### Working with missing values

# In[80]:


#information about data
df.info()


# #### check how many column have missing values
# do whatever you want to handle these missing values

# #### fill Embarked column values

# In[81]:


#find rows for null values
df[df.Embarked.isnull()]


# In[82]:


#find how many type of Embarked, or categorical feature
df.Embarked.value_counts()


# In[83]:


#which embarked point has highest survived counts
pd.crosstab(df[df.Survived != -888].Embarked, df[df.Survived != -888].Survived).plot(kind='bar')


# In[84]:


#set Embarked value on basis of survived count
#df.loc[df.Embarked.isnull(), 'Embarked']='S'
#df.Embarked.fillna('S', inplace=True)


# In[85]:


#options: categories on basis of Pclass and fare
df.groupby(['Pclass', 'Embarked']).Fare.median().plot(kind='bar', rot=0)


# In[86]:


#fill value of Embarked with 'C'
df.Embarked.fillna('C', inplace=True)


# In[87]:


len(df.Embarked.isnull().values)


# In[88]:


df[df.Embarked.isnull()]


# #### we can see there is no null values in Embarked column

# In[89]:


df.info()


# #### next munging for Fare column

# ## Munging: Fare

# In[90]:


df[df.Fare.isnull()]


# In[91]:


df.groupby(['Pclass', 'Embarked']).Fare.median()


# In[92]:


mean_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'C'), 'Fare'].median()
print("mean fare value where class is 3 and Embarked value is C: {0}".format(mean_fare))


# In[93]:


#fill missing value of fare with median values
df.Fare.fillna(mean_fare, inplace=True)


# In[94]:


df[df.Fare.isnull()]


# In[95]:


df.info()


# In[96]:


#set maximum number of raws to display in case of large rendered data
pd.options.display.max_rows = 15


# ### Munging : Age

# In[97]:


df[df.Age.isnull()]


# #### option1: replace all missing age values with mean

# In[98]:


#histogram of age ranges
df.Age.plot(kind='hist', bins=20)


# In[99]:


df.Age.plot(kind='kde')


# In[102]:


#mean value of age
mean_age = df.Age.mean()
median_age = df.Age.median()
print("mean of Age: {0}".format(mean_age))
print("median of Age: {0}".format(median_age))


# #### option2:replace with median age of gender

# In[103]:



df.groupby(['Sex']).Age.median()


# In[104]:


df[df.Age.notnull()].boxplot('Age', 'Sex');


# In[106]:


#age_sex_median = df.groupby('Sex').Age.transform('median')
#df.Age.fillna(age_sex_median, inplace=True)


# #### this is also not a proper discrimption on basis of Sex

# ### option3: replace values with median Age of Pclass 
# 

# In[107]:


df[df.Age.notnull()].boxplot('Age', 'Pclass')


# In[108]:


#age_pclass_median = df.groupby('Pclass').Age.transform('median)
#df.Age.fillna(age_sex_median, inplace=True)


# In[109]:


df.head()


# #### option4: replace age with title of name
# 

# In[110]:


#get title of name
def getTitle(name):
    name_with_title = name.split(',')[1]
    title_of_name = name_with_title.split('.')[0]
    title = title_of_name.strip().lower()
    return title


# In[111]:


#testing of getTitle function
name = "BLR, Mr. Pulkit Agrawal"
print(getTitle(name))


# In[112]:


#we need unquie title for these data sets
df.Name.map(lambda x: getTitle(x)).unique()


# In[113]:


#get specified category title for name
def getSpecifiedTitle(name):
    title_category ={
        'mr': 'Mr',
        'mrs': 'Mrs',
        'miss': 'Miss',
        'master': 'Master',
        'don': 'Sir',
        'rev': 'Sir',
        'dr': 'Officer',
        'mme': 'Mrs',
        'ms': 'Mrs',
        'major': 'Master',
        'lady': 'Lady', 
        'sir': 'Sir', 
        'mlle': 'Lady', 
        'col': 'Officer', 
        'capt': 'officer', 
        'the countess': 'Lady',
        'jonkheer': 'Sir',
        'dona': 'Lady'
    }
    name_with_title = name.split(',')[1]
    title_of_name = name_with_title.split('.')[0]
    title = title_of_name.strip().lower()
    return title_category[title]


# In[114]:


#create a new Title column
df['Title']=df.Name.map(lambda x: getTitle(x))


# In[115]:


df.info()


# In[116]:


df[df.Age.notnull()].boxplot('Age', 'Title')


# In[117]:


#replace missing Age values with median of title
age_title_median = df.groupby('Title').Age.transform('median')
df.Age.fillna(age_title_median, inplace=True)


# In[118]:


df.info()


# ## Working with Outliers

# #### Feature: Age

# In[119]:


df.Age.plot(kind='hist',bins = 20)


# we can easily see that there are some outliers who have more than 70

# In[120]:


df.loc[df.Age > 70]


# #### Feature: Fare

# In[121]:


df.Fare.plot(kind='hist', bins=20)


# In[122]:


df.Fare.plot(kind='box')


# we can see in Fare there is many outliers.
# we will try to reduce skewness of this by taking the log

# In[123]:


logFare = np.log(df.Fare + 1)


# In[124]:


logFare.plot(kind='hist', bins = 20)


# In[125]:


logFare.plot(kind='box')


# #### we can also use qcut for creating the bins for remove outliers

# In[126]:


#binning
pd.qcut(df.Fare, 4)


# In[127]:


#discritization
pd.qcut(df.Fare, 4, labels = ['very_low', 'low', 'high', 'very_high'])


# In[128]:


pd.qcut(df.Fare, 4, labels = ['very_low', 'low', 'high', 'very_high']).value_counts().plot(kind='bar', rot=0)


# In[129]:


#create new feature column
df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels = ['very_low', 'low', 'high', 'very_high'])


# In[130]:


df.info()


# ## Feature Engineering

# #### Feature: AgeState

# In[131]:


#create a AgeState feature
df['AgeState'] = np.where(df['Age'] >= 18, 'Adult', 'Child')


# In[132]:


df.info()


# In[133]:


df.AgeState.value_counts()


# In[134]:


pd.crosstab(df.loc[df.Survived != -888].Survived, df.loc[df.Survived != -888].AgeState)


# #### Feature: FamilySize

# In[135]:


# create a familysize feature


# In[136]:


df['FamilySize'] = df.Parch + df.SibSp +1


# In[137]:


df.FamilySize.plot(kind='hist')


# In[138]:


df.loc[df.FamilySize == df.FamilySize.max() , ['Age', 'Ticket', 'FamilySize', 'Survived']]


# In[139]:


pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].FamilySize)


# As we can easily see that family size with less persons survived 
# and large family not survived

# #### Feature: Deck

# In[140]:


df.Cabin.unique()


# In[141]:


def getDeck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')
df['Deck'] = df['Cabin'].map(lambda x: getDeck(x))


# In[142]:


df.Deck.value_counts()


# In[143]:


pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived !=-888].Deck)


# In[144]:


df.info()


# ### categorical feature encoding

# In[145]:


#change categorical feature into values
df['isMale'] = np.where(df.Sex=='Male', 1, 0)


# In[146]:


#create one-hot encoding 
df = pd.get_dummies(df, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])


# In[147]:


df.info()


# ### Drop and Reordered columns

# In[148]:


#drop unused columns
df.drop(['Cabin', 'Name', 'Parch', 'SibSp', 'Ticket', 'Sex'], axis=1, inplace=True)


# In[153]:


#reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]


# In[154]:


df.info()


# In[155]:


#write train data
df_train = df.loc[df.Survived != -888]

#write test data 
columns = [column for column in df.columns if column != 'Survived']
df_test = df.loc[df.Survived == -888, columns]


# In[156]:


df_train.info()


# # Build Prediction Model

# ### split training data for crossvalidation

# In[158]:


#convert input and output features
X = df_train.loc[:,'Age':].as_matrix().astype('float')
y = df_train['Survived'].ravel()


# In[159]:


print(X.shape)
print(y.shape)


# In[160]:


#split data into 80/20 using train_test_split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[161]:


print("train: {0}, {1}".format(X_train.shape, y_train.shape))
print("test: {0}, {1}".format(X_test.shape, y_test.shape))


# ## Logistic Regression for analysis****

# In[163]:


#import logistic regression
from sklearn.linear_model import LogisticRegression
logisticRg_model = LogisticRegression(random_state = 0)
logisticRg_model.fit(X_train, y_train)


# In[164]:


print("score of the Logistic Regression model: {0:.3f}".format(logisticRg_model.score(X_test, y_test)))


# In[167]:


logistic_predicted_model = logisticRg_model.predict(X_test)


# In[168]:


#imports performance matrices
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_curve


# In[170]:


#confusion metices
print("Confusion Metrices of Logistic Regression model : \n {0}".format(confusion_matrix(y_test, logistic_predicted_model)))


# In[ ]:




