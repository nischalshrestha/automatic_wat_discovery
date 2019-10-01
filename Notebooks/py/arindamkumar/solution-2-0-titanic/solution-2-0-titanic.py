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



import matplotlib as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


path='../input/'


# In[ ]:


train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')


# In[ ]:


# save the passenger id for the final submission
passengerId=test.PassengerId

# merge train and test
titanic = train.append(test, ignore_index=True)

## we use the ignore_index as in test data we have the labels columns which is not present in the train data.


# In[ ]:


train_id=len(train)
test_id=len(titanic)-len(test)


# In[ ]:


train_id


# In[ ]:


test_id


# In[ ]:


len(titanic)


# In[ ]:


len(test)


# In[ ]:


titanic.head()


# In[ ]:


titanic.info()


# It looks like we have a few NaNs in the dataset across a few features. We will use the data to try and fill in the gaps. The info() method reveals that the Age, Cabin, Embarked, and Fare all have a few entries missing. Technically the Survived column also has entries missing, but this is actually correct since we merged the train and test together for future feature engineering and the test data doesn't have a Survived column.
# 
# Additionally, from looking at the features, it looks like we can just drop PassengerId from the dataset all together since it isn't really a helpful feature, but rather simply a row identifier.

# In[ ]:


titanic.drop(['PassengerId'],1,inplace=True)


# In[ ]:


titanic.head()


# Now we create a title feature which extracts the honorifc from the Name feature.Simply put, an honorific is the title or rank of a given person such as “Mrs” or “Miss”. The following code takes a value like “Braund, Mr. Owen Harris” from the Name column and extracts “Mr”.

# In[ ]:


titanic['Title']=titanic.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip() )


# In[ ]:


titanic.head()


# In[ ]:


## title counts
#print("There are {} unique title.".format(titanic.Title.nunique))
print("There are {} unique titles.".format(titanic.Title.nunique()))
print("\n", titanic.Title.unique())


# In[ ]:


titanic.head()


# In[ ]:


# normalize the titles
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}

def convert(val):
    return normalized_titles[val]


# In[ ]:


titanic.head()


# In[ ]:


type(titanic.Title.values[0])


# In[ ]:


# view value counts for the normalized titles
print(titanic.Title.value_counts())


# In[ ]:


titanic.Title = titanic.Title.map(normalized_titles)


# In[ ]:


titanic.head()


# In[ ]:


# view value counts for the normalized titles
print(titanic.Title.value_counts())


# For our next step, we are going to assume that their is a relationship between a person's age and their title since it makes sense that someone that is younger is more likely to be a titled a "Miss" vs a "Mrs".
# 
# With this in mind, we will group the data by Sex, Pclass, and Title and then view the median age for the grouped classes.

# In[ ]:


#groupby sex,Pclass and Title
grouped=titanic.groupby(['Sex','Pclass','Title'])
grouped.Age.median()


# In[ ]:


titanic.Embarked.value_counts()


# In[ ]:


grouped1=titanic.groupby(['Sex','Pclass','Title','Embarked'])
grouped1.Age.median()


# As expected, those passengers with a title of "Miss" tend to be younger than those titled "Mrs". Also, it looks like we have some age variability amongst the different passenger classes as well as between the sexes, so this should help us more accurately estimate the missing ages for the observations that do not have an age recorded.

# In[ ]:


## applying the grouped median age value
titanic.Age=grouped.Age.apply(lambda x:x.fillna(x.median()))

titanic.info()


# In[ ]:


titanic.head(10)


# In[ ]:


titanic.Embarked.value_counts()


# In[ ]:


most_embarked=titanic.Embarked.value_counts().index[0]


# In[ ]:


most_embarked


# In[ ]:


titanic.info()


# In[ ]:


titanic.Embarked=titanic.Embarked.fillna(most_embarked)
titanic.info()


# In[ ]:


titanic.head()


# In[ ]:


titanic.info()


# In[ ]:


##percentage of death vs percentage of survival
titanic.Survived.value_counts()


# In[ ]:


titanic.Survived.value_counts(normalize=True)


# In[ ]:


## lets dig deeper and determine the survival rates based on the gender
groupbysex=titanic.groupby(['Sex'])
groupbysex.Survived.value_counts(normalize=True)


# In[ ]:


##survival rates based on their sex
groupbysex.Survived.mean()


# For those who have seen the fateful story of titanic we know that the women and children were given priority oven men.Even though it is very astounding that only 19% of the men survived compared the 75% women.

# In[ ]:


## group by passenge Pclass and sex
group_class_sex=titanic.groupby(['Pclass','Sex'])
group_class_sex.Survived.mean()


# 
# It appears that 1st class females had an incredible 97% survival rate while 1st class males only still had a 37% chance of survival. Even though you only had a 37% chance of surviving as a 1st class male, you still were almost 3 times more likely to survive than a 3rd class male who had the lowest survival rate amongst sex and class at 13.5%.

# The social status gives us a pretty good idea about the survival chance.

# In[ ]:


##get stats on all other metrics
titanic.describe()


# # Creating new features from the data

# The first feature we will look at building is FamilySize. This is important to look at because we want to see if having a large or small family affected someone's chances of survival.
# 
# The relevant features that will help with this are Parch (number of parents/children aboard) and SibSp (number of siblings/spouses aboard). We combine the Parch and SibSp features and add 1 as well as we want to count the passenger for each observation.

# In[ ]:


## size of the family including the passenger.
titanic['FamilySize']=titanic['Parch']+titanic['SibSp']+1


# In[ ]:


titanic=titanic.drop(['Parch','SibSp'],1)
titanic.head()


# In[ ]:


titanic.info()


#  we can also generate info from the Cabin as cabins near the life boats will have higher chance of suvival compared to the others located elsewhere.So we extract the first letter from the cabin and generate features.

# # Cabin

# 
# In theory, the deck can increase the probability of survival, as this includes the distance to the lifeboats. We've already seen this in the 'Class' attribute. The room cannot be assigned directly. In general it can be said that the variable is very experimental and needs further research. For example, it would be possible to predict the missing values using the ticket number, the fare and the surname. For the beginning of the missing cabins we assign an own deck and the mean value (50).

# In[ ]:


#Deck
titanic["Deck"] = titanic["Cabin"].str.slice(0,1)
titanic["Deck"] = titanic["Deck"].fillna("N")

#Room
titanic["Room"] = titanic["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
titanic["Room"] = titanic["Room"].fillna(titanic["Room"].mean())
titanic["Room"] = titanic.Room.astype(int)


# In[ ]:


# ## map the first letter of the cabin to the cabin.
# titanic.Cabin=titanic.Cabin.map(lambda x:x[0])

# ## view the normalized count
# titanic.Cabin.value_counts(normalize=True)
titanic=titanic.drop(['Cabin'],1)


# In[ ]:


titanic.head()


# In[ ]:


titanic.Deck.value_counts()


# # Tickets
# 
# If the 'Ticket' column is examined randomly, some details will be noticed. Some tickets have a letter in front of them and others consist only of numbers. Furthermore, the tickets have a different length. Let's take a closer look at this. (Based on https://www.kaggle.com/zlatankr/titanic-random-forest-82-78)
# 
# First we take the first character from the ticket number and in the next step we also determine the ticket length and look at the connection with the variable 'Survive'.

# In[ ]:


titanic['Ticket_len']=titanic['Ticket'].apply(lambda x:len(x))
titanic.groupby(['Ticket_len']).Survived.mean()


# In[ ]:


titanic['Ticket_lett']=titanic['Ticket'].apply(lambda x:str(x)[0])
titanic.groupby(['Ticket_lett']).Survived.mean()


# So there is a different distribution between the different ticket letters as well as the length of the ticket number. Therefore, we use both variables in the model.
# 
# I'm forming two groups to split the ticket letters. However, you can also test with a different classification.

# In[ ]:


#Ticket Letter Encoding
replacement = {
    'A': 0,
    'P': 1,
    'S': 0,
    '1': 1,
    '2': 0,
    'C': 0,
    '7': 0,
    'W': 0,
    '4': 0,
    'F': 1,
    'L': 0,
    '9': 1,
    '6': 0,
    '5': 0,
    '8': 0,   
    '3': 0,
}

titanic['Ticket_lett']=titanic['Ticket_lett'].map(replacement)
titanic.head()


# In[ ]:


titanic=titanic.drop(['Ticket'],1)


# In[ ]:


titanic.info()


# We can use the length of the name as the features.

# In[ ]:


titanic['Name_len']=titanic['Name'].apply(lambda x:len(x))
titanic.groupby(['Name_len']).Survived.mean()


# We can conclude from the above result that people with bigger names such as the dignitaries have higher rates of survival.

# In[ ]:


titanic=titanic.drop(['Name'],1)


# In[ ]:


titanic['Fare'].value_counts()


# In[ ]:


type(titanic['Fare'][0])


# In[ ]:


titanic.head()


# In[ ]:


titanic.info()


# In[ ]:


titanic['Fare']=titanic['Fare'].fillna(titanic['Fare'].median())
titanic.info()


# In[ ]:


titanic.loc[titanic['Fare']<=7.896,'Fare_grouped']=0
titanic.loc[(titanic['Fare']>7.896) & (titanic['Fare']<=14.454),'Fare_grouped']=1
titanic.loc[(titanic['Fare']>14.454) & (titanic['Fare']<=31.275),'Fare_grouped']=2
titanic.loc[(titanic['Fare']>31.275),'Fare_grouped']=3
titanic['Fare_grouped']=titanic['Fare_grouped'].astype('int')


# In[ ]:


titanic.head()


# In[ ]:


def handle_non_numeric_data(df):
	columns=df.columns.values
	for column in columns:
		text_digit_vals={}
		def convert_to_int(val):
			return text_digit_vals[val] 

		if df[column].dtype!= np.int64 and df[column].dtype!= np.float64:
			column_contents=df[column].values.tolist()		#.values is used to get the values of a function
			unique_elements=set(column_contents)	#converting to a set
			x=0
					
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique]=x
					x+=1

			df[column]=list(map(convert_to_int,df[column]))		#we are resetting the df column by mapping the function here to the value in the column

	return df


# In[ ]:


titanic=handle_non_numeric_data(titanic)


# In[ ]:


titanic.head()


# In[ ]:


titanic=titanic.drop(['Fare'],1)


# In[ ]:


titanic.head()


# In[ ]:



train=titanic[:train_id]
test=titanic[test_id:]


# In[ ]:


## convert the survived back to int
train.Suvived=train.Survived.astype(int)


# In[ ]:


train.head()


# # Modelling

# In[ ]:


# create X and y for data and target values
X = train.drop('Survived', axis=1).values
y = train.Survived.values


# In[ ]:


test.head()


# In[ ]:


X_test=test.drop('Survived',1).values


# # Logistic Regression

# The first model we will try is a Logistic Regression model which is a binary classifier algorithm. We will be using GridSearchCV to fit our model by specifying a few paramters and return the best possible combination of those parameters.

# In[ ]:


# The parameters that we are going to optimise
parameters = dict(
    C = np.logspace(-5, 10, 15),
    penalty = ['l1', 'l2']
    #solver =[‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
    
)


# In[ ]:


## instantiate the logistic regression
clf=LogisticRegression()

# Perform grid search using the parameters and f1_scorer as the scoring method
grid_search=GridSearchCV(estimator=clf,param_grid=parameters,cv=6,n_jobs=-1)
# here cv is used for the cross-validation strategy.


# In[ ]:


grid_search.fit(X,y)


# In[ ]:


clf1=grid_search.best_estimator_        # get the best estimator(classifier)
print(clf1)


# In[ ]:


# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(grid_search.best_params_)) 
print("Best score is {}".format(grid_search.best_score_))


# In[ ]:


## prediction on test set
pred=grid_search.predict(X_test)
print(pred)


# # Random Forest Model

# The best score using logistic regression was ~82% which wasn't bad. But let's see how we can fare with a Random Forrest Classifier algorithm instead.

# In[ ]:


# create param grid object
forrest_params = dict(
    max_depth = [n for n in range(7, 14)],
    min_samples_split = [n for n in range(4, 12)],
    min_samples_leaf = [n for n in range(2, 6)],
    n_estimators = [n for n in range(10, 60, 10)],
)


# In[ ]:


forest=RandomForestClassifier()


# In[ ]:


# build and fit model
forest_cv = GridSearchCV(estimator=forest, param_grid=forrest_params, cv=5)
forest_cv.fit(X, y)


# In[ ]:


print("Best score: {}".format(forest_cv.best_score_))
print("Optimal params: {}".format(forest_cv.best_estimator_))


# In[ ]:


# random forrest prediction on test set
forrest_pred = forest_cv.predict(X_test)


# Random forest classifier has a better accuracy than the logistic regression as deduced above.

# # For submission on kaggle

# In[ ]:


sub=pd.DataFrame({'PassengerId':passengerId,'Survived':forrest_pred})


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('prediction1_titanic.csv',index=False)   
## we initialise the index as false as we donot need the index


# In[ ]:





# In[ ]:





# In[ ]:




