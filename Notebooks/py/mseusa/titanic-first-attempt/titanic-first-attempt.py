#!/usr/bin/env python
# coding: utf-8

# **First Kaggle kernel - Building a sensible, robust solution to the Titanic problem**
# 
# *This is by no means an exhaustive study, however it should serve as a good overview of building a simple first model which performs well without succumbing to overfitting.
# This method resulted in a public score of 0.80382 using a random forest model.
# Inspiration for much of the feature engineering came from [this](https://www.kaggle.com/erikbruin/titanic-2nd-degree-families-and-majority-voting) very thorough kernel.
# *
# 1. Reading the data and an initial look
# 2. Dealing with missing age data
# 3. Intuitions and high level feature analysis
# 4. Calculating group size, family or friends?
# 5. Building our features and fitting a model
# 6. Potential improvements
# 
# 

# **1. Reading the data and an initial look**
# 
# First, we can glance at the data frames to check they were read correctly
# 

# In[ ]:



import numpy as np 
import pandas as pd 

df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_full = pd.concat([df,df_test])



df.head()




# In[ ]:


df_test.head()


# Check for missing data by counting the nulls in each dataframe

# In[ ]:


df.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# The cabin sparsity is the biggest concern here, with 687 out of 891 records missing there's very little chance that it will be of interest to us compared to other features. 
# 
# Age has a significant rate of missing data (~20%) so some care will need to be taken when imputing this feature.
# 
# If embarked is to be used then it would most likely be sufficient to just use the most frequent entry as there is no missing data in the test set and only 2 missing rows in the training set. 

# **2. Dealing with missing age data**
# 
# With age likely a significant indicator of survival, we need to impute missing data with a little more care than a blunt mean/median approach. 
# Using a bit of intuition one would assume that:
# - Males with the title 'Master' are children
# - Females with the title 'Miss' would be younger than the title 'Mrs'
# - Those in first class were more likely to be older than those in second and third
# 
# The first step will be extracting and simplifying the titles of the passengers.

# In[ ]:


#Title list was extracted using the substrings in string method and combined into a unique list.
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']

def substrings_in_string(string, substrings):

    for substring in substrings:
        if str(string).find(substring) != -1:
            return substring
    
    return np.nan


#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
    
df_full['Title']=df_full['Name'].map(lambda x: substrings_in_string(x, title_list))
df_full['Title']=df_full.apply(replace_titles, axis=1)

df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))
df['Title']=df.apply(replace_titles, axis=1)

df_test['Title']=df_test['Name'].map(lambda x: substrings_in_string(x, title_list))
df_test['Title']=df_test.apply(replace_titles, axis=1)


# With the titles of passengers aggregated into a simple list, we can now inspect the spread of age across these titles for each class to confirm our intuition.

# In[ ]:


import matplotlib.pyplot as plt

titles = ['Mrs','Mr','Miss','Master']
plt.figure(figsize=(12,12))
for title in titles:
    data = []
    for Pclass in [1,2,3]:
        data.append(df_full.loc[(df_full['Title'] == title) & 
                                (df_full['Pclass'] == Pclass) & 
                                ~np.isnan(df_full['Age'])]['Age'].values)
    plt.subplot(2,2,titles.index(title)+1)
    plt.boxplot(data)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Age')
plt.tight_layout()
plt.show()


# In[ ]:


df[['Title','Pclass','Age']].groupby(['Title','Pclass']).median()


# From the above summary and plots we can see there is indeed an impact of class on age, as well as title.
# 
# In the final model, passengers were categorised as 'Male, Female, Child' , so rather than perform a regression it should be sufficient for now to just take the median age for each title / class as shown in the table above. 
# 
# The only problem that may arise from this method is labelling those with the title 'Miss' as adults when they are children. 
# 
# 
# 
# 

# In[ ]:


#Add a boolean feature to flag whether this passenger's age was originally missing. 
def age_missing(x):
    if (np.isnan(x['Age'])):
        return 1
    else:
        return 0

df['Age_Missing'] = df.apply(age_missing,axis=1)
df_test['Age_Missing'] = df_test.apply(age_missing,axis=1)

#Age imputation
def age_imputation(x, age_lookup):
    age = x['Age']
    pclass = x['Pclass']
    title = x['Title']

    if np.isnan(age):
        lookup_age = age_lookup['Age'][age_lookup['Title'] == title][age_lookup['Pclass'] == pclass].values
        
        return lookup_age[0].astype(int)
    else:
        return age
    
age_lookup = df_full[['Pclass','Title','Age']].groupby(['Pclass','Title'], as_index=False).median().sort_values(by='Age', ascending=False)
df['Age'] = df.apply(lambda x: age_imputation(x,age_lookup), axis=1)
df_test['Age'] = df_test.apply(lambda x: age_imputation(x,age_lookup), axis=1)


# **3. Intuitions and high level feature analysis**
# 
# With age data imputed in we can look at some of our assumptions about who is more likely to survive or perish.
# 
# * We expect those in higher classes are more likely to survive
# * Females are more likely to survive than males
# * Children are more likely to survive than adults
# 
# * Port of embarkation shouldn't have a significant impact on survival (unless different ports had different socio-economic levels in which case it could just be another way of phrasing class)
# * Title shouldn't make a significant difference to survival rates (especially with our simplified set of titles)
# * Fare paid should also have a minimal impact (as this would most likely just map back to class)
# 
# The last significant factors to examine are the Sibsp and Parch counts (i.e. group size), and our new feature indicating whether the age was missing from the passenger record.
# 
# These can all be combined on a pearson correlation plot to see how they impact survival and how we might simplify things further by combining features. 

# In[ ]:


import seaborn as sns
df_interesting = df.loc[:,['Age','SibSp','Parch','Sex','Pclass','Age_Missing','Survived']]
df_interesting['Sex'] = df_interesting['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



plt.figure(figsize=(10,10))
sns.heatmap(df_interesting.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True, linecolor='white', annot=True)
plt.show()


# As expected, Pclass and Sex have a strong correlation to survival. 
# 
# 
# Parch appears to have a stronger correlation than SibSp, though both are still somewhat weak compared to Sex and Pclass. It would be worth simplifying these into a single group size feature (more on that in the next section). 
# 
# Age *appears* to have a weaker than expected correlation to survival, (in fact it is more strongly correlated to other features than it is to survival.)
# 
# We can investigate this surprising Age correlation a little further by looking at the age distribution among those who survived and did not. 
# 
# 

# In[ ]:


plt.figure(figsize=(20,12))
plt.title('Age distribution among survived and perished')
sns.distplot(df.loc[df['Survived'] == 1]['Age'].values,label = 'Survived')
sns.distplot(df.loc[df['Survived'] == 0]['Age'].values,label = 'Perished')
plt.legend()
plt.show()


# From the above distribution plot we can assume that the survival rate is much higher for children, but otherwise age does not really factor in (ignoring the peak around 25-30 which I suspect would be the males in 2nd and 3rd class as they made up a significant proportion of the ship and had a much lower rate of survival.) 
# 
# What we really want is a feature which identifies children (say under the age of 14), while we're there we might was well make this feature identify men and women too. 
# 
# So what we can do now is simplifiy our model and introduce a new categorical feature representing 'Person Class' (i.e. male/female/child in 1st, 2nd, or 3rd class). 
# 
# *NB: this feature initially did not have Pclass included (i.e. Pclass stayed as its own feature) but this lead to a significantly lower public score of 0.77990*

# In[ ]:


#Apply person class to encompass gender, whether the passenger is a child, and class
def person_class(x):
    sex = x['Sex']
    age = x['Age']
    pclass = str(x['Pclass'])
    if (age <= 14):
        return 'child'+'_'+pclass
    else:
        return sex+'_'+pclass

df['Person_Class'] = df.apply(person_class,axis=1)

df_test['Person_Class'] = df_test.apply(person_class,axis=1)

#Rebuild pearson correlation plot
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_interesting = df.loc[:,['SibSp','Parch','Person_Class','Age_Missing','Survived']]
df_interesting['Person_Class'] = encoder.fit_transform(df_interesting['Person_Class'])
plt.figure(figsize=(10,10))
sns.heatmap(df_interesting.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, linecolor='white', annot=True)
plt.show()


# **4. Calculating group size, family or friends?**
# 
# As mentioned previously, the SibSp and Parch factors weren't amazingly helpful on their own, but could perhaps be combined into a 'group size' feature. 
# However, it is entirely possible that there are people on board travelling together who would not show up with anything in those columns (i.e. friends, extended family).
# 
# One way to modify the group size variable to include these cases is to count the number of people on each unique ticket number, then take the maximum of this and SibSp + Parch. 
# 
# 
# 

# In[ ]:


#Adding group size feature
def modify_group_size(x,ticket_numbers):
    ticket = x['Ticket']
    num_on_ticket = ticket_numbers['count'][ticket_numbers['Ticket'] == ticket].values[0]
    group_size = x['Group_Size']
    return max(num_on_ticket,group_size)

ticket_numbers = df_full.groupby(['Ticket']).size().reset_index(name='count')

df['Group_Size'] = df['Parch'] + df['SibSp'] + 1
df['Group_Size'] = df.apply(lambda x: modify_group_size(x,ticket_numbers), axis=1)

df_test['Group_Size'] = df_test['Parch'] + df['SibSp'] + 1
df_test['Group_Size'] = df_test.apply(lambda x: modify_group_size(x,ticket_numbers), axis=1)

#Look at the survival rate for each group size
df[['Group_Size','Survived']].groupby(['Group_Size']).mean().reset_index()


# We can probably make this group size feature a bit simpler by grouping into 3  categories: 
# * Solo
# * Small group (2-4)
# * Large group (5+)
# 

# In[ ]:


#Categorizing group size for simplicity
def categorize_group(x):
    size = x['Group_Size']
    if(size== 1):
        return 'solo'
    
    elif (size >= 2 and size <= 4):
        return 'small_group'
    else:
        return 'large_group'

   
df['Group_Category'] = df.apply(categorize_group,axis=1)

df_test['Group_Category'] = df_test.apply(categorize_group,axis=1)


#Rebuild pearson correlation plot
df_interesting = df.loc[:,['Group_Category','Person_Class','Age_Missing','Survived']]
df_interesting['Person_Class'] = encoder.fit_transform(df_interesting['Person_Class'])
df_interesting['Group_Category'] = encoder.fit_transform(df_interesting['Group_Category'])
plt.figure(figsize=(10,10))
sns.heatmap(df_interesting.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, linecolor='white', annot=True)
plt.show()


# **5. Building our features and fitting a model**
# 
# Before fitting our model and analysing the accuracy, we can take a look at survival rates within combinations of our new (entirely categorical) features. 
# 

# In[ ]:


df[['Person_Class','Group_Category','Survived']].groupby(['Person_Class','Group_Category']).mean()


# There's a couple of  interesting things to note here that may impact how accurate we can get our model:
# 
# * Females in 3rd class fare a lot worse than females in 1st or 2nd class
# * Males in 1st class fare much better than males in 2nd or 3rd class (with the exception of large groups but I would take this with a grain of salt given how few large groups there are)
# 
# 
# Several classifiers were trialled, with features tuned using grid search and k-fold cross validation, resulting in a random forest with 20 decision trees. (Scores were very slightly lower for a single decision tree classifier, and overall there was not a significant difference in average accuracy - I assume this is due to the relatively small number of entirely categorical features).
# 

# In[ ]:


#Building feature matrices from relevant dataframe columns
X = df[['Group_Category','Person_Class','Age_Missing']].values
X_test= df_test[['Group_Category','Person_Class','Age_Missing']].values

#Label encode textual features
from sklearn.preprocessing import OneHotEncoder


X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 1] = encoder.fit_transform(X[:, 1])

X_test[:, 0] = encoder.fit_transform(X_test[:, 0])
X_test[:, 1] = encoder.fit_transform(X_test[:, 1])

"""One hot encode categorical features (this does not make a significant difference to average 10-fold cross validation score, but has been left in 
as it improved public leaderboard score.
"""

onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder.fit_transform(X).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

#Set y variable for fitting
y = df['Survived'].values

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, 
                                    criterion = 'entropy', 
                                    max_features = 10,
                                    min_samples_split = 2,
                                    min_samples_leaf = 1,
                                    max_depth = 6,
                                    random_state = 2018)
classifier.fit(X, y)

# Applying 10-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
meanAccuracy = accuracies.mean()
stdAccuracy = accuracies.std()

print(str(accuracies) + '\nMean: ' + str(meanAccuracy) + '\nStDev: ' + str(stdAccuracy))


# **6. Potential improvements**
# 
# Overall this is not too bad an accuracy given the simplicity of the model, with a public score of 0.80382 (a touch outside the top 10% at time of writing). 
#    
# However, there is one limitation of the current model that becomes quite obvious when plotting the actual vs predicted survival rates on the training set.
#     
# 

# In[ ]:


#Append the prediction of survival back on to the training set
y_pred = classifier.predict(X)
df['Pred_survived'] = y_pred

incorrect_predictions = df.loc[~(df['Survived'] == df['Pred_survived'])]
correct_predictions = df.loc[(df['Survived'] == df['Pred_survived'])]

group_counts = df[['Person_Class','Group_Category']].groupby(['Person_Class','Group_Category']).size().reset_index(name = 'Count')
survived_by_group = df[['Person_Class','Group_Category','Survived','Pred_survived']].groupby(['Person_Class','Group_Category']).mean().reset_index()
group_counts['Survived'] = survived_by_group['Survived']
group_counts['Pred_survived'] = survived_by_group['Pred_survived']
group_counts['Diff'] = group_counts['Pred_survived']-group_counts['Survived']

group_counts



# Essentially the way the features and model have been set up means that within each unique combination, the model will either predict 0 or 100% survival rate, so our biggest rate of error as a percentage comes when one of our groups has a 50% rate of survival (essentially a coin toss when we only consider the features we have.) 
# 
# Any group with a true survival rate below 0.5 has a predicted survival rate of 0. Interestingly this completely binary result for the shown groups indicates that the "Age_Missing" feature is having no impact on our predictions.
# 
# If we assume the training set's true survival rate was the same in the test set, we can produce a ballpark estimate for our error rate when using all 418 test data records.
# 

# In[ ]:


group_counts_test = df_test[['Person_Class','Group_Category']].groupby(['Person_Class','Group_Category']).size().reset_index(name = 'Count')

group_counts_test

def count_potential_errors(x,train_group_counts):
    person_class = x['Person_Class']
    group_category = x['Group_Category']
    error_rate = train_group_counts['Diff'][train_group_counts['Person_Class'] == person_class][train_group_counts['Group_Category'] == group_category].values
    if (len(error_rate) == 0):
        return x['Count']
    else:
        return abs((x['Count']*error_rate[0]).astype(int))
    
group_counts_test['Est_Num_Errors'] = group_counts_test.apply(lambda x:count_potential_errors(x,group_counts),axis=1)

print('Estimated maximum accuracy (based on 418 entries in test set): ')
print(1 - group_counts_test['Est_Num_Errors'].sum()/418)

    


# This seems to be the logical conclusion for a simple one classifier model based on as few features as possible (in the end only really 2 features were required, but the creation of these features was the challenging part). 
# 
# That being said there could be some things to try next if I revisit this competition after some more experience:
# * Stacking of models that behave in *different* ways (I did play around with some stacking / voting ensembles but the models behaved too similary to have an impact)
# * Introducing new features to further subdivide groups until the rate of survival for each unique combination is closer to 0 or 1 (not likely given the number of features we have / random noise / the risk of overfitting) 
# * Splitting data and applying different models to different subsets
