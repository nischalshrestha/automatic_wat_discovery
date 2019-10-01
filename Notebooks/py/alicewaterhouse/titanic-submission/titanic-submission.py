#!/usr/bin/env python
# coding: utf-8

# This is my first Kaggle competition! I am just using it to experiment with some models and techniques. A rough plan is as follows:
# 
# * Overview, data cleaning and some feature engineering
# * Correlation matrix and missing values
# * Basic visualisation
# * More feature engineering and pre-modelling
# * Modelling and submission
# 
# **1. Overview, data cleaning and some feature engineering**
# 
# First I import some relevant libraries along with the data, and take a look at where there are missing values.

# In[ ]:


#Import relevant modules
import numpy as np # linear algebra
import pandas as pd # data processing, (e.g. pd.read_csv)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Read the data and take a look at the first few lines
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(10)


# In[ ]:


#Find the number of missing values in each column of the training data and print if it is non-zero
missing_vals_train = (train.isnull().sum())
print('Missing values in training data:\n')
print(missing_vals_train[missing_vals_train > 0])

#Do the same for test data
print('\nMissing values in test data:\n')
missing_vals_test = (test.isnull().sum())
print(missing_vals_test[missing_vals_test > 0])


# A lot of cabin data is missing, but it seems a shame to drop the data that is there because it may contain useful information. Also, it might have a reasonable effect on survival if it determines where passengers were on the ship when the iceberg hit. I will drop the numeric part and add an extra category 'N' for those passengers without cabin data, and remove the numeric part for those with cabin data.
# 
# Age, fare and embarkation point can be imputed. I will decide how to do this a little later on. I drop passenger ID and ticket number as I don't expect these to affect survival.

# In[ ]:


#Keep PassengerId for submission later
passengerid = test.PassengerId

#Drop irrelevant columns from both test and training data
Columns_to_drop = ['Ticket','PassengerId']
train.drop(Columns_to_drop, axis=1, inplace=True)
test.drop(Columns_to_drop, axis=1, inplace=True)

#Add 'N' for passengers without cabin data
train.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)

#Take the first character of the string to remove the numeric part of the cabin data
train.Cabin = [i[0] for i in train.Cabin]
test.Cabin = [i[0] for i in test.Cabin]

train.head()


# To investigate whether useful information can be obtained from the name column I will take the title and length. This will tell me, for example, whether the women on the ship are married, which might be relevant. Then I can drop the Name feature.

# In[ ]:


#Add a column for name length
train['NameLength'] = [len(i) for i in train.Name]
test['NameLength'] = [len(i) for i in test.Name]

#Add a column for title
train['Title'] = [i.split('.')[0] for i in train.Name]
train['Title'] = [i.split(', ')[1] for i in train.Title]
test['Title'] = [i.split('.')[0] for i in test.Name]
test['Title'] = [i.split(', ')[1] for i in test.Title]

#Drop Name
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

train.head()


# I'm now going to look at the titles that appear to (a) check that this worked as I expected and (b) see which are the main ones and whether any are obscure ones that only appear once and hence aren't very helpful.

# In[ ]:


print('Title value counts in training data: \n')
print(train.Title.value_counts())
print('\nTitle counts in test data: \n')
print(test.Title.value_counts())


# I will assume Mlle and Ms mean the same as Miss and Mme means Mrs so these are easily replaced. I will also group military titles and titles indicating nobility. Maybe this is still too many categories. I don't know much about the pros and cons of having a lot of categories to one-hot encode. Something to look into another time. I do know that the fact that number of passengers is still much greater than number of features, which is encouraging.

# In[ ]:


#Simplify titles in training data
train["Title"] = [i.replace('Ms', 'Miss') for i in train.Title]
train["Title"] = [i.replace('Mlle', 'Miss') for i in train.Title]
train["Title"] = [i.replace('Mme', 'Mrs') for i in train.Title]
train["Title"] = [i.replace('Col', 'Military') for i in train.Title]
train["Title"] = [i.replace('Major', 'Military') for i in train.Title]
train["Title"] = [i.replace('Don', 'Military') for i in train.Title]
train["Title"] = [i.replace('Jonkheer', 'Nobility') for i in train.Title]
train["Title"] = [i.replace('Sir', 'Nobility') for i in train.Title]
train["Title"] = [i.replace('Lady', 'Nobility') for i in train.Title]
train["Title"] = [i.replace('Capt', 'Military') for i in train.Title]
train["Title"] = [i.replace('the Countess', 'Nobility') for i in train.Title]

#Simplify titles in test data
test["Title"] = [i.replace('Ms', 'Miss') for i in test.Title]
test["Title"] = [i.replace('Col', 'Military') for i in test.Title]
test["Title"] = [i.replace('Dona', 'Nobility') for i in test.Title]

train.head()


# Finally, I reformat Sex so that 1 is male and 0 is female:

# In[ ]:


train["Sex"] = train["Sex"].replace({"female":0, "male":1})
test["Sex"] = test["Sex"].replace({"female":0, "male":1})
train.head()


# Summary so far: we dropped irrelevant columns and extracted name length and title from the Name column. We also engineered the cabin data, dropping the numerical part and adding 'N' where no cabin data was available.
# 
# Still to do: decide on the best way to impute age, fare and embarkation point. But first, I want to gain some basic statistical understanding of the data.
# 
# **2. Correlation matrix and missing values**
# 
# The first thing I want to do is look at the correlations between different features. Note: this will ignore categorical features that have not yet been one-hot-encoded.

# In[ ]:


#Change options to make pandas display all the columns
pd.options.display.max_columns = 99

#Print the correlation matrix
print(train.corr())


# Survival correlations that stand out:
# * Survival is positively correlated with name length (!)
# * Survival is negatively correlated with class
# * Being male is negatively correlated with survival
# * I am surprised how small the correlation is between age and survival. Perhaps this is because both children and the elderly had preferential access to lifeboats.
# 
# Other correlations that stand out:
# * Class and age
# * Class and fare (makes sense)
# * Sex and name length (interesting??)
# * Age and number of siblings/spouses you are travelling with
# * Number of parents/children and number of siblings/spouses you are travelling with
# 
# Now that I have this information, my first priority is imputation of age, fare and embarkation point. I waited until now because I want to use some knowledge of the correlations to inform my imputation.
# 
# We can impute using both the training and test data, since this gives us more information. In the real world this would be cheating because we would be unlikely to have access to test data at the time of training. Also, imputing based on all the data causes a so-called leaky pipeline. Technically when cross-validation later I should do the imputation for each fold. This will result in my accuracy on the test set being lower than expected from cross-validation scores in the training set. But that's okay since I still expect an increase in cross-validation score to reflect a better model fit.
# 
# I create a new dataframe combine train and test data.
# 

# In[ ]:


#Make a copy of the training data
train1 = train.copy(deep=True)
#Add a new column stating that 
train1['Dataset'] = 'Training Data'

#Do the same for the test data. Add 'Unknown' for Survived
test1 = test.copy(deep=True)
test1['Dataset'] = 'Test Data'
test1['Survived'] = 'Unknown'

#Combine the training and test data
combined_data = pd.concat([train1,test1],ignore_index=True)
combined_data.tail()


# Now, embarkation. We might expect this to be related to fare. Firstly, who are the two passengers whose embarkation point we don't know about? And what is the modal embarkation point?

# In[ ]:


#Examine the modal value for embarked in both training and test data
print('Embarked value counts in training data: \n')
print(train.Embarked.value_counts())
print('\nEmbarked value counts in test data: \n')
print(test.Embarked.value_counts())

#Extract passengers whose Embarkation data is missing
train[train.Embarked.isnull()]


# Most people board at Southampton, and both passengers for whom the data is missing paid 80.0. Let's look at how Fare is distributed with Embarked.

# In[ ]:


#Import plotting libraries
import seaborn as sns #Statistical visualisation
import matplotlib.pyplot as plt #Plotting

#Make a boxplot of Fare against embarked for each dataset
plt.figure(figsize=(15,9))
sns.set(style="whitegrid")
ax = sns.boxplot(y="Embarked", x="Fare", hue="Dataset", data=combined_data);
plt.title('Fare Boxplots for each Embarkation Point', fontsize=18)


# The first thing we notice is there are some massive outliers in fare. Possibly these should be removed. I will think more about that later. The second thing we notice is that passengers embarking at C in general paid higher fares, meaning it makes the most sense to set Embarked to C for our two passengers whose fares were 80.0.

# In[ ]:


#Fill in missing Embarked values with 'C'
train.Embarked.fillna("C", inplace=True)


# Now for Fare. Whose fare value is missing?

# In[ ]:


test[test.Fare.isnull()]


# We already saw that Fare is correlated with Pclass, so it would be silly not to use the fact that this passenger's Pclass is 3. I take the mean fare of those passengers who embarked at Southampton and have Pclass 3.

# In[ ]:


#Impute Fare with the mean for the subset described above
missing_fare = combined_data[(combined_data.Pclass == 3) & (combined_data.Embarked == "S")].Fare.mean()
#Replace the test.Fare null values with missing_fare
test.Fare.fillna(missing_fare, inplace=True)
test.iloc[152]


# Cool so his fare was filled in as 14.4354.
# 
# Last to be imputed is age. I wait until after my visualisation to do this, so that it doesn't mess up the plots involving age.

# **3. Basic visualisation**
# 
# Time to look in more detail at the qualitative effects of the features on survival.
# 
# **Survival vs Sex**
# 
# I examine this using a simple bar chart of percentage survival for each Sex.
# 

# In[ ]:


plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", y = "Survived", data=train)
plt.title("Fraction of Passengers that Survived by Sex", fontsize = 18)
plt.ylabel("Fraction of Passengers that Survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);


# Really I should change the labels for Sex back to male and female but recall that 0 is female and 1 is male. Women (as one might expect) were more likely to survive.
# 
# **Survival vs Pclass**
# 
# This can be investigated using a similar chart.

# In[ ]:


plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Pclass", y = "Survived", data=train)
plt.title("Fraction of Passengers that Survived by Pclass", fontsize = 18)
plt.ylabel("Fraction of Passengers that Survived", fontsize = 15)
plt.xlabel("Pclass",fontsize = 15);


# First class passengers were more likely to survive than second, and second class passengers were more likely to survive than third.
# 
# **Survival vs Age**
# 
# I am using a KDE plot to examine differences in the age distribution between passengers who survived and passengers who didn't.

# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.kdeplot(train[(train['Survived'] == 1)].Age , shade=True, label='Survived')
ax = sns.kdeplot(train[(train['Survived'] == 0)].Age , shade=True, label='Did not survive')
plt.title('Estimated probability density of age given survival', fontsize = 18)
plt.xlabel('Age', fontsize = 15)
plt.ylabel('Probability density', fontsize = 15);


# This is clearly a bit ridiculous because it has a non-zero probability of survival for negative ages. This is because of the smoothing the KDE plot does. But anyway, the small peak below 10 in the survived data which is not present in the data of those who did not survive suggests that children were prioritised over adults for lifeboat space, as we might expect. When feature engineering, I might categorise age according to this distinction.
# 
# **Survival and Fare**

# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.kdeplot(train[(train['Survived'] == 1)].Fare, label='Survived', shade=True)
ax = sns.kdeplot(train[(train['Survived'] == 0)].Fare, label='Did not survive', shade=True)
plt.title('Estimated probability density of fare given survival', fontsize = 18)
plt.xlabel('Fare', fontsize = 15)
plt.ylabel('Probability density', fontsize = 15);


# The sharp peak at low fares for those who didn't survive, because it is not present in the data for those who did survive, suggests those with lower fares were more likely to die. Similarly, the larger tail on the RHS for survivors suggests paying a higher fare made you more likely to survive.
# 
# Now that this is done, I impute age.

# In[ ]:


#Use the mean age from the combined data to fill in unknown ages
train.Age.fillna(combined_data.Age.mean(), inplace=True)
test.Age.fillna(combined_data.Age.mean(), inplace=True)


# **4. More feature engineering and pre-modelling**
# 
# I want to create an extra feature for whether or not a passenger is a child. Since the peak in the KDE plot above is below age 10,  my definition of child will be the same.

# In[ ]:


train['Child'] = [1 if i<10 else 0 for i in train.Age]
test['Child'] = [1 if i<10 else 0 for i in test.Age]
train.head(10)


# The last thing I need to do before modelling is add dummy variables for the categorical data and split into dependent and independent variables.

# In[ ]:


#Get dummies for one-hot-encoding of categorical features
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

#Drop the T Cabin...
train.drop(['Cabin_T'], axis=1, inplace=True)

#Split training data into dependent and independent variables
X = train.drop(['Survived'], axis=1)
y = train["Survived"]


# **5. Modelling and submission**

# Now to implement cross-validation in order to better compare different models.

# In[ ]:


from sklearn.model_selection import cross_val_score


# Okay I want to test a few models against eachother, then tune the best one. I am going to use logistic regression, support vector classifier random forest classifier, and XGBoost. For each one I calculate its cross-validation score.

# In[ ]:


#Import the models and then construct a pipeline for each including imputation
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=1)
from sklearn.svm import SVC
SVC = SVC(random_state=1)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=1)
from xgboost import XGBClassifier
XGB = XGBClassifier(random_state=1)

#Function that takes a model and returns its cross-validation score
def cv_score(model):
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=10)
    return scores.mean()

#Calculate cross-validation scores using this function
CV_scores = pd.DataFrame({'Cross-validation score':[cv_score(LR),cv_score(SVC),cv_score(RF),cv_score(XGB)]})
CV_scores.index = ['LR','SVC','RF','XGB']
print(CV_scores)


# Let's tune the random forest classifier, since it did well and I understand it better than XGB. I use a grid search.

# In[ ]:


#Import GridSearchCV
from sklearn.model_selection import GridSearchCV

#Define parameter grid on the number of decision trees used and the maximum depth of the trees
parameters = {'n_estimators':[100,120,150], 'max_depth':[5,10,15,20,25,30]}
#Peform gridsearch
RF_grid = GridSearchCV(RF, param_grid=parameters)
RF_grid.fit(X,y)

#Print the best parameters and what they scored
print('Best parameters:'+str(RF_grid.best_params_))
print('Best score:'+str(RF_grid.best_score_))


# Now I use the optimised random forest to predict based on the test data.

# In[ ]:


#Make prediction and create a dataframe for submission
predictions = RF_grid.predict(test)
submission = pd.DataFrame({'PassengerId': passengerid, 'Survived': predictions})
submission.head(10)


# Now to write these predictions to csv.

# In[ ]:


submission.to_csv('submission.csv', index=False)

