#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# As many others before me, this is my first "dive" into a kaggle project. With its popularity and relative ease of manipulation, I felt the "Titanic: ML from Disaster" was a great place to start. Join me on this voyage through cleaning, visualizing, and modeling the Titanic survivals and please feel free to leave comments below. All criticisms are well received.
# 
# Bon Voyage!

# In[ ]:


#Imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#Read in the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
full = pd.concat([train,test])


# In[ ]:


#Check head
full.head()


# In[ ]:


#Check info
full.info()


# It's worth taking note of the number of missing values:
# - Age: 263
# - Cabin: 1014!!!
# - Embarked: 2
# - Fare: 1
# - Survived: 418 (But this corresponds to our test set, so no worries here)

# # Data Cleaning
# 
# Now that we have a sense of our data, I'm going to go through column by column, thinking about each variable and organizing it into a form we can use for analysis.

# ### 1.Age: Age of Passenger
#     - We'll have to deal with some missing values, but I'm going to back to this last.

# ### 2.Cabin: Cabin the passenger stayed in.
#     - Nearly 80% missing, so I'm just going to drop it.

# In[ ]:


#Drop 'Cabin'
full.drop('Cabin',axis=1,inplace=True)


# ### 3.Embarked: Indicates port that the passenger embarked from.
#     - Fill in the na values with whichever port was most prevalent.
#     - Convert into a dummy variable.

# In[ ]:


#Check which port most of the passengers came from
full['Embarked'].value_counts()


# In[ ]:


#Fill na values with 'S'
full['Embarked'].fillna('S',inplace = True)

#Convert 'Embarked' into dummy variables and drop 'Embarked'
full = pd.concat([full,pd.get_dummies(full['Embarked'],drop_first=True,prefix='Port')],axis=1)
full.drop('Embarked',axis=1,inplace=True)


# ### 4.Fare: Price of the ticket the passenger paid for.
#     - Fill missing Fare value in with average Fare.

# In[ ]:


#Fill na values with average Fare
full.loc[full['Fare'].isnull(),'Fare'] = full['Fare'].mean()


# ### 5.Name: Passenger name.
#     - Title may signify importance.
#     - Reduce Name to title.

# In[ ]:


#Split names into a list
full_title = full['Name'].apply(lambda x: x.split()[1])

#Function: Takes in a title and returns the title if it's among those specified; Else returns 'No Title'
def impute_title(title):
    if title not in ['Mr.','Miss.','Mrs.','Master.']:
        return 'No title'
    else:
        return title

#Assign titles, convert into dummy variables, and drop 'Name'
full_title = full_title.apply(impute_title)
full = pd.concat([full,pd.get_dummies(full_title,drop_first=True)],axis=1)
full.drop('Name',axis=1,inplace=True)


# ### 6.Parch: # of parents or children a passenger had. 

# ### 7.PassengerId: Passenger Index. 
#     - Not a predictor.
#     - Store Test IDs before dropping.

# In[ ]:


#Store test ids for later use and drop 'PassengerId'
test_id = test['PassengerId']
full.drop('PassengerId',axis=1,inplace=True)


# ### 8.Pclass: Numeric representation of class. 
#     - Effectively a categorical variable, so let's represent it as such.
# 

# In[ ]:


#Convert 'Pclass' into dummy variables and drop 'Pclass'
full = pd.concat([full,pd.get_dummies(full['Pclass'],drop_first=True,prefix='Pclass')],axis=1)
full.drop('Pclass',axis=1,inplace=True)


# ### 9.Sex: Passenger sex. 
#     - Assign a binary variable.

# In[ ]:


#Convert 'Sex' into dummy variable and drop 'Sex'
full = pd.concat([full,pd.get_dummies(full['Sex'],drop_first=True)],axis=1)
full.drop('Sex',axis=1,inplace=True)


# ### 10.SibSp: # of siblings or spouses a passenger had.

# ### 11.Survived: Whether or not the passenger survived. 
#     - This is our target.

# ### 12.Ticket: Ticket number of passenger.
#     - Has potential for feature engineering, but I'm going to ignore for this analysis.

# In[ ]:


#Drop 'Ticket'
full.drop('Ticket',axis=1,inplace=True)


# In[ ]:


#Take a look at the data frame now
full.head()


# # Returning to Age.
# 
# We're going to fill in Age based on a linear regression performed on the rest of the variables. It would likely be fine for the analysis to fill in with simply average age or even average age of Pclass or Sex, but we have the tools to make a more detailed prediction, so might as well!

# In[ ]:


#Import Linear Regression Model
from sklearn.linear_model import LinearRegression

#Fit model on data that does have an Age entry
impute_age = LinearRegression()
impute_age.fit(full[full['Age'].isnull()==False].drop(['Survived','Age'],axis=1),
               full[full['Age'].isnull()==False].drop('Survived',axis=1)['Age'])

#Impute ages for those that were missing
ages = impute_age.predict(full[full['Age'].isnull()].drop(['Survived','Age'],axis=1))


# In[ ]:


#Compare Age Distributions with and without imputed ages
plt.figure(figsize=(13.5,6))
plt.subplot(1,2,1)
plt.hist(full[full['Age'].isnull()==False].drop('Survived',axis=1)['Age'],
         bins=range(0,80,5),edgecolor='white')
plt.title('Without Age Imputations')
plt.xlabel('Age')

plt.subplot(1,2,2)
plt.hist(list(full[full['Age'].isnull()==False].drop('Survived',axis=1)['Age']) + list(ages),
         bins=range(0,80,5),edgecolor='white',alpha=.5)
plt.title('With Age Imputations')
plt.xlabel('Age')


# Visually, the distribution appears unchanged. We've added roughly 100 passeners in their late 20s, which appears to be the most significant change. This looks to be about where the average age of the distribution without age imputations lies, though, so I'm actually pleased with this result. I feel confident that these results won't skew the outcome, so we can proceed with filling in the missing age values with the imputed ones.

# In[ ]:


#Fill dataframe in with imputed ages
full.loc[full['Age'].isnull(),'Age'] = ages


# # Visualizations
# With our data cleaned up and in a workable format, it's time to take a look at it!

# ### Any strong correlations?

# In[ ]:


#Produce heatmap of correlations
plt.figure(figsize=(16,8))
sns.heatmap(full.corr(),annot=True,cmap='viridis')
plt.tight_layout


# Wow! Who would've thought 'Mr.' and 'male' would be so strongly correlated?? Jokes aside, there is some useful insight to be had here. First and foremost, I'm curious what's most strongly correlated with 'Survived'. It appears the largest contributors are 'male' (or 'Mr.'...) and 'Pclass_3'. 'Miss.' and 'Mrs.' are prevalent as well, but it's fair to assume that the sex is really what's contributing to this relationship as opposed to the title. Another couple standouts are the relationships between 'Pclass_3' and 'Age' and 'Fare'. It's easy to see how the 3rd class is going to be cheaper, but it's also interesting that they tend to be younger as well. For passengers buying their own tickets, this makes sense, but let's think about those who aren't, the children. If class 3 tended to be younger, it means that they either had more children or the higher classes had fewer. The question worth asking then is whether or not family size impacted survival rates.

# ### Family Size Exploration

# In[ ]:


#Illustrate relationship between family size and survival rate
plt.figure(figsize=(13.5,6))
sns.countplot((full['SibSp'] + full['Parch'] + 1),hue=full['Survived'],palette='viridis')
plt.xlabel('Family Size')


# If you were alone, you had a much greater chance of dying. If you were in a large family (greater than 4 members), you had a greater chance of dying. But if you were in a small family (2-4 members), you actually had a better chance of surviving. Let's go ahead then and create features for this in our dataframe.

# In[ ]:


#Create new column for family size
full['Family'] = (full['SibSp'] + full['Parch'] + 1)

#Function: Takes in family size and returns corresponding description
def impute_alone(x):
    if x == 1:
        return 'Alone'
    elif x > 4:
        return 'Large Family'
    else:
        return 'Small Family'

#Label each passenger's family size
full['Family'] = full['Family'].apply(impute_alone)


# In[ ]:


#Again, illustrate relationship between family size and survival rate
plt.figure(figsize=(13.5,6))
sns.countplot(full['Family'],hue=full['Survived'],palette='viridis')


# In[ ]:


#Convert into dummy variable and drop 'SibSp', 'Parch', and 'Family'
full = pd.concat([full,pd.get_dummies(full['Family'],drop_first=True)],axis=1)
full.drop(['SibSp','Parch','Family'],inplace=True,axis=1)


# ### One last look at the data

# In[ ]:


#Take a look
full.head(10)


# # Model Fitting

# In[ ]:


#Standardize Age and Fare
full['Age'] = (full['Age'] - full['Age'].mean()) / full['Age'].std()
full['Fare'] = (full['Fare'] - full['Fare'].mean()) / full['Fare'].std()


# In[ ]:


#Split the data back into train and test sets
train = full.iloc[0:len(train)]
test = full.iloc[len(train):len(full)]


# In[ ]:


#Scikit imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


#Split the data 'train' data into train and test sets
X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)


# ### Logistic Regression

# In[ ]:


#Fit and predict with Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred_lr = lr.predict(X_test)
print(confusion_matrix(y_test,pred_lr))
print('\n')
print(classification_report(y_test,pred_lr))


# ### K Nearest Neighbors w/ optimization

# In[ ]:


#Create empty array to hold errors
error_rate = []

#Iterate through different K values, fit, predict, and store error rates
for i in range(1,100,2):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

#Plot error rates with respect to K value
plt.figure(figsize=(10,6))
plt.plot(range(1,100,2),error_rate,color='blue',ls='dashed',marker='o',
        markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error')


# In[ ]:


#Min at k = 3
#Re-run model with new k value
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test,pred_knn))
print('\n')
print(classification_report(y_test,pred_knn))


# ### Support Vector Classification w/ optimization

# In[ ]:


#Iterate through various parameter values of SVC and apply optimal model
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,.01,.001,.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=1)
grid.fit(X_train,y_train)
pred_grid = grid.predict(X_test)
print(grid.best_params_)
print('\n')
print(confusion_matrix(y_test,pred_grid))
print('\n')
print(classification_report(y_test,pred_grid))


# ### Decision Tree

# In[ ]:


#Fit and predict with Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred_dtree = dtree.predict(X_test)
print(confusion_matrix(y_test,pred_dtree))
print('\n')
print(classification_report(y_test,pred_dtree))


# ### Random Forest

# In[ ]:


#Fit and predict with Random Forest
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test,pred_rfc))
print('\n')
print(classification_report(y_test,pred_rfc))


# # Conclusions & Extensions
# It was a very narrow margin, but the Support Vector model with parameter optimization produced the highest accuracy. Often times it can be imperative to consider the precision and recall of the predictions, but for this project our primary concern was simply accuracy. Therefore, I'm going to retrain the model on the entirety of data at our disposal, then use it to make predictions on our test set! Before I do so, though, I want to take the time to list some potential extensions of the analysis:
#     - Feature engineer 'Cabin' and 'Ticket': Perhaps information on passenger's location on the boat can be found.
#     - Dive deeper into the family dynamics, looking at whether Mothers and Daughters perhaps had higher survival rates.
#     - There are always more models to try!
#         - Does giving greater weight to the passengers that were incorrectly classified improve the predictions?

# In[ ]:


#Fit optimal SVC model on the entirety of the training set and predict on test set
grid.fit(train.drop('Survived',axis=1),train['Survived'])
pred_final = grid.predict(test.drop('Survived',axis=1))


# In[ ]:


#Create Submission
submission = pd.DataFrame(
    {'PassengerId' : test_id,
     'Survived' : pred_final}
)


# In[ ]:


#Make sure everything looks good
submission.head()


# In[ ]:


#Store it
submission.to_csv('Submission',index=False)


# # Thank you!
# Land, ho! (hopefully...). Thanks for making it with me this far. Again, please feel free to leave comments. I'm continually trying to improve my coding skill, analysis, and presentation, so all advice or criticism is welcomed and well received. Cheers!

# In[ ]:




