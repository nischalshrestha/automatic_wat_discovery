#!/usr/bin/env python
# coding: utf-8

# ## Machine learning from disaster by Olabode Alamu

# The sinking of the Titanic is one of the most tragic maritime accidents of all time, it sank on the 15th of April 1912 in the North Atlantic Ocean leaving 1500 people out of it 2224 passengers dead.
# 
# The Titanic was on its maiden voyage from Southampton to Newyork with stops at Cherbourg and Queenstown to pick up more passengers.
# 
# This project is concerned with exploring the data available for the passengers and applying machine learning algorithms to predict if a particular passenger would survive or not.
# 
# The data was gotten from Kaggle [here](https://www.kaggle.com/c/titanic/data) .
# 
# 
# 

# In[ ]:





# In[ ]:


# Import the libraries
import numpy as np
import pandas as pd


# In[ ]:


# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set(style="darkgrid")


# In[ ]:


# import Machine learning 
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# Import the data into a dataframe

titanic = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Explore the dataset
titanic.head(10)


# In[ ]:


test.head()


# In[ ]:


# Basic statistics
titanic.describe()


# In[ ]:


test.describe()


# In[ ]:


titanic.info()


# In[ ]:


test.info()


# In[ ]:


# Check for missing values
titanic.isnull().sum()


# In[ ]:


test.isnull().sum()


# As can be seen above, the Age column has lots of missing values, a suitable inputation technique would need to be implemented to fill these missing values.

# ## Function for calculating percentages in different categories in this dataset

# Several times in this project, I had to determine the percentage of each category in this dataset. In order to do this in an expedite fashion, the function below was written and would be used several times in this notebook.

# In[ ]:



def count(dataframe, column_name):
    """
    Written by Olabode Alamu 15th Jan 2017
    Function calculates the percentage of the different categorical variables.
    The function takes in two parameters- dataframe and column_name and returns 
    the percentage of each category in any particular column.
    
    dataframe = Name of the dataframe under consideration
    column_name = string of column name of the column with categorical variables
    """
    try:
        total = len(dataframe[column_name]) # Counts the number of rows
        for i in dataframe[column_name].unique():
            # Counts the number of rows in each class
            Count = len(dataframe[dataframe[column_name]==i])
            print('Percentage in column '+column_name +' with value '+ str(i)+ ' is' , (Count/total)*100, '%')
    
    except:
        print('Column name not found in dataset or Dataframe doesnt exist.')
            


# In[ ]:





# ## Data Visualization

# Basic data visualization would be carried out in order to investigate the effect of the different parameters on the survival of the passengers.

# ### Effect of gender on survival chances

# From the portrayal of the events of the 15th of April in movies and documentaries, it is generally believed that women and children were allowed to get on the lifeboats first before men. Lets explore the data to see what it says.

# In[ ]:


sns.countplot(x = 'Sex', data = titanic, palette='Set1')


# What percentage of the passengers are women? 

# In[ ]:


count(dataframe=titanic,column_name='Sex')


# In[ ]:


# Visualize the proportion of males to females survived
sns.countplot(x='Sex' , hue= 'Survived', data = titanic,  palette='Set1')


# In[ ]:


# What is the exact number of men that died?
M = titanic[titanic['Sex']=='male']
len(M[M['Survived']== 0])


# In[ ]:


len(titanic[titanic['Sex']=='male'][titanic[titanic['Sex']=='male']['Survived']== 0])


# In[ ]:


# What is the exact number of women that died?
F = titanic[titanic['Sex']=='female']
len(F[F['Survived']== 0])


# In[ ]:





# The red column shows those who died while the blue column shows those who survived. It can be seen that over 450 men died as compared to 100 men who survived, while on the other hand, less than a 100 women died and above 200 survived.
# 
# Could the larger number of males onboard be a reason for their larger number of casualities? The ratio of men to women is nearly 2:1, however, when we compare the number of casualities of men (468) to women (81), we see a ratio of nearly 6:1.
# 
# This is in agreement with the movies... Sex is therefore a strong predictor of survival.

# ### Effect of passenger class on survival

# The titanic had 3 passenger classes- First class, Second class and Third class. More information about this can be found [here](https://en.wikipedia.org/wiki/RMS_Titanic#Passenger_facilities).
# 
# The class is represented by the column Pclass in the dataframe. 
# 
# Lets investiage the role Passenger class had on the survival chances of the passengers.

# In[ ]:


# Visualize the distribution amongst the different classes
sns.countplot(x = 'Pclass', data= titanic , palette='Set1')


# The vast majority of passengers were in the third class on the ship, next to the first class passengers and last to those in second class.

# In[ ]:


# Shows the percentage of passengers in each class
count(titanic,'Pclass')


# What effect did the class a passenger belong to have on their survival?

# In[ ]:


sns.countplot(x = 'Pclass', data = titanic, hue= 'Survived', palette='Set1')


# The red bars represents the count of passengers that died in each class. The 1st class passengers had the lowest death toll, next to passengers in the 2nd class, while those in the 3rd class suffered the most casualities. 
# 
# 
# This shows that being among the first class passengers gave you a better chance of surviving the incident.
# 
# Would you stand a better chance of surviving if you were female and in first class?

# In[ ]:


sns.factorplot(x = 'Pclass', col = 'Survived',hue = 'Sex'
               , data = titanic, kind = 'count',palette='Set1')


# In[ ]:


# How many women in First class died?
F = titanic[titanic['Sex']=='female'] # Dataframe showing females onboard
Female_died = F[F['Survived']== 0] # Dataframe showing females that died
# Dataframe showing females in first class that died
Female_died[Female_died['Pclass']==1] 


# Looks like there is an interaction between sex and the passenger class. From the above diagram, it can be seen that being female (blue bars) and also being in the first class gave you a much better chance of surviving.
# 
# Infact, there were only 3 women that were in first class that died out of the 81 women that died in this dataset.
# 
# Being a man in third class put you at a very high likehood of not surviving the incident, but your odds of survival increased as you went up a class, with men in first class having the best chances of survival amongst the men.
# 
# 

# In[ ]:





# ### Effect of Number of siblings on survival

# In a moment of chaos like that on the titanic, could having siblings onboard spell doom for you? or would it help improve your chances of survival?
# 
# The number of siblings / spouse is represented by the column SibSp in the dataset.
# 
# What effect does the number of siblings present have on the probability of survival of the passenger?

# In[ ]:


# Visualize the distribution
sns.countplot(x = 'SibSp', data = titanic,palette='Set1')


# A good number of the passengers had no siblings onboard with them.

# In[ ]:


count(titanic,'SibSp')


# The vast majority of passengers (91.7 %) were either by themselves or had a spouse/sibling onboard.

# In[ ]:


sns.countplot(x = 'SibSp', data = titanic, hue= 'Survived', palette='Set1')


# Being single was actually risky! A very large number of single men and women died, while those with a spouse or sibling had a better chance of surviving.
# 
# Having 3 or more siblings was actually bad news, maybe because it would have been difficult to find where everyone was amidst the chaos. 
# 
# Infact, having siblings greater than 4 in number meant you weren't going to survive.

# In[ ]:


# What group had 8 siblings?
titanic[titanic['SibSp']==8]


# You can read about the Sage family [here](https://www.encyclopedia-titanica.org/titanic-victim/thomas-henry-sage.html)

# How many passengers have parents onboard?

# In[ ]:


sns.countplot(x = 'Parch', data = titanic, palette='Set1')


# In[ ]:


sns.countplot(x = 'Parch', data = titanic, hue= 'Survived', palette='Set1')


# ### What effect does family size have on the probability of survival of a passenger.

# In[ ]:


# Create a column that shows the size of the family
titanic['Family Size'] = titanic['SibSp']+titanic['Parch']+ 1
# Create a column that shows the size of the family
test['Family Size'] = test['SibSp']+test['Parch']+ 1


# In[ ]:


titanic.head(3)


# In[ ]:


sns.countplot(x = 'Family Size', data = titanic, hue= 'Survived', palette='Set1')


# This shows a very interesting relationship, being alone on the titanic leaves you at a greater chance of not surviving, whereas, as the family size increases to a size of 4, the chance of survival increases. Beyond a size of 4, there was a greater chance of not surviving the incident.

# ### Port of Embarkment

# The image shows the route of the Titanic. https://commons.wikimedia.org/wiki/File:Titanic_voyage_map.png#/media/File:Titanic_voyage_map.png

# The Titanic began its journey from Southampton to Cherbourg and then Queenstown.

# In[ ]:


titanic['Embarked'].fillna(value = 'S', inplace = True)


# In[ ]:


# What was the percentage of passengers that embarked at the different ports?
count(dataframe=titanic, column_name='Embarked')


# The majority of passengers got on the Ship from Southampton while the least got on from Queenstown. 
# Lets see what information this gives us.

# In[ ]:


sns.countplot(x = 'Embarked', data = titanic, hue= 'Survived', palette='Set1')


# The relative number of casualities follows the trend of relative number of people that embark on the journey at each port which makes sense.
# 
# However, at Cherbourg, a larger number of passengers survived amongst those that embarked at that city.
# 
# What could be responsible? Could class have a hand in this?

# In[ ]:


sns.factorplot(col = 'Embarked', x = 'Pclass', data = titanic
               , kind = 'count', palette='Set1')


# A greater proportion of passengers that got on the titanic from Southampton and Queenstown were 3rd class passngers. As for Cherbourg, the 1st class passengers were the largest proportion, this might explain the improved chance of survival experienced by passengers that embarked in this port.

# ### Effect of fare paid

# Could your fare price have had an effect on your chance of survival? In order to investigate this, we need to create a new column called Fare per person.

# In[ ]:


titanic['Fare per person']= titanic['Fare']/ titanic['Family Size']


# In[ ]:


test.isnull().sum()


# In[ ]:


# fill the missing Fare value in the test dataset with the median fare value
test['Fare'].fillna(value = test['Fare'].median(), inplace = True)


# In[ ]:


test['Fare per person']= test['Fare']/ test['Family Size']


# In[ ]:


sns.distplot(a= titanic['Fare per person'], bins=50, )


# In[ ]:


# What was the average fare paid per class?
sns.barplot(data = titanic, x = 'Pclass', y = 'Fare per person', palette='Set1')


# Third class passengers paid on average a fare less than #10 , while first class passengers paid a fare above #50.

# In[ ]:



sns.factorplot(data = titanic, x = 'Pclass', col = 'Sex'
            ,y='Fare per person',hue='Survived' , palette='Set1', kind = 'bar')


# On average, the survivors from the first class paid more in fare.

# ### Effect of Age

# The age column in the training and test dataset has missing values which need to be filled.

# In[ ]:


titanic.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


# Check the distribution of the age before filling the holes
sns.distplot(a = titanic['Age'].dropna(), bins=40)


# In[ ]:


sns.factorplot(y= 'Age', data = titanic, x = 'Pclass', kind = 'box', col = 'Survived', palette='Set1')


# The age column has lots of missing values, we need to input those values.

# In[ ]:


def age_input(dataframe):
    """
    This function fills the missing age values with random numbers generated between two values.
    The first value is the mean - standard deviation while the second value is the mean plus the standard deviation.
    
    Accepts the name of the dataframe as input and returns a dataframe with no missing values in the Age column
    """
    Number_missing_age = dataframe['Age'].isnull().sum() # Counts the number of nan values
    Mean_age = dataframe['Age'].mean() # calculates the mean values
    Std_age = dataframe['Age'].std() # calculates the standard deviation values
    
    # Generates random numbers the size of the missing values
    random_age = np.random.randint(low =(Mean_age-Std_age) , high= (Mean_age+Std_age), size= Number_missing_age)
    dataframe.loc[:,'Age'][np.isnan(dataframe['Age'])]= random_age
    #df.loc[:,'B'][np.isnan(df['B'])]= filll
    


# In[ ]:


age_input(titanic)


# In[ ]:


titanic.isnull().sum()


# In[ ]:


age_input(test)


# In[ ]:


test.isnull().sum()


# In[ ]:


sns.factorplot(y= 'Age', data = titanic, x = 'Pclass', kind = 'box', col = 'Survived', palette='Set1')


# In[ ]:


# Check the distribution of the age after filling the holes
sns.distplot(a = titanic['Age'].dropna(), bins=40, )


# In[ ]:


sns.factorplot(y= 'Age', data = titanic, x = 'Sex', kind = 'bar', col = 'Survived', palette='Set1')


# In[ ]:


test.head()


# ### Preparation for Machine learning

# In order to prepare the dataset for machine learning algorithms, we would need to perform the following tasks:
# 1) Create dummy variables for the Sex column and the Embarked columns.
# 2) Drop non numeric columns.

# In[ ]:


def embarked_dummy(dataframe, label_drop):
    
    """
    Function creates dummy variable for the Embarked column, drops one of the column in the dummy column 
    and joins to the previous dataframe. This function also drops the columns with the names in the list
    column_drop in the passed dataframe.
    
    dataframe = Name of the dataframe
    label_drop = string: String of categorical value in Sex column which you would like to drop
    column_drop = list: list of column labels which you want to be dropped
    
    """
    import pandas as pd
    embarked_dummy = pd.get_dummies(data = dataframe['Embarked'])
    # Drop column in dummy column
    embarked_dummy.drop(labels= label_drop, inplace=True, axis=1)
    
    # Merge to the dataset 
    dataframe= dataframe.join(embarked_dummy)

    return dataframe
    


# In[ ]:


titanic = embarked_dummy(dataframe=titanic, label_drop='S')


# In[ ]:


titanic.head()


# In[ ]:


test = embarked_dummy(dataframe=test, label_drop='S')


# In[ ]:


test.head()


# In[ ]:


def gender_dummy(dataframe, label_drop, column_drop):
    
    """
    Function creates dummy variable for the Sex column, drops one of the column in the dummy column 
    and joins to the previous dataframe. This function also drops the columns with the names in the list
    column_drop in the passed dataframe.
    
    dataframe = Name of the dataframe
    label_drop = string: String of categorical value in Sex column which you would like to drop
    column_drop = list: list of column labels which you want to be dropped
    
    """
    import pandas as pd
    gender_dummy = pd.get_dummies(data = dataframe['Sex'])
    # Drop column in dummy column
    gender_dummy.drop(labels= label_drop, inplace=True, axis=1)
    
    # Merge to the dataset 
    dataframe= dataframe.join(gender_dummy)
    # Drop Sex column
    dataframe.drop(labels = column_drop, axis = 1, inplace = True )
    
    return dataframe
    
    


# In[ ]:


dropped = ['Cabin', 'Sex', 'SibSp', 'Parch', 'Ticket','Fare', 'Embarked','Name']


# In[ ]:


# apply to train dataset
titanic = gender_dummy(dataframe=titanic, label_drop='male', column_drop=dropped)


# In[ ]:


titanic.head(3)


# In[ ]:


# apply to train dataset
test = gender_dummy(dataframe=test, label_drop='male', column_drop=dropped)


# In[ ]:


test.head(3)


# Split into training and test

# In[ ]:


# define training and testing sets

X_train = titanic.drop(["Survived",'PassengerId'],axis=1)
Y_train = titanic["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()


# ## Machine learning algorithm

# Several machine learning algorithms would be tested and scored, the algorithm with the highest score would then be used for prediction on the test set.

# In[ ]:


# Names of classifiers
names = ['LogisticRegression',"Nearest Neighbors", "Linear SVM", "RBF SVM","Random Forest", "Naive Bayes"]


# In[ ]:


classifiers = [LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB()]


# In[ ]:


def classifer_iterate(names, classifiers, X_train, Y_train):
    import matplotlib.pyplot as plt
    plt.Figure(figsize=(20,6))
    D = {}
    for name, clf in zip(names, classifiers):
        clf.fit(X = X_train,y = Y_train)
        score = clf.score(X_train, Y_train)
        D[name]= score
        print('Score of ' + name + ' is ', score)
        
    print('-------------------------------------------------------------------------')
    plt.bar(range(len(D)), list(D.values()))
    


# In[ ]:


classifer_iterate(names=names, classifiers=classifiers,X_train=X_train, Y_train=Y_train)


# Prediction off the test set

# In[ ]:


#  Support Vector Machine
SVM_RBF = SVC(gamma=2, C=1)
SVM_RBF.fit(X_train, Y_train)
Y_pred = SVM_RBF.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


submission


# In[ ]:


submission.to_csv('submit.csv', index=False)


# In[ ]:


print('The end')


# In[ ]:




