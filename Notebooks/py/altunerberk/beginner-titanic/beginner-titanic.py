#!/usr/bin/env python
# coding: utf-8

# **Titanic Project** 
# 
# 
# Anıl Berk Altuner
# 
# 
# 
# Today, we will learn to:
# 
# 1. Question definition
# 
# 2. Exploring the data
# 
#     2.1. Use some pandas functions on the data
#    
#     2.2. Make comparisons between values
#    
#     2.3. Determining valuable information on purpose
#     
# 3. Wrangling with data
# 
#     3.1. These valuable informations can be more effective. We use some functions for about that.
#     
#     3.2. We will make the data cleaner
#     
# 4. Model and predict
# 
#     4.1 We try some Machine Learning models and find a best solution.
# 
# 5. Visualize problem solution
# 
# 
# 
# **Question**
# 
# The Titanic was one of the most tragic wrecks in history. Today we will work on t if a passenger survived the sinking of the Titanic or not. 
# For each PassengerId in the test set, we predict a 0 or 1 value for the Survived variable.

# 

# In[108]:


import pandas as pd #Data processing
from sklearn.linear_model import LogisticRegression #For the machine learning modelling
from sklearn.metrics import accuracy_score #Calculating accuracy for end of the evulation
from sklearn.neighbors import KNeighborsClassifier #KNeighborsClassifier Algorithm in Sklearn Library
from sklearn.tree import DecisionTreeClassifier #Decision Tree Algorithm in Sklearn Library
from sklearn.ensemble import RandomForestClassifier #RandomForest Algortihm in Sklearn Library
from sklearn.naive_bayes import GaussianNB #Naive Bayes Algorithm
from sklearn.svm import SVC #Support Vector Machine
from sklearn.model_selection import cross_val_score,train_test_split #Data proccessing for the modelling
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt #Visual modelling
import seaborn as sns


# **Input Data to program**
# 
# We take tests and training data separately to ensure that the program is healthy. We perform our operations on the training data and compare it with the test data at the end of the program. This is how we achieve a healthy end result. The data we will guess is "Survival" data. We use pandas library. Lets use [pd.read_csv()](http://https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html). We can import the .csv data with this function and we can look shape of these data with [Dataframe.shape](http://http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.shape.html)

# In[109]:



test_df = pd.read_csv("../input/test.csv", sep=",")
train_df = pd.read_csv("../input/train.csv", sep=",")
# Input data files in program.



# **Explore the Data**
# 
# Our data has 7 columns. We analyze these columns.   
# 
# PassangerId : This column is the unique number that identifies the passengers
# 
# Survived : This column has 0 and 1. These numbers are value of the passangers survived or died. "1" mean passanger survive, "0" mean passanger died.
# 
# Pclass : Ticket class 
# 
# Age :  The passenger's age in years
# 
# SibSp : The number of siblings or spouses the passenger had aboard the Titanic
# 
# Parch : The number of parents or children the passenger had aboard the Titanic
# 
# Ticket : The passenger's ticket number
# 
# Fare :The fare the passenger paid
# 
# Cabin : The passenger's cabin number
# 
# Embarked : The port where the passenger embarked (C=Cherbourg, Q=Queenstown, S=Southampton)

# In[110]:


train_df.info()
print("_"*50)
test_df.info()
#For the comparisons train and test.


# * Train data has 891 passanger's information, test data has 418
# * PassangerId, Pclass, Age,SibSp,Parch has a numerical values.
# * We can see our train and test data informations but something missed. Age and Cabin columns are less than the others. Probably Age and Cabin columns have some NaN values. 

# In[111]:


train_df.describe(include='all') #Statical values from train_df


# In[112]:


train_df.head() #First 5 value from to train_df


# We see first 5 information on data. We need to find effective columns for train our data. We can use Age and Sex for about that. Because they have chance to first preference for a lifeboats. We also use Pclass. If your ticket at first class, you have preference.We can use Embarked too. City's economic value can say something about passangers rich or poor. Fare can give it to us too. We can determine the richness or poverty of the traveler according to the value of the Fare. In this case, we can give us a rate according to the probability of live or dead . We use Age,Sex and Pclass,Embarked,Fare.

# In[113]:


#Creat the chosen column and survive value comparision
def chart(feature):
  survived = train_df[train_df["Survived"] == 1] #Transfer survival values of 1 to a new series
  died = train_df[train_df["Survived"] == 0]#Transfer survival values of 0 to a new series
  survived[feature].plot.hist(alpha=0.5,color='red',bins=25)#Survived[column we chose].plot.hist(alpha=Visibilty value, color=color of graphic,bins=width of boxes)
  died[feature].plot.hist(alpha=0.5,color='blue',bins=25)
  plt.legend(['Survived','Died'])
  plt.show()
#We write "Age" for age-survive value comprasion    
chart("Age")


# Lets look a graph of the relationship between Age and Survived. It seems a little complicated but we can see that most of the babies survive, and the majority of the population in the 20-40 age range is dead.
# 
# Now we can look a graph of the relationship between Sex and Survived

# In[114]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived') #Comprasion female and male about alive or dead.


# As you see when we comparision beetween Sex and Survived, we can see more female alive. Becuase we said female has preference for a lifeboats.
# 
# **Wrangling Data**
# 
# We need to more cleaner data for analyze. First we need to focus what we need. We use Age,Pclass and Sex for modelling. After that we can drop the other columns and cutting age column will be more clean data for us. If you remember we said "Age" column had some NaN values. These values are missing. We create new variable, its name "Missing". These NaN values replaced to -0.5. After that, missing values between -1 , 0. For this process we use [df[columnname].fillna()](http://http://pandas.pydata.org/pandas-docs/stable//generated/pandas.DataFrame.fillna.html) function. For cutting process, we use [pd.cut](http://https://stackoverflow.com/questions/45751390/pandas-how-to-use-pd-cut) function.

# In[115]:


#Cut the age and create new age categories. These catagories are more cleaner our data
def cutting_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5) #Some age values are missing. We fill -0.5 these NaN values.
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names) # pd.cut function cut from the points we want
    return df

cut_points = [-1,0,5,12,18,35,60,100] #cut points we chose
label_names = ["Missing","Baby","Child","Teenager","Young Adult","Adult","Old"] #Every 2 range will describe one label in order.

train_df = cutting_age(train_df,cut_points,label_names) #Cut Age column in train data
test_df = cutting_age(test_df,cut_points,label_names) #Cut Age column in test data

sns.barplot(x="Age_categories", y="Survived", data=train_df)


# In[116]:


#Cutting fare values for clean data.
def cutting_fare(df,label_names): 
    df["Fare_categories"] = pd.qcut(df["Fare"],4,labels=label_names) # pd.qcut() function cut from the equal points. We choose 4 for that value.
    return df

label_names = ["Least","Less-Middle","Middle","High"]

train_df = cutting_fare(train_df,label_names)
test_df = cutting_fare(test_df,label_names)

sns.barplot(x="Fare_categories", y="Survived", data=train_df)


# In[117]:


train_df[["Fare_categories", "Survived"]].groupby(['Fare_categories'], as_index=False).mean().sort_values(by='Survived') #Every fare catagories survive values.


# Now we can more clean if we create dummies. If we extend the divided age categories to each dummy column, we can increase the prediction ratio. We can create dummies with [pd.get_dummies](http://https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) function. Pandas can easily do seperated every diffrent values to diffrent columns. After extend process we must use [pd.concat](http://https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html) for comibe to dataframes.

# In[118]:


#We create dummies for every created values. After this process we have a lot of columns for every value.
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name) #Seperate to columns for every diffrent value
    df = pd.concat([df,dummies],axis=1) # created columns adding to  actual data 
    return df

for column in ["Pclass","Sex","Age_categories","Embarked","Fare_categories"]: #Columns who diveded
    train_df = create_dummies(train_df,column)
    test_df = create_dummies(test_df,column)


# Now we drop unnecessary columns in the train data. We use for that [Dataframe.drop()](http://https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)

# In[120]:


train_df = train_df.drop(['Name','Pclass','Age','Ticket','Sex','SibSp','Parch','Fare','Cabin','Embarked','Age_categories','Fare_categories'], axis=1) 
#We dont need the other columns. We drop columns for cleaner data


# In[121]:


train_df.head() # First 5 value of new train_df 


# **Model and Predict**
# 
# Let's use logistic regression as model of machine learning.  [LogisticRegression](http://http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) function in the Sklearn library makes it very easy to use. We will use the [LogisticRegression.fit(](http://http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)) command to fit the columns we have created. After creating the model we must separate the test data to make predicts. Again, [train.test.split()](http://http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) command in the Sklearn library performs this operation.

# In[122]:


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male','Age_categories_Missing','Age_categories_Child', 'Age_categories_Teenager',
           'Age_categories_Young Adult', 'Age_categories_Adult','Age_categories_Old','Embarked_C','Embarked_S','Embarked_Q','Fare_categories_Least','Fare_categories_Less-Middle',
           'Fare_categories_Middle','Fare_categories_High'] #We use these columns


#Splitting data
X = train_df[columns] #Train data
y = train_df['Survived'] #Targer data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20,random_state=15) #Splitting data to train and test datas. We chose the test size %20.
#Random state is change rate to data when program started.


# **K-Fold Cross Validation**
# 
# We split to data %80 train, %20 test and this datas randomized but we may not get the real performance. We can not get maximum performance because the randomly selected data is not selected very well. In the K-Fold Cross Validation process, the meaning of "K" determines how many times we extend the test data to the all dataframe. For example, if we take K as 5, %20 of the data is distributed 5 times, and the average performance of each distribution is determined as the overall performance. During this process we use [cross_val_score()](http://http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) from sklearn library

# In[123]:


#Cross Validation function. We call this function when try every model.
def cr_val(model,tr_data,test_data):
    accuracy = (cross_val_score(model, tr_data, test_data, cv=10)).mean() #cross_val_score(ModelWeUse, TrainData, TestData, cv=Number of pieces data will we divide).
    # .mean() is for take the average after getting the results
    return accuracy


# In[124]:


#Logistic Regression process
lr=LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)

#Now call the Cross Validation Function
accuracy = cr_val(lr,X,y)

print(accuracy)


# **K-Nearest Neighbor**
# 
# K-Nearest Neighbor  Classifier take K nearest item and return value of main item. Value of item is a majority of other items values. We use[ KNeighborsClassifier()](http://http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) in Sklearn library for it.

# In[143]:



hyperparameters = {
    "n_neighbors": range(1,20),
}
knn=KNeighborsClassifier()
grid=GridSearchCV(knn,param_grid=hyperparameters,cv=10)
grid.fit(X,y)
best_score=grid.best_score_

print(best_score)


# **Random Forest Classifier**
# 
# Random Forest Classifier is small tree groups. Every groups have questions and end of the process, our item has value of majority to the answers. We use [RandomForestClassifier()](http://http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) in Sklearn library for it.

# In[142]:



hyperparameters={ "criterion":["entropy","gini"], 
                "max_depth":[5,10],
                "max_features":["log2","sqrt"],
                "min_samples_leaf":[1,5],
                "min_samples_split":[3,5],
                "n_estimators":[6,9],
               }

clf=RandomForestClassifier(random_state=1)
grid=GridSearchCV(clf,param_grid=hyperparameters,cv=10)
grid.fit(X,y)
best_params=grid.best_params_
best_score=grid.best_score_

print(best_score)


# **Decision Tree**
# 
# Decision Tree Classifier is bigger than Random Forest. In Decision Tree, the value of end of the chained question is equal to our item's value. We use [DecisionTreeClassifier()](http://http://scikit-learn.org/stable/modules/tree.html) in Sklearn library for it

# In[128]:


clf=DecisionTreeClassifier()

#Cross Validation Function
accuracy=cr_val(clf,X,y)
print(accuracy)


# **Naive Bayes Algorithm**
# 
# 
# 

# In[129]:


clf=GaussianNB()

#Cross Validation Function
accuracy=cr_val(clf,X,y)
print(accuracy)


# **Support Vector Machine**
# 
# The support vector machine is grouped together with an equal line from the middle of the graphical two sides of the graph and this graph becomes clear with a clear line. We use [SVC()](http://http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) from Sklearn library.

# In[130]:


clf=SVC()

#Cross Validation Function
accuracy=cr_val(clf,X,y)
print(accuracy)


# **Submission File**
# 
# We got the best accuracy from K-Neighbor Classifier. Now we should predict with Kaggle's test data and after that we create submission file.

# In[145]:



best_rf=grid.best_estimator_
test_predictions=best_rf.predict(test_df[columns])
submission_df = {"PassengerId":test_df["PassengerId"],
                 "Survived": test_predictions}
submission = pd.DataFrame(submission_df)
print(submission)


# In[ ]:


submission.to_csv("submission.csv",index=False)


# 
