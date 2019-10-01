#!/usr/bin/env python
# coding: utf-8

# # Titanic survival predictions
# This kernel is inspired from click [here](https://www.kaggle.com/samsonqian/titanic-guide-with-sklearn-and-eda)
# Hi, I am a graduate Electronics and Communications Engineer who decided to shift to Data sciecne and Machine learning and in this kernel i will be disscussing explicitly the problem of prediction the survival/non survival people on board of titanic, i will come across several concepts and techniques spanning data visualization, Machine learning and Deep learning trying to state the best model working on this specific dataset.
# 
# *Please upvote and share if this helps you!! Also, feel free to fork this kernel to play around with the code and test it for yourself. If you plan to use any part of this code, please reference this kernel!* I will be glad to answer any questions you may have in the comments. Thank You! 
# 
# *Make sure to follow me for Future Kernels even better than this one!*
# 

# ## Supervised vs. Unsupervised learning
# ### Supervised
# Instead of stating concepts to give you an inuition about these terms i will go to the more practical way, this problem is supervised why? as we are given a data with labels stating whom survived and whom not our job is to do the feature extraction and find patterns in the data and do modeling in order to predict where a person survived or not then test this on a test data without labels to see how good our model is.
# ### Unsupervised
# Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. The most common unsupervised learning method is cluster analysis, which is used for exploratory data analysis to find hidden patterns or grouping in data. so let's have an example let's you have a dataset with like 100 people and their interests and you would like to group them according to interests, see no labels are here. simple?

# ## Classification vs. Regression
# As you know, predicting Titanic survivors is a supervised classification Machine Learning problem, where you classify a passenger as either survived, or not survived. Whereas in regression, you predict a continuous value like house price or the temperature, i you like to see a regression problem that i addressed go check click [here](https://www.kaggle.com/omarayman/random-forest-experiment) where i predicted the prices of bulldozers offered by Blue Book stores

# # Contents
# 1. [Importing Libraries and Packages](#p1)
# 2. [Loading and Viewing Data Set](#p2)
# 3. [Evaluation](#p3)
# 4. [Data Cleaning(Handling missing values)](#p4)
# 5. [Feature Engineering](#p5)
# 6. [Feature Rescaling](#p6)
# 7. [Data Visualization](#p7)
# 8. [Model Fitting, Optimizing, and Predicting](#p8)

# <a id="p1"></a>
# # 1. Importing Libraries and Packages
# We will use these packages to help us manipulate the data and visualize the features/labels as well as measure how well our model performed. Numpy and Pandas are helpful for manipulating the dataframe and its columns and cells. We will use matplotlib along with Seaborn to visualize our data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

get_ipython().magic(u'matplotlib inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="p2"></a>
# # 2. Loading and Viewing Data Set
# With Pandas, we can load both csv files including the training and testing set that we wil later use to train and test our model. Before we begin, we should take a look at our data table to see the values that we'll be working with. We can use the head and describe function to look at some sample data and statistics as mean,std and count values . We can also look at its keys and column names.

# In[ ]:


train =pd.read_csv('../input/train.csv')
test =pd.read_csv('../input/test.csv')
train.sample(50)


# #### Columns 
# * survival 	Survival 	0 = No, 1 = Yes
# * pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
# * sex 	Sex 	
# * Age 	Age in years 	
# * sibsp 	# of siblings / spouses aboard the Titanic 	
# * parch 	# of parents / children aboard the Titanic 	
# * ticket 	Ticket number 	
# * fare 	Passenger fare 	
# * cabin 	Cabin number 	
# * embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
# #### Variable Notes
# 
# * pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# * age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# * parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.
# 

# Now we have a better intuition about what our dataset is having, so our eyes should be searching for the numerical values that will make our job easier as our machine learning/deep learning models so here the continous numerical values are found in columns [Age,Fare] don't be fooled by other columns they are represented in numbers but in fact they are  representing just discret values that describe categories as for example sibsp is representing the number of siblings a passenger is having.
# ## let's demonstrates the differences between different variables
# 
# ### Categorical variable
#     Categorical variables contain a finite number of categories or distinct groups. Categorical data might not have a logical order. For example, categorical predictors include gender, material type, and payment method. 
# ### Discrete variable
#     Discrete variables are numeric variables that have a countable number of values between any two values. A discrete variable is always numeric. For example, the number of customer complaints or the number of flaws or defects. 
# ### Continuous variable
#     Continuous variables are numeric variables that have an infinite number of values between any two values. A continuous variable can be numeric or date/time. For example, the length of a part or the date and time a payment is received. 

# <a id="p3"></a>
# # 3.Evaluation
# after all we have done we should see how Kaggle will evaluate our work.
# ### Goal
# 
# It is your job to predict if a passenger survived the sinking of the Titanic or not.
# For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.
# ### Metric
# 
# Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.
# ***so all we will have to take care of is our accuracy on test data*****

# In[ ]:


print("The types of data our dataset has")
train.dtypes


# In[ ]:


print('lets see the statistical values of our dataset')
train.describe()


# In[ ]:


print("let's see the number of non values in our data and their types")
train.info()


# <a id="p4"></a>
# # 4. Data Cleaning(Handling missing values) 
# The first thing I do when I get a new dataset is take a look at some of it. This lets me see that it all read in correctly and get an idea of what's going on with the data. In this case, I'm looking to see if I see any missing values, which will be reprsented with NaN or None.
# ## See how many missing data points we have
# 

# In[ ]:


missing_values_count = train.isnull().sum()


# In[ ]:


total_cells = np.product(train.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100


# Around 8% of the data is missing, Ok, now we know that we do have some missing values. Let's see how many we have in each column.

# In[ ]:


print("The percentage of NaN values in descending order")
print((train.isnull().sum().sort_values(ascending=False)/len(train)*100))


# 
# # Figure out why the data is missing
# 
# This is the point at which we get into the part of data science that I like to call "data intution", by which I mean "really looking at your data and trying to figure out why it is the way it is and how that will affect your analysis". It can be a frustrating part of data science, especially if you're newer to the field and don't have a lot of experience. For dealing with missing values, you'll need to use your intution to figure out why the value is missing. One of the most important question you can ask yourself to help figure this out is this:
# 
#     Is this value missing becuase it wasn't recorded or becuase it dosen't exist?
# 
# If a value is missing becuase it doens't exist  then it doesn't make sense to try and guess what it might be. These values you probalby do want to keep as NaN. On the other hand, if a value is missing becuase it wasn't recorded like for example here in Embarked, Age columns, then you can try to guess what it might have been based on the other values in that column and row. (This is called "imputation" and we'll learn how to do it next! :)
# 
# Let's work through an example. Looking at the number of missing values in the **train** dataframe, I notice that the column **Cabin** has a lot of missing values in it:
# 

# In[ ]:


print('lets see in this column in our training set')
train.Cabin


# It seems that it's too noise for the handy techniques of handling missing values so we are gonna just drop it for both training and testing sets

# In[ ]:


train.drop(labels=['Cabin'],axis=1,inplace=True)
test.drop(labels=['Cabin'],axis=1,inplace=True)


# In[ ]:


print('lets see in the age columns instances')
train.Age


# Nth is clear right so the righ tool at this moment is to plot the data and see what it represents, so let's plot the Age data

# In[ ]:


copy = train.copy()
copy.dropna(inplace=True)
sns.distplot(copy.Age)


# Looks like the distribution of ages is slightly skewed right. Because of this, we can fill in the null values with the median for the most accuracy. 
# > **Note:** We do not want to fill with the mean because the skewed distribution means that very large values on one end will greatly impact the mean, as opposed to the median, which will only be slightly impacted.

# In[ ]:


train.Age.fillna(train.Age.median(),inplace=True)
sns.distplot(train.Age)
test.Age.fillna(test.Age.median(),inplace=True)


# why is this plot is better than the earlier one so this will lead us to 
# ## Normal distribution 
# First reason the normal distribution is important is that many psychological and educational variables are distributed approximately normally. Measures of reading ability, introversion, job satisfaction, and memory are among the many psychological variables approximately normally distributed. although the distributions are only approximately normal, they are usually quite close.
# 
# It is very helpful in forecasting .We can calculate the estimated length of the bones of animals and woods and leaves . because if animals are one type their numbering is normally distributed .
# 
# Second reason the normal distribution is so important is that it is easy for mathematical statisticians to work with.Normal distribution is very useful for controlling the quality in business. With this we can fix the limit of quality . that will helpful for controlling the quality .
# 
# If we take one sample out of the universe and calculate the mean size of growing then it will normal distribution This means that many kinds of statistical tests can be derived for normal distributions. Almost all statistical tests discussed in this text assume normal distributions. Fortunately, these tests work very well even if the distribution is only approximately normally distributed. Some tests work well even with very wide deviations from normality.

# In[ ]:


print("lets see the values in Embarked column")
train.Embarked


# Let's see how much categories are weighted much in this column as it can't be plotted and we don't wanna just drop some useful data without giving it a try

# In[ ]:


train.Embarked.value_counts()


# In[ ]:


#seems that S has the more counts and it's less than 1 percent that is missing in this column so we just going to place S in the NaN instances
train.Embarked.fillna("S",inplace=True)


# In[ ]:


#horaaaay no missing in training set
train.isnull().sum()


# In[ ]:


print("lets check missing data in the testing dataset to move on to step")
test.isnull().sum()


# In[ ]:


#only one instance will just put the median
test.Fare.fillna(test.Fare.median(),inplace=True)


# but there is other problem, we are having a bunch of columns which having string variables represents a certain category so it's our job to convert it to representations includes numbers so that our model can deal with it and this leads us to feature engineering. 

# <a id="p5"></a>
# # 5. Feature Engineering
# 
# 

# In[ ]:


train.sample(10)


# In[ ]:


test.sample(10)


# In[ ]:


#Most easy column is the Sex column as it just consists of 2 values let' begin with it so that you can have the intuition 
#let's say put the value=1 to represent a male and value 0 to represent a women
train.loc[train['Sex']=='male','Sex'] = 1
train.loc[train['Sex']=='female','Sex'] = 2
# and the same for test data
test.loc[test['Sex']=='male','Sex'] = 1
test.loc[test['Sex']=='female','Sex'] = 2


# Same for Embark give them the values you like , i will give them the values 1,2,3 since i actually bored of the fact that the index in numbering in python begins with 0 :d

# In[ ]:


#let's see what values Embarked has as i forgot :D
train.Embarked.unique()
#aha ok sorry
train.loc[train['Embarked']=='S','Embarked'] = 1
train.loc[train['Embarked']=='C','Embarked'] =2
train.loc[train['Embarked']=='Q','Embarked'] =3
#test data
test.loc[test['Embarked']=='S','Embarked'] =1
test.loc[test['Embarked']=='C','Embarked'] =2
test.loc[test['Embarked']=='Q','Embarked'] =3


# In[ ]:


#Done let's see what we have done so far
train.sample(10)


# In[ ]:


test.sample(10)


# ## Improving Classifier Performance With Synthetic Features
# When you train a classifier on a dataset, often that you are stuck at a certain performance and are unable to get past it. At this time, you may add new training data, use a more complex model, etc to boost the performance of the classifier. However, you may not always be able to add more training data, or may have a large dataset such that using a more complex model is infeasible computationally.
# So, what can I do?
# 
# We can add synthetic features to the dataset, which is a form of feature engineering. You can view it as adding a higher level representation of the raw dataset.
# For instance, if you have to predict whether it is feasible to drive between two pairs of latitude and longitude points. Let’s say feasible is that we are driving our car and we can’t drive for many days. Then, the dataset would look something like this: we have in our datasets 2 columns somehow related to each other like siblings column which has the number of sibling with a certain person and parents column which is the number of accompaning parents with a certain person and if he is alone or not.

# In[ ]:


#just adding the values in Parents and Siblings to get the family size
#and adding 1 counts for the person himself
train['FamSize'] = train['Parch'] + train['SibSp'] + 1
test['FamSize'] = test['Parch'] + test['SibSp'] + 1


# In[ ]:


#See if famsize is 1 then the person is considered alone
train["IsAlone"] = train.FamSize.apply(lambda x: 1 if x == 1 else 0)
test["IsAlone"] = test.FamSize.apply(lambda x: 1 if x == 1 else 0)


# In[ ]:


# inspect the correlation between Family and Survived
train[['FamSize', 'Survived']].groupby(['FamSize'], as_index=False).mean()


# We can see that the survival rate increases with the family size, but not beyond Family = 4. Also, the amount of people in big families is much lower than those in small families. I will combine all the data with Family > 4 into one category. Since people in big families have an even lower survival rate (0.161290) than those who are alone, I decided to map data with Family > 4 to Family = 0, such that the survival rate always increases as Family increases.

# In[ ]:


train.FamSize = train.FamSize.map(lambda x: 0 if x > 4 else x)
train[['FamSize', 'Survived']].groupby(['FamSize'], as_index=False).mean()


# In[ ]:


#let's see what we have done so far
train.sample(10)


# Last feature is the name the technique i am about to inroduce is a bit counter intuitive but trust me it worked, we can extract some useful information from names column , actual names are actually means nth  so why don't we extract these info from the names columns what if we just extracted Mr., Mrs,Doc so we can see if Doc are more likely to survive or its Mr or Mrs and so on don't you think it will help our classifier to achieve a better accuracy, let's find out
# 

# In[ ]:


for name in train["Name"]:
    train["Title"] = train["Name"].str.extract("([A-Za-z]+)\.",expand=True)
    
for name in test["Name"]:
    test["Title"] = test["Name"].str.extract("([A-Za-z]+)\.",expand=True)


# In[ ]:


train.head()


# See we have extracted all the Misters and miss out of the Names column but remeber when i told you that machine learning models can't handle any types but numerical data so we have to just put it in a string form, so now let's see the unique values we are having in Title column 

# In[ ]:


print("Unique values in the Title column")
unique_titles=train.Title.unique()
unique_titles = list(unique_titles)
unique_titles


# In[ ]:


print("let's see the frequencies of each title in our dataset")
title_list = list(train["Title"])
frequency_titles = []

for i in unique_titles:
    frequency_titles.append(title_list.count(i))
    
print(frequency_titles)


# In[ ]:


print("integrating both title as a string and its frequency to see which title most occured and which is least")
title_dataframe = pd.DataFrame({
    "Titles" : unique_titles,
    "Frequency" : frequency_titles
})

print(title_dataframe.sort_values(by='Frequency',ascending=False))


# So it's clear that some Titles are quiet rare in our title column so we will stack them up and replace them with Title "Rare", also replacing some values that are having the same meaning like Mlle,mme, as The term Mademoiselle is a French familiar title, abbreviated Mlle, traditionally given to an unmarried woman. The equivalent in English is "Miss",In France, one traditionally calls a young, unmarried woman Mademoiselle – Mlle for short – and an older, married woman Madame, whose abbreviation is Mme. As in English, there is only one term to describe males: Monsieur, or M for short.

# In[ ]:


#instead of repeating my steps for training and test sets will just put them as a list and iterate through them
#
datasets = [train,test]
for dataset in datasets:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


print("unique values after working on them")
test.Title.unique()


# In[ ]:


print("assigning an easy number to indicate it to each title")
for dataset in datasets:
    dataset.loc[dataset["Title"] == "Miss", "Title"] = 0
    dataset.loc[dataset["Title"] == "Mr", "Title"] = 1
    dataset.loc[dataset["Title"] == "Mrs", "Title"] = 2
    dataset.loc[dataset["Title"] == "Master", "Title"] = 3
    dataset.loc[dataset["Title"] == "Rare", "Title"] = 4


# In[ ]:


train.Title


# In[ ]:


test.Title


# Perfect we are done with titles so now feel free to drop the Name column as it did the job it meant to do also PassengerId, Ticket, these columns will just deflects our learning process

# In[ ]:


for dataset in datasets:
    dataset.drop(columns=['Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train


# Perfect? but wait a second don't you think that the values in Age and Fare are a bit in a different range than the rest of our columns , wouldnt it be a problem? yes  of course we should put all our datasets in the same range in order to achieve normalization and  normalization in statistics basically means:
# I have dataset 1 with possible values from 0-1000 randomly distributed.
# I have dataset 2 with possible values from 0-1 randomly distributed.
# 
# If I want to compare the distributions between dataset 1 and dataset 2 I would normalize the data by either dividing every value in dataset 1 by 1000 or multiplying every value in dataset 2 by 1000. Then the 2 datasets are comparable and can be merged. so this leads us to what's called feature rescaling.

# <a id="p6"></a>
# # 6. Feature Rescaling
#  It would be beneficial to scale them so they are more representative. We will do this with sklearn's MinMaxScaler function. This function also requires us to reshape our data so that it accepts the input. The steps are shown below.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#scaler requires arguments to be in a specific format shown below
#convert columns into numpy arrays and reshape them 
for dataset in datasets:
    ages_train = np.array(dataset["Age"]).reshape(-1, 1)
    fares_train = np.array(dataset["Fare"]).reshape(-1, 1)
#we replace the original column with the transformed/scaled values
    dataset["Age"] = scaler.fit_transform(ages_train)
    dataset["Fare"] = scaler.fit_transform(fares_train)


# In[ ]:


train


# Don't you see the difference? now let's move on to the next step
# 

# <a id="p7"></a>
# # 7.Data Visualization 
# It is very important to understand and visualize any data we are going to use in a machine learning/Deep learning model. By visualizing, we can see the trends and general associations of variables like Sex and Age with survival rate so you can see what how the dependent target which here in this case is (survival/non survival) depends on some independent features which are actually the columns in our training data. We can make several different graphs for each feature we want to work with to see the entropy and information gain of the feature. 

# ## An easy way to directly visualize whether a certain feature is affecting  our target is Correlation heat maps  

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# Before i may proceed as long as the square infront of a specific feature lies on the left  getting more navy and the number in it is closer to 1 that means that it's strongly correlated with the feature lies down. One thing that that the Pearson Correlation plot can tell us is that there are not too many features strongly correlated with one another. This is good from a point of view of feeding these features into your learning model because this means that there isn't much redundant or superfluous data in our training set and we are happy that each feature carries with it some unique information. Here are two most correlated features are that of Family size and Parch (Parents and Children). I'll still leave both features in for the purposes of this exercise.

# 
# ## let's see how gender affects the survival rate 
# 

# In[ ]:


sns.barplot(x='Sex',y='Survived',data=train)
plt.title("Distribution of Survival based on Gender")
plt.show()

total_survived_females = train[train.Sex == 2]["Survived"].sum()
total_survived_males = train[train.Sex == 1]["Survived"].sum()

print("Total people survived is: " + str((total_survived_females + total_survived_males)))
print("Proportion of Females who survived:") 
print(total_survived_females/(total_survived_females + total_survived_males))
print("Proportion of Males who survived:")
print(total_survived_males/(total_survived_females + total_survived_males))


# Gender appears to be a very good feature to use to predict survival, as shown by the large difference in propotion survived so as you can see in the plot if you are a woman then your likelyhood to survive is more than double the men likelyhood . Let's take a look at how class plays a role in survival as well. 

# Let's see if we tried to compare between different sexes and different classes

# In[ ]:


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")


# So we can see that women from the first class are the most likely to survive then comes women from class 2 

# In[ ]:


sns.barplot(x="Age",y='Survived',data=train)


# Couldn't see a thing ha?
# since these types are continous types of data need other plots, let's see

# In[ ]:


survived_ages = train[train.Survived == 1]["Age"]
not_survived_ages = train[train.Survived == 0]["Age"]
plt.subplot(1, 2, 1)
sns.distplot(survived_ages, kde=False)
plt.axis([0, 1, 0, 100])
plt.title("Survived")
plt.ylabel("Proportion")
plt.subplot(1, 2, 2)
sns.distplot(not_survived_ages, kde=False)
plt.axis([0, 1, 0, 100])
plt.title("Didn't Survive")
plt.subplots_adjust(right=1.7)
plt.show()


# In[ ]:


sns.stripplot(x="Survived", y="Age", data=train, jitter=True)


# Don't be scared of the Ages, as we ranged the data from 0 to 1 , you can get the real age if muliply the age value by 81 let's say 0.27*81 = 21.8 approx. to 22 (you can check that) then you will get the real age, no magic this is the value by which our data is scalled. It seems that passengers with older ages are least likely to survive

# Final cumulative graph

# In[ ]:


sns.pairplot(train)


# <a id="p8"></a>
# # 8. Model Fitting, Optimizing, and Predicting

# Defining our training/testing feature we will be using in training

# In[ ]:


X_train = train.drop(labels=["Survived","PassengerId"], axis=1) #define training features set
y_train = train["Survived"] #define training label set
X_test=test.drop('PassengerId',axis=1)
#we don't have y_test, that is what we're trying to predict with our model


# Dividing our training data to train/validation in order to have an intuition about how our model performe on data it has never seen before putting it on the test data

# In[ ]:


from sklearn.model_selection import train_test_split #to create validation data set

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets


# I always trust **Neural network **in solving my problems so i will be using a 10 fully connected layers only the first layer has 149 neurons, next 8 have 20 and the last one has only 1 as we are i a binary classification problem and for the activation function "relu" is commonly used, last one is sigmoid as it should be so if you are facing a binary classifcation problem(i.e if its multiclassification problem you should be using softmax instead). When it comes to **optimizers** i always prefer Adam as its commonly used in addition to RMS prop, if you need a better intuition about optimizations algorithms you can read my article on Medium click [here](https://medium.com/@omaraymanomar/optimization-algorithms-feb9dcb57116). In **fitting** batch_size is an important parameter and its just how many data the model should run through before it update its weights, epochs are the iteration through data

# In[ ]:


from keras.layers import Dense
from keras.models import Sequential
# Initialising the NN
model = Sequential()
#149-20-20-20-20-20-20-20-20-1
# layers
model.add(Dense(units = 149, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN,
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_train, y_train, batch_size = 32, epochs = 1000,validation_data=(X_valid,y_valid))


# In[ ]:


submission_predictions = model.predict(X_test)
#checking when a probality is bigger than 0.5 so if so convert the "True" it should spit out to 1 and reshape 
#submission to be like test data
y_final = (submission_predictions > 0.5).astype(int).reshape(X_test.shape[0])


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_final
    })

submission.to_csv("titanic.csv", index=False)
print(submission.shape)


# If you made it this far, congratulations!! You have gotten a glimpse at an introduction to data visualization, analysis and Deep Learning. You are well on your way to become a Data Science expert! Keep learning and trying out new things, as one of the most important things for Data Scientists is to be creative and perform analysis hands-on. Please upvote and share if this kernel helped you!
