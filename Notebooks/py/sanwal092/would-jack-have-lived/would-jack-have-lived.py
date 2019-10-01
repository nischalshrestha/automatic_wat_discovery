#!/usr/bin/env python
# coding: utf-8

# # <font color = 'purple'>  TITANIC: AN EXPLORATORY DATA ANALYSIS AND PREDICTING IF JACK WOULD HAVE SURVIVED </font>
#   
# The goal of this Jupyter Notebook is to do some exploratorty data analysis on the people who were on board the RMS Titanic which met a very guesome end. I intend to use this Notebook to work on Feature Engineering and slicing data in DataFrame to prepare it for Machine Learning libraries. 
# 
# The second goal of this Jupyter Notebook is to see if we can predict with reasonable accuracy whether the character of Jack Dawson(portrayed by Leonardo DiCaprio) from James Cameron's Titanic would have survived. I have scoured the Internet chatrooms to find trivia information about the character and seen the movie again just to see if I can get information which related to the variables looked at in this dataset. 
#  
# I was able to find only segmented information regarding the character mostly because he isn't supposed to be on Titanic. He wins tickets from unnamed Swedes in gambling. The tickets however are for 3rd class and is a man of age 20 years and has no faily on board. 
# 
# So, to recap, this Notebook is split into two parts:
# * Exploratory Data Analysis of the Titanic Dataset 
# * Setting up Machine Learning Classifers for four major algorithms used for classification problems and use each of them to see whether Jack would have survived or not.

# # <font color='Purple'>IMPORT LIBRARIES FOR DATA ANALYSIS</font>

# In[ ]:


# Libraries for data analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# # <font color = 'purple'> Import data </font>
# 

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# ## <font color = 'purple'> There are different number of columns in train and test data. I want to see which ones are the different columns </font>
# 

# In[ ]:


test_columns = test_data.columns.tolist()
train_columns = train_data.columns.tolist()
print('There are ',len(test_columns),' test columns ')
print('There are ',len(train_columns),' train columns \n')
print('***********columns in test data***********')
print(test_data.columns.values,'\n')
print('***********columns in train data***********')
print(train_data.columns.values,'\n')
diff= list(set(train_columns)-set(test_columns))
print('*************************************************')
print('The extra column in train data DataFrame is',diff)


# ## <font color='purple'>Check null or NaN values in training data </font>

# In[ ]:


# There are 891 values in every column in training data except for age which has 714 values 
print('*******The NaN values in each column******* \n')
print(train_data.isnull().sum(),'\n')


# In[ ]:


print('*******NaN values as % of 891 ******* \n')
print((train_data.isnull().sum()/891)*100)


# ## <font color= 'purple'>I am going to drop any column where more than 60% of the values are NaN </font>
# 
# ## <font color='red'> Observations </font >
# * So there was a column called Cabin which had 77.1% values that were NaN .  Since I chose 60 % of the values as acceptable NaN to keep the columns. Since Cabin had 77.1% values which were NaNs, so i dropped it. 
# 
# * There are other columns where some rows have NaN. Age has 19.86% values  and Embarrked has 0.225% values which are NaN  so I am dropping only the missing ROWS, NOT the columns.

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
train_clean = train_data 
for col in train_clean.columns:
    if train_clean[col].isnull().sum()/891 >= 0.6:
        train_clean =train_clean.drop(col,axis =1)
        
# train_clean=train_clean.dropna(axis=0)        
#Need to check if there are any more NaNs in the new dataframe 
train_clean.isnull().sum()/891
train_clean.head()
train_clean.isnull().sum()


# 
# 
# ## <font color ='purple'>LET'S LOOK AT THE CLEANED DATA FRAME FOR TRAINING DATA</font>

# In[ ]:


train_clean.describe()
# train_clean.isnull().values.any()


# In[ ]:


train_clean.sample(5)


# ## <font color ='purple'>Let's take care of the missing values in Age column </font>
# 
# I am filling the missing values in the Age column by taking the mean of the column and plugging that value back into the the missing rows in the Age column. Pandas provides you the option to choose the values that you want to replace the missing values with.
# 

# In[ ]:


'''
There are 177 NaN values in Age. This is because there are empty Age values for whatever reason. I am going to fill these
empty values with the mean age
'''
train_clean['Age'] =  train_clean['Age'].fillna(train_clean['Age'].mean())
train_clean.isnull().sum()


# <font color ='purple'>**There are still some NaN values in Embarked column. I am going to repeat the proceedure that I used with Age and fill the missing rows in Embarked.**</font>
# 

# In[ ]:


train_clean= train_clean.fillna(train_clean.mean())

train_clean.isnull().sum()
train_clean=train_clean.dropna(axis=0)
train_clean['Embarked'].isnull().sum()


# # <font color ='purple'> TIME TO CLEAN THE TEST DATA. </font>
# 
# **Observations**
# * 25.9% of all the Age values in the TEST dataset are all NaNs. **I am going to fill them with the mean value of the Age column in the testing dataset.**
# * 0.239% of all values in Fare are NaN. **I will fill these as well with the mean of the column in the test set.**
# * The rest of the columns don't have any empty values.
# * I got rid of the Cabin column since I took it out in the train dataset as well.

# In[ ]:


test_data = pd.read_csv('../input/test.csv')
test_data.sample(5)
# test_data.isnull().sum()
test_data = test_data.drop('Cabin', axis=1)
test_data.describe()
# test_data['Age'].count()
# test_data.columns


# In[ ]:


print('*******NaN values as % of the count of the values in each column ******* \n')
# test_data.isnull().sum()/test_data[]
for col in test_data.columns:
    print(col,':',(test_data[col].isnull().sum()/test_data[col].count())*100,'% values are NaNs')


# In[ ]:


# FIX AGE AND FARE COLUMN IN THE TEST SET.

## AGE 
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data.isnull().sum() # no NaNs in the Age column
# test_data.sample(5)

## FARE
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
# test_data.isnull().sum()  # no NaNs in the Fare column

test_clean = test_data
test_clean.sample(10)


# In[ ]:


test_clean.isnull().sum()


# ## <font color ='red'>So, we just cleaned the TEST dataset. It doesn't have any more NaNs and has the same columns as the TRAIN set, so we should be set to move on.
# </font>

# ## <font color ='Mauve'>Below, I am creating temporary variables to store training and test set. This is because down the line while creating Fare segmentation has the tendency to mess up my saved and cleaned trained/test set. By creating temporary variables, I get around this.</font>

# In[ ]:


# temp variables just so I don't have to reload training, testing data and then cleaning it
train_try = train_clean
test_try = test_clean
data_all = [train_try,test_try]


# # <font color= 'purple'>I was reading through other kernels and read the approach taken in this kernel towards Fare and using it: </font>
#     https://www.kaggle.com/startupsci/titanic-data-science-solutions
#     
# **The author here us creating a new Feature called FareBand by changing the Fare values to ordinal values. I find this approach interesting and hence am incorporating this in.**
# 
# * The author sets off by using the *pd.qcut* function to built quantiles and then group the survived based on the quantiles of the fare prices instead of  looking at the 'Fare' values as a continuous range of values.

# In[ ]:


train_try['FareBand'] = pd.qcut(train_try['Fare'], 4)
test_try['FareBand'] = pd.qcut(test_try['Fare'],4)
train_try[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
# test_clean[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


# train_clean.sample(5)
# train_try  = train_clean
# test_try = test_clean
type(train_try)
test_try.sample(5)


# In[ ]:


for dataset in data_all:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)



# In[ ]:


train_try = train_try.drop(['FareBand'], axis=1)
test_try = test_try.drop(['FareBand'], axis =1)


# In[ ]:


train_clean = train_try 


# ## <font color='purple'>Great! So now we have changed the Fare variable into  a categorical variable. Below are the category delineations </text>
# 
# * For fare value greater than 7.91, the class is 0.
# * For fare values between 7.91 and 14.454, the class is 1.
# * For fare values between 14.454 and 31, the class is 2.
# * For fare values greater than 31, the class is 3.
# 

# In[ ]:


train_clean.sample(10)


# ## <font color = 'purple'>The test data has also had its Fare values converted into categories instead of a range of numbers.</font>

# In[ ]:


test_clean = test_try 


# In[ ]:


test_clean.sample(10)


# # <font color ='purple'> LET'S DO SOME CORRELATION TESTING </font>
# 
# ## <font color ='green'>Since our main question revolves around whether people survived or not, the correlations will be considered with regards to their relation to the Survived column.
#     
#   <font color='red'>**OBSERVATIONS**  </font>
#   
# 
# 
# * Obviously Survived has the strongest correlation with itself.
# * The correlation values can be anywhere from -1 to 1. 
# * The magnitude of the correlation tells us of how strongly the variables are related.
# * A negative sign implies inverse relationship.
# * **Based on the following results, the strongest relationship that 'Survived' has is with Pclass**
# 
# **These variables share the strongest correlation with Survived.**
# 
# *  PClass = -0.335549
# *  Fare     =   0.295875
# * Parch    =   0.083151
# 
# Since Survived is so strongly defined by Pclass, let's explore that. .
# 
# 
# 
#   
#   **Correction: I saw that Sex wasn't included in the correlation, so I looked and had to encode into numerical data from its categorical origins**
#  * Sex = 0.541585
# ## <font color='Red'>I will be using the above mentioned Four variables to construct a machine learning model</font>
# 

# In[ ]:


# pd.DataFrame(train_clean.corr()['Survived']).reset_index()
train_clean.corr()['Survived']


# ## <font color ='Mauve'>Now you might notice that Sex, Ticket and Embarked are missing from the correlation table printed above. That is because they are all categorical values. Correlation is based on the idea of how one variable changes in regards to change in another variables. What happens to variables *a* when variable *b* increases or decreases. Since Sex, Ticket and Embarked not numerical values, we can't generate this correlation.</font>
# 
# # So we will have to convert them into categorical values.
# **In this case, the categories wouuld be as follows:**
# * 1= Female
# * 0= Male

# In[ ]:


train_clean['Sex'] =  train_clean['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_clean['Sex'] = test_clean['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


test_clean.sample(5)
# train_clean.sample(5)


# In[ ]:


train_clean.sample(5)
# train_clean.corr()['Survived']


# # <font color='Purple'> Let's make the heat map of these correlations </font>

# In[ ]:


sns.heatmap(train_clean.corr(),cmap='RdYlGn')
fig = plt.gcf()
plt.show()


# # <font color = 'green'> LET'S DO SOME VISUALIZATION FOR THE TRAINING DATA </font>

# In[ ]:


# Passenger class with survived. 
class_surv = pd.DataFrame(train_clean[['Pclass','Survived']]).groupby('Pclass', as_index=False).count()
plt.hist(class_surv['Pclass'], weights=class_surv['Survived'])
plt.xticks(np.arange(1,3))
plt.grid()
plt.title('People survived based on their Passenger class')


# ## <font color='green'> Let's take a look at the test data set and see how men fared against women </font>
# **Taking a look at the data below, we can see that  **
# 
# * Women had a mean age of 28.077 years 
# * Men had mean age of 30.506 years

# In[ ]:


train_clean.sample(5)
train_clean[["Survived", "Sex"]].groupby(['Sex'], as_index=False).count()#.mean().sort_values(by='Survived',ascending=False)
train_clean[["Survived", "Sex"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)


# # <font color ='purple'> LET'S GET READY TO PREPARE THIS DATA FOR JACK </font>
# 

# ## <font color ='purple'> Since we saw that the 4 variables which are most strongly correlated with Survival are <font color='red'>Pclass, Fare and Parch, Sex</font> so let's construct a dataFrame to deal with that. Oh and I  am adding in age as well just because why not</font>
# 
# ## <font color ='red'>Below, I construct a data Frame with only the variables which I am interested in exploring further. </font>
# 

# In[ ]:


jack_df = train_clean[["Pclass","Sex","Age", "Parch","Fare", "Survived"]]


# In[ ]:


jack_df.sample(5)


# In[ ]:


test_clean = test_clean[["Pclass","Sex","Age", "Parch","Fare"]]


# In[ ]:


test_clean.sample(5)


# # <font color ='Purple'>I am going to build some test data for Jack that I have been able to find over the Internet related to these variables</font>
# 
# **How did I collect this data??**
# 
# This thing called the Internet!!! LOL
# 
# No, really this is how!
# 
# **Pclass:** At 0:47 mark of this video, you can see that the tickets which Jack won were 3rd class tickets. 
# 
#                     https://www.youtube.com/watch?v=k2p_5FHMONU
# 
# **Sex:** Obviously!!!
# 
# **Age:** Per Wiki and James Cameron himself, Jack was 20 years old at the time of events of Titanic. 
#                 http://jamescameronstitanic.wikia.com/wiki/Jack_Dawson
#                 
# **Parch:** He had no children or parents with him.
# 
# **Fare**: The way we engineered our Fares into quartiles, his **Pclass** would make his **Fare** the categorical value of **0**
# 
# <font color ='red'>**Observation**</font>
# 
# *  I am trying to build a datFrame from a dictionary. This is an interesting way of data entry and something I will use in my future kernels
# 
# 

# In[ ]:


# jack_ = {'Pclass':3, 'Sex':0, 'Age': 20, 'Parch':0, 'Fare':0}
jack_ = {'Pclass':[3], 'Sex':[0], 'Age': [20], 'Parch':[0], 'Fare':[0]}
jack_test= pd.DataFrame(jack_, columns=['Pclass','Sex','Age','Parch','Fare'])
jack_test


# #  <font color = 'purple'>We are going to use different machine learning models that deal with classification, and see how they compare on this problem.</font>
# **The algorithms and their scores are listed below**
# 
# * Logistic Regressions**(78.74)** 
# * KNN**(85.04)**
# * Support Vector Classifiers**(86.61)**
# * Decision Tree Classifiers**(93.81)**
# 
# ## <font color = 'green'>All of these algorithms predict that based on the data I provided for Jack, he would have died</font>
# 

# # <font color ='purple'> Let's set up test/train for the dataset and import the machine learning libraries</font>

# In[ ]:


## MACHINE LEARNING LIBRARIES 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# Let's set up data
# all_features.head(5)
x_train = jack_df.drop('Survived', axis=1)
y_train = jack_df[["Survived"]]
# x_test  = test_clean.drop("PassengerId", axis=1).copy()
x_test = test_clean.copy()
x_train.shape,y_train.shape, x_test.shape#,  y_test.shape


# In[ ]:


x_train.head(5)


# In[ ]:


x_test.head(5)


# In[ ]:


y_train.head()


# In[ ]:


def jack_output(jack_p):
    for i in jack_p:
        print(jack_p[i])
        if jack_p == 0:
            result = "JACK DIDN'T SURVIVE :'(" 
        else:
            result = 'JACK SURVIVED!!!!'
    return result 


# ## LOGISTIC REGRESSION
# 
# **Basic intro for Logistic Regression**
# * Logistic Regression is used binary classification problems.
# * Based on Sigmoid function which classifies a new point as class A or B depending on whether it is above or below a certain threshold value X.
# * This serves as a powerful way of classifying binary problems becuase it only cares about whether a new value is above or below a certain threshold. 
# * Hence, I used this algorithm for this problem
# ![](http://cdn-images-1.medium.com/max/800/1*RqXFpiNGwdiKBWyLJc_E7g.png)
# 
# 

# In[ ]:


# Logistic Regression 
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
jack_log = clf.predict(jack_test)
jack_logVal = jack_output(jack_log)
log_score = round(clf.score(x_train,y_train)*100,2)
print('\n Per Logistic Regression ', jack_logVal,'\n')
print('Logistic Regression score: ',log_score)


# # KNN 
# **Basic Intro for KNN**
#  * KNN is another classification algorithm.
#  * It works by using clusters as the delineation for the known and unknown datasets. 
#  * It, first of all, starts by creating random clusters(you can choose clusters based on meta knowledge of the dataset).
#  * Now, when a new datapoint comes in, it sees what is the closest cluster to the data point. 
#  * Once that is done, it recalculates the clusters centers adn recalculates the repeats the process until the best possible classification has been found.

# In[ ]:


# K Nearest Neighbors 
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
jack_nn = knn.predict(jack_test)
jack_knn_out = jack_output(jack_nn)
k_score = round(knn.score(x_train, y_train) * 100, 2)
print('\nPer KNN ',jack_knn_out,'\n')
print('The score for KNN is: ',k_score)
# print(y_pred)


# # Support Vector Classifiers
# 
# **Basic Intro to SVC**
# 
# * Once considered the ultimate machine learning algorithm.
# * It creates classification by seperating the classes best  by using a hyperplane to seperate them.

# In[ ]:


# Support Vector Classifiers
svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
jack_svc = svc.predict(jack_test)
jack_svc_out = jack_output(jack_svc)
svc_score = round(svc.score(x_train,y_train)*100,2)
print('Per SVC ', jack_svc_out)
print('The score for SVC is: ', svc_score)


# # Decision Tree
# 
# **Basic Intro to  Decision Tree**
# 
# * Classifies an unknown data point by slicing data across the dimensions. 
# * it continues this until there is an optimal level of classification achieved or the some preset conditionn is met. 
# 
# ![](http://cdn-images-1.medium.com/max/800/1*1CchuZc1nLM3B60zS7A1yw.png)

# In[ ]:


# Decision Tree Classifiers

tree= DecisionTreeClassifier(max_depth = 100, random_state = 42)
tree.fit(x_train,y_train)
tree_pred = tree.predict(x_test)
jack_tree = tree.predict(jack_test)
jack_tree_out = jack_output(jack_tree)
tree_score = round(tree.score(x_train, y_train)*100,2)
print('\nPer Decision Tree ', jack_tree_out,'\n')
print('The score for Decision Tree classifer is: ',tree_score)


# ## Let's keep track of the progress we are making with the Algorithms 
# 

# In[ ]:


algo_scores = {'Logistic Regression': [log_score], 'KNN':[k_score], 'SVC': [svc_score], 'Decision Tree':[tree_score]}
algo_df = (pd.DataFrame(algo_scores, columns = list(algo_scores.keys())).T)
algo_perform = algo_df.reset_index()
algo_perform.columns = ['Algorithm', 'Score']
algo_perform


# In[ ]:


plt.plot(algo_perform['Algorithm'],algo_perform['Score'])
plt.title('The performance of the algorithms')
plt.grid()



# # <font color='red'> Conclusion</font>
# * If Jack existed in real life, these models predict that he would have died more likely than not and not from hypothermia because he Rose wasn't willing to take turns on that make shift raft.
# * Based on the fact that, Jack had 
#     * Third class tickets.
#     * 20 years old male.
#     * Didn't have any parents or children with him.
#     * Had no siblings with him. 
#     
#     ## HE WOULD HAVE MORE LIKELY THAN NOT  DIED. 
#     
