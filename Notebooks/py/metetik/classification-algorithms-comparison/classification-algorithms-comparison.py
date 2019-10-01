#!/usr/bin/env python
# coding: utf-8

# # Introduction<a id="1"></a> <br>
# 
# Hello everyone, this is my first competition's solution.The data is about titanic casualties.There are features about casualities in dataset.The aim is predict situation of people who don't known survive or not survive.
# 
# * [Introduction](#1)
#     * [Import Libraries](#2)
#     * [Load Data](#3)
# * [Exploratory Data Analysis](#4)
#     * [A Quik View of data](#5)
#     * [Cleaning data for Data Visualisation](#6)
#     * [Data Visualization](#7)
#         * [Line Plot](#8)
#         * [Count Plots](#9)
# 
# * [Classification](#10)
#     * [Preparing data for Classification](#11)
#     * [Implementing Classification Algorithms](#12)
#         * [Logistic Regression](#13)
#         * [K-Nearest Neighbors](#14)
#         * [Support Vector Machine](#15)
#         * [Naive Bayes](#16)
#         * [Decision Tree](#17)
#         * [Random Forest](#18)
#     * [Compare and Compound Classificaton Algorithms](#19)
#     * [Preparing Output](#20)
# 

# ## Import Libraries<a id="2"></a> <br>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #count plot

import plotly.plotly as py #plotly library
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


import os
print(os.listdir("../input"))


# **We have 3 dataset in working directory : **
# 
# * train : To train classification models.We know output.
# * test : To test classification models and create a prediction output.
# * gender_submission : A sample of output format.

# ## Load  data<a id="3"></a> <br>

# In[ ]:


data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')


# # Exploratory Data Analysis<a id="4"></a> <br>

# ## A Quik View of Data <a id="5"></a> <br>
# 
# Let's look at data quickly to understand the content.
# 

# In[ ]:


data_train.info()

print()

data_test.info()


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()

#As you see, there isn't "Survived" column in test data because it requested from us. 


# ## Cleaning Data <a id="6"></a> <br>
# 
# As you see the datasets have NaN values(this means like empty),unneeded features and object datatype.We should fix them and clean data in order to use classification algorithms.Because the algorithms don't understand 'male' or 'female' .If we transform these to mathematical form (1 and 0) the algorithms work well.

# In[ ]:


#Drop unneed columns and save

data_train.drop(["Name","Cabin","Ticket","Embarked"],axis = 1,inplace = True)

data_test.drop(["Name","Cabin","Ticket","Embarked"],axis = 1,inplace = True)


# In[ ]:


#Split dataframe into 'survived' and 'not survived' so we will use these easily at data visualization

data_survived = data_train[data_train['Survived'] == 1].sort_values('Age') #dataframe that only has datas from survived peoples 

data_not_survived = data_train[data_train['Survived'] == 0].sort_values('Age')

#We will use this serie at line plot

survived_age_number = data_survived.Age.value_counts(sort = False,dropna = True)#How many survived people are from which age

not_survived_age_number = data_not_survived.Age.value_counts(sort = False,dropna = True)

display(survived_age_number)

not_survived_age_number


# In[ ]:


#0.42,0.67 .. values at tail of serie and this is a wrong sort.Lets fix it.

a = survived_age_number.tail(4)#put values into a.

survived_age_number.drop([0.42,0.67,0.83,0.92],inplace = True)#delete these values from tail of serie

survived_age_number = pd.concat([a,survived_age_number],axis=0)#attach a to head of serie

survived_age_number #Done


# ## Data Visualization<a id="7"></a> <br>
# 
# Our datas are ready for data visualization.Let's make plots to better understand data.

# ### Line Plot<a id="8"></a> <br>

# In[ ]:


#trace1 is green line and trace2 is red line.

trace1 = go.Scatter(
    x = survived_age_number.index,
    y = survived_age_number,
    opacity = 0.75,
    name = "Survived",
    mode = "lines",
    marker=dict(color = 'rgba(0, 230, 0, 0.6)'))

trace2 = go.Scatter(
    x = not_survived_age_number.index,
    y = not_survived_age_number,
    opacity=0.75,
    name = "Not Survived",
    mode = "lines",
    marker=dict(color = 'rgba(230, 0, 0, 0.6)'))

data = [trace1,trace2]
layout = go.Layout(title = 'Age of Survived and not-Survived People in Titanic',
                   xaxis=dict(title='Age'),
                   yaxis=dict( title='Count'),)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Count Plots<a id="9"></a> <br>

# In[ ]:


sns.countplot(data_survived.Pclass)
plt.title('Passenger Class of Survived People')
plt.show()


# In[ ]:


sns.countplot(data_not_survived.Pclass)
plt.title('Passenger Class of Not Survived People')
plt.show()


# In[ ]:


sns.countplot(data_survived.Sex)
plt.title('Gender of Survived People')
plt.show()


# In[ ]:


sns.countplot(data_not_survived.Sex)
plt.title('Gender of Not Survived People')
plt.show()


# # Classification<a id="10"></a> <br>

# ## Preparing data for Classification<a id="11"></a> <br>

# In[ ]:


data_train.head()


# In[ ]:


data_train_x = data_train #We should prepare x and y data for train classification

data_train_x.Sex = [1 if i == 'male' else 0 for i in data_train_x.Sex] #Transform strings to integers

data_train_y = data_train_x.Survived #y is our output  

data_train_x.drop(['PassengerId','Survived'], axis = 1,inplace = True)#drop passengerıd and survived because they will not use while training

data_train_x.fillna(0.0,inplace = True) #fill NaN values with zero.We write '0.0' because we want to fill with float values 

#normalization :  i encountered 'to make conform to or reduce to a norm or standard' definition when i search normalization on google.
#But if you ask simply definition i say that : 'to fit values between 0 and 1'
#Normalization formula : (data - min)/(max-min) 

data_train_x = (data_train_x - np.min(data_train_x))/(np.max(data_train_x) - np.min(data_train_x)).values


# In[ ]:


#We repeat same process to test dataset

data_test.Sex = [1 if i == 'male' else 0 for i in data_test.Sex]

PassengerId = data_test['PassengerId'].values

data_test.drop(['PassengerId'], axis = 1,inplace = True)

data_test.fillna(0.0,inplace = True)

data_test = (data_test - np.min(data_test))/(np.max(data_test) - np.min(data_test)).values


# In[ ]:


#Split train data in order to reserve %80 of train data for test .You don't confuse this test data is for check.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_train_x,data_train_y,test_size = 0.2,random_state=1)

score_list = [] #to keep scores of algorithms


# ## Implementing Classification Algorithms<a id="12"></a> <br>

# ### Logistic Regression<a id="13"></a> <br>

# In[ ]:


from sklearn.linear_model import LogisticRegression #importing logistic regression model

lr = LogisticRegression()

lr.fit(x_train,y_train)#fit or train data

print('Logistic Regression Score : ',lr.score(x_test,y_test))#Ratio of correct predictions

score_list.append(lr.score(x_test,y_test))


# In[ ]:


#this is our real prediction part

lr.fit(data_train_x,data_train_y)

lr_prediction = lr.predict(data_test)


# ### K-Nearest Neighbors<a id="14"></a> <br>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

print('K-Nearest Neighbors Score : ',knn.score(x_test,y_test))

score_list.append(knn.score(x_test,y_test))


# In[ ]:


knn.fit(data_train_x,data_train_y)

knn_prediction = knn.predict(data_test)


# ### Support Vector Machine<a id="15"></a> <br>

# In[ ]:


from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print('Super Vector Machine Score : ',svm.score(x_test,y_test))

score_list.append(svm.score(x_test,y_test))


# In[ ]:


svm.fit(data_train_x,data_train_y)

svm_prediction = svm.predict(data_test)


# ### Naive Bayes<a id="16"></a> <br>

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print('Naive Bayes Score : ',nb.score(x_test,y_test))

score_list.append(nb.score(x_test,y_test))


# In[ ]:


nb.fit(data_train_x,data_train_y)

nb_prediction = nb.predict(data_test)


# ### Decision Tree <a id="17"></a> <br>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print('Decision Tree Score : ',dt.score(x_test,y_test))

score_list.append(dt.score(x_test,y_test))


# In[ ]:


dt.fit(data_train_x,data_train_y)

dt_prediction = dt.predict(data_test)


# ### Random Forest <a id="18"></a> <br>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 22,random_state = 40)

rf.fit(x_train,y_train)

print('Random Forest Score : ',rf.score(x_test,y_test))

score_list.append(rf.score(x_test,y_test))


# In[ ]:


rf.fit(data_train_x,data_train_y)

rf_prediction = rf.predict(data_test)


# ## Compare and Compound Classificaton Algorithms<a id="19"></a> <br>
# 
# To determine the best predict,we will compare all predictions and select final prediction by scores.

# In[ ]:


pr_dict = {'Logistic Regression' : lr_prediction,'KNN' : knn_prediction,'SVM' : svm_prediction,
           'Naive Bayes' : nb_prediction,'Decision Tree' : dt_prediction, 'Random Forest' : rf_prediction}

all_predictions = pd.DataFrame(pr_dict)

all_predictions


# In[ ]:


final_prediction = [] #final prediction list

#i : range columns , j : range rows

for i in all_predictions.values:
    sum_zero_score = 0 #summary of zero scores
    
    sum_one_score = 0 #summary of one scores
    
    for j in range(5):
        if i[j]==0:
            sum_zero_score += score_list[j]
        else:
            sum_one_score += score_list[j]
    
    if sum_zero_score >= sum_one_score:
        final_prediction.append(0)
    else:
        final_prediction.append(1)
    


# ## Preparing Output<a id="20"></a> <br>

# In[ ]:


output = {'PassengerId' : PassengerId,'Survived' : final_prediction}

submission = pd.DataFrame(output)

submission.to_csv('output.csv', index = False)


# # The End
# 
# 
# If you see any mistake or lack please tell me with comment. Especially  mistakes about language . Thank you.  😊
