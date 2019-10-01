#!/usr/bin/env python
# coding: utf-8

# **<H1>Titanic Dataset Analysis: Visualization and Prediction<H1>**
# 
# ![Titanic](http://https://goo.gl/images/9kGf1u)
# 
# **Titanic dataset** is a very famous set for the beginners. Also, this is my first kernel at kaggle. I would like to start from a easy dataset and try to make a **comprehensive analysis** which will contain **data visualization**, **data refromation**, **supervised learning methods**, **unsupervised learning method**, even **simple deep learning model**. For each method, I am try to use **raw code** to introduce the basic idea. Then, I use exisiting **functions** from "sklearn" or others to achieve easy implementation. Also, I try to **tune the model parameters** to achieve better preformance.<br>
# 
# Apologize my poor Engilish and let's start~! 

# In[ ]:


import numpy as np # linear algebra
from numpy import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visulization
get_ipython().magic(u'matplotlib inline')

import seaborn as sns # data visulization
import missingno as msno # missing data visualization
import math # Calcuation 
from math import log

import operator # Operation
import sys
#import treePlotter # Visualization tool for decision tree
from time import time # time info

from sklearn.cross_validation import train_test_split # dataset split

from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.ensemble import BaggingClassifier # Bagging
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import linear_model 
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.linear_model import Perceptron # Perceptron
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent 
from sklearn.svm import SVC, LinearSVC # Support Vector Machine (Normal, linear)
from sklearn.naive_bayes import GaussianNB # Naive Bayes

from sklearn.cluster import KMeans # K-means

from sklearn.neural_network import MLPClassifier # Multiple Layers Perceptron

from sklearn.metrics import accuracy_score # Accuracy Calculation
from sklearn.metrics import precision_score, recall_score # calculate precision and recall
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import tensorflow as tf # Deep Learning Library
from keras.models import Sequential
from keras.layers import Dense, Activation

np.random.seed(2)

import os
print(os.listdir("../input"))


# First of the first, the dataset is **loaded**.<br>
# We can check each data and their discription.<br>
# Moreover, **the missing records** can be viusalized.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#print (train)
#print (test)
#print (train.head())
#train.describe(include="all")
#train.isnull().any()
#train.isnull().sum()
msno.matrix(train,figsize=(12,5))


# **<H1>Data Visualization<H1>**
# 
# This is the first step I would like to do. It will help me to understand the data better.
# 
# **Totally we have 12 labels:** (Most of them are discrbed in Kaggle data info. I just add some my thoughts)
# 
# * PassengerID: I feel that this is just ID number, do not relate with analysis, will throw it;
# 
# * Survival: It can be considered as results:  
# 0 = No; 1 = Yes;
# 
# * Pclass:	Ticket class, it can indicate the SES as mentioned in the data discription.
# 1 = 1st(Upper); 2 = 2nd(Middle); 3 = 3rd(Lower);
# 
# * Name: Passenger name. Do not relate with analysis, will throw it;
# 
# * Sex:	Sex. I think this lbel is enssential since the women and child have a high priority for the lifeboat boarding;	
# 
# * Age:	Age in years. I think this lbel is enssential since the women and child have a high priority for the lifeboat boarding. the problem with this label isthat, 177 records are missing;
# 
# * SibSp: This dataset defines family relations. 
# Sibling means brother, sister, stepbrother, stepsister; Spouse means husband, wife(mistresses and fiances were ignored);
# 
# * Parch: This dataset defines family relations as well, but different from the label "SibSp". 
# Parent means father and mother; Child means daughter, son, stepdaughter, stepson. if children travelled only with a nanny, therefore parch=0 for them;
# 
# * Ticket: I feel that this is just number, do not relate with analysis, will throw it;
# 
# * Fare: Passenger Fare. I do not comfirm this label is usefull or not. in my opinion, this label is related with passenger class. Normally, higher class with higher fare. So, in the next seesions, I will check the relationship between passenger class and fare. if they are positive correlation, I think this bale can be thrown as well;
# 
# * Cabin: Cabin number. I feel that this is just recoards. Moreover, most of this records are missing. So I will throw it;
# 
# * Embarkation: This dataset defines the port of embarkation. 
# C means Cherbourg; Q means Queenstown;S means Southampton;
# 
# Based on the analysis above, we will focus on the labels: Survived, Pclass, Sex, Age, SibSp, Parch. moreover, I will check labels: Fare and Embarkation first, then deciede they will be thrown or not.
# 
# OK, first, let us check how many people is **"Survived"**.
# I will use **pie chart** to visualize it.

# In[ ]:


survived = train["Survived"]
total = survived.shape[0]
result_survived = pd.value_counts(survived)
print (result_survived)

labels_survived = 'Survived', 'Dead'
size_survived = [result_survived[1]/total, result_survived[0]/total]
explode_survived = [0.1, 0]

plt.figure(figsize = (5,5))
plt.pie(size_survived, explode = explode_survived, labels = labels_survived, center = (0, 0), labeldistance=1.1, autopct='%1.2f%%', pctdistance=0.5, shadow=True)
plt.title("Survived")

plt.show()


# **Over 60%** passengers are dead in this **tragedy**. 
# 
# Next, we will check the **passenger class**. Then, I will **combine** the labels: class and survived to check the relationship between them.
# 
# I will use **pie chart** to visualize label "**Pclass**" as well. 
# <br>
# Then, I will use **bar chart** to visualize the combination. **Seabone** is another lisualization library I use.
# <br>

# In[ ]:


passenger_class = train["Pclass"]
result_class = pd.value_counts(passenger_class)

labels_class = 'Class 1', 'Class 2', 'Class 3'
size_class = [result_class[1]/total, result_class[2]/total, result_class[3]/total]
explode_class = [0.1, 0.1, 0.1]

plt.figure(figsize = (5,4.5))
plt.pie(size_class, explode = explode_class, labels = labels_class, center = (0, 0), labeldistance=1.1, autopct='%1.2f%%', pctdistance=0.5, shadow=True)
plt.title("Passenger class")

plt.show()


# In[ ]:


train[["Pclass", "Survived"]].groupby(["Pclass"]).mean().plot.bar()
sns.countplot("Pclass", hue = "Survived", data = train)

plt.show()


# Last figure indicates the relation between class and survived. **First class has the largest survived number**, followed by the third class. The difference between three classes survived amounts is not so large. It seems like the importnce of class is not high. But, one we need notice that, the people in the third class is largest.
# 
# Next , I will check label "**Sex**" by **pie chart**.

# In[ ]:


passenger_sex = train["Sex"]
result_sex = pd.value_counts(passenger_sex)
   
labels_sex = 'Male', 'Female'
size_sex = [result_sex['male']/total, result_sex['female']/total]
explode_sex = [0.1, 0]

plt.figure(figsize = (5,4.5))
plt.pie(size_sex, explode = explode_sex, labels = labels_sex, center = (0, 0), labeldistance=1.1, autopct='%1.2f%%', pctdistance=0.5, shadow=True)
plt.title("Sex")

plt.show()


# In[ ]:


train[["Sex", "Survived"]].groupby(["Sex"]).mean().plot.bar()
sns.countplot("Sex", hue = "Survived", data = train)

plt.show()


# Among the rescued people, the **female** is aound **1.5 times** more than **male**.  This indicates the truth that, women have a high priority to board the lifeboat.
# 
# I try to add the **class** info into the result. I would like to check is there the order among the class.

# In[ ]:


sns.catplot(x = "Pclass", y = "Survived", hue = "Sex", data = train, height = 5, kind = "bar")

plt.show()


# Although the number of rescued peopel is nearly the same among the three class. But, the last figure show that, **over 90%** femals are survived in **class 1 & 2**. Only **half** femals are rescued in **class 3**. So, "Class" is the important factor.
# 
# Next, I will check label "**Age**" by **bar chart**.
# 
# **Notice**: There are **177 records are "nan"**. For data visualization, I throw them first.

# In[ ]:


age = train["Age"]
result_age = pd.value_counts(age)
x = np.arange(0,90,0.1)

#age.isnull().sum()
age = age.dropna(axis = 0, how = "any") # Delete "nan" recoards
#print (age)

plt.bar(x,result_age[x])
plt.show


# It looks like a "**Gaussian Distribution**". <br>
# Let us add "**Survived**" label and **class & sex** factors.

# In[ ]:


sns.violinplot(x = "Pclass", y = "Age", hue = "Survived", split = True, inner = "quart",data = train)

plt.show()


# In[ ]:


sns.violinplot(x = "Sex", y = "Age", hue = "Survived", split = True, inner = "quart",data = train)

plt.show()


# For each plot, the age factor always look like  "**Gaussian Distribution**", which is same as the original age distribution.  The **expected value in this distribution is around 30**, which is similaer with teh original distribution as well. It seems like the childs are not send to the liftboat first.
# 
# Next, I will look into the factor "**SibSp**" and "**Parch**" since they indicate the same factor. Let's check the data by **bar chart** first.

# In[ ]:


sibsp = train["SibSp"]
result_sibsp = pd.value_counts(sibsp)
x_1 = np.arange(0,10,1)

parch = train["Parch"]
result_parch = pd.value_counts(parch)
x_2 = np.arange(0,10,1)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

sns.barplot(x_1, result_sibsp[x_1], ax = ax1)
sns.barplot(x_2, result_parch[x_2], ax = ax2)
plt.show


# Let us introduce "**Survived**" into previous figure. <br>
# First, we check the **SibSp/Survived** and **Parch/Survived** table.<br>
# Then, use **bar chart** to indicate the data.

# In[ ]:


sibsp_survived = pd.crosstab([train.SibSp],train.Survived)
print (sibsp_survived)

parch_survived = pd.crosstab([train.Parch],train.Survived)
print (parch_survived)


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), )

sns.barplot('SibSp','Survived', data=train, ax = ax1)
sns.barplot('Parch','Survived', data=train, ax = ax2)


# **Passengers who travel alone** are the largest rescued number. But, **passenger with one family member** have the highest rescued percentage. <br>
# It make sense since no matter parents or husband will give the lifeboat boarding chance to their child or wife.
# 
# Last label I want to check is "**Fare**". I think it is related with the label "Pclass". Normally, the higher class, the more fare passenger need to pay.<br>
# Let's check it is correct or not.

# In[ ]:


class_fare = pd.crosstab([train.Pclass],train.Fare)
print (class_fare)


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), )
sns.boxenplot(x = "Pclass", y = "Fare", color = "blue", scale = "linear", data = train, ax = ax1)
sns.violinplot(x = "Pclass", y = "Fare", hue = "Survived", split = True, inner = "quart",data = train, ax = ax2)
plt.show()


# Yup, "**Fare**" is related with "**Pclass**". The above left one shows the price pattern of class 1, 2 nd 3. **Class 1 has the highest price. Class 2 price is higher than class 3, but the differenceis quite small.** <br>
# 
# Till now, all the necessary data visualization is done. I think I have a understanding of this data.<br>
# Next, I will start to **build the machine learning model.**
# 

# **<H1>Data Preparation<H1>**
# 
# Before we start the prediction, we need to do the **data preparation**.<br>
# If we look back to the beginning, we found that there are **177 records** missing under label "**Age**" and **over 650 records** missing under label "**Cabin**". Moreover, what we found from the visualization is that, **some label is not useful** for the prediction. Based on this, we do the data preparation by the following steps:
# 
# * Delete the column "Cabin", as we visualize it and find it not important; 
# 
# * Convert the 'nan' to 'S' under label "Embarked". Further convert label "Embarked" as:<br>
# "S" -> "0";
# <br>
# "C" -> "1";
# <br>
# "Q' -> "2";
# 
# * Convert 'male' to '0' and 'femal' to '1' under label "Sex";
# 
# * Fill missing records under label "Age". Normally, the mediation number is used for the refill. Moreover, I consider sex and class fsctor as well.
# 
# * Split train data set into two parts: learning and evaluation.
# 

# In[ ]:


train = train.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

train.isnull().sum()


# In[ ]:


train.fillna({"Embarked":"S"},inplace=True)
train.isnull().sum()


# In[ ]:


ports = {"S": 0, "C": 1, "Q": 2}
train['Embarked'] = train['Embarked'].map(ports)

train.Embarked.describe()


# In[ ]:


genders = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(genders)

train.Sex.describe()


# In[ ]:


age_med = train.groupby(["Pclass","Sex"]).Age.median()
train.set_index(["Pclass","Sex"],inplace = True)
train.Age.fillna(age_med, inplace = True)
train.reset_index(inplace = True)

train.Age.describe()


# In[ ]:


train_test, train_eval = train_test_split(train, test_size = 0.2)

print (train_test)
print (train_eval)


# **<H1>KNN<H1>**
# 
# I try the KNN method first.
# I will follow the following steps to build the KNN model:
# 
# * Split the train dataset into two part: raw data and result list;
# 
# * Calculate [eculidean distance](http://https://en.wikipedia.org/wiki/Euclidean_distance). The distance between each test record and learning record will be calculated;
# 
# * Find the K nearest distance and return the largest value;
# 
# * Based on the largest value, generate the prediction results;
# 
# * Evaluate the results.
# 

# In[ ]:


train_test_learning = train_test.drop("Survived", axis = 1)
train_test_results = train_test["Survived"] # generate the results list

train_eval_learning = train_eval.drop("Survived", axis = 1)
train_eval_results = train_eval["Survived"] # generate the results list

print (train_test_learning, train_test_results)


# In[ ]:


# Eculidean distance calculation
def euclideanDistance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance = pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)
 
# Return K nearest distance
def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance) -1
    # Calculate test record to each train records
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
    # Sort of all distance
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    # Return K nearest value
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
 
# Merge all KNN and find the largest value
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    # Sort of the KNN
    sortedVotes = sorted(classVotes.items(),key = operator.itemgetter(1),reverse =True)
    return sortedVotes[0][0]
 
# Evaluate the model
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
    return (correct/float(len(testSet))) * 100.0

# Convert the dataframe to array
trainingSet = pd.concat([train_test_learning,train_test_results],axis=1).values
testSet = pd.concat([train_eval_learning,train_eval_results],axis=1).values

# Generate the prediction list
predictions = []

# Define K value
k = 5

#print (trainingSet)

# Main Part
for x in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print (">predicted = " + repr(result) + ",actual = " + repr(testSet[x][-1]))
accuracy = getAccuracy(testSet, predictions)
print ("Accuracy:" + repr(accuracy) + "%")


# The above is the **KNN fundmental code** I find from the internet. It helps me to undertand the KNN better.<br>
# However, the KNN algorithm is not as simply as above.<br>
# Actully,  library "**sklearn**" contains the [KNN module](http://http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). I can use it directly. <br>
# Please take care of the following parameters:
# * "n_neighbors" is the **K value**;
# 
# * "weight" is the way to consider the distance. it can be considered as equal or the nearer the more importance;
# 
# * "algorithm" contains the following 4 types:
# >* brute: heavy calculation method;
# >*  kdtree: reduce the calculation pressure. It has a good performance if the dimension is less than 20;
# >* balltree: If the dimension is higher than 20, the efficiency of KD tree is reduced. This is called "Curse of Dimensionality". Balltree is proposed to solve this problem;
# >*  auto: Module "fit" function will auto decide which method will be used;<br>
# 
# * "leaf_size" will pass to KDtree or Balltree. **It will not affect the prediction results but the calculaton speed and store space**. Normaly, the store space is **the number of samples divided leaf_size**. Also, the number of samples should be located **between one leaf_size and double leaf_size**;
# 
# * "metric" is the method to calculat distance.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 20, weights = 'uniform', algorithm = 'auto', leaf_size = 30, p = 2, metric = 'minkowski', metric_params = None, n_jobs = 1)
knn.fit(train_test_learning, train_test_results)  
eval_pred_knn = knn.predict(train_eval_learning)  
acc_knn = round(knn.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred)
#print (train_eval_results)
print (acc_knn)

accuracy_score(train_eval_results, eval_pred_knn)


# **<H1>Booststrap Aggregation (Bagging)<H1>**
# 
# **Bagging** is a **classic prediction** method. It is the fundamental for many other advanced predict algorithm. In facts, bagging is the advanced method from **bootstrap** since the sampleing idea based on it. Different from the KNN, "**class**" is ued for module implementation, so this module can be used for other advanced method. General speaking from the publications, baggins is a good method for **unstable classification**. It has a better perforamce for **high square error/low bias error model**.
# 
# The module is built by the following steps:
# * Initialization:
# 
# * Sampling: Suppose there is a **original set "S"** which contains the **n** samples. Each time, I pick a sample and put it into a **new set "Si"**, after this, the sample will be **put back to the original set**. Repeat this step, we can generate **i new set** and each set contains **j elements**. We can define hw many new set we want to generate which is parameter i; same we can define how many elements in the new set which is parameter j. If j is less than n, the sampling method is called "**Under-Sampling**"; If j is larger than n, the sampling method is called "**Over-Sampling**"; If j is equal to n, the sampling method is called "**Bootstrap**". The reference code is defined other method as well. Please note, the elements in new set **can be the same** since we put the sample back after pick it out.
# 
# * Simple model precess: For each new set, it will be processed by the simple model. Then each model have a result or output.
# 
# * Voting: For **classification purpose**, voting is a good way. The results from simple model will be summeraized to define the finnal result. However, if we want to do **regression or others**, we can choose **averaging** method or other advanced methods.
# 

# In[ ]:


class Bagging(object):
    # Initialization
    def __init__(self,n_estimators,estimator,rate=1.0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.rate = rate

    def Voting(self,data):          # Define voting method
        term = np.transpose(data)   
        result = list()            

        def Vote(df):               # vote for each raw or each simple model output
            store = defaultdict()
            for kw in df:
                store.setdefault(kw, 0)
                store[kw] += 1
            return max(store,key = store.get)

        result = map(Vote,term)      # Generate results
        return result

    # Define Under-Sampling
    def UnderSampling(self,data):
        #np.random.seed(np.random.randint(0,1000))
        data = np.array(data)
        np.random.shuffle(data)    # Personally think shuffle is important          
        newdata = data[0:int(data.shape[0] * self.rate),:]   # Define the number of elements in new set
        return newdata   

    def TrainPredict(self,train,test):          # Build simple model
        clf = self.estimator.fit(train[:,0:-1],train[:,-1])
        result = clf.predict(test[:,0:-1])
        return result

    # General sampling method
    def RepetitionRandomSampling(self,data,number):     
        sample = []
        for i in range(int(self.rate * number)):
             sample.append(data[random.randint(0,len(data)-1)])
        return sample

    def Metrics(self,predict_data,test):        # Evaluation
        score = predict_data
        recall = recall_score(test[:,-1], score, average = None)    # Recall
        precision = precision_score(test[:,-1], score, average = None)  # Precision
        return recall,precision


    def MutModel_clf(self,train,test,sample_type = "RepetitionRandomSampling"):
        print ("self.Bagging Mul_basemodel")
        result = list()
        num_estimators = len(self.estimator)   

        if sample_type == "RepetitionRandomSampling":
            print ("Sample Method：",sample_type)
            sample_function = self.RepetitionRandomSampling
        elif sample_type == "UnderSampling":
            print ("Sample Method：",sample_type)
            sample_function = self.UnderSampling 
            print ("Sampling Rate",self.rate)
        elif sample_type == "IF_SubSample":
            print ("Sample Method：",sample_type)
            sample_function = self.IF_SubSample 
            print ("Sampling Rate",(1.0-self.rate))

        for estimator in self.estimator:
            print (estimator)
            for i in range(int(self.n_estimators/num_estimators)):
                sample = np.array(sample_function(train,len(train)))       
                clf = estimator.fit(sample[:,0:-1],sample[:,-1])
                result.append(clf.predict(test[:,0:-1]))      # Summerize simple model output

        score = self.Voting(result)
        recall,precosoion = self.Metrics(score,test)
        return recall,precosoion  

train_r = Bagging(trainingSet,100,10)

print (train_r)


# Same as KNN,  sklearn provide "Bagging" function. We can use it directly. So the problem is how to set parameters. The following s are the definitions for each parameter.
# * base_estimator: basic parameter for simple model, defualt settingis "None" which means **decision tree**. Also, youcan change it to **random forest** or others;
# 
# * n_estimators: No. of estimator. Normally, **the more estimator, the lower variance**;
# 
# * max_samples: the number of parameter j. Please note, thea value an be **integer or float**.But, if you set "1" pnot "1.0" and there is only one sample in the new set, the error will occur;
# 
# * max_features: number of new set features;
# 
# * bootstrap & bootstrap_features: **replacement** samples and sample features or not;
# 
# * oob_score: "oob" stands of "**out of bag**". Whether to use out-of-bag samples to estimate the **generalization error**.
# 
# * warm_start: **Reuse** the solution of the previous call to fit and add more estimators to the ensemble if the setting is Ture;
# 
# * n_jobs: How many jobs working parally;
# 

# In[ ]:


bagging = BaggingClassifier(base_estimator = None, n_estimators = 10, max_samples = 1.0, max_features = 1.0, bootstrap = True, bootstrap_features = False, oob_score = False, warm_start = False, n_jobs = 1, random_state = None, verbose = 0)
bagging.fit(train_test_learning, train_test_results)

eval_pred_bg = bagging.predict(train_eval_learning)
acc_bg = round(bagging.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_bg)
print (acc_bg)

accuracy_score(train_eval_results, eval_pred_bg)


# **<H1>Decision Tree<H1>**
# 
# Next, let's try other method, Decision Tree. Follow the bollowing steps to build decision tree model.
# * Define **entropy** calculation;
# 
# * Calculate **infomation gain ratio** and choose the feature with largest gain ratio;
# 
# * Split the data based on the feature;
# 
# * Iterate the above steps to **generate the decision tree**;
# 
# * If it possible, **visualize** the decision tree;
# 
# * Prepare the test data set and **run** the model.
# 

# Same as above other methods, "sklearn" provide the function as well. Let us take a look at parameters.
# 
# * criterion: This determin the algorithom. For **gini impurity**,  which is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset,  simply as **CART**, the setting should be "**gini**". For **imformation gain** as I use for raw code, you can choose "**entropy**"; 
# 
# * splitter: Two options, "**best**" for best split; "**random**" for random split. If the set is not large, best is good, otherwise, please use "random";
# 
# * max_depth: How deep the tree goes to.  If the **data is not big** or the **number of featrues** is not big, can use defalut value "**None**". Otherwise, the randm value between **10-100** is better.
# 
# * min_samples_split: Defalut value is** 2**, which hs a good performance for **small data sample**. If the **set is large**, increase this value a little bit. The maximum valueI used for this parameter is **10**;
# 
# * min_samples_leaf: Same as above features, **if the data set is samll, keep default value "1"**. If the set is large, increase this value a little bit. The maximum valueI used for this parameter is **5**;
# 
# * min_weight_fraction_leaf: If we **ignore the weight issue**, set value "**0**". If not, or the** some samples have missing features**, or** the deviation of the distribution category is large**, please consider this value;
# 
# * max_features: Very important parameter, I just copy from official website, I think it is very clear;
# > The number of features to consider when looking for the best split:
# > * If **int**, then consider max_features features at each split;
# > * If **float**, then max_features is a percentage and int(max_features * n_features) features are considered at each split;
# > * If **“auto”**, then max_features=sqrt(n_features);
# > * If **“sqrt”**, then max_features=sqrt(n_features);
# > * If **“log2”**, then max_features=log2(n_features);
# > * If **None**, then max_features=n_features;
# 
# * random_state: Not use too much;
# 
# * max_leaf_nodes: For set, if the feature is not to much, keep defalut value. Otherwise, set a value will have a better performance. Use cross validation to choose the value;
# 
# * min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value;
# 
# * min_impurity_split: Threshold for early stopping in tree growth;
# 
# * class_weight: This parameter is used for **blancing the set smaples**. For example, if in the set, one type of data record is too many, so the decision tree predicion is strongly related with this type of data, which is not good. In this case, we need set as "**balance**". If not, choose "**None**";
# 
# * presort: It is related with the speed. Normally I ignore this parameter.
# 

# In[ ]:


decision_tree = DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_depth = None, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = None, random_state = None, max_leaf_nodes = None, min_impurity_decrease = 0.0, min_impurity_split = None, class_weight = None, presort = False)
decision_tree.fit(train_test_learning, train_test_results)

eval_pred_dt = decision_tree.predict(train_eval_learning)
acc_dt = round(decision_tree.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_dt)
print (acc_dt)

accuracy_score(train_eval_results, eval_pred_dt)


# **<H1>Random Forest<H1>**
# 
# If two methods, bagging and decision tree, are combined, there will be an new method, Random Forest. Tuning the parameters based on the decsion tree and bagging I mentioned above.

# In[ ]:


random_forest = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = 'auto', max_leaf_nodes = None, min_impurity_decrease = 0.0, min_impurity_split = None, bootstrap = True, oob_score = True, n_jobs = 1, random_state = None, verbose = 0, warm_start = False, class_weight = None)
random_forest.fit(train_test_learning, train_test_results)

eval_pred_rf = random_forest.predict(train_eval_learning)

acc_rf = round(random_forest.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_rf)
print (acc_rf)

accuracy_score(train_eval_results, eval_pred_rf)


# **<H1>Logistic Regression<H1>**
# 
# **Linear regression** is basic prediction method. However, the regression equation of linear regression is **linear function**. If we want to accurate regression or classification, the function may not be linear. So, **logistic regression** is introduced as advance method.
# 
# To implement the logistic regression, the following aspects are necessary:
# 
# * Regression/Classification equation: a propoer function is necessary for the classification, like **step, sigmoid**....
# 
# * Cost function: The **variation** between prediction and realistic. 
# 
# * Using **gradient decrese or other advanced method** to minimize the cost function and find **the best model parameters**.
# 

# "sklearn" function implementation.
# 
# * penalty: This parameter is related with normalization. If the purpose is **solve overfitting problem**, choose "**l2**"; if the **predict result is not good**, try "**l1**";
# 
# * dual: Normally is "false". **Only if "penalty = 'l2'" and "solve = 'liblinear'"**, choose "dual"; 
# 
# * tol: critaria on stopping;
# 
# * C: Inverse of lambda;
# 
# * fit_intercept: Normally set "True";
# 
# * intercept_scaling: **Only if "solve='liblinear'" and "fit_intercept='True'"**, this parameter affect the prediction;
# 
# * class_weight: same as desicion tree;
# 
# * random_state: **Only if "solve='sag'" or "solve='liblinear'"**, this parameter affect the prediction. Otherwise, keep it as default;
# 
# * solver: Define the method on cost function optimization:
# > * For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
# > * For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# > * ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.
# 
# * max_iter: maximum iter number, default is 10. ** Only if "solve='sag'" or "solve='newton-cg'" or "solve='lbfgs'"**, this parameter affect the prediction. 
# 
# * multi_class : Two types: "**ovr**" means "one to rest", and "**mvm**" means "many to many". "ovr" is simple and fast but perforamce is not as good as "mvm", but "mvm" is slower.
# 

# In[ ]:


logistic_regression = LogisticRegression(penalty = 'l2', dual = False, tol = 0.0001, C = 1.0, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None, solver = 'liblinear', max_iter = 100, multi_class = 'ovr', verbose = 0, warm_start = False, n_jobs = 1)
logistic_regression.fit(train_test_learning, train_test_results)


eval_pred_lr = logistic_regression.predict(train_eval_learning)

acc_lr = round(logistic_regression.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_lr)
print (acc_lr)

accuracy_score(train_eval_results, eval_pred_lr)


# **<H1>Stochastic Gradient Descent (SGD)<H1>**
# 
# One of the advanced method for logistic regression optimize the SGD. **SGD has a fast speed** when the traning set is large since it only use a part of the data to optimize the cost function. However, SGD has some shortages as **large noise**. Among them, the essential disadvantege is the **accuracy**. Since SGD only some of the data to optimize, technically SGD achieve local optima not global optima. It will sffect the accuracy. The simple raw code is as below.

# In[ ]:


def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data:
        n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] 
                for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data),n_test))
            else:
                print ("Epoch {0} complete".format(j))


# SGD implemetation by "sklearn".
# 
# 1. loss: Define the loss function. Can choose from "**hinge**", "**modified_huber**", "**log**", "**squared_loss**", "**epsilon_insensitive**" and "**huber**";
# 
# 2. penalty: same as logistic regression;
# 
# 3. alpha: penalty function parameter; 
# 
# 4. l1_ratio: The mixtrue value for "l1" and 'l2". **"0" means "l2"; "1" means "l1"**;
# 
# 5. fit_intercept: Whether the intercept should be estimated or not;
# 
# 6. max_iter: The maximum number of passes over the training data (aka epochs);
# 
# 7. epsilon: Official description as follow:
# > Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.
# 
# 8. learning_rate:
# > The learning rate schedule:
# > * ‘**constant**’: eta = eta0
# > * ‘**optimal**’: eta = 1.0 / (alpha * (t + t0)) [default]
# > * ‘**invscaling**’: eta = eta0 / pow(t, power_t);
# 
# 9.  n_iter: The number of passes over the training data (aka epochs).
# 

# In[ ]:


sgd = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 0.0001, l1_ratio = 0.15, fit_intercept = True, max_iter = None, tol = None, shuffle = True, verbose = 0, epsilon = 0.1, n_jobs = 1, random_state = None, learning_rate = 'optimal', eta0 = 0.0, power_t = 0.5, class_weight = None, warm_start = False, average = False, n_iter = None)
sgd.fit(train_test_learning, train_test_results)

eval_pred_sgd = sgd.predict(train_eval_learning)

acc_sgd = round(sgd.score(train_test_learning, train_test_results) * 100, 2)

print (acc_sgd)

accuracy_score(train_eval_results, eval_pred_sgd)


# **<H1>Perceptron<H1>**
# 
# Perceptron is the **basic** of **neural network**. It is a simplest **forward propagation network** and a **binary linear classification**. It only contains two layers: **input and output**.
# 
# Preceptron can be built based on the following steps:
# 
# 1. Define a **node** which contains two layers:input and output;
# 
# 2. Define the node **activation function**;
# 
# 3. Define an **input list** for the node;
# 
# 4. Define the **tranfer function** frominout to output,the function contains weights and bias;
# 
# 5. Define the **optimizer** for the parameter.
# 

# Preceptron can be implemented by "sklearn" as well. Most of the parameters are same as other method I introduced.

# In[ ]:


perceptron = Perceptron(penalty = None, alpha = 0.0001, fit_intercept = True, max_iter = None, tol = None, shuffle = True, verbose = 0, eta0 = 1.0, n_jobs = 1, random_state = 0, class_weight = None, warm_start = False, n_iter = None)
perceptron.fit(train_test_learning, train_test_results)


eval_pred_pp = perceptron.predict(train_eval_learning)

acc_pp = round(perceptron.score(train_test_learning, train_test_results) * 100, 2)

#print (eval_pred_pp)
print (acc_pp)

accuracy_score(train_eval_results, eval_pred_pp)


# **<H1>Linear Supported Vector Classification<H1>**
# 
# SVM is a binary classification method. Simply speaking, **SVM is to build a hyperplane or a set of hyperplane to achieve classifiction or regression**. Make sure the distance between the hyperplane bounary and each point as large as possible. The basic one is linear SVC, which means the linear function is used for hyperplane implementation.

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(train_test_learning, train_test_results)

eval_pred_liner_svc = linear_svc.predict(train_eval_learning)

acc_linear_svc = round(linear_svc.score(train_test_learning, train_test_results) * 100, 2)

print (acc_linear_svc)

accuracy_score(train_eval_results, eval_pred_liner_svc)


# **<H1>Supported Vector Classification<H1>**
# 
# Extend the linear SVC to general method. SVC is proposed.
# 
# The **differences** betweem lineasr SVC and SVC are as follows:
# 
# 1. Linear SVC is to minimize the square of "hinge loss". SVC is to minimize "hinge loss";
# 
# 2. Linear SVC use "one to rest", but SVC use "one to one";
# 
# 3. SVC can choose kernel function, but linear SVC can not;
# 
# 4. Linear SVC can choose penalty function, but SVC can not.
# 

# In[ ]:


svc = SVC(C = 1.0, kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, shrinking = True, probability = False, tol = 0.001, cache_size = 200, class_weight = None, verbose = False, max_iter = -1, decision_function_shape = 'ovr', random_state = None)
svc.fit(train_test_learning, train_test_results)

eval_pred_svc = svc.predict(train_eval_learning)

acc_svc = round(svc.score(train_test_learning, train_test_results) * 100, 2)

print (acc_svc)

accuracy_score(train_eval_results, eval_pred_svc)


# **<H1>Naive Bayes<H1>**
# 
# Naive bayes is a very simple predict method. It is based on **Bayes' theorem**, which is a famous theorem in probability theory. 
# 
# Simple speaking, the idea of naive bayes is **calcuate all the potential prediction resluts probability based on the bayes' theorem, then choose the result with highest probability**. It is a good method for text mining.

# In[ ]:


gaussian_naive_bayes = GaussianNB(priors = None)
gaussian_naive_bayes.fit(train_test_learning, train_test_results)

eval_pred_gnb = gaussian_naive_bayes.predict(train_eval_learning)

acc_gnb = round(gaussian_naive_bayes.score(train_test_learning, train_test_results) * 100, 2)

print (acc_gnb)

accuracy_score(train_eval_results, eval_pred_gnb)


# **<H1>K-Means<H1>**
# 
# All the above prediction methods are **supervised learning**. Let us take a look at **unsupervised learning** one.
# 
# K-means is a typical unsupervised learning method.  It is suitablefor **text mining** as well. The followings are the steps of K-means:
# 
# 1. Define the number of clustering center, K. Randomly choose the clustering center;
# 
# 2. Calculate the distance between each point and each clustering center;
# 
# 3. Based on the distance, implemnt K clustering;
# 
# 4. Calculate the clustering center again;
# 
# 5. Repeat step 1-3, tilll the center is not change.
# 
# Based on the steps, it is obvious to see that the essential parameters is **number of clustering** and clustering center.

# In[ ]:


def kmeans(data,k=2):
    def _distance(p1,p2):
        """
        Return Eclud distance between two points.
        p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
        """
        tmp = np.sum((p1-p2)**2)
        return np.sqrt(tmp)
    def _rand_center(data,k):
        """Generate k center within the range of data set."""
        n = data.shape[1] # features
        centroids = np.zeros((k,n)) # init with (0,0)....
        for i in range(n):
            dmin, dmax = np.min(data[:,i]), np.max(data[:,i])
            centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(k)
        return centroids
    
    def _converged(centroids1, centroids2):
        
        # if centroids not changed, we say 'converged'
         set1 = set([tuple(c) for c in centroids1])
         set2 = set([tuple(c) for c in centroids2])
         return (set1 == set2)
        
    
    n = data.shape[0] # number of entries
    centroids = _rand_center(data,k)
    label = np.zeros(n,dtype=np.int) # track the nearest centroid
    assement = np.zeros(n) # for the assement of our model
    converged = False
    
    while not converged:
        old_centroids = np.copy(centroids)
        for i in range(n):
            # determine the nearest centroid and track it with label
            min_dist, min_index = np.inf, -1
            for j in range(k):
                dist = _distance(data[i],centroids[j])
                if dist < min_dist:
                    min_dist, min_index = dist, j
                    label[i] = j
            assement[i] = _distance(data[i],centroids[label[i]])**2
        
        # update centroid
        for m in range(k):
            centroids[m] = np.mean(data[label==m],axis=0)
        converged = _converged(old_centroids,centroids)    
    return centroids, label, np.sum(assement)


# "sklearn" has the k-means function as well.
# 
# 1. n_clusters: Number of clustering;
# 
# 2. init: The method of choosing initial clustering center;
# 
# 3. n_init: Number of time the k-means algorithm will be run with different centroid seeds, which is related with initial clustering center;
# 
# 4. precompute_distances：Need pre-calculate the distance or not. If it is "True", model will put the whole matrix in ram; if it is "auto", model willl choose "false" if n_samples * n_clusters > 12 million; if it is "false",the core algorithm is "Cpython";
# 
# 5. copy_x: When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True, then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean;
# 
# 6. algorithm: K-means algorithm.
# 

# In[ ]:


#Trainset = train_test_learning.values

#print (Trainset)

kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init = 10, max_iter = 300, tol = 0.0001, precompute_distances = 'auto', verbose = 0, random_state = None, copy_x = True, n_jobs = 1, algorithm = 'auto')
kmeans.fit_predict(train_test_learning)
label_pred = kmeans.labels_
centroids = kmeans.cluster_centers_ # Clustering center
inertia = kmeans.inertia_ # Clustering inertia summary

#print (label_pred)
print (centroids)
print (inertia)

acc_k = accuracy_score(train_test_results, label_pred)*100

print (acc_k)


# **<H1>Neural Network (N.N)<H2>**
# 
# Neural network, even deep learning, is widely used technology. It **simulate the human brain** working, especilly in some specify area which the conventional method is not good at, as **computer vision**, **speed recognition**...
# 
# The N.N contains **three components**:
# 1. Topology, weights and biases: These parameters define the N.N **architechture**;
# 
# 2. Activity rule: It defines the how to activate the nodes (neurons) and transfer the info between each nodes (neuons);
# 
# 3. Learning rule or propagation rule: This is very important in the N.N. It defines the how to optimize the model parameters to achieve the best performance. If the activitry rule is considered as short-term dynamic rule, the propogation can be considered as a long-term dymanic rule.
# 
# To implement the raw N.N model, the above three components are necessary. Strongly recommend the ccousera course to build the N.N, even deep learning model step by step. It is worth to do it.

# "sklearn" provide the N.N function as well. As menthioned, perceptron is the basic of N.N, so the "sklearn" model is called **multi-layer perceptron (MLP)**. However, "sklearn" model is not intend to **handle big data**. 
# 
# 1. hidden_layer_sizes: Define the hidden laye size, in the other words, it defines the topology. For example, (10, 5) means there are two hidden layers, the first hidden layer contains 10 neurons, the second hidden layer contains 5 neurons;
# 
# 2. activation: Define activation function. Can only choose **"identity", "logistic" (sigmoid), "tanh" and "relu";**
# 
# 3. solver: Optimization method. Can choose **"lbfgs", "sgd" and "adam"**;
# 
# 4. alpha: Normalize parameter;
# 
# 5. batch_size: Related with mini-batch;
# 
# 6. learning_rate & learning_rate_init: Define the learning rate;
# 
# 7. power_t: The exponent for inverse scaling learning rate;
# 
# 8. max_iter: Maximum number of iterations;
# 
# 9. shuffle: Whether to shuffle samples in each iteration;
# 
# 10. momentum: Momentum for gradient descent update;
# 
# 11. nesterovs_momentum: Whether to use Nesterov’s momentum.
# 

# In[ ]:


mlp = MLPClassifier(hidden_layer_sizes = (50, ), activation = 'relu', solver = 'adam', alpha = 0.0001, batch_size = 'auto', learning_rate = 'constant', learning_rate_init = 0.001, power_t = 0.5, max_iter = 200, shuffle = True, random_state = None, tol = 0.0001, verbose = False, warm_start = False, momentum = 0.9, nesterovs_momentum = True, early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
mlp.fit(train_test_learning, train_test_results)

eval_pred_mlp = mlp.predict(train_eval_learning)

acc_mlp = round(mlp.score(train_test_learning, train_test_results) * 100, 2)

print (acc_mlp)

accuracy_score(train_eval_results, eval_pred_mlp)


# Besides the "sklearn", there are some other library can be used for N.N , even deep learning, model built. **Keras** is one of them. The following is the keras N.N model. I add come comments in the code to explain the code meaning.

# In[ ]:


start = time() # use "time" function to calculate the model process time

model = Sequential() # very improtent, it defines the model is built one layer by one layer
model.add(Dense(input_dim=7, output_dim=1)) # .add means add a layer into model; dense is the layer I added, dense layer is fully connected layer
model.add(Activation("relu")) # add activation function, I choose "relu" for classification
# this is a single layer model, it is a simple one. If need, you can add more layers by using code .add
# take care, before you built the model, it is better to have a whole model topology 
# after this, we define the model topology

# next, we need activate model by using code .complie
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# we define the optimizer, loss function
# optimize can be custimize first, than add into the complie function
# metrics is used for evaluate the model, can put accuracy, score or cost into it

# train the model by using .fit
model.fit(train_test_learning, train_test_results)
# train data, train label
# can add the number of epoch and batch size as well.

loss, accuracy = model.evaluate(train_test_learning, train_test_results)
acc_nn = 100*accuracy

print (loss, accuracy)
print ('\ntime taken %s seconds' % str(time() - start))

# predic the test data
dp1_pred = model.predict_classes(train_eval_learning)
#print (dp1_pred)
#print (train_eval_results)
print ("\n\naccuracy", np.sum(dp1_pred == train_eval_results.values) / float(len(train_eval_results.values)))


# Till now, I try different types of prediction method to predict survived person. But, which method is the best?
# 
# Let us create a final results to check which one is the best.
# 

# In[ ]:


comparesion = pd.DataFrame({
    'Model': ['KNN', 'bagging', 'Decision Tree', 'Random Forest', 'Logistic Regression', 
              'Stochastic Gradient Decent', 'Perceptron', 'Linear Support Vector Machines', 
              'Support Vector Machines', 'Naive Bayes', 'K-Means', 'N.N(sklearn)', 'N.N(keras)'],
    'Score': [acc_knn, acc_bg, acc_dt, acc_rf, acc_lr, acc_sgd, acc_pp, acc_linear_svc, acc_svc,
              acc_gnb, acc_k, acc_mlp, acc_nn
              ]})
comparesion_df = comparesion.sort_values(by='Score', ascending=False)
comparesion_df = comparesion_df.set_index('Score')
comparesion_df.head(14)


# Based on the above table, it is obvious to find that, **random forest** and **decision tree** has the best performance. Because of the** data set limitation**, unsupervised learning and netural network do not good performance.

# For random forest, draw **precision & recall curve** and **ROC**

# In[ ]:


predictions = cross_val_predict(random_forest, train_test_learning, train_test_results, cv=3)
print (precision_score(train_test_results, predictions), recall_score(train_test_results, predictions))


# In[ ]:


y_scores = random_forest.predict_proba(train_test_learning)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(train_test_results, y_scores)

def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(10, 5))
plot_precision_vs_recall(precision, recall)
plt.show()


# In[ ]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(train_test_results, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(10, 5))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# In[ ]:


final = roc_auc_score(train_test_results, y_scores)

print (final)


# ROC score is very important value for performance evaluation. The more cvlose to 1, the better performance. Now, we can find that the random forest value is over 99%, which is good for the prefdiction.

# Last step is to predict the test data set and submission.

# In[ ]:


test_mod = test.drop(columns = ['Name', 'Ticket', 'Cabin'])

age_med_test = test_mod.groupby(["Pclass","Sex"]).Age.median()
test_mod.set_index(["Pclass","Sex"],inplace = True)
test_mod.Age.fillna(age_med_test, inplace = True)
test_mod.reset_index(inplace = True)

fare_med_test = test_mod.groupby(["Pclass"]).Fare.median()
test_mod.set_index(["Pclass"],inplace = True)
test_mod.Fare.fillna(fare_med_test, inplace = True)
test_mod.reset_index(inplace = True)

test_mod['Embarked'] = test_mod['Embarked'].map(ports)
test_mod['Sex'] = test_mod['Sex'].map(genders)

test_mod.isnull().sum()

test_mod_pred = test_mod.drop("PassengerId", axis = 1)
test_mod_id = test_mod["PassengerId"] # generate the results list

pred = random_forest.predict(test_mod_pred)

print (pred)

submission = pd.DataFrame({
        "PassengerId": test_mod_id,
        "Survived": pred
    })

submission.to_csv("submission.csv",index=False)

