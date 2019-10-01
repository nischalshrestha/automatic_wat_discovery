#!/usr/bin/env python
# coding: utf-8

# Here is simple introduction of Data Science via Python, will cover the following main work steps*, include
# 1. CSV file read
# 2. First galance of data
# 3. Handel Missing data
# 4. Encoding categorical data and create dummies vairable
# 5. Splitting the dataset into the Training set and Test set
# 6. Feature Scaling
# 7. Cross validation
# 
# Classfication Model will cover
# 1. Logistic Regression
# 2. Kernel SVM
# 3. Random Forest Classification 
# 4. Artificial neural network  (ANN, Deep Learnning)
# 
# 
# 

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML


# In[2]:


# Function Tools
def group_by_mean(dataset,group, value):
    result=dataset[[group,value]].groupby([group],as_index=False).mean()    .sort_values(by=value,ascending=False)
    display(result)


# In[3]:


# Importing the dataset
dataset = pd.read_csv('../input/train.csv')
dataset_t = pd.read_csv('../input/test.csv')


# In[4]:


dataset.head(10)


# **Data Dictionary**
# 
# Variable	Definition	Key
# survival	 Survival	0 = No, 1 = Yes
# pclass	    Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	           Sex	
# Age	          Age in years	
# sibsp	     # of siblings / spouses aboard the Titanic	
# parch	     # of parents / children aboard the Titanic	
# ticket	     Ticket number	
# fare	       Passenger fare	
# cabin	     Cabin number	
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
#  
#  Variable Notes
# 
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[5]:


#drop the unlessary column, don't select to prevent chain update
dataset=dataset.drop(["PassengerId","Ticket","Cabin"], axis=1)
dataset_t=dataset_t.drop(["PassengerId","Ticket","Cabin"], axis=1)


# **Overview the Data**

# In[6]:


dataset.describe(include="all")
#only 38% people survive
#Age，Embarked have missing value
#Age have a lot missing value(13%) and can use name to predict
# there are some infant or child
# Most people abord from S Southampton


# In[7]:


dataset.hist(bins=50, figsize=(14,8))


# In[8]:


dataset.corr()
# Survived is highly related to Pclass and Fare


# Scanning the feature relationship with Survival rate

# In[9]:


from pandas.plotting import scatter_matrix
attributes=["Survived","Pclass","Age","Fare"]
scatter_matrix(dataset[attributes],figsize=(14,8))


# In[10]:


# Feature Placss Sex, they both have signicant impact
group_by_mean(dataset,"Pclass","Survived")
sns.barplot(x="Sex",y="Survived", hue="Pclass", data=dataset)


# In[11]:


# Age is not very significant, 
group_by_mean(dataset,"Survived","Age")
sns.swarmplot(x="Survived", y="Age", data=dataset)


# In[12]:


sns.boxplot(x="Survived", y="Age", data=dataset)


# In[13]:


# for continue value, Age can be split to differnt group latre
sns.distplot(dataset["Age"].dropna())


# In[14]:


# few VIP (very high fare) seems have better survived rate
group_by_mean(dataset,"Survived","Fare")
sns.distplot(dataset["Fare"].dropna())


# In[15]:


sns.barplot(x="Survived",y="Fare", hue="Pclass", data=dataset)


# In[16]:


sns.boxplot(x="Pclass", y="Fare", data=dataset)
#sns.swarmplot(x="Survived", hue="Fare", data=dataset)


# In[17]:


# Embarked
sns.barplot(x="Embarked", y="Survived", data=dataset)


# **Taking care of missing data **

# In[18]:


dataset.info()
#Age and Embarked have missing values
#For Age, the name is a good predictor


# In[19]:


#Age and Embarked have missing values
#For Age, the name is a good predictor
def handel_missing(dataset):
    dataset["Title"]=dataset.Name.str.extract("([A-Za-z]+)\.", expand=False)
    pd.crosstab(dataset["Title"],dataset["Sex"])
    
    dataset["Title"]=dataset["Title"].replace(["Lady","Countess","Capt","Col",
            "Don","Dr","Major","Rev","Sir","Jonkheer"],"Rare")
    dataset["Title"] = dataset["Title"].replace("Mlle","Miss")
    dataset["Title"] = dataset["Title"].replace("Ms","Miss")
    dataset["Title"] = dataset["Title"].replace("Mme","Mrs")
    
    group_by_mean(dataset,"Title","Age")
    sns.boxplot(x="Title", y="Age", data=dataset)
    
    dataset["Age"].fillna(dataset.groupby("Title")["Age"].transform("mean"),inplace= True)
    dataset=dataset.drop(["Title","Name"], axis=1)
    dataset.info()
    return dataset

#for trainning dataset
dataset=handel_missing(dataset)
dataset["Embarked"].fillna(dataset["Embarked"].dropna().mode()[0], inplace=True)    
dataset.info()

#for testing dataset
dataset_t=handel_missing(dataset_t)
dataset_t["Fare"].fillna(dataset_t["Fare"].dropna().mode()[0], inplace=True)    


# **Feature engineering**

# In[20]:


def featureengineer(dataset):
    dataset["FamilySize"] = dataset["SibSp"] + dataset['Parch'] +1
    dataset=dataset.drop(["SibSp","Parch"],axis=1)
    
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"]==1,"IsAlone"]=1

    #Age bin
    age_bin = [0,10,18,40,100]
    agegroup_name = ["Child","Yong","Adult","Old"]
    dataset["AgeBin"] = pd.cut(dataset["Age"],age_bin,labels=agegroup_name)
    dataset=dataset.drop(["Age"], axis=1)
    #Fare bin
    fare_bin = [-1,80,600]
    faregroup_name = ["Normal","VIP"]
    dataset["FareBin"] = pd.cut(dataset["Fare"],fare_bin,labels=faregroup_name)
    dataset=dataset.drop(["Fare"], axis=1)
    return dataset
    
dataset=featureengineer(dataset)
dataset_t=featureengineer(dataset_t)


# In[21]:


dataset.head(10)


# **Encoding categorical data and create dummies vairable**

# In[22]:


# Encoding the Independent Variable

def encode(dataset):
    dataset = pd.get_dummies(dataset,columns=["Pclass","Sex","Embarked","AgeBin","FareBin"], drop_first = True)
    #dataset=dataset.drop(["Embarked_Q","FareBin_expenceive","FareBin_normal"], axis=1)
    return dataset
dataset=encode(dataset)
dataset_t=encode(dataset_t)

dataset.info()
dataset_t.info()


# In[23]:



# Seprate X and y
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

X_t=dataset_t.values


# **Feature Scaling**

# In[24]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#X_test = sc.transform(X_test)
X_t = sc.transform(X_t)


# **Cross validation**

# In[25]:


from sklearn.cross_validation import cross_val_score
model_score = pd.DataFrame(columns=["Model","Score","Score Variation"])
def model_score_add(modelname,model):
    scores = cross_val_score(model,X,y,cv=5)
    name=modelname
    score=scores.mean()
    score_std = scores.std()
    model_score.loc[len(model_score)]=[modelname,score,score_std]


# **Model Select**

# In[26]:


#---------------------------------------------------------
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_logistic = LogisticRegression(random_state = 0)
model_score_add("Logistics",classifier_logistic)



# In[27]:


classifier_logistic.fit(X,y)
coeff_df = pd.DataFrame(dataset.columns.delete(0))
coeff_df.columns = ["Feature"]
coeff_df["Correlation"] = pd.Series(classifier_logistic.coef_[0])
coeff_df.sort_values(by="Correlation",ascending=False)

coeff_df


# In[28]:


#---------------------------------------------------------
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_svck = SVC(kernel = 'rbf')
model_score_add("Kernel SVM",classifier_svck)

#---------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
model_score_add("Naive Bayes",classifier_nb)

#--------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
classifier_dc = DecisionTreeClassifier(criterion = 'entropy')
model_score_add("Decision Tree",classifier_dc)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_rfc = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
model_score_add("Random Forest Classification",classifier_rfc)


# In[29]:


#--------------------------------------------------------
# Artificial Neural Network
# Importing the Keras libraries and packages
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
import keras

def build_classifier():
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    # Adding the second thrid hidden layer
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier_ann = KerasClassifier(build_fn = build_classifier, batch_size = 5, epochs = 50)

model_score_add("Artificial Neural Network",classifier_ann)


# In[30]:


model_score


# In[31]:


# Kernel SVM is better


# In[32]:


classifier_svck.fit(X,y)
y_submit = classifier_svck.predict(X_t)
#y_submit = (y_submit > 0.5)
#y_submit = y_submit.reshape(-1,1)
dataset_t2 = pd.read_csv('../input/test.csv')
passengerid=dataset_t2["PassengerId"]
results=pd.DataFrame({"PassengerId":passengerid,"Survived":y_submit})
results.to_csv("submission.csv",index=False)

