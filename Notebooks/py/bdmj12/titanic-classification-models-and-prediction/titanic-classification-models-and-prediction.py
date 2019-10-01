#!/usr/bin/env python
# coding: utf-8

# This kernel explores the Titanic dataset, trains several models and outputs a prediction.

# ## Imports

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().magic(u'matplotlib inline')


# ## Get and Clean Data

# In[ ]:


folder ="../input/"
data = pd.read_csv(folder + "train.csv")
data.head()


# The following features are not numerical, so we drop them (for now).

# In[ ]:


clean_data = data.drop(["PassengerId","Name", "Ticket","Cabin"], axis=1)


# Lets encode the sex category as a numerical label:

# In[ ]:


clean_data["Sex"] = clean_data["Sex"].map({"female": 1, "male": 0})


# In[ ]:


clean_data.head()


# In[ ]:


clean_data.info()


# We see that there are missing values for Age and Embarked. Lets deal with Embarked first:

# In[ ]:


clean_data["Embarked"].value_counts()


# There are considerably more 'S' instances than the rest, so lets fill in the 2 missing values with that.

# In[ ]:


for i in range(len(clean_data["Embarked"])):

    if clean_data["Embarked"][i]!="S":
          if clean_data["Embarked"][i]!="Q":
            if clean_data["Embarked"][i]!="C":
                #change the null entries:
                clean_data["Embarked"][i]= "S"
    


# Now lets change these entries to categorical data. We will create 3 new columns, the reason being that there is no numeric connection between the categories (i.e. if (S,Q,C) = (0,1,2), the computer may think that S is closer to Q than C, which might not be true).

# In[ ]:


em_encoded,em_cats= clean_data["Embarked"].factorize()


# In[ ]:


clean_data[em_cats]=pd.get_dummies(em_encoded)


# In[ ]:


clean_data = clean_data.drop("Embarked",axis=1)


# Now lets impute the missing age values with the mean.

# In[ ]:


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean')


# In[ ]:


imp.fit(clean_data)


# In[ ]:


imp_data =pd.DataFrame(imp.transform(clean_data.values))
imp_data.columns = clean_data.columns
clean_data = imp_data


# In[ ]:


clean_data.head()


# In[ ]:


clean_data.info()


# Lets scale certain columns to improve effectiveness of certain ML models.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_data=clean_data

scaled_data[["Fare","Age"]] = scaler.fit_transform(scaled_data[["Fare","Age"]])


# In[ ]:


clean_data = scaled_data


# Lets combine all the preprocessing into a single function, in order to apply it to the test set:

# In[ ]:


def cleanup_data(data):
    data = data.drop(["PassengerId","Name", "Ticket","Cabin"], axis=1)
    data["Sex"] = data["Sex"].map({"female": 1, "male": 0})
    
    for i in range(len(data["Embarked"])):

        if data["Embarked"][i]!="S":
              if data["Embarked"][i]!="Q":
                if data["Embarked"][i]!="C":
                    #change the null entries:
                    data["Embarked"][i]= "S"
                
    em_encoded,em_cats= data["Embarked"].factorize()
    data[em_cats]=pd.get_dummies(em_encoded)
    data = data.drop("Embarked",axis=1)
    
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='mean')
    imp.fit(data)
    imp_data =pd.DataFrame(imp.transform(data.values))
    imp_data.columns = data.columns
    data = imp_data
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data=data
    scaled_data[["Fare","Age"]] = scaler.fit_transform(scaled_data[["Fare","Age"]])
    data = scaled_data
    return data


# In[ ]:


cleanup_data(data).head()


# ## Performance

# What is going to be our performance measure?
# 
# Accuracy: percent correct predictions out of total predictions <<<<<< I think this is the competition metric on Kaggle
# 
# Precision: percent predicted survived correct out of total predicted
# 
# Recall: percent predicted right out of total survived
# 

# Let's look at Accuracy for a random prediction

# In[ ]:


randpred = []
n_correct=0
for i in range(len(clean_data)):
    randpred.append(np.random.randint(low=0,high=2))
    if randpred[i]== clean_data["Survived"][i]:
        n_correct = n_correct+1
print("Accuracy: ", n_correct/len(clean_data))


# We see that the random prediction is roughly 50%, as we would expect. Now lets train some real models:

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(clean_data.drop("Survived",axis=1),clean_data["Survived"] , test_size=0.33, random_state=42)
X_train =(clean_data.drop("Survived",axis=1))
y_train = clean_data["Survived"]


# Lets define a couple of useful evaluation functions:

# In[ ]:


from sklearn import metrics, model_selection

def eval_model_withC(model, n, init, increment):
    values = dict()
    for x in range(n):
        c_val = init+x*increment
        predicted = model_selection.cross_val_predict(model(C=c_val), X_train, y_train, cv=10)
        values["C="  + str(round(c_val,4))] = (metrics.accuracy_score(y_train, predicted))
#for k, v in values.items():
#   print(f'{k:<4} {v}')

    maximum = max(values, key=values.get) 
    print(maximum, "  ", "Accuracy is", values[maximum])
    
def eval_model(model):
    predicted = model_selection.cross_val_predict(model(), X_train, y_train, cv=10)
    print(metrics.accuracy_score(y_train, predicted))
    

    


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

eval_model(LogisticRegression)
eval_model_withC(LogisticRegression, 100,0.001, 0.001)



# In[ ]:


#pred = model_selection.cross_val_predict(LogisticRegression(C=0.01+4*0.02), X_train, y_train, cv=10)
#print(metrics.classification_report(y_train, pred) )


# ## SVM

# In[ ]:


from sklearn import svm

eval_model_withC(svm.SVC,50,0.01, 0.01)


# ## Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB

eval_model(GaussianNB)


# ## KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

eval_model(KNeighborsClassifier)


# ## Decision Trees

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

def eval_tree(model, n, init, increment):
    values = dict()
    for x in range(n):
        val = init+x*increment
        predicted = model_selection.cross_val_predict(model(min_impurity_decrease=val), X_train, y_train, cv=10)
        values["min_impurity_decrease="  + str(round(val,4))] = (metrics.accuracy_score(y_train, predicted))
#for k, v in values.items():
#   print(f'{k:<4} {v}')

    maximum = max(values, key=values.get) 
    print(maximum, "  ", "Accuracy is", values[maximum])


# In[ ]:


scaled_data = pd.concat([scaled_data, clean_data[["Survived","Sex","S","C","Q"]]],sort=True)


# In[ ]:


eval_tree(DecisionTreeClassifier,100,0,0.00005)


# ## Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

eval_model(RandomForestClassifier)
eval_tree(RandomForestClassifier,100,0.001,0.0001)


# ## Neural Nets

# To be added...

# In[ ]:





# ## Finalize Predictions

# Our best model has been the Random Forest Classifier, so lets use that for our final predictions:

# In[ ]:


test_data = pd.read_csv(folder + "test.csv")
passID = test_data["PassengerId"]
clean_test_data = cleanup_data(test_data)


# In[ ]:


clean_test_data.head()


# In[ ]:


X_train.head()


# We need to swap the S,C,Q columns in clean_test_data to match those in X_train

# In[ ]:


X_train.columns


# In[ ]:


clean_test_data = clean_test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'S', 'C', 'Q']]
clean_test_data.head()


# In[ ]:


RFClassifier = RandomForestClassifier(min_impurity_decrease=0.0013)
                                         
RFClassifier.fit(X_train,y_train)

pred = pd.DataFrame(RFClassifier.predict(clean_test_data))
        
pred["PassengerId"] = passID

pred.columns = ["Survived","PassengerId"]
    
pred = pred[["PassengerId","Survived"]]

#needs to be int or wont submit properly!
pred = pred.astype(int)

pred.head()
                                         


# In[ ]:


pred.to_csv("predictions.csv", index=False)


# In[ ]:




