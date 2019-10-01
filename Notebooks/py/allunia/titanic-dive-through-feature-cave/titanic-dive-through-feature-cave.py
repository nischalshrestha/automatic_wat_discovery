#!/usr/bin/env python
# coding: utf-8

# ## Welcome Kaggler!
# 
# With this interactive diving course I invite you to learn some machine learning basics. This course is designed as a series of kernels that guides you through different topics and hopefully you can discover some hidden treasures that push you forward on your data science road. You don't have to pick the courses in sequence if you are only interested in some topics that are covered. If you are new I would recommend you to take them step by step one after another. ;-)
# 
# Just fork the kernels and have fun! :-)
# 
# * [Prepare to start](https://www.kaggle.com/allunia/titanic-dive-through-prepare-to-start): Within this kernel we will prepare our data such that we can use it to proceed. Don't except nice feature selection or extraction techniques here because we will stay as simple as possible. Without a clear motivation we won't change any features. Consequently we are only going to explore how to deal with missing values and how to turn objects to numerical values. In the end we will store our prepared data as output such that we can continue working with it in the next kernel.
# * [MyClassifier](https://www.kaggle.com/allunia/titanic-dive-through-myclassifier): Are you ready to code your own classifier? Within this kernel you will build logistic regression from scratch. By implementing the model ourselves we can understand the assumptions behind it. This knowledge will help us to make better decisions in the next kernel where we will use this model and build some diagnosis tools to improve its performance.
# * **The feature cave**: By using our own logistic regression model we will explore how we can improve by adding a bias term and why we should encode categorical features. 
# * [Feature scaling and outliers](https://www.kaggle.com/allunia/titanic-dive-through-feature-scaling-and-outliers): Why is it important to scale features and to detect outliers? By analyzing the model structure we will discover how our gradients and our model performance are influenced by these topics. 

# ## Get your equipment
# ...
# 
# Titanic! 
# ...
# 
# We are on the way to your wrack!
# ...
# 
# With our own classifier at hand we are now able to analyse every step we take while playing with our data. But before doing so, let's get our equipment:

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set()


# We will use the prepared data we made during the first course:

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/titanicdivethrough/prepared_train.csv", index_col=0)
test = pd.read_csv("../input/titanicdivethrough/prepared_test.csv", index_col=0)
train.head()


# And of course we will use our own classifier we build during the second course: 

# In[ ]:


class MyClassifier:
    
    def __init__(self, n_features):
        self.n_features = n_features
        np.random.seed(0)
        self.w = np.random.normal(loc=0, scale=0.01, size=n_features)
        self.losses = []
    
    def predict(self, x):
        y = sigmoid(np.sum(self.w*x, axis=1))
        return y
    
    def loss(self, y, t):
        E = - np.sum(t * np.log(y) + (1-t) * np.log(1-y))
        return E
        
    def gradient(self, x, y, t):
        grad = np.zeros(self.w.shape[0])
        for d in range(self.w.shape[0]):
            grad[d] = np.sum((y-t)*x[:, d])
        return grad
        
    def update(self, eta, grad):
        w_next = np.zeros(self.w.shape) 
        for d in range(self.w.shape[0]):
            w_next[d] = self.w[d] - eta * grad[d]
        return w_next
    
    def learn(self, x, t, eta, max_steps, tol):
        y = self.predict(x)
        for step in range(max_steps):
            error = self.loss(y, t)
            grad = self.gradient(x, y, t)
            self.w = self.update(eta, grad)
            self.losses.append(error)
            y = self.predict(x)
            error_next = self.loss(y, t)
            if (error - error_next) < tol:
                break
                
    def decide(self, y):
        decision = np.zeros(y.shape)
        decision[y >= 0.5] = 1
        decision[y < 0.5] = 0
        return decision.astype(np.int)
    
    def accuracy(self, y, t):
        N = y.shape[0]
        return 1/N * np.sum(1 - np.abs(t-y))
        
    def score(self, x, t):
        y = self.predict(x)
        y = self.decide(y)
        return self.accuracy(y, t)

def sigmoid(x):
    result = 1/(1+np.exp(-x))
    return result


# Now, we are ready to dive! ;-)

# ## Station 1 - Validation
# 
# So far, we haven't thought about the topic "how to measure the performance" of our model. In the last course we used the accuracy score, namely the percentage of right predictions, solely based on the train set. Consequently we can only say: "We learned how to make more or less good predictions for the passengers in the train set". Instead we would like to estimate how good our model makes predictions on unseen data. Thus it would be great to split our data into a train and validation set. By looking at the performance on the validation set we can adjust our strategies such that we find an optimal solution for unknown data. But of course this is again some kind of "fitting" and the validation data becomes part of the training as well. For this reason we need a third data set where we can measure the final performance of our model: the test data. In our case the test data is already given by the competition. But the validation data is still missing.
# 
# Do generate it we will use [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from scikit-learn (sklearn), an opensource tool for data science and machine learning.   

# In[ ]:


from sklearn.model_selection import train_test_split

features = train.drop(["PassengerId", "Survived"], axis=1).columns

X = train[features].values
Y = train.Survived.values

x_train, x_val, t_train, t_val = train_test_split(X, Y, random_state=0)


# Let's see how good our model is on the train and validation set. 

# In[ ]:


# your task: call the score method to find out how well your model performs 
# on the train and validation data
classifier = MyClassifier(x_train.shape[1])
classifier.learn(x_train, t_train, 0.000001, 100000, 0.00001)
#train_score = <your code>
#test_score = <your code>


# In[ ]:


#assert(np.round(train_score, 4) == 0.7994)
#assert(np.round(test_score, 4) == 0.7803)


# Luckily our predictions are similar good on the train and validation set and we will proceed to measure our performance this way until we run into problems. Let's check if the train loss converged to a minimum during our learning procedure:

# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(classifier.losses)
plt.xlabel("Iteration steps")
plt.ylabel("cross entropy loss")


# That's ok for us. We don't expect the classifier to be much better with more iteration steps. 

# ## Station 2 - The bias term
# 
# The example submission we have given is based just on one feature: the sex. The submission predicts "not survived - 0" if the passenger was a male and "survived - 1" otherwise. Let's perform an experiment and make predictions with our model solely on this feature. 

# In[ ]:


x_sex_train = x_train[:, 1].reshape(-1,1)
x_sex_val = x_val[:, 1].reshape(-1,1)

classifier = MyClassifier(1)
classifier.learn(x_sex_train, t_train, 0.000001, 100000, 0.00001)
print(classifier.score(x_sex_train, t_train))
print(classifier.score(x_sex_val, t_val))


# Ok, now let's have a look if we would directly pass 0 for males and 1 for females:

# In[ ]:


classifier.accuracy(x_val[:,1], t_val)


# Obviously our model is much worst than the example submission even if we use the sex feature alone. Something is wrong here!  Let's have a look at the predictions our model can make after learning:

# In[ ]:


predictions = classifier.predict(x_sex_train)
# It's your turn now: Plot the survival probabilites given by the predictions our classifier made:
plt.figure(figsize=(15,4))
#<your code>
plt.xlabel("predicted survival probability")
plt.ylabel("frequency")
plt.title("Distribution of predictions")


# What do you observe? The predicted probability is always >= 0.5. Consequently our model always predicts that the passengers survived. To understand why that happened, look closer at the prediction function we used:
# 
# $$ y = \sigma (w_{sex} \cdot x_{sex}) $$
# 
# This sum has vanished as we only use one feature. Given a male we would have $x_{sex} = 0$ and in this case $y$ would always yield 0.5 due to the sigmoid function. This is of course not what we want. To overcome this problem we need a term that enables us to shif the sigmoid function such that we can map some other value than $x=0$ to $y=0.5$. This can be done by including the so called bias term:
# 
# $$ y = \sigma(w_{sex} \cdot x_{sex} + w_{bias})$$
# 
# This course already provides a skeleton for an improved classifier. It's your turn now! Add a bias parameter to self.w and adjust the method to make predictions and to compute the gradients. 

# In[ ]:


class ImprovedClassifier(MyClassifier):
    
    def __init__(self, n_features):
        super().__init__(n_features)
        np.random.seed(0)
        self.w = np.random.normal(loc=0, scale=0.01, size=n_features + 1)
    
    def predict(self, x):
        # your task: replace the compuation of y by including the bias term self.w[-1] * 1
        y = sigmoid(np.sum(self.w[:-1]*x, axis=1))
        return y
        
    def gradient(self, x, y, t):
        grad = np.zeros(self.w.shape[0])
        for d in range(self.w.shape[0]):
            if d == self.n_features:
                # replace the computation of the gradient of E with respect to the bias term
                grad[d] = 0
            else:
                grad[d] = np.sum((y-t)*x[:, d])
        return grad


# In[ ]:


iclassifier = ImprovedClassifier(1)
iclassifier.learn(x_sex_train, t_train, 0.000001, 100000, 0.00001)
train_score = iclassifier.score(x_sex_train, t_train)
val_score = iclassifier.score(x_sex_val, t_val)
print("train accuracy: %f , validation accuracy: %f" %(train_score, val_score) )


# After your improvement the model is as good as if we would directly pass "0 for males" and "1 for females":

# In[ ]:


#assert(train_score == iclassifier.accuracy(x_train[:,1], t_train))


# ## Station 2 - Categorical features
# 
# In the last section we have already seen that we have to take care about our sigmoid function. By playing with a binary categorical feature we have seen that we need to introduce a bias term to shift the sensitive region close to zero of the sigmoid function. But what about categorical features with more than two values? Let's have a look at the Embarked feature:

# In[ ]:


train_df = pd.DataFrame(x_train, columns=features)
val_df = pd.DataFrame(x_val, columns=features)
train_df["Survived"] = t_train


# Let's pass only the Embarked feature to train our model and have a look at the predictions we can make this way:

# In[ ]:


x_embarked_train = train_df.Embarked.values.reshape(-1,1)
x_embarked_val = val_df.Embarked.values.reshape(-1,1)

iclassifier = ImprovedClassifier(1)
iclassifier.learn(x_embarked_train, t_train, 0.000001, 100000, 0.00001)
train_score = iclassifier.score(x_embarked_train, t_train)
val_score = iclassifier.score(x_embarked_val, t_val)
predictions_proba = iclassifier.predict(x_embarked_train)
predictions = iclassifier.decide(predictions_proba)

print("train accuracy: %f , validation accuracy: %f" %(train_score, val_score))

train_df["predictions_proba"] = np.round(predictions_proba, 2)
train_df["predictions"] = predictions


# If you have done the bias term station right, you should see:

# In[ ]:


#assert(np.round(train_score, 4) == 0.6138)
#assert(np.round(val_score, 4) == 0.6233)


# In[ ]:


fig, ax = plt.subplots(1,4, figsize=(20,5))
sns.countplot(train_df.Embarked, ax=ax[0])
ax[0].set_title("Embarkation counts")
sns.countplot(x="Embarked", hue="Survived", data=train_df, ax=ax[1])
ax[1].set_title("True Survival")
sns.countplot(x="Embarked", hue="predictions", data=train_df, ax=ax[2])
ax[2].set_title("Predicted Survival")
sns.countplot(x="Embarked", hue="predictions_proba", data=train_df, ax=ax[3])


# The decisions of our model for embarkation of value zero and two are ok as more passengers in those cases died. But what about embarkation with value one? In this case our model should predict that the passengers survived even though the true distribution shows only slightly differences. 
# 
# By considering the bahaviour of the sigmoid function, can you explain what went wrong?
# 
# ...
# 
# What would have happend if we would change the original map of:
# 
# ```python
# embarked_map = {"S": 0, "C": 1, "Q": 2}
# ```
# 
# to
# 
# ```python
# embarked_map = {"S": 0, "C": 2, "Q": 1}
# ```
# ?
# 
# By choosing numerical values we already assumed some kind of order that has to fit to our target variables. That's bad! Let's solve this problem by assigning binary values to each property per categorical variable:

# In[ ]:


train_df["S"] = train_df.Embarked.apply(lambda l: np.where(l==0, 1, 0))
train_df["C"] = train_df.Embarked.apply(lambda l: np.where(l==1, 1, 0))
train_df["Q"] = train_df.Embarked.apply(lambda l: np.where(l==2, 1, 0))
val_df["S"] = val_df.Embarked.apply(lambda l: np.where(l==0, 1, 0))
val_df["C"] = val_df.Embarked.apply(lambda l: np.where(l==1, 1, 0))
val_df["Q"] = val_df.Embarked.apply(lambda l: np.where(l==2, 1, 0))

train_df.drop("Embarked", axis=1, inplace=True)
val_df.drop("Embarked", axis=1, inplace=True)


# In[ ]:


x_embarked_train = train_df[["S", "C", "Q"]].values
x_embarked_val = val_df[["S", "C", "Q"]].values

iclassifier = ImprovedClassifier(x_embarked_train.shape[1])
iclassifier.learn(x_embarked_train, t_train, 0.000001, 100000, 0.00001)
train_score = iclassifier.score(x_embarked_train, t_train)
val_score = iclassifier.score(x_embarked_val, t_val)
predictions_proba = iclassifier.predict(x_embarked_train)
predictions = iclassifier.decide(predictions_proba)

print("train accuracy: %f , validation accuracy: %f" %(train_score, val_score))

train_df["predictions_proba"] = np.round(predictions_proba, 2)
train_df["predictions"] = predictions


# If you have done the bias-term station right, you should see:

# In[ ]:


#assert(np.round(train_score, 4) == 0.6287)
#assert(np.round(val_score, 4) == 0.6592)


# Using binary representations per property of embarkation we improved!!! :-) 
# 
# **This way we made sure that we don't introduce an order of survival by passing the numerical values of a categorical feature through the sigmoid function.**
# 
# Our improvement should turn passengers that embarked in Cherbourg from "not survived - 0" to "survived - 1". Let's check this:

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(15,4))
sns.countplot(x="C", hue="Survived", data=train_df, ax=ax[0])
ax[0].set_title("True Survival")
sns.countplot(x="C", hue="predictions", data=train_df, ax=ax[1])
ax[1].set_title("Predicted Survival")
sns.countplot(x="C", hue="predictions_proba", data=train_df, ax=ax[2])


# Yup! :-)
# 
# Now, it's your turn: Use binary representation for all remaining categorical features in our data sets! 
# Even though there are opensource solutions out there let's use our own encoder that is build on the procedure we used before:

# In[ ]:


class MyEncoder:
    
    def __init__(self, features):
        self.features = features
        self.feature_levels = []
    
    def fit(self, df):
        for feature in self.features:
            levels = df[feature].unique()
            self.feature_levels.append(levels)
    
    def transform(self,df):
        for f in range(len(self.features)):
            feature = self.features[f]
            levels = self.feature_levels[f]
            for level in levels:
                new_name = feature + "_" + str(level)
                df[new_name] = df[feature].apply(lambda l: np.where(l==level, 1, 0))
        return df


# To make sure that we don't miss any property per categorical feature of train and test, we should fit our encoder on the combined data set and transform train and test separately:

# In[ ]:


combined = train.drop("Survived", axis=1).append(test)
combined.head()


# In[ ]:


to_encode = ["Embarked"] # put your categorical features into the list!!!
encoder = MyEncoder(to_encode)
encoder.fit(combined)
train = encoder.transform(train)
test = encoder.transform(test)


# If you are not sure taking one feature or not, try:
# 
# ```python
# sns.countplot(x=featuretotry, hue="Survived", data=train)
# ```
# 
# Does the order of the numerical values fit to the target and predictions you would obtain by passing through sigmoid? Hint: There are more than Embarked! ;-)

# After extracting the new features clean your data and drop the old ones:

# In[ ]:


for feature in to_encode:
    train.drop(feature, axis=1, inplace=True)
    test.drop(feature, axis=1, inplace=True)


# For the next stattion, make sure that you gain the right score:

# In[ ]:


features = train.drop(["PassengerId", "Survived"], axis=1).columns

X = train[features].values
Y = train.Survived.values

x_train, x_val, t_train, t_val = train_test_split(X, Y, random_state=0)

iclassifier = ImprovedClassifier(x_train.shape[1])
iclassifier.learn(x_train, t_train, 0.000001, 100000, 0.00001)
train_score = iclassifier.score(x_train, t_train)
val_score = iclassifier.score(x_val, t_val)
print("train accuracy: %f , validation accuracy: %f" %(train_score, val_score))


# You should obtain:

# In[ ]:


#assert(np.round(train_score, 4) == 0.8114)
#assert(np.round(val_score, 4) == 0.8072)

