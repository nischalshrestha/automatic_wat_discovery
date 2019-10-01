#!/usr/bin/env python
# coding: utf-8

# # Introduction & References used

# ### Dataset was obtained from https://www.kaggle.com/c/titanic/
# 
# ### Goal: Train a model with 891 training examples to predict the survival outcome (0,1) of 418 test examples
# 
# ### 1. Importing of Libraries and dataset
# 
# ### 2. Categorical Variable Analysis
# 
# ### 3. Numerical Variable Analysis
# 
# ### 4. Feature Engineering & Correlation
# 
# ### 5. Transformation of test set
# 
# ### 6. Modeling & Predictions
# 
# #### - Created and used a logistic regression model from scratch via gradient descent
# 
# #### - Applied SVM & Random Forest by using the sklearn library with some hyperparameter tuning
# 
# #### - Applied a simple artifical neural network with Keras/Tensorflow
# 
# ### 7. References
# 
# #### - https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling on feature engineering
# 
# #### - A - Z Machine Learning Udemy Course by Krill on Artifical Neural Network

# # 1. Importing of libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sb

from pylab import rcParams
rcParams['figure.figsize'] =5,4

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['figure.edgecolor']='black'
plt.rcParams['patch.force_edgecolor']= True

sb.set_style('whitegrid')


# In[ ]:


dataset = pd.read_csv("../input/train.csv")
# Dropping passenger id and ticket which have little impact on the survival
dataset = dataset.drop(['PassengerId','Ticket'],axis=1)
dataset.describe()


# # 2. Categorical Variable Analysis

# ## 2A. Pclass Variable

# In[ ]:


sb.countplot(x = dataset['Pclass'], hue = dataset['Survived'])
plt.legend(loc='best')
plt.title('No of Survivals based on Pclass',weight='bold')
plt.annotate(s='Pclass 3 highest\nnumber of survival',xy=(1.5,200),xytext=(0,250),
             arrowprops=dict(facecolor='black', shrink = 0.1),fontsize=11)

sb.factorplot(x='Pclass',y='Survived',data = dataset,kind='bar')
plt.title('Percentage of Survivals based on Pclass',weight='bold')
plt.annotate(s='Pclass 1 highest\n% of survival',xy=(0.45,0.55), xytext=(1.35,0.62),
             arrowprops=dict(facecolor='black',shrink = 0.05),fontsize=11)


# ## 2B. Name Variable

# In[ ]:


# Extracting title from name and giving rare titles as 'rare'
namelist = ['Mr.','Miss.','Mrs.','Master.']
rows,cols=dataset.shape
for i in range(0,rows):
    for n in namelist:
        if (n in dataset.iloc[i,2]) == True:
            dataset.iloc[i,2] = n
count = 0
for i in (~dataset['Name'].isin(namelist)):
    if i == True:
        dataset.iloc[count,2] = 'Rare'
        count = count + 1
    else:
        count = count + 1


# In[ ]:


sb.countplot(x=dataset['Name'],hue=dataset['Survived'])
plt.title('Survival Count based on title',fontweight='bold')
plt.xlabel('Title',fontsize = 12)
plt.ylabel('Survival Count',fontsize=12)
plt.annotate(s='Males have lowest\nsurvival count',xy=(0.1,250),xytext=(1.5,310),fontsize=12,
             arrowprops=dict(facecolor='black',shrink =0.1))
plt.annotate(s='Females have higher\nsurvival count',xy=(2.3,130),xytext=(1.5,240),fontsize=12,
             arrowprops=dict(facecolor='black',shrink=0.1))                                                                                                          
plt.annotate(s='',xy=(1.2,110),xytext=(1.7,220),fontsize=12,arrowprops=dict(facecolor='black',shrink=0.1))                                                                                                          


sb.factorplot(x = 'Name',y='Survived',data=dataset, kind ='bar')
plt.title('Survival Percentage based on title',fontweight='bold')
plt.ylabel('Survival Probability',fontsize=12)
plt.xlabel('Title',fontsize=12)


# ## 2C. Sex Variable

# In[ ]:


# Females have a higher survival probability than males
sb.factorplot(x='Sex',y='Survived',data=dataset,kind='bar')
plt.ylabel('Survival Probability',fontsize=12)
plt.xlabel('Sex',fontsize =12)


# ## 2D. Embarked Variable

# In[ ]:


# Filling in the two missing embarked as Q
dataset['Embarked']=dataset['Embarked'].fillna('Q')


# In[ ]:


# Survival Probability is dependent on the passenger's embark location
sb.countplot(dataset['Embarked'],hue=dataset['Survived'])
plt.ylabel('Survival Count')
sb.factorplot('Embarked','Survived',data=dataset,kind='bar')
plt.ylabel('Survival Probability')


# # 3. Numerical Variables

# ## 3A. Family Size Variable + 1 for himself

# In[ ]:


dataset['Family']=dataset['SibSp'] + dataset['Parch'] +1
dataset = dataset.drop(['SibSp','Parch'],axis=1)


# In[ ]:


# Outlier present in the family size variable
sb.boxplot(dataset.Family)
dataset.Family.value_counts()


# In[ ]:


# Dropping family size of >=7 (outliers)
dataset=dataset[dataset['Family']<7]
dataset = dataset.reset_index()
dataset = dataset.drop(['index'],axis=1)


# In[ ]:


sb.countplot(dataset['Family'],hue = dataset['Survived'])
plt.title('Survival Count based on family size',fontweight='bold')
plt.ylabel('Survival Count',fontweight='bold')
plt.xlabel('Family Size',fontweight='bold')
plt.annotate(s='People who are alone\nare less likely\nto survive',xy=(0.1,190),xytext=(2.8,230),fontsize=11,
             arrowprops=dict(facecolor='black',shrink=0.1))


sb.factorplot('Family','Survived',data=dataset,kind='bar',aspect = 1.2)
plt.title('Survival Percentage based on family size',fontweight='bold')
plt.ylabel('Survival Count',fontweight='bold')
plt.xlabel('Family Size',fontweight='bold')

plt.annotate(s='Family of 4 most\n likely to survive',xy=(2.6,0.7),xytext=(0,0.72),fontsize=11,
             arrowprops=dict(facecolor='black',shrink=0.1))


# ## 3B. Fare Variable

# In[ ]:


# Checks for the distribution first, displot
plt.subplot(1,2,1)
sb.distplot(dataset['Fare'])
plt.title ('Before log transformation')

# After taking the log
dataset['Fare'] = dataset['Fare'].map(lambda i: np.log(i) if i >0 else 0)
plt.subplot(1,2,2)
sb.distplot(dataset['Fare'])
plt.title ('After log transformation')


# In[ ]:


# Outlier removal for fares
lq = np.percentile(dataset['Fare'],25)
uq = np.percentile(dataset['Fare'],75)
IQR = uq-lq
limit1 = lq-1.5*IQR
limit2= uq+1.5*IQR
dataset= dataset[(dataset['Fare']<limit2) & (dataset['Fare']>limit1) ]
dataset= dataset.reset_index(drop=True)


# In[ ]:


# from https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
sb.kdeplot(dataset['Fare'][dataset['Survived']==1],color='blue',shade='True')
sb.kdeplot(dataset['Fare'][dataset['Survived']==0],color='red',shade='True')
plt.legend(['Survived','Died'])
plt.xlabel('log(Fare)')
plt.ylabel('Frequency')
plt.annotate(s='Higher fares, \ngreater survival rate', xy = (4,0.25),xytext=(3,0.5),fontsize=11,fontweight='bold',
            arrowprops=dict(facecolor='black',shrink = 0.1))


# ## 3C: Age Variable

# In[ ]:


# Attempt to fill in the missing age of 160 using existing variables
plt.subplot(2,2,1)
sb.boxplot(dataset['Sex'],dataset['Age'])
plt.subplot(2,2,2)
sb.boxplot(dataset['Family'],dataset['Age'])
plt.subplot(2,2,3)
sb.boxplot(dataset['Embarked'],dataset['Age'])
plt.subplot(2,2,4)
sb.boxplot(dataset['Pclass'],dataset['Age'])

# Using medians for each individual PClass to predict age
medianages = dataset[['Pclass','Age']].groupby(by='Pclass').median()

def agefiller(x):
    if x == 1:
        y = medianages.iloc[0,0]
    elif x == 2:
        y = medianages.iloc[1,0]
    else:
        y = medianages.iloc[2,0]
    return y 

# Replace missing age with these medians based on PClass
missingage = dataset['Age'].isnull()
rows,cols = dataset.shape

for i in range(0,rows):
    if (missingage[i] == True):
        dataset.iloc[i,4] = agefiller(dataset.iloc[i,1])
plt.subplots_adjust(wspace=0.5,hspace=0.5)


# In[ ]:


plt.subplots_adjust(wspace=0.4)
rcParams['figure.figsize'] =7,4

plt.subplot(1,2,1)
sb.distplot(dataset['Age'])
plt.ylabel('Frequency')

plt.subplot(1,2,2)
sb.kdeplot(dataset['Age'][dataset['Survived']==1],color='blue',shade = 'True')
sb.kdeplot(dataset['Age'][dataset['Survived']==0],color='red',shade='True')
plt.legend(['Survived','Died'],loc='best')
plt.xlabel('Age')
plt.ylabel('Frequency')


# ## 3D. Cabin Variable

# In[ ]:


# Cabin Variable , 0 and 1. Assume that those with Nan in cabin column have no cabins
rcParams['figure.figsize'] =5,4
dataset.Cabin = dataset.Cabin.fillna(0)
dataset.Cabin = dataset.Cabin.astype(bool).astype(int)
sb.countplot(dataset.Cabin,hue=dataset.Survived)
sb.factorplot('Cabin','Survived',data=dataset,kind='bar')
plt.ylabel('Survival Probability')
plt.title('Survival Probability based on cabins',fontweight='bold')
plt.annotate(s=' Those with no\ncabin have lower\n  survival rate', xy = (0,0.35),xytext=(-0.4,0.6),fontsize=11,fontweight='bold',
            arrowprops=dict(facecolor='black',shrink = 0.1))


# # 4. Feature Selection / Correlation

# In[ ]:


# 1 Hot Encoding for the categorical datas, namaely Sex,Embarked and Name
dataset = pd.get_dummies(dataset, columns=['Name'],prefix =['Name'])
dataset = pd.get_dummies(dataset, columns=['Embarked'],prefix =['Embarked'])
dataset = pd.get_dummies(dataset, columns=['Sex'],prefix =['Sex'])
dataset = pd.get_dummies(dataset, columns=['Pclass'],prefix =['Pclass'])

dataset = dataset.drop(['Name_Rare','Name_Mr.','Embarked_S','Sex_female','Pclass_1'],axis =1)


# In[ ]:


rcParams['figure.figsize'] =7,5
sb.heatmap(dataset.corr(method='spearman'),annot=True,fmt='.0g')


# # 5. Transforming Kaggle test dataset

# In[ ]:


kaggle = pd.read_csv("../input/test.csv")

# To be used when submitting predictions
passengerid = kaggle['PassengerId']

kaggle = kaggle.drop(['PassengerId','Ticket'],axis=1)

# Extracting title out from name
namelist = ['Mr.','Miss.','Mrs.','Master.']
rows,cols=kaggle.shape
for i in range(0,rows):
    for n in namelist:
        if (n in kaggle.iloc[i,1]) == True:
            kaggle.iloc[i,1] = n
count = 0
for i in (~kaggle['Name'].isin(namelist)):
    if i == True:
        kaggle.iloc[count,1] = 'Rare'
        count = count + 1
    else:
        count = count + 1

# Extracting family size from Sibsp and Prch
kaggle['Family']=kaggle['SibSp'] + kaggle['Parch'] +1

# Assuming those with no cabin assigned are given a value of 0
kaggle['Cabin'] = kaggle['Cabin'].fillna(0)
kaggle.Cabin = kaggle.Cabin.astype(bool).astype(int)

# Log transformation of Fare, for better distribution (normal)
kaggle['Fare'] = kaggle['Fare'].fillna(kaggle['Fare'].mean())
kaggle['Fare'] = kaggle['Fare'].map(lambda i: np.log(i) if i >0 else 0)

# Applying median ages (based on PClass) as seen in the training set
missingage = kaggle['Age'].isnull()
rows,cols = kaggle.shape

for i in range(0,rows):
    if (missingage[i] == True):
        kaggle.iloc[i,3] = agefiller(kaggle.iloc[i,0])

# One hot encoding for the categorical variables
kaggle = pd.get_dummies(kaggle, columns=['Name'],prefix =['Name'])
kaggle = pd.get_dummies(kaggle, columns=['Embarked'],prefix =['Embarked'])
kaggle = pd.get_dummies(kaggle, columns=['Sex'],prefix =['Sex'])
kaggle = pd.get_dummies(kaggle, columns=['Pclass'],prefix =['Pclass'])
kaggle = kaggle.drop(['Embarked_S','Sex_female','Name_Rare','Name_Mr.','SibSp','Parch','Pclass_1'],axis=1)


# In[ ]:


dataset.head(1)


# In[ ]:


kaggle.head(1)


# # 6. Modeling and predictions

# ## 6A. Creation and testing of a logistic regression model from scratch

# In[ ]:


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def logistic_regression(x,y,lr,iterations):
    # obtain the training numbers and number of variables
    training_number, variables = x.shape

    # insert a row of ones into x as the bias
    x = np.append(arr=np.ones((training_number,1)).astype(int),values = x, axis=1)

    # randomise the weights for all the variables
    weights = (np.random.randn(1,variables+1)).T

    inputs = np.matmul(x,weights)

    # Apply sigmoid function and convert to binary 0 or 1, based on the >0.5 = 1
    outputs = sigmoid(inputs)

    # Backpropagation step
    for i in range(iterations):
        # D_error/d(output)
        D_error_output = -1*(np.multiply(y,(1/(outputs+0.0001))))  + np.multiply((1-y),(1/(1-outputs+0.0001)))

        # D_output_d_input (sigmoid derivative), element wise multiplication!
        matrix1 = 1- outputs
        D_output_d_input = np.multiply(matrix1,outputs)

        # Combination of D_error/d(output) & D_output_d_input & all the x variables
        error_missing_x = (np.multiply(D_error_output,D_output_d_input))

        # this computes a matrix with rows being training example, and columns x1,x2,x3. a gradient is calculated for each
        # x variable as well as training example
        D_error_d_weight = np.multiply(error_missing_x,x)

        # takign the average of the gradient for each variable, that is each column average across all the training set
        gradient = (D_error_d_weight.mean(axis=0)).T
        gradient = gradient.reshape(variables+1,1)

        # update the weights using the alpha learning rate

        weights = weights - gradient*lr

        inputs = np.matmul(x,weights)

        outputs = sigmoid(inputs)

    outputs = np.where(outputs>=0.5,1,0)

    return outputs,weights

def fit(x,weights):
    training_number, variables = x.shape
    x = np.append(arr=np.ones((training_number,1)).astype(int),values = x, axis=1)
    xtest = np.matmul(x,weights)
    ypred = sigmoid(xtest)
    ypred = np.where(ypred>=0.5,1,0)
    return ypred


# In[ ]:


X = dataset.iloc[:,1:13].values
y = dataset.iloc[:,0:1].values

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
sc = StandardScaler()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

train_pred,weights = logistic_regression(X_train,y_train,0.05,10000)
test_pred = fit(X_test,weights)


print ("The training accuracy score is %.3f" % accuracy_score(y_train,train_pred))
print ("The test accuracy score is %.3f" % accuracy_score(y_test,test_pred))


# ## Sklearn Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train,y_train.ravel())
yscore = log_model.predict(X_test)
print ("The test accuracy score is %.3f" % accuracy_score(y_test,yscore))


# ## 6B. Sklearn Kernel SVM

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.05)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
svm_model = SVC(kernel='rbf',degree =4,gamma='auto')
svm_model.fit(X_train,y_train.ravel())

from sklearn.model_selection import cross_val_score
svmcv = cross_val_score(estimator = svm_model, X = X, y = y.ravel(),cv = 5,scoring='accuracy')
print ("The mean cross-validation training accuracy score is %.3f" % svmcv.mean())


# In[ ]:


# Checking for overfitting or underfitting via train and test scores (y-axis) against training set size (x-axis)
from sklearn.model_selection import learning_curve

# Splits overall dataset into training set & test set, and varying the training set size from 100 to max possible (step by 10)
max_training_size, nil = X.shape
cv = 15
max_training_size = int((max_training_size*((cv-1)/cv))-1)
train_sizes, train_scores, test_scores = learning_curve(svm_model, X, y.ravel(), train_sizes=range(100,max_training_size,10), 
                                                         cv=cv, shuffle=True)

# Taking the mean of train and test score for each cross validation test/traing set and for each training set size
# train_scores, cols = testset1--->testset10, rows = trainingsize100 --> trainingsize(max)
train_scores = [i.mean() for i in train_scores]
test_scores = [i.mean() for i in test_scores]

plt.subplot(1,2,1)
errorplot = pd.concat([pd.DataFrame(train_sizes,columns = ['Training Size']),
                       pd.DataFrame(train_scores,columns=['Training Score']),
                       pd.DataFrame(test_scores, columns =['Test Score'])],axis=1)

plt.plot(errorplot.iloc[:,0],errorplot.iloc[:,1], color='red', label='Training Set')
plt.plot(errorplot.iloc[:,0],errorplot.iloc[:,2],color='blue',label='Test Set')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')

difference = errorplot
plt.subplot(1,2,2)
difference['Difference']= errorplot.iloc[:,1]- errorplot.loc[:,'Test Score']
difference = difference.drop(['Training Score','Test Score'],axis=1)
plt.plot(difference.iloc[:,0],difference.iloc[:,1],color='green',label='Score Difference')
plt.xlabel('Training Set Size')
plt.legend(loc='best')

plt.subplots_adjust(hspace =1.6)

print (str(round((1/cv*100),1)) +'% '+ 'of the overall training data set was used as the cross validation test set')
print ("")
print ('From the graph, the SVM model has a maximum accuracy of ~81% and is obtained by taking only 5% as the test set')
print ("")
print ('Taking the usual 15-20% of the training set as test set yield lower acc (seen above), likely due to insufficent training data')
print ("")
print ('One way to improve the model is to increase the training size since the test score < training score')
print("")
print('Decided to use SVM to train on 95% training set and applying it to kaggle test set. 80.61% accuracy returned')


# # 6C. Sklearn Random Forest

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
# number of trees, min_samples_split,

ranforest_model = RandomForestClassifier(n_estimators = 20, criterion = 'entropy',min_samples_split=8,min_samples_leaf=2)
ranforest_model.fit(X_train,y_train.ravel())

rany= ranforest_model.predict(X_test)
print ("The test accuracy score is %.3f" % accuracy_score(y_test,rany))


# In[ ]:


# Checking for overfitting or underfitting via train and test scores (y-axis) against training set size (x-axis)
from sklearn.model_selection import learning_curve

# Splits overall dataset into training set & test set, and varying the training set size from 100 to max possible (step by 10)
max_training_size, nil = X.shape
cv = 10
max_training_size = int((max_training_size*((cv-1)/cv))-1)
train_sizes, train_scores, test_scores = learning_curve(ranforest_model, X, y.ravel(), train_sizes=range(100,max_training_size,10), 
                                                         cv=cv, shuffle=True)

# Taking the mean of train and test score for each cross validation test/traing set and for each training set size
# train_scores, cols = testset1--->testset10, rows = trainingsize100 --> trainingsize(max)
train_scores = [i.mean() for i in train_scores]
test_scores = [i.mean() for i in test_scores]

plt.subplot(1,2,1)
errorplot = pd.concat([pd.DataFrame(train_sizes,columns = ['Training Size']),
                       pd.DataFrame(train_scores,columns=['Training Score']),
                       pd.DataFrame(test_scores, columns =['Test Score'])],axis=1)

plt.plot(errorplot.iloc[:,0],errorplot.iloc[:,1], color='red', label='Training Set')
plt.plot(errorplot.iloc[:,0],errorplot.iloc[:,2],color='blue',label='Test Set')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')

difference = errorplot
plt.subplot(1,2,2)
difference['Difference']= errorplot.iloc[:,1]- errorplot.loc[:,'Test Score']
difference = difference.drop(['Training Score','Test Score'],axis=1)
plt.plot(difference.iloc[:,0],difference.iloc[:,1],color='green',label='Score Difference')
plt.xlabel('Training Set Size')
plt.legend(loc='best')

plt.subplots_adjust(hspace =1.6)


# In[ ]:


# Tuning the parameters

from sklearn.model_selection import GridSearchCV

min_samples_split=[6,7,8,9,10]
min_samples_leaf=[1,2,3,4,5]
parameters = dict(min_samples_split = min_samples_split,min_samples_leaf=min_samples_leaf)
optimizer = GridSearchCV(estimator=ranforest_model
                         ,param_grid = parameters,
                         scoring = 'accuracy',
                         cv = 5)
optimizer = optimizer.fit(X_train,y_train.ravel())
print('The best training score obtained is %.3f' % optimizer.best_score_)
print(optimizer.best_params_)

rany= ranforest_model.predict(X_test)
print ("The test accuracy score is %3f" % accuracy_score(y_test,rany))
print ("")
print ('While test accuracy is high, the graph above shows the high variability in performance over larger training sets')


# # 6D. Artifical Neural Network

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# Creation of the ann model that links all the nodes together
ann_model = Sequential()

# Adding the first hidden layer (20 nodes) with the input layer (having 12 nodes). Glorot uniform initializer for weight init
ann_model.add(Dense(units= 20, kernel_initializer='uniform',activation='relu', input_dim = 12))

# Adding the second hidden layer (20 nodes) with the input layer. Glorot uniform initializer for weight init
ann_model.add(Dense(units = 20, kernel_initializer='uniform',activation='relu'))

# Adding the third hidden layer (20 nodes) with the input layer. Glorot uniform initializer for weight init
ann_model.add(Dense(units = 20, kernel_initializer='uniform',activation='relu'))

# Adding the output layer (1 node for binary classifcation). Glorot uniform initializer for weight init
ann_model.add(Dense(units= 1, kernel_initializer='uniform',activation='sigmoid'))

# Compiling the ANN and selecting the optimizer,cost function
ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the Model

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the ann model, batch_size which takes x number of training examples before updating the weights. 
# Epochs is the number of times the entire dataset passes through the ann network
ann_model.fit(X_train,y_train,batch_size =20, epochs =15)

anny = ann_model.predict(X_test)
anny = np.where(anny>=0.5,1,0)
print("The test score is %3f" % accuracy_score(y_test,anny))
print ("")
print("Overall performance did not beat SVM. Tuning with grid search did not improve accuracy significantly as there are many factors involved.")


# # Using Kaggle Test Dataset

# ## SVM Kaggle Submission

# In[ ]:


kaggle = sc.transform(kaggle)
kaggletest =svm_model.predict(kaggle)
# kaggletest = fit(kaggle,weights)
t1 = pd.DataFrame(passengerid, columns = ['PassengerId'])
s1 = pd.DataFrame(kaggletest,columns=['Survived'])
final = pd.concat([t1,s1],axis = 1)

final.to_csv('results.csv', index=False)
final.Survived.value_counts()

