#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1> Danbury AI June 2018: Workshop Part 2</h1>
# <h2>The Titanic Disaster</h2>
# </center>
# 
# **About this Notebook **
# 
# This notebook explores the process of using a neural network to produce a predictive model of the titanic dataset hosted on kaggle. There are many steps in this process that must not be overlooked or it will be more difficult to optimize the performance of your network. In particular, taking time to do EDA, feature engineering, and be considerate when preprocessing the data will stave of many potential modeling headaches later on. It is our aim to make you comfortable with the basics of each of these stages, so you can effectively use neural networks on your data. 
# 
# Learning Objectives:
# * Learn how to use neural networks on small datasets. 
# * Become familiar with the data science process. 
# * How to use a neural network on datasets with categorical and quantitative features. 
# 
# **Competition Overview  **
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# [Kaggle Competition Source](https://www.kaggle.com/c/titanic)
# 
# **Key Challenges of Using a Neural Network**
# 
# The number of parameters a statistical model has determines the range of functions it can represent. We call this range the *capacity* of a model. Models with greater capacity have a tendency to overfit the training data. Neural Networks are the epitome of high-capacity models. High-capacity models generally don't do well on small datasets. Large datasets balance out a neural network's tendency to overfit. With small datasets, we must pay great attention to how we regularize the network in order to fight this overfitting behavior. While there are many ways of doing this, in this notebook we will be using the dropout method. 
# 
# The Titanic data is a very small dataset of about 800 or so samples. This means our success is dependent upon proper regularization of the network. Lower capacity models like random forests or svms would do much better out of the box because they will not overfit as reaidly; however, if we tune the network well, we can generally beat their performance. 
# 
# 
# # Library Imports

# In[1]:


# General Libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

# Scikit-Learn Libraries
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split,learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model,svm,tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Keras Libraries 
from keras.models import Model
from keras.layers import Input, Dense, Dropout

# Interactive Widgets
from ipywidgets import interact_manual
from IPython.display import display

# Print out the folders where our datasets live. 
print("Datasets: {0}".format(os.listdir("../input")))


# ## 1. Exploratory Data Analysis ( EDA )
# We will begin by exploring the dataset. Getting a feel for the data helps us determine what features we want to use in our model and how we may process them in the ETL stage. In addition, we will compute some baselines based on our initial understanding of the data and use them later when we are evaluating our model's performance. 

# In[2]:


# Load Test Data
testDF = pd.read_csv("../input/titanic/test.csv")

# Load Training Data
titanicDF = pd.read_csv("../input/titanic/train.csv")
titanicDF.head(10)


# Let's visualize the data. 

# In[3]:


nRows = 3
nCols = 3

plt.figure(figsize=(6*3,6*3))
plt.subplot(nRows,nCols,1)

titanicDF["Survived"].value_counts().plot.pie(autopct="%.2f%%")

plt.subplot(nRows,nCols,2)
titanicDF["Sex"].value_counts().plot.pie(autopct="%.2f%%")

plt.subplot(nRows,nCols,3)
titanicDF["Pclass"].value_counts().plot.pie()

plt.subplot(nRows,nCols,4)
titanicDF["SibSp"].value_counts().plot.pie()

plt.subplot(nRows,nCols,5)
titanicDF["Parch"].value_counts().plot.pie()

plt.subplot(nRows,nCols,6)
titanicDF["Embarked"].value_counts().plot.pie()

plt.subplot(nRows,nCols,7)
plt.title("Age Histogram")
titanicDF["Age"].hist()

plt.subplot(nRows,nCols,9)
plt.title("Fare Histogram")
titanicDF["Fare"].hist()

plt.show()


# **Workshop Question:** What other visualizations would help us understand the dataset? 

# # 2. Feature Engineering
# 
# When we use features from a dataset to produce new features it is called *feature engineering*. From our our understanding of the problem, children were given a high precedence on life boats, yet we do not have an is_child feature. How would we engineer this feature? Let's explore. 
# 
# **1.) How many passengers are children?** 
# 
# We ask this question because we would like to determine which men were children, since we know being a woman or child has a better chance of being on the lifeboats and therefore has a greater chance of survival. The most obvious fields for deriving this feature are: Age, PArch, and SibSp. We will use age here. We must, however, make an assumption as to what age marks a child.

# In[4]:


# Plot
def plotChilds(ageThresh=10):
    df = titanicDF["Age"].dropna()
    childFeat = df.map(lambda x: x<ageThresh)
    
    plt.figure(figsize=(12,12))
    
    plt.subplot(221)
    plt.title("Size of Age Group")
    childFeat.value_counts().plot.pie(autopct="%.2f%%")
    
    
    newDF = titanicDF.assign(isChild=df.map(lambda x: x<ageThresh))
    res = newDF.groupby("isChild")["Survived"]
    adultC,childC = res.count()
    adultS,childS = res.sum()
    
    plt.subplot(222)
    plt.title("Age Group Survival")
    plt.pie([childC-childS,childS],labels=["Died","Survived"],autopct="%.2f%%")
    
    plt.subplot(223)
    plt.title("Age Group Gender Distribution")
    newDF.groupby("isChild")["Sex"].value_counts()[1].plot.pie(autopct="%.2f%%")
    
    plt.subplot(224)
    plt.title("Age Group Class Distribution")
    newDF.groupby("isChild")["Pclass"].value_counts()[1].plot.pie(autopct="%.2f%%")
    
    plt.show()
    display(newDF.head())
    
interact_manual(plotChilds,ageThresh=(0,100))


# **Workshop Questions:** How do we interpret the above visualizations? At what age would we consider a passenger a child? Should we use the presence of parents via the *Parch* feature as well? 

# ## 3. Preprocessing
# Here we do some simple preprocessing. Mainly we translate categorical variables to a [one-hot representation](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) and fill missing values with zeros. We do not use the cabin, ticket, or name features in this notebook in order to keep things simple.

# In[5]:


# Our prepprocessing function which is applied to every row of the target dataframe. 
def preprocessRow(row):
    # Process Categorical Variables - One-Hot-Encoding
    sex      = [0,0]
    embarked = [0,0,0]
    pclass   = [0,0,0]
    
    if row["Sex"] == "male":
        sex = [0,1]
    elif row["Sex"] == "female":
        sex = [1,0]
    
    if row["Embarked"] == "S":
        embarked = [0,0,1]
    elif row["Embarked"] == "C":
        embarked = [0,1,0]
    elif row["Embarked"] == "Q":
        embarked = [1,0,0]
    
    if row["Pclass"] == 1:
        pclass   = [0,0,1]
    elif row["Pclass"] == 2:
        pclass   = [0,1,0]
    elif row["Pclass"] == 3:
        pclass   = [1,0,0]
    
    return pclass+sex+[row["Age"],row["SibSp"],row["Parch"],row["Fare"]]+embarked

# Labels for the feature columns. 
featureLabels = ["3 Class","2 Class","1 Class","Female","Male","Age","SibSp",
                 "Parch","Fare","Q Embarked","C Embarked","S Embarked"]

# Fill Missing Values
titanicDF = titanicDF.fillna(0).sample(frac=1)

# Preprocess Data
titanicMat = np.stack(titanicDF.apply(preprocessRow,axis=1).values)

# View what the training vectors look like. 
tmp = pd.DataFrame(titanicMat)
tmp.columns = featureLabels
tmp.head()


# ** Workshop Questios ** 
# * What type of preprocessing may we want to do on ticket, name, and cabin to include them in our model? Would the length of the passengers name be helpful? 
# * When we use features from a dataset to produce new features it is called *feature engineering*. What features may we engineer to make modeling easier? From our our understanding of the problem, children were given a high precedence on life boats, yet we do not have an is_child feature. How would we engineer this feature? 
# 
# In order to plot our learning curves, which allow us to understand how well our model is performing, we need a training and validation set. We train on the training set and hold out the validation set. Since our model is not exposed to the validation data during training, we can use it to determine how well our model generalizes to new data; however, for a better measure of how our model will perform we must use a cross-validation method. 

# In[6]:


# Size of validation set. 
splitSize = 0.2

titanic_X, titanic_y = [titanicMat, titanicDF["Survived"].values]
titanic_train_x, titanic_validation_x, titanic_train_y , titanic_validation_y = train_test_split(titanic_X,titanic_y, test_size=splitSize)


# ## 4. Naive Baselines
# Before you begin modeling it is a good idea to hand write some baseline models based on your understanding of the problem and basic dataset statistics. A common baseline model in binary classification is to simply predict the the most represented class. In our case, this simply means: given any feature vector representing a passenger, we predict they die because that is the most common occurrence. This baseline is very helpful in the context of deep learning because if we catch our models performing similarly to our mean baseline, by having a similar loss, we know that no useful representation has been learned. 

# In[7]:


def meanBaseline(df):
    nRows = df["Survived"].shape[0]
    mean = df["Survived"].sum()/nRows
    return log_loss(df["Survived"],np.full(nRows,mean))

print("Mean Baseline: {0}".format(meanBaseline(titanicDF)))


# We also know from history that women and children were given precedence on the life boats. Being able to predict the latent  variable which represents a passenger getting on a lifeboat is the same as predicting survival -- to my knowledge, all that successfully boarded these boats survived. We use this knowledge to write a gender model: if a passenger is a female they survive, otherwise they die. 

# In[8]:


def genderBaseline(df):
    nRows = df["Survived"].shape[0]
    pred = df["Sex"].map(lambda x: 1 if x == "female" else 0)
    res = df["Survived"]==pred
    return res.sum()/res.count()

print("Gender Baseline: {0}".format(genderBaseline(titanicDF)))


# **Workshop Problem: ** What other naive baselines can we derive from our understanding of the problem?  

# In[9]:


# Put your code here. 


# ## 5. Non-Neural Modeling
# In this section we will train many scikit models in order to gain some perspective on how different models perform on the data. What is absent here is hyperparameter adjustment, so we must take these results with a grain of salt. The results here do not represent the optimal result we can get with these models -- we'd need to tune hyperparams for that.

# In[ ]:


models = {"Linear Regression":linear_model.LinearRegression(),"Logistic Regression":linear_model.LogisticRegression(),
         "Ridge Regression":linear_model.Ridge(),"Lasso Regression":linear_model.Lasso(),
         "Bayesian Ridge Regression":linear_model.BayesianRidge(),"Perceptron":linear_model.Perceptron(max_iter=1000),
         "Support Vector Machine":svm.SVC(gamma="auto"),"Gaussian Naive Bayes":GaussianNB(),"Decision Tree":tree.DecisionTreeClassifier(),
         "Random Forest":RandomForestClassifier(),"AdaBoost":AdaBoostClassifier(),
         "Gradient Boosting":GradientBoostingClassifier()}
kFolds = 40

def scoringFN(model,X,y):
    pred = model.predict(X)
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    return np.sum(y == pred)/y.shape[0]

for mod in models:
    model = models[mod]
    cross_val = cross_val_score(model, titanic_X, titanic_y,cv=kFolds,scoring= scoringFN).mean()
    print("{0:30} {1} Fold Cross-Validation Accuracy: {2:7f}".format(mod,kFolds,cross_val))


# From the results displayed above we can see that the gradient boosting method achieved the best results with the default hyperparameters. We will now adjust the parameters of the gradient boosting model and analyze the resulting learning curves.

# In[10]:


# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(20,12))
    plt.title(title,size=30)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples",size=15)
    plt.ylabel("Score",size=15)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Gradient Boosting Classifier Parameters (Partial List)
learning_rate     = 0.1
n_estimators      = 100
max_depth         = 3
min_samples_split = 2
subsample         = 1

# Define the gradient boosting classifier. 
gradBoost = GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators,
                                       max_depth=max_depth,min_samples_split=min_samples_split,subsample=subsample)

# Plot the learning curve of the gradient boosting classifier. 
plot_learning_curve(gradBoost,"Gradient Boosting Learning Curve",titanic_X,titanic_y,cv=5).show()

# Fit model on the full dataset. 
gradBoost.fit(titanic_X,titanic_y)


# ## 6. Simple Feed-Forward Neural Network
# We begin our neural network adventure by defining a very simple five layer feed-forward neural network. The architecture is as follows: 
# 
# 1. Input Layer: 12 inputs.
# 2. Hidden Layer: 20 sigmoid neurons. 
# 3. Hidden Layer: 20 sigmoid neurons. 
# 4. Hidden Layer: 20 sigmoid neurons. 
# 5. Output Layer: 1 sigmoid neuron. 
# 
# We will be using the [adam optimizer](https://www.coursera.org/learn/deep-neural-network/lecture/w9VCZ/adam-optimization-algorithm) with its default keras parameters and the [binary crossentropy](https://en.wikipedia.org/wiki/Cross_entropy) loss function. While a good understanding of the optimizer and loss function is essential for deep learning research, for our purposes we will pass over these details. 
# 
# We will also use the classification accuracy metric to evaluate how our network is performing: 
# $$\huge\frac{1}{N}\sum_{i=1}^N [\hat{y}_i == r(y_i)] $$
# $$
# \huge r(x) = 
# \left\{
# \begin{array}{ll}
#       0 & x\leq 0.5 \\
#       1 & otherwise \\
# \end{array} 
# \right. 
# $$

# In[12]:


# Network Definition.
# Here we use the keras functional api.
inputs = Input(shape=(titanic_train_x.shape[1],),name="input")
x = Dense(20,activation="sigmoid")(inputs)
x = Dense(20,activation="sigmoid")(x)
x = Dense(20,activation="sigmoid")(x)
out = Dense(1,activation="sigmoid", name="output")(x)

# Instantiate the network.
simpleModel = Model(inputs=inputs, outputs=out)

# Compile the network. 
simpleModel.compile(optimizer="adam",loss="binary_crossentropy", metrics=['acc'])

# Pretty print the details of the network. 
simpleModel.summary()


# ** Workshop question: ** Why does the first dense layer have 260 parameters? Why does the second layer have 420 parameters? Explain why we are seeing this number of parameters. 
# 
# We will now train the network on the training split and validate on the validation split.

# In[13]:


hist = simpleModel.fit(titanic_train_x, titanic_train_y,validation_data=(titanic_validation_x,titanic_validation_y), batch_size=30,epochs=30, verbose=1)


# In[14]:


def learningCurves(hist):
    histAcc_train = hist.history['acc']
    histLoss_train = hist.history['loss']
    histAcc_validation = hist.history['val_acc']
    histLoss_validation = hist.history['val_loss']
    maxValAcc = np.max(histAcc_validation)
    minValLoss = np.min(histLoss_validation)

    plt.figure(figsize=(12,12))
    epochs = len(histAcc_train)
    plt.plot(range(epochs),np.full(epochs,meanBaseline(titanicDF)),label="Unbiased Estimator", color="red")

    plt.plot(range(epochs),histLoss_train, label="Training Loss", color="#acc6ef")
    plt.plot(range(epochs),histAcc_train, label="Training Accuracy", color = "#005ff9" )

    plt.plot(range(epochs),histLoss_validation, label="Validation Loss", color="#a7e295")
    plt.plot(range(epochs),histAcc_validation, label="Validation Accuracy",color="#3ddd0d")

    plt.scatter(np.argmax(histAcc_validation),maxValAcc,zorder=10,color="green")
    plt.scatter(np.argmin(histLoss_validation),minValLoss,zorder=10,color="green")

    plt.xlabel('Epochs',fontsize=14)
    plt.title("Learning Curves",fontsize=20)

    plt.legend()
    plt.show()

    print("Max validation accuracy: {0}".format(maxValAcc))
    print("Minimum validation loss: {0}".format(minValLoss))
    
learningCurves(hist)


# ** Workshop Questions**  
# * Do the learning curves indicate that we have a good model? What would make a good model? 
# * Can you explain the difference between the loss and accuracy curves? Why are there periods where the loss is decreasing, but the accuracy is not? 
# * Depending on the train/validation split you may see the validation loss decrease faster than the training loss. What would explain this behavior? 

# ## 7. Adjustable Feed-Forward Neural Network
# In this section you will experiment with different neural network architectures and evaluate the resulting learning curves. 
# 
# **Workshop Activity** 
# 
# Adjust the hyperparameters below and evaluate how the impact the learning curves. 

# In[25]:


#### Training Hyperparameters ####
batch_size = 300
epochs = 1000

#### Model Hyperparameters  ####
nLayers = 3
layerSize = 80
dropoutPercent = 0.87# Regularization 

# Possible loss fuctions: https://keras.io/losses/
# mean_squared_error, mean_absolute_error,mean_absolute_percentage_error,mean_squared_logarithmic_error
# squared_hinge, hinge, categorical_hinge, logcosh, kullback_leibler_divergence, poisson, cosine_proximity
lossFn = 'binary_crossentropy'

# Possible optimizers: https://keras.io/optimizers/
# SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
optimizer = 'adam'

# Possible Activation Functions: https://keras.io/activations/
# elu, selu, softplus, softsign, relu, tanh, hard_sigmoid, linear
# Possible Advanced Activations: https://keras.io/layers/advanced-activations/
# LeakyReLU, PReLU, ELU, ThresholdedReLU
activationFn = 'sigmoid'


# Here we have a more sophisticated way of constructing networks parameterized by architectural hyperparameters you set above. 

# In[26]:


# Model Architecture 
def makeModel(inputShape,nLayers,layerSize,dropoutPercent,lossFn,optimizer):
    inputs = Input(shape=(inputShape,),name="input")
    x = None 
    
    for layer in range(nLayers):
        if x == None:
            x = inputs

        x = Dense(layerSize, activation=activationFn,name="fc"+str(layer))(x)
        x = Dropout(dropoutPercent,name="fc_dropout_"+str(layer))(x)

    out = Dense(1,activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizer,
                  loss=lossFn,
                  metrics=['acc'])
    
    return model

modelMain = makeModel(titanic_train_x.shape[1],nLayers,layerSize,dropoutPercent,lossFn,optimizer)
modelMain.summary()


# In[27]:


hist = modelMain.fit(titanic_train_x, titanic_train_y,validation_data=(titanic_validation_x,titanic_validation_y), batch_size=batch_size,epochs=epochs, verbose=0)
learningCurves(hist)


# ** Workshop Questions**  
# * Do the learning curves indicate that we have a good model? What would make a good model? 
# * Can you explain the difference between the loss and accuracy curves? Why are there periods where the loss is decreasing, but the accuracy is not? 
# * Depending on the train/validation split you may see the validation loss decrease faster than the training loss. What would explain this behavior? 

# ## 8. Cross-Validation
# If our learning curve indicates healthy learning ( i.e. the training and validation metrics do not diverge ), we may want to use a cross-validation method in order to get a more accurate estimation of how well our model generalizes to new data. We will use a technique called called k-fold cross-validation

# In[ ]:


# Cross-Validation Parameter 
kFolds = 3

kfold = StratifiedKFold(n_splits=kFolds, shuffle=True)
means = []
stds = []
lossesLs = []
accuracyLs = []

runningLoss = []
runningAccuracy = []

# Train on k-folds of the data. 
for train, test in kfold.split(titanic_X, titanic_y):
   
   # Create new instance of our model. 
   model = makeModel(titanic_X.shape[1],nLayers,layerSize,dropoutPercent,lossFn,optimizer)
   
   # Train the model on this kfold. 
   model.fit(titanic_X[train], titanic_y[train],batch_size=batch_size,epochs=epochs, verbose=0)

   # Evaluate the model
   loss,acc = model.evaluate(titanic_X[test], titanic_y[test], verbose=0)
   
   # Log Cross-Validation Data
   lossesLs.append(loss)
   accuracyLs.append(acc)
   mean = np.mean(lossesLs)
   std = np.std(lossesLs)
   
   accuracyMean = np.mean(accuracyLs)
   accuracyStd = np.std(accuracyLs)
   
   runningLoss.append(mean)
   runningAccuracy.append(accuracyMean)
   
   print("Loss: %.2f%% (+/- %.2f%%) | Accuracy: %.2f%% (+/- %.2f%%)" % (mean*100,std,accuracyMean*100,accuracyStd))

plt.show()


# **Workshop Questions** 
# * Should we use more k-folds? How would you make this decision? 
# * Does the cross-validated accuracy and loss differ from the validation/training metrics above? Can you explain why the are different or the same? 

# ## 9. Application to Test Set
# Now that we have designed, trained, and evaluated our neural network, we will apply it to the training set and save the results to a csv file which can be uploaded to kaggle for scoring.

# In[23]:


# The thrshold function which assigns a class of 0 or 1 based on the sigmoid output of the network. 
def thresholdFn(x):
    if(x < 0.5):
        return 0
    else:
        return 1

pred = modelMain.predict(np.stack(testDF.apply(preprocessRow,axis=1)))
    
# Save the predictions to a CSV file in the format suitable for the competition. 
data_to_submit = pd.DataFrame.from_items([
    ('PassengerId',testDF["PassengerId"]),
    ('Survived', pd.Series(np.hstack(pred)).map(thresholdFn))])

data_to_submit.to_csv('neuralNet.csv', index = False)


# ## 10. Next Steps
# This notebook introduced a simple workflow of the stages involved in modeing categorical and categorical tabular data with neural networks. Here are some of the next steps you can take to make your analysis more robust:
# 
# * Implement a hyperparameter search procedure. A common approach is grid search.
# * Add more features from the training set. In this notebook we ignore several columns like name and cabin. Figuring out how to include these features may help make better predictions. 
# 
