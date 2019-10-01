#!/usr/bin/env python
# coding: utf-8

# **Machine Learning innovation Lab Manifest**
# 
# With this "paper" or so to call it, I would like to state my objectives and general introduction of myself and what i'll try to achieve. 
# I am studying business on Faculty of organizational sciences at Belgrade, Serbia. Altough im enthusiastic about business with focus on several different parts of it (finance, startups, tech, green energy), i wanted to understand how software development works in order to better understand people i want to work with so i learned some of python programming. After spending some time coding and learing basic principles, i naturally found out the benefits of python compared to other langagues. I then read about machine learning, statistics and math required to understand how can you use algorithms on data your business creates, to improve it. I did some reading and several minor projects, just enought to get interested and realize it's potential. I am terrible coder and most of software engineers would probably feel sick looking at my code, but with all things i want to do, I don't have time focusing on optimization. I would like to focus on the code readability so that even non-software engineers would be able to read some of the code and understand it's simplicity. Enough with the introduction, for all further questions, you can contact me or see some of my work:
# 
# Mail: nemanja.boskovic17@gmail.com
# LinkedIn: https://www.linkedin.com/in/nemanja-boskovic/
# Facebook: https://www.facebook.com/boskovicnemanja17
# Github: https://github.com/nemanjab17
# Kaggle: https://www.kaggle.com/nemanjab17
# 
# What do I want to achieve?
# 
# There are several types of people I want to address by doing these (research, blog, comments, what is this anyway?):
# 
# **1. Business students, etrepreneurs**
# 
# We hear "Machine learning", "AI", "Deep learning" and we know that is has huge potential but most of us think of it as some kind of rocket science reserved for PhD's and scientists that are waaay more educated and briliant in quantitative fields than us. And don't get me wrong, there are plenty of people with this profile that are working in this field and provides us with new models, algorithms and apporaches on a daily basis, these guys are creating something that later drives business innovation. But not all of us has to know every bit (see what i did there?) and peace of those algortihms and models. Since they spend years coming up with new apporaches, it is normal that they do not focus on creating companies and exploiting the merrits these things gives us. There has to be new wave of innovators and business thinkers educated about tech just enough to spark some real innovation. There is more to it than this initial set up I layed so I would like to address **how important it is for anyone to contact me and give me thoughts on this innovation driving model of researchers and entrepreneurs.** 
# 
# **2. Engineers, engineering students, coders etc**
# 
# There is some basic understanding of machine learning needed to be able to understand the application of ML in real life. But in order to properly execute innovative projects, there has to be bridge made between business application and models and algorithms available. This is done by coders who are especially good at creating so called "piplines" I will explain later. In plain english, there has to be real code created that puts your data into right algorithm and prints the output so that you acctually generate some insights or forecasts in the project you are doing.** I would invite coders to contact me if they have any interest in Machine Learning, since there are many opportunities to collaborate.
# **
# **3. Teachers, universities which are backbone of this model**
# 
# Since we are depending here on knowledge, academia is very important in this set up. There has to be lots of mentorship, guidance, advices and general friendships involved between students, academia and industry in order to create successful Machine learning innovation lab. **I would like to address all academic enthusiasts to be kind to contact me with any piece of advice on how to implement this kind of model, what could be some challenges and how to overcome them?**
# 
# **4. Finally, companies**
# 
# All participants up to this point are irrelevant if there is no strong relationship with the industry. The companies are the driving force of innovation as they own the infrastructure to bring innovations to the users. Retail, Media, Entertainment, Healthcare, Transportation, Financial services and many more industries are enough to spark your imagination and realize industry importance**. I would like to invite industry to join this educational series in order to better understand potential application of ML in current business conditions. **
# 
# **Why am i writing on a platform that has wierd stuff around the text box?**
# 
# Because it allows me to code, show results and type with relative ease. In this article (i agreed with myself on how to call this) I want to show example of ML use case with detailed explanation of every single step.
# 
# **What is kaggle?**
# 
# Kaggle is simple website that organizes competitions on which companies can provide data, tasks and rewards for the winners. Kaggle then hosts the competitions and you and I can all create a team that competes. There is always 10ish competitions around. Some data scientists use kaggle as a great career pusher because winning a competiton here is really a big deal. 
# 
# The other benefits of using Kaggle is that it is free, it offers lots and lots of different datasets(data usually stored in excel tables form) provided by people and u can code using Kaggle's resources, which means you only need Google Chrome! You can also upload your own datasets and play around. Since Kaggle team is not sponsoring this, i will stop with braggig about Kaggle. 
# 
# 
#  I would like to lay feedback form right here, for all those who actualy enjoyed the first part and are not interested in the code and summary of this article. Even if you don't like the code if you keep reading, you will be able to pick up on this concept I am talking about, which is my goal with this article. 
# 
# If you do not wish to continue, i would kindly ask you to fill the form:
# <insert form here
# 
# I truly believe that it's writers obligation to keep readers interested, so if you kept on reading, I promise that i'll try. 
# 
# In order to explain the concept of innovation lab, i want to explain the use case of machine learning with Titanic data. As we all know there has been a tragedy at titanic and some of the passangers didn't survive. We have data about a part of the passangers and kaggle team has the other. They want us to learn on the data they gave us, to predict who is a surviver from the data they have (like a puzzle). 
# 
# So to generalize from titanic situation, we have **the problem:**
# 
# "Can we predict who is likely to survive shipwrecks?"
# 
# **Results:**
# 
# Using data gathered from titanic dataset (which is probably fictive), i used one of the algorithms to predict outcome of each passanger (survived or not survived) with around 78% accuracy (not officialy stamped by kaggle team, take this with certain degree of suspicion). Some of the guys on Kaggle managed to do it with around 85% accuracy, and I can also provide information on how they did that.
# 
# So let's look at the process of producing this kind of information(in this case, predictions) that is valuable to many organizations:
# 
# 1. Gather the data
# 2. Analyze the data
# 3. Make changes to data(more on this later)
# 4. Try algorithms 
# 5. Get the results
# Iterate steps 3 - 4 - 5 until you solved the problem successfuly
# 
# Now, many might argue about this, I would like to say this is my own process made to simplify everything. 
# Now i'll try to use this framework to explain how i solved titanic passengers prediction problem with certain degree of success.
# 
# 1. Gathering the data:
# In real setting, gathering the data is much more difficult, it involves gogin through company databases, playing with SQL and other database technolgies to extract all the data and get it ready. In this case, we have a dataset that look like this:
# 
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
import itertools

warnings.filterwarnings('ignore') #ignore warning messages 

import os
data_train_file = "../input/train.csv"
data_test_file = "../input/test.csv"

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)

#SibSP values (1,0,3,4,2,5,8)
#891 observations in train set and 418 in test set
#test set has no outcome variable
#there is 248 unique fares, how do they impact their probability of death? Higher fares, better cabins? Or did they get boats because they were "VIP"?
#male and female change to 1 and 0
#ticket columns random numbers and letter
#

df_train.Sex.replace(to_replace = dict(male = 1, female = 0), inplace = True)
df_test.Sex.replace(to_replace = dict(male = 1, female = 0), inplace = True)
df_train.dropna()
df_test.dropna()
dummies = pd.get_dummies(df_train['Embarked'])
df_train = df_train.join(dummies, how='outer')
df_train = df_train.drop('Embarked',axis=1)
df_test = df_test.join(dummies, how='outer')
df_test = df_test.drop('Embarked',axis=1)
df_train.head()
# Any results you write to the current directory are saved as output.


# Each row represents a unique passanger and each column is a so called feature, or characteristic. I'll explain some of the features, SibSp is the number of siblings and spouses a peson has onboard, Parch is number of parents and/or children a person has onboard. Feeatures can be nubers, categories and other types, and this part is important for engineers. The key thing here when it comes to success of the whole project is to think about all the factors that affect some variable, in this case survival. When you think about it, is the age really affecting survival? Did old people have difficulties finding boats, did the crew coordinate boarding and they gave priority to babies? These are the things we might find out, but not necessarilly, there are some models that take all this to consideration and figure things out on their own. Now it's time to run some code and get some visual distributions of the data. Now is the time to lay out some coding principles. This is important to acknowledge in order to understand the degree of coding needed to run projects like this. There are libraries written in python which we use to everything we do. That means someone did a lot of coding and made a simple manual on how to do things. For visualisation i am using seaborn library. I won't get to specific on each line of the code but get this, I am simply selecting type of the graph and selecting the data i want to show. It' really that easy. 
# 
# 1. Gather the data
# 
# **2. Analyze the data**
# 3. Make changes to data(more on this later)
# 4. Try algorithms 
# 5. Get the results

# In[ ]:


def plot_distribution(data_select) : 
    sns.set_style("ticks")
    s = sns.FacetGrid(df_train, hue = 'Survived',aspect = 2.5, palette ={0 : 'lightskyblue', 1 :'gold'})
    s.map(sns.kdeplot, data_select, shade = True, alpha = 0.5)
    s.set(xlim=(0, df_train[data_select].max()))
    s.add_legend()
    s.set_axis_labels(data_select, 'proportion')
    s.fig.suptitle(data_select)
    plt.show()
    
def plot_distribution2(data_select) : 
    sns.set_style("ticks")
    s = sns.FacetGrid(df_train, hue = 'Survived',aspect = 2.5, palette ={0 : 'lightskyblue', 1 :'gold'})
    s.map(sns.kdeplot, data_select, shade = True, alpha = 0.5)
    s.set(xlim=(0, 200))
    s.add_legend()
    s.set_axis_labels(data_select, 'proportion')
    s.fig.suptitle(data_select)
    plt.show()


# In[ ]:


plot_distribution('Age')
plot_distribution('SibSp')
plot_distribution('Parch')
plot_distribution2('Fare')


# We can now take a low at the data for explanatory purposes. I created graphs that show values of some features for "survived" and "not survived". On the first graph we can see that most of the youngs survived, and that the curve of distribution for those who not survived is more centered around young healthy individuals in their 20s. On the graph for "Fare" we can see that the group that didn't survive is really grouped around low fare amount. 
# 
# As part of my research, i did some more analyses that i didn't want to share in order to avoid confusion because of general purpose of this article, but if anyone is interested, I am more than willing to share more. 

# In[ ]:


# Correlation matrices by outcome
f, (ax1, ax2) = plt.subplots(1,2,figsize =( 18, 8))
sns.heatmap((df_train.loc[df_train['Survived'] ==1]).corr(), vmax = .8, square=True, ax = ax1, cmap = 'magma', linewidths = .1, linecolor = 'grey');
ax1.invert_yaxis();
ax1.set_title('Yes')
sns.heatmap((df_train.loc[df_train['Survived'] ==0]).corr(), vmax = .8, square=True, ax = ax2, cmap = 'YlGnBu', linewidths = .1, linecolor = 'grey');
ax2.invert_yaxis();
ax2.set_title('No')
plt.show()


# In[ ]:


palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'grey'


# In[ ]:


fig = plt.figure(figsize=(12,12))

plt.subplot(221)
ax1 = sns.scatterplot(x = df_train['Age'], y = df_train['Fare'], hue = "Survived",
                    data = df_train, palette =palette, edgecolor=edgecolor)
plt.title('Age vs Fare')


# In[ ]:


# PCA
df_train.dropna()
df_train = df_train[np.isfinite(df_train['Age'])]
#df_train = df_train[~np.isnan(df_train)]

target_pca = df_train['Survived']
data_pca = df_train.drop(['Survived', 'PassengerId','Pclass','Name','Ticket','Cabin'], axis=1)

print(data_pca.columns.values)

#To make a PCA, normalize data is essential
X_pca = data_pca.values

X_std = StandardScaler().fit_transform(X_pca)

# Select 6 components
pca = PCA(n_components = 8)
pca_std = pca.fit(X_std, target_pca).transform(X_std)


# In[ ]:


colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgrey','gold','gold','gold']
explode = (0.1, 0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1)
labels = ['comp 1','comp 2','comp 3','comp 4','comp 5','comp 6','comp 7','comp 8']

plt.figure(figsize=(25,12))
plt.subplot(121)
ax1 = plt.pie(pca.explained_variance_ratio_, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', textprops = { 'fontsize': 12}, shadow=True, startangle=140)
plt.title("PCA : components and explained variance (5 comp)", fontsize = 20)
plt.show()


# 1. Gather the data
# 2. Analyze the data
# 
# **3. Make changes to data**
# 4. Try algorithms 
# 5. Get the results
# 
# In this third step, we want to think about the algorithms we can use to solve our problem. There are many many algorithms around, all developed by researchers in the last 50-60 years. I recommend reading "The Introduction to Statistical Learning" By Gareth James et. al. It teaches overall concepts that are important for engineers, but i recommend entrepreneurs also do quick reading in order to better understand problems engineers face in the development process. Key here is to understand that different algorithms require different types of data. Lets first try and use Logistic regression. It's fairly simple algorithm that assumes linear relationship between features and outcome variable. It is based on linear regression with some adjustments made to be usable on predicting categorical values (0 and 1). Let's try and explain linear regression. 
# If we wanted to predict Salary of a football player, we could simply use data about number of goals he score so far. I know, you're already thinking, but there are many other factors here that affect salary. Yes, you're right but it's easy to explain. We could simply create a small function 
# $$ y= a+ b*x $$
# 
# Y is our desired number, salary, and x is number of goals. So if let's say football managers gave 100 bucks to their players for each goal they scored, we could perfectly predict their salary right? In that case b would equal 100 and it's solved, bye bye. In reality it's not like that so we need to try and come up with this estimate and see what happens. The first a in equation is helping us set up a base, because players who do not score anything must also get some salary. We now have another problem at hand, what a and b should we set in order to predict salary the best? We can try each of the millions of combinations of these "parameters" to come up with best, and we will right? So we need to somehow see how good the parameters are. Let's say our client gives us 1000 examples of players, their salaries and number of goals scored. We can make predictions based on equation and see how much did we miss right? We can simply make 1000 predictions and see the absolute difference between real values and the ones we predicted. It's really that simple, sum up all the salaries in the league and deduce sum of all predicted salaries, we get total error we made. It's probably millions? So we can now change a and b (instead of thinking its 100 per goal, lets try 1000) and see the difference now. Well we can create and algorithm that will try many possibilities and get the best parameters based on this kind of thinking. Well in reality, people made several sophisticated algorithms that don't need to do all that to get to the results quicker. Now here lies some important insight, researchers really focus on finding new algorithms that do same things but quicker, none of the practicioners who wan't to focus on application of machine learning, doesn't have to bother with writing and adjusting algorithms. Here lies the **first example of knowledge depth that im thinking about. There are people who know lots of machine learning, and people who know nothing about it, but there is a lack of people who know some basic stuff about it, enough to make exciting things happen.**
# 
# We now understand the overall concept of linear regression, all we have to know is that it assumes that if someone is 2 times better at something then someone else, linear regression would predict the value of the first person in that fashion, it is really strict and not flexible at all. Now think of the problems that we could solve with these. There should be genuine number of people who get to this kind of dept and never get any deeper, but then research about other alogorithms and their premises. Logistic regression has several differences compared to linear regression but the logic is similar. 
# 
# Now we create some visual representation of the results that we will use later to show how good was our logistic regression. You can find plenty of code online, on other Kaggle Kernels (kaggle books) and adjust code to your dataset. Read some of the comments in code sections if you are interested. 

# In[ ]:


#creating confusion matrix, it shows number of rows survived and not survived,
#compared to our predicted values, look it up online or just wait for a clear picture, it is understandable
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix"',
                          cmap = plt.cm.Blues) :
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
#choosing between different options, selecting data shown etc.. 
#really technical part
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Show metrics 
def show_metrics():
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Precision =     {:.3f}'.format(tp/(tp+fp)))
    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))
    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/
                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))


# In[ ]:


# Def X and Y
y = np.array(df_train.Survived.tolist())
X = np.array(data_pca.as_matrix())
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 1. Gather the data
# 2. Analyze the data
# 3. Make changes to data
# 
# **4. Try algorithms** 
# 5. Get the results
# 
# Key here is to remember that this code is completely reusable and found online very easily. It's not something that requires much work. Now we want to load algorithm from external library. This is the part I talked about, we already know there is code written and stored in library. We simply use the line /from sklearn.linear_model import LogisticRegression/ and look at instructions on how to use it. This is generaly done by engineers, and this is the first part of the pipline I already mentioned. Engineers navigate data thorugh algorithms. 

# In[ ]:


#import model from the library
from sklearn.linear_model import LogisticRegression
#random generator class, this part is not of concern to general public
random_state = 600
#selecting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state = random_state)
#call upon algorithm
log_clf = LogisticRegression(random_state = random_state)
#give it parameters, this is engineers only concern
param_grid = {
            'penalty' : ['l2','l1'],  
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
#engineers only concern
CV_log_clf = GridSearchCV(estimator = log_clf, param_grid = param_grid , scoring = 'accuracy', verbose = 1, n_jobs = -1)
CV_log_clf.fit(X_train, y_train)
#give us the best a and b in football players analogy
best_parameters = CV_log_clf.best_params_
print('The best parameters for using this model is', best_parameters)


# In[ ]:


#this is where we connect our parameters givven with matrix we created, so now we can see the results.
#Log with best hyperparameters
CV_log_clf = LogisticRegression(C = best_parameters['C'], 
                                penalty = best_parameters['penalty'], 
                                random_state = random_state)

CV_log_clf.fit(X_train, y_train)
y_pred = CV_log_clf.predict(X_test)
y_score = CV_log_clf.decision_function(X_test)

# Confusion maxtrix & metrics
cm = confusion_matrix(y_test, y_pred)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, 
                      classes=class_names, 
                      title='Logistic Confusion matrix')
plt.savefig('6')
plt.show()

show_metrics()



# So now we get to understand confusion matrix, on the left axis we have true labels or true outcome, and on the X axis we have predicted, 41 in the matrix means, there are 41 person which were 0 and we predicted 0. If you take a look at the whole matrix, you can understand other fields, For a 100% accurate model, there would be no situations where we predicted 0 and it was acctualy 1. In this example we have accuracy of 0.79 which is really nice. There are some considerations here of course, but getting into more details would probably confuse a whole lot of people. 
# 
# Let's take a look at this prossess from the entrepreneur's aspect. Image that you are using Google Adwords to promote your campaigns. You have several geographic markets and need to map out how much money you want to invest in each market. Your outome is simple, number of items sold in each market at a given day. But every marketer knows that even if you put large amount of money in a certain campaign, it doesn't really increase sales linearly. There are factors affecting buying decision. Lets look at another example. Image that you own a hotel, and you have to adjust prices depending on the demand and supply. One weekend there is nothing special going on in Belgrade, so the demand for room is not very high, prices should be set lower than usual. But on other occasions, there are major premier league games going on, city is full of tourists, demand is very high, so if you leave the price as it is, you earn average revenue but later you find out that your competing hotel rose its room prices and still sold all the rooms because demand was so high! If only you knew about all the factors that affect room demand, you would know how much to price and earn maximum revenue. Entrepreneur with ML mindest could make assumptions about which data would be sufficient to predict demand and could contact peers from ML Innovation Lab who have data engineering skills and they could together assess possibilities. Colaborative innovation design. 

# In[ ]:


log2_clf = LogisticRegression(random_state = random_state)
param_grid = {
            'penalty' : ['l2','l1'],  
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            }

CV_log2_clf = GridSearchCV(estimator = log2_clf, param_grid = param_grid , scoring = 'recall', verbose = 1, n_jobs = -1)
CV_log2_clf.fit(X_train, y_train)

best_parameters = CV_log2_clf.best_params_
print('The best parameters for using this model is', best_parameters)


# In[ ]:


CV_log2_clf = LogisticRegression(C = best_parameters['C'], 
                                 penalty = best_parameters['penalty'], 
                                 random_state = random_state)


CV_log2_clf.fit(X_train, y_train)

y_pred = CV_log2_clf.predict(X_test)
y_score = CV_log2_clf.decision_function(X_test)
# Confusion maxtrix & metrics
cm = confusion_matrix(y_test, y_pred)
class_names = [0,1]


# In[ ]:


#Voting Classifier
voting_clf = VotingClassifier (
        estimators = [('log1', CV_log_clf), ('log_2', CV_log2_clf)],
                     voting='soft', weights = [1, 1])
    
voting_clf.fit(X_train,y_train)

y_pred = voting_clf.predict(X_test)
y_score = voting_clf.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [0,1]
show_metrics()


# If we assume there is no linear relationship between features and outcome, we must find algorithms that lay this non-linear hypothesis. One easy to understand non-linear algorithm is Random Forest. This part  ismore interesting to engineers than others, but i encourage everyone who is interested in the field to continue on reading. We first introduce Decision tree as a basc building block. Decision tree is very simple concept. It allows us to create division pattern that gives us unique cateogries of data which we use to classify. For titanic example, we can take a loook at the data and see that the greatest decision factor is maybe location of passengers cabin. We then find cabins that are close to the exit and make a rule: if passenger is in the cabin close to exit, predict survival, if its very far, predict death. We could then take these subgroups and again divide them on the horizon of another feature. We create divison patterns that let us classify our passangers. This is where things get especially interesting for the engineers, but Decision Trees are proun to overfitting, that means they adapt to the training data set (passangers we use to find our model parameters) and if we change the dataset, maybe add some more passangers or leave some out, tree can be totaly different. So how do we decide on which data to include or not? I encourage everyone interested to contact me for further information and discussion, as well as reading material I can recommend.  Now let's share some results from decision trees: 

# In[ ]:


def GridSearchModel(X, Y, model, parameters, cv):
    CV_model = GridSearchCV(estimator = model, param_grid = parameters, cv = cv)
    CV_model.fit(X, Y)
    CV_model.cv_results_
    print("Best Score:", CV_model.best_score_," / Best parameters:", CV_model.best_params_)
    
# Learning curve
def LearningCurve(X, y, model, cv, train_sizes):

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv = cv, n_jobs = 4, 
                                                            train_sizes = train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std  = np.std(train_scores, axis = 1)
    test_scores_mean  = np.mean(test_scores, axis = 1)
    test_scores_std   = np.std(test_scores, axis = 1)
    
    train_Error_mean = np.mean(1- train_scores, axis = 1)
    train_Error_std  = np.std(1 - train_scores, axis = 1)
    test_Error_mean  = np.mean(1 - test_scores, axis = 1)
    test_Error_std   = np.std(1 - test_scores, axis = 1)

    Scores_mean = np.mean(train_scores_mean)
    Scores_std = np.mean(train_scores_std)
    
    _, y_pred, Accuracy, Error, precision, recall, f1score = ApplyModel(X, y, model)
    
    plt.figure(figsize = (16,4))
    plt.subplot(1,2,1)
    ax1 = Confuse(y, y_pred, classes)
    plt.subplot(1,2,2)
    plt.fill_between(train_sizes, train_Error_mean - train_Error_std,train_Error_mean + train_Error_std, alpha = 0.1,
                     color = "r")
    plt.fill_between(train_sizes, test_Error_mean - test_Error_std, test_Error_mean + test_Error_std, alpha = 0.1, color = "g")
    plt.plot(train_sizes, train_Error_mean, 'o-', color = "r",label = "Training Error")
    plt.plot(train_sizes, test_Error_mean, 'o-', color = "g",label = "Cross-validation Error")
    plt.legend(loc = "best")
    plt.grid(True)
     
    return (model, Scores_mean, Scores_std )

def ApplyModel(X, y, model):
    
    model.fit(X, y)
    y_pred  = model.predict(X)

    Accuracy = round(np.median(cross_val_score(model, X, y, cv = cv)),2)*100
 
    Error   = 1 - Accuracy
    
    precision = precision_score(y_train, y_pred) * 100
    recall = recall_score(y_train, y_pred) * 100
    f1score = f1_score(y_train, y_pred) * 100
    
    return (model, y_pred, Accuracy, Error, precision, recall, f1score)  
    
def Confuse(y, y_pred, classes):
    cnf_matrix = confusion_matrix(y, y_pred)
    
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis = 1)[:, np.newaxis]
    c_train = pd.DataFrame(cnf_matrix, index = classes, columns = classes)  

    ax = sns.heatmap(c_train, annot = True, cmap = cmap, square = True, cbar = False, 
                          fmt = '.2f', annot_kws = {"size": 20})
    return(ax, c_train)

def PrintResults(model, X, y, title, limitleafs=2):
    
    model, y_pred, Accuracy, Error, precision, recall, f1score = ApplyModel(X, y, model)
    
    _, Score_mean, Score_std = LearningCurve(X, y, model, cv, train_size)
    Score_mean, Score_std = Score_mean*100, Score_std*100
    
    
    print('Scoring Accuracy: %.2f %%'%(Accuracy))
    print('Scoring Mean: %.2f %%'%(Score_mean))
    print('Scoring Standard Deviation: %.4f %%'%(Score_std))
    print("Precision: %.2f %%"%(precision))
    print("Recall: %.2f %%"%(recall))
    print('f1-score: %.2f %%'%(f1score))
    print('Limited leafs:' + str(limitleafs))
    print(' ')
    
    
    Summary = pd.DataFrame({'Model': title,
                       'Accuracy': Accuracy, 
                       'Score Mean': Score_mean, 
                       'Score St Dv': Score_std, 
                       'Precision': precision, 
                       'Recall': recall, 
                       'F1-Score': f1score,
                       'Limited leafs': limitleafs}, index = [0])
    return (model, Summary)


# In[ ]:


train_size = np.linspace(.1, 1.0, 15)
cv = ShuffleSplit(n_splits = 100, test_size = 0.25, random_state = 0)

classes = ['Dead','Survived']
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
model = DecisionTreeClassifier()
model,Summary_DT = PrintResults(model, X_train, y_train, 'DT')


# We can see confusion matrix and how well our algorithm performed! But there is a catch, this matrix shows errors in training dataset, that means, trees adapted to dataset beautifully, it found pattern of division that makes very good classificication groups. But if we were to add new passengers or take different boat accident, we would get poor results. On the graph above, you can see error rate for "Cross-validation error", it's also very easy concept but without getting to technical, it simulates real prediction situations and we can take it's result as average accuracy we would get if we used this model to predict future observations. Scoring accuracy of 73% is not really good, it's even lower than logistic regression. The reason why I introduced this concept is to outline the relevance of data engineers in this Lab proposal. Using written code to do Machine Learning leads to a conclusion that everyone can just pipeline data to the algorithms and thats it! But in reality there is much more to it. For example, depth of knowledge i provided by now is not enough to optimize the algorithm to perform better which is almost always essential to a successful ML Project. Now I would like to state that my domain of knowledge is limited and would like to kindly ask readers to take further experiments and conclusions with caution, as I am not engineer and my only goal is to help everyone interested understand what layers of knowlodge can create, not teach concepts and show any kind of ML work.  

# As we can see from the graph, Cross-validation error is still very high and we are dedicating our attention to tuning our model to decrease this error.

# I already introduced some intermediate concepts to the story, so showing code could really impact readability. Let's focus on the process itself. It is now required from the coder to understand how Random Forrests work and what parameters it uses. Advanced practicioners can use several optimzation techniques to tune the parameters and find best performing Random Forest model to predict the data.

# In[ ]:



#model = DecisionTreeClassifier(criterion='entropy',min_samples_split=5)
#model,Summary_DT = PrintResults(model, X_train, y_train, 'DT')
accu = []
prec= []
rec = []
tre = []
for i in range(2,25):
    model = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=i)
    model,Summary_DT = PrintResults(model, X_train, y_train, 'DT',i)
    prec.append(float(Summary_DT['Precision']))
    rec.append(float(Summary_DT['Recall']))
    tre.append(float(Summary_DT['Limited leafs']))
    accu.append(float(Summary_DT['Accuracy']))
print(len(accu),len(prec),len(rec),len(tre))

data = pd.DataFrame(
    {'Limited leafs': tre,
     'Precision': prec,
     'Recall': rec,
     'Accuracy': accu
    })
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Limited leafs',y='Precision',data=data,color='lime',alpha=0.8)
sns.pointplot(x='Limited leafs',y='Recall',data=data,color='red',alpha=0.8)
sns.pointplot(x='Limited leafs',y='Accuracy',data=data,color='blue',alpha=0.8)
plt.title('PRA',fontsize = 20,color='blue')
plt.grid()

    
    


# In[ ]:


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# **Engineers only**
# We can now use RandomizedSearchCV, algorithm that for a given set of RF parameters such as leaf limit, max depth, number of features etc.. and value range for each parameter, tries every parameter combination thorugh Cross-Validation and gives back parameter set with max success rate, in literature this is called Tree pruning. We just import from SKLearn and select parameters and ranges.  I recommend reading Elements of Statistical Learning for more details about Random Forest.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
param_dist = {"max_depth": [3, 5],
              "max_features": sp_randint(1, 6),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11)
             }
# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(X_train,y_train)
report(random_search.cv_results_)


# We can see top tree parameter combinations. Now we can use parameter set given by top ranked model.

# In[ ]:


model = DecisionTreeClassifier(max_depth=5,max_features=4,min_samples_leaf=4,min_samples_split=3)
model,Summary_DT = PrintResults(model, X_train, y_train, 'DT')


# As we can see, our scoring accuracy increased by 4%. 

# In[ ]:




