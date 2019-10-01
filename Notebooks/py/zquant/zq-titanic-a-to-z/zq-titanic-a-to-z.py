#!/usr/bin/env python
# coding: utf-8

# <h2> Titanic: Machine Learning from Disaster </h2>
# 
# <p> In this Kernel, we explore the Titanic dataset and look to divide those that are statistically favourable to survive amongst all travelers. In the following report, we cover;
#         <ol>
#             <li>Import/Split the data </li>
#             <li>Data Visualation & Interpretation </li>
#             <li>Data Cleaning & Filtering </li>
#             <li>Feature Engineering/Selection </li>
#             <li>Model Selection & Performance </li>
#             <li>Out of Sample Tests </li>
#             <li> Conclusion </li>
#         </ol>
#         
#    <i>All comments and suggestions are welcomed and appreciated! Enjoy :-D</i>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
#Preprocessing
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
#Models & Metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.model_selection import GridSearchCV
get_ipython().magic(u'matplotlib inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# <h1> 1. Import/Split the data </h1>
#   
# <p>We want to immediately split the dataset into testing and training data. It is good practice to set the indexing as 'PassengerId'.</p>

# In[ ]:


dataset = pd.read_csv('../input/train.csv').set_index('PassengerId', drop=True)
testset = pd.read_csv('../input/test.csv').set_index('PassengerId', drop=True)


# In[ ]:


dataset.head()


# In[ ]:


testset.head()


# <p> From the dataset previews above, we gain an understanding of the features we have in our dataset and the kind of features they are. Let's keep it clean and set the feature types appropriately. </p>

# In[ ]:


dataset.info()


# In[ ]:


dataset.Sex = dataset.Sex.astype('category')
testset.Sex = testset.Sex.astype('category')

dataset.Embarked = dataset.Embarked.astype('category')
testset.Embarked = testset.Embarked.astype('category')

dataset.info()


# <p> Immediately, we see that Age and Cabin have less rows than others. We should check these for NaN
#  values. </p>
#  
#  <h2>a. What we dont know about the data </h2>

# In[ ]:


f, ax = plt.subplots(1, 2, figsize=(14, 8))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis', ax=ax[0])
sns.heatmap(testset.isnull(), cbar=False, cmap='viridis', ax=ax[1])


# In[ ]:


print('NaN Value Counts\n\nTraining Set:\n%s\n\nTesting Set:\n%s' %       (dataset.isnull().sum(), testset.isnull().sum()))


# <p> We'll need to do some hole plugging in our datasets seeing as there are missing values in both the training & testing portions. The heatmap and counts above help us understand the missing pieces of data. Columns like Cabin have too many NaN values to be able to fill the rest with anything accurate. It may be wise to drop this column all together. </p>
# 
# <h2> b. What we do know about the data </h>
# 
# <p> <br>We'll start by looking at the class imbalance between Didn't Survive(0) and Survived(1).</p>

# In[ ]:


train_deceased = dataset[dataset.Survived==0]
train_survived = dataset[dataset.Survived==1]
labels = 'Deceased', 'Survived'
sizes = [train_deceased.shape[0], train_survived.shape[0]]
colors = ['lightcoral', 'lightskyblue']
explode = (0.05, 0)  # explode 1st slice
 
fig = plt.figure(figsize=(14, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Deceased vs Survived Class Imbalance')
 
plt.axis('equal')
plt.show()


# <p> We'll have to adjust our models to account for the class imbalance seen in the training set. These proportions can be used as the prior probability of classes. We hope for the testing set to have a similar probability of occurence so that our model generalizes well out of sample. 
# <br><br> Next, we look at the individual features and their correspondance to the classes. These plots should give us an understanding of a certain features ability to predict if the person survived. </p>

# In[ ]:


fig = plt.figure(figsize=(24, 15))

plt.subplot(331)
sns.distplot(train_deceased.Age.dropna(), kde=False, color='Red')
sns.distplot(train_survived.Age.dropna(), kde=False, color='Blue')
plt.title('Survival Age Distribution')
plt.ylabel('Frequency')

plt.subplot(332)
sns.distplot(train_survived.Fare, kde=False, color='Blue')
sns.distplot(train_deceased.Fare, kde=False, color='Red')
plt.title('Survival Fare Distribution')
plt.ylabel('Frequency')
plt.xlim(0, 200)

fig, ax = plt.subplots(1, 3, figsize=(14, 8))

g = sns.factorplot(x='Embarked', kind='count', data=dataset, hue='Survived', ax=ax[0], legend=False)
g.ax.set_axis_off()
ax[0].set_title('Embarkment Location Survival')

g = sns.factorplot(x='Pclass', kind='count', data=dataset, hue='Survived', ax=ax[1], legend=False)
g.ax.set_axis_off()
ax[1].set_title('Ticket Class Survival')

g = sns.factorplot(x='Sex', kind='count', data=dataset, hue='Survived', ax=ax[2], legend=False)
g.ax.set_axis_off()
ax[2].set_title('Sex Survival')


# In[ ]:


sns.factorplot(x='Parch', kind='count', hue='Survived', data=dataset, orient='v')
plt.title('Parent/Children Survival')

sns.factorplot(x='SibSp', kind='count', hue='Survived', data=dataset, orient='v')
plt.title('Sibling/Spouse Survival')


# <p> In our analysis of individual features benchmarked against their classes, we will use the class imabalance exhibited as an indication of features that have predictive power. Features that show a more balanced or inversed class imbalance than shown above (61% deceased vs 39% survived), are indicators of predictive power. <i>Note** Blue represents Survived where Red represents deceases.
# <br><br>
#     <ol> 
#         <li><b>Age</b></li>
#                     <p><i> A change in imbalance is seen in the young age category. The largest discrepancy is in children younger than 5 years old. The second largest discrepancy is seen in the 10-15 year old category. </p>
#          <li><b>Fare</b></li>
#                      <p><i> The fare distribution is hard to analyze because of how oddly distributed it is. We cannot draw much of a conclusion from its distribution. Later, we will transform the distribution to get something more gaussian in nature.</i></p>
#          <li><b>Embarkment Location</b></li>
#                      <p><i> The Cherbourg location shows a higher survival rate than other locations.</i></p>
#          <li><b>Ticket Class</b></li>
#          <p><i>The most interesting and obvious of the features, higher ticket classes show a smaller chance of survival. Intuitively, those that are closer to the ground floors may have had better access to the safety boats on the Titanic. It is a known problem that the Titanic was not equipped with enough safety boats. Implying this logic, the imablance for 3rd class ticket holders is to no surprise. </i></p>
#          <li><b>Sex</b></li>
#          <p><i>We see here that the females are those with the biggest survival rate. This seems to be true to do "the mans responsibility of protecting his wife and children". Following this adage, men were last on the list of priorities.</i> </p>
#          <li><b>Family Members</b></li>
#          <p><i>We can combine the last two metrics in the same analysis. These two metrics give us a look at how many loved ones or family members they had on board. It looks like those who traveled alone or with many family members had a low chance of survival where those with few family members had a higher than normal chance of survival.</i> </p> </p>

# In[ ]:


#Look at the Fare feature's distribution.
f, ax = plt.subplots(1, 2, figsize=(14, 8))

sns.kdeplot(dataset.Fare[dataset.Pclass==1], ax=ax[0], legend=False)
sns.kdeplot(dataset.Fare[dataset.Pclass==2], ax=ax[0], legend=False)
sns.kdeplot(dataset.Fare[dataset.Pclass==3], ax=ax[0], legend=False)
ax[0].set_title('Fare per Ticket Class')
ax[0].legend(['1', '2', '3'])
ax[0].set_xlim(-50, 300)

sns.kdeplot(dataset.Fare[dataset.Embarked=='C'], ax=ax[1], legend=False)
sns.kdeplot(dataset.Fare[dataset.Embarked=='S'], ax=ax[1], legend=False)
sns.kdeplot(dataset.Fare[dataset.Embarked=='Q'], ax=ax[1], legend=False)
ax[1].set_title('Fare per Embarkment Location')
ax[0].legend(['C', 'S', 'Q'])
ax[1].set_xlim(-50, 300)


# <p>The fare features seems to follow a lognormal distribution. We can transform the data by taking log(Fare) to get a distribution that is more gaussian in nature. <br><br>Last but not least, we have to take a look at the cabin and ticket columns.</p>

# In[ ]:


print('Cabins')

print('We only know %.2f%% of the cabin numbers in the training set.' % ((len(dataset[dataset.Cabin.notnull()])/                                                                        len(dataset))*100))
print('We only know %.2f%% of the cabin numbers in the testing set.' % ((len(testset[testset.Cabin.notnull()])/                                                                        len(testset))*100))

print('\nTickets')
print('There are %s unique versus %s duplicate tickets.' % (dataset.Ticket.nunique(), len(dataset)-dataset.Ticket.nunique()))


# In[ ]:


fig = plt.figure(figsize=(10, 10))
font = {'size' : 22}
plt.rc('font', **font)
sns.heatmap(data=dataset.loc[:, dataset.columns != 'Survived'].corr(), cmap='plasma',annot=True, fmt='.2f', linewidth=0.5)


# <p>Looking at the correlation heatmap, there is nothing of particular concern in terms of abnormal positive correlations. SibSp and Parch, the family features, have the highest positive correlation so we might look to combine the features to reduce model complexity.</p>
# 
# <h2> c. Out of Sample Invariance Testing </h2>

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(14, 8))

sns.kdeplot(dataset.Age[dataset.Pclass==1], ax=ax[0], legend=False)
sns.kdeplot(dataset.Age[dataset.Pclass==2], ax=ax[0], legend=False)
sns.kdeplot(dataset.Age[dataset.Pclass==3], ax=ax[0], legend=False)

sns.kdeplot(testset.Age[testset.Pclass==1], ax=ax[1], legend=False)
sns.kdeplot(testset.Age[testset.Pclass==2], ax=ax[1], legend=False)
sns.kdeplot(testset.Age[testset.Pclass==3], ax=ax[1], legend=False)

#fig.legend(['1', '2', '3'], loc=5)
fig.suptitle('Training vs Testing Age per Class Distribution')

fig, ax = plt.subplots(1, 2, figsize=(14, 8))

sns.kdeplot(dataset.Age[dataset.Sex=='male'], ax=ax[0], legend=False)
sns.kdeplot(dataset.Age[dataset.Sex=='female'], ax=ax[0], legend=False)

sns.kdeplot(testset.Age[testset.Sex=='male'], ax=ax[1], legend=False)
sns.kdeplot(testset.Age[testset.Sex=='female'], ax=ax[1], legend=False)

#fig.legend(['male', 'female'], loc=5)
fig.suptitle('Training vs Testing Age per Sex Distribution')

fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharex=True)
ax[0].set_xlim(-50, 300)

sns.kdeplot(dataset.Fare[dataset.Embarked=='C'], ax=ax[0])
sns.kdeplot(dataset.Fare[dataset.Embarked=='S'], ax=ax[0])
sns.kdeplot(dataset.Fare[dataset.Embarked=='Q'], ax=ax[0])

sns.kdeplot(testset.Fare[testset.Embarked=='C'], ax=ax[1])
sns.kdeplot(testset.Fare[testset.Embarked=='S'], ax=ax[1])
sns.kdeplot(testset.Fare[testset.Embarked=='Q'], ax=ax[1])


#fig.legend(['C', 'S', 'Q'], loc=5)
fig.suptitle('Training vs Testing Fare per Embarkment Location')


# <p>I think its important to see that our training and testing set are representative of each other. Typically we would want to reshuffle the training and testing sets if they are vastly different. Looking at the distribution of certain features over both sets, the distributions are quite similar. <br><br>One notable change is the distribution of <i>Ticket Class 3 Ages</i>. The distribution is almost perfectly Gaussian in the testing sample where it shifts to a multimodal gaussian shaped distribution. The change is not drastic and should not break the models. </p> 

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharex=True, sharey=True)
sns.pointplot(x='Parch', y='SibSp', data=dataset, ax=ax[0])
sns.pointplot(x='Parch', y='SibSp', data=testset, ax=ax[1])

fig.suptitle('Training v Testing Parch to SibSp Samples')


# <h1>2. Missing Values & Feature Configuration </h1>

# In[ ]:


#Convert Sex column to encoded labels
le = LabelEncoder()
dataset.Sex = le.fit_transform(dataset.Sex)
testset.Sex = le.fit_transform(testset.Sex)

#Since there are only 2 missing values in the embarked row, we can just drop these for now
dataset = dataset[dataset.Embarked.notnull()]

#Get dummy variables for Embarked column & drop one of the dummy variables
dataset = pd.concat([dataset.drop('Embarked', axis=1), pd.get_dummies(dataset.Embarked, drop_first=True)],         axis=1)
testset = pd.concat([testset.drop('Embarked', axis=1), pd.get_dummies(testset.Embarked, drop_first=True)],         axis=1)

#Due to the overwhelmindly large amount of missing cabin values, we will drop the row all together
dataset.drop('Cabin', inplace=True, axis=1)
testset.drop('Cabin', inplace=True, axis=1)

dataset.head()


# <p> One of the bigger challenges in the dataset is to come up with some method to predict and fill the values of the Age feature. Looking back at the data exploration section, we saw that Age followed 3 different distributions based on the ticket class. This might be a hint towards using different ages per ticket class. It is also confirming  to see the testing set followed a similar gaussian distribution. </p> 

# In[ ]:


avg_1 = dataset.Age[dataset.Pclass==1].dropna().mean()
avg_2 = dataset.Age[dataset.Pclass==2].dropna().mean()
avg_3 = dataset.Age[dataset.Pclass==1].dropna().mean()

def fill_na(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if(np.isnan(Age)):
        if(Pclass==1):
            return avg_1
        elif(Pclass==2):
            return avg_2
        elif(Pclass==3):
            return avg_3
    else:
        return Age
        
dataset['Age'] = dataset[['Age', 'Pclass']].apply(fill_na, axis=1)
testset['Age'] = testset[['Age', 'Pclass']].apply(fill_na, axis=1)


# In[ ]:


imp = Imputer(missing_values ='NaN',              strategy='mean',              axis=0)
#print(dataset.columns)
#print(testset.columns)

imp.fit(dataset.iloc[:, 8:9])
testset.Fare = imp.transform(testset.iloc[:, 7:8])


# In[ ]:


#Scale the features
ss = StandardScaler()

i = np.argwhere('Age'==dataset.columns)[0][0]

dataset.Age = ss.fit_transform(dataset.iloc[:, i:i+1])
testset.Age = ss.transform(testset.iloc[:, i-1:i])


# <p>Above we filled the Age NaN values with the mean of each Ticket Class, scaled the age values in the training and testing sets and filled the missing Fare value in the testing set with the average. <br><br>Let's take a look at our NaN value heatmap to see if we cleaned the data properly. We should see a blank heatmap with no values highlighted in yellow if and only if everything was properly cleaned and formatted. </p>

# In[ ]:


f, ax = plt.subplots(1, 2, figsize=(14, 8))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis', ax=ax[0])
sns.heatmap(testset.isnull(), cbar=False, cmap='viridis', ax=ax[1])


# <p> Mission <i>Success</i>!! We can move to feature engineering. <br><br><br>
# <h1> 3. Feature Engineering </h1>

# In[ ]:


def str_freq(cols):
    
    ticket = cols
    return sum(bytearray(ticket, 'utf8'))


dataset.Ticket = dataset.Ticket.apply(str_freq)
testset.Ticket = testset.Ticket.apply(str_freq)


# In[ ]:


'''
dataset['Family'] = dataset.SibSp + dataset.Parch
testset['Family'] = testset.SibSp + testset.Parch

dataset.drop(['SibSp', 'Parch', 'Name', 'Ticket'], inplace=True, axis=1)
testset.drop(['SibSp', 'Parch', 'Name', 'Ticket'], inplace=True, axis=1)
'''
dataset.drop(['Name', 'Ticket'], inplace=True, axis=1)
testset.drop(['Name', 'Ticket'], inplace=True, axis=1)


# <p> As stated earlier, the fare feature seems to follow a lognormal distribution. An interesting characteristic lognormal distributions is the ability to be converted to a gaussian distribution. By taking the log of all the Fare values, we can accomplish just that. <br><br>
# Below we will transform and scale all its values.</p>

# In[ ]:


def is_neginf(cols):
    Fare = cols[0]
    
    if(np.isneginf(Fare)):
        return 0
    else:
        return Fare

dataset.Fare = np.log(dataset.Fare)
testset.Fare = np.log(testset.Fare)

dataset.Fare = dataset[['Fare', 'Age']].apply(is_neginf, axis=1)
testset.Fare = testset[['Fare', 'Age']].apply(is_neginf, axis=1)

i = np.argwhere('Age'==dataset.columns)[0][0]

dataset.Fare = ss.fit_transform(dataset.iloc[:, i:i+1])
testset.Fare = ss.transform(testset.iloc[:, i-1:i])

fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharex=True)

sns.kdeplot(dataset.Fare, ax=ax[0])
sns.kdeplot(testset.Fare, ax=ax[1])

fig.suptitle('Training vs Testing Log Fare Distribution')


# In[ ]:


#Make sure all features are in the same order
dataset.info()
print('\n\n')
testset.info()


# <p>Do note that I do all of my model performance out of the kernel. My method of parameter selection is too computationally complex to run in a kernel. I basically iterate through the set of possible parameters. Each configuration outputted shows the average cross validation score as well as the standard deviation of those scores. The configuration with the highest score and lowest standard deviation is chosen. A low standard deviation shows an invariant model to changes in the data which is what we want when testing out of sample. </p>
# 
# <h1>4. Model Selection & Performance </h1>

# In[ ]:


y = dataset.Survived
X = dataset.loc[:, dataset.columns != 'Survived']

'''
clf = RandomForestClassifier(n_estimators=25, max_depth=4, \
                             min_samples_split = 25, \
                             min_samples_leaf = 7, \
                             random_state=48, 
                            class_weight={0:0.5, 1:0.5})
'''
clf = RandomForestClassifier(n_estimators=25, max_depth=7,                              min_samples_split = 30,                              min_samples_leaf = 14,                              random_state=6, 
                             class_weight={0:0.5, 1:0.5})
clf.fit(X, y)

scoring = {'acc': 'accuracy',
                   'prec_macro': 'precision_macro',
                   'rec_micro': 'recall_macro'}
        
score = cross_validate(clf, X, y,
                       scoring=scoring, 
                       cv=KFold(n_splits=3, \
                               shuffle=True, random_state=0))
        
print('Accuracy: %.2f STD: %.3f' % (score['test_acc'].mean(), score['test_acc'].std()))
print('Precision: %.2f STD: %.3f' % (score['test_prec_macro'].mean(), score['test_prec_macro'].std()))
print('Recall: %.2f STD: %.3f' % (score['test_rec_micro'].mean(), score['test_rec_micro'].std()))


# In[ ]:


def build_classifier():
    clf = keras.Sequential()
    
    clf.add(Dense(output_dim = 8, input_dim = 8 , init = 'uniform', activation = 'relu'))
    
    clf.add(Dense(output_dim = 8, init = 'uniform', activation = 'tanh'))
    
    clf.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return clf

clf = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs = 100)


# In[ ]:


scoring = {'acc': 'accuracy',
                   'prec_macro': 'precision_macro',
                   'rec_micro': 'recall_macro'}

score = cross_validate(clf, X, y, scoring=scoring,                       cv=KFold(n_splits=3,                                shuffle=True, random_state=0))


# In[ ]:


print('Accuracy: %.2f STD: %.3f' % (score['test_acc'].mean(), score['test_acc'].std()))
print('Precision: %.2f STD: %.3f' % (score['test_prec_macro'].mean(), score['test_prec_macro'].std()))
print('Recall: %.2f STD: %.3f' % (score['test_rec_micro'].mean(), score['test_rec_micro'].std()))


# In[ ]:


knn = KNeighborsClassifier()

params = {'n_neighbors' : [2**i for i in range(6)],          'weights' : ['uniform', 'distance']}
clf = GridSearchCV(knn, param_grid=params, scoring='accuracy', cv = KFold(n_splits=3,                                                                       shuffle=True,                                                                       random_state=0))


# In[ ]:


clf.fit(X, y)
clf.best_params_


# In[ ]:


clf = KNeighborsClassifier(n_neighbors=8, weights='uniform')

scoring = {'acc': 'accuracy',
                   'prec_macro': 'precision_macro',
                   'rec_micro': 'recall_macro'}
        
score = cross_validate(clf, X, y,
                       scoring=scoring, 
                       cv=KFold(n_splits=3, \
                               shuffle=True, random_state=0))
        
print('Accuracy: %.2f STD: %.3f' % (score['test_acc'].mean(), score['test_acc'].std()))
print('Precision: %.2f STD: %.3f' % (score['test_prec_macro'].mean(), score['test_prec_macro'].std()))
print('Recall: %.2f STD: %.3f' % (score['test_rec_micro'].mean(), score['test_rec_micro'].std()))


# In[ ]:


'''my_submission = pd.DataFrame({'PassengerId': testset.index.values, 'Survived':                               clf.predict_classes(testset).reshape(1, -1)[0]})
my_submission.to_csv('submission.csv', index=False)'''

