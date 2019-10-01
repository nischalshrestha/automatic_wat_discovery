#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# 
# On April 15th, 1912, the Titanic collided with an Iceberg and sank. Who survived and who perished? We know the answer for 891 of the 1,309 passengers on board.  Our task is to predict which of the remaining 418 passengers survived. Our conclusions and takeaways:
# 
# 1. Survival rates are highest for women and children, high-paying / first-class passengers, and smaller families. 
#   
# 2. With some tuning, RandomForest performs the best out of the three algorithms that we tried. 
#   
# 3. Although overfitting is hard to avoid, the following helped minimize the impact:
#   - Processing training and test data separately to avoid data leakage
#   - Using feature engineering to avoid high cardinality variables (which get overweighted in decision trees)
#   - Identifying overfitting using learning curves 
#   
# In Part I, we deliberately apply a very simple model; in Part II, we leverage the workflow and graph functions to do a lot more feature engineering and fine tuning.

# | NOTEBOOK SECTION  | ANALYSIS   |
# | :--- |:---|
# | PART 1: <br> INTRODUCTORY <br> MODEL | 1.1 Feature Engineering <br> 1.2 Feature Selection <br> 1.3 Running the classification model (Naive Bayes) |
# | PART 2:  <br> FULL <br> MODEL | 2.1 Feature Engineering <br> 2.2 Feature Selection <br> 2.3 Running and tuning the models (SVM and RandomForest) |

# # PART 1 - INTRODUCTORY MODEL
# ## Loading packages and data

# In[ ]:


# Load packages
import warnings
warnings.simplefilter(action='ignore')                             # suppress warnings
import numpy as np                                                 # linear algebra
import pandas as pd                                                # data analysis
import matplotlib.pyplot as plt                                    # visualization
import seaborn as sns                                              # visualization
from sklearn.svm import SVC                                        # Support Vector Machine classifier
from sklearn.naive_bayes import GaussianNB                         # Naive Bayes classifier
from sklearn.ensemble import RandomForestClassifier                # ensemble classifier
from sklearn.preprocessing import StandardScaler                   # scaler (for SVM model)
from sklearn.model_selection import GridSearchCV, cross_val_score  # parameter tuning
pd.set_option('display.float_format', lambda x: '%.0f' % x)        # format decimals
sns.set(font_scale=1.5) # increse font size for seaborn charts
get_ipython().magic(u'matplotlib inline')
RANDOM_STATE = 42


# Let's load the data and take a look at how it's structured:
#   
# (Note: SibSp = # of Sibling-Spouses, Parch = # of Parents-Children, S = Southampton, C = Cherbourg, Q = Queenstown)

# In[ ]:


def read_data(filename):
    df = pd.read_csv(filename,index_col="PassengerId")
    return df
train = read_data('../input/train.csv')
test = read_data('../input/test.csv')
train.head()


# ## 1.1 Feature Engineering

# Let's start with the simplest (and ultimately most important) variable - male / female. Although most of the passengers are male...

# In[ ]:


train['Sex'].groupby(train['Sex']).count()


# More than two-thirds of the survivors are female:

# In[ ]:


train[train['Survived']==1]['Survived'].groupby(train['Sex']).count()


# To help visualize survival rates for other variables, we've created a stacked 100% bar chart function:

# In[ ]:


def PredictorPlot(df,var,freqvar,freqvalues,ticks=True, print_table=False,show_title=True):
    if  var in df.columns:
        df2 = df.loc[~df[freqvar].isnull()]
        n = len(freqvalues)
        Freq = []
        Pcent = []
        Total = df2[var].groupby(df2[var]).count()
    
        for i in range (0,n):
            Freq.append( df2[df2[freqvar]==freqvalues[i]][freqvar].groupby(df2[var]).count() )
            Pcent.append ( Freq[i].div(Total,fill_value=0))
            
        df3 = Pcent[0]
        for i in range (1,n):
            df3 = pd.concat([df3, Pcent[i]], axis=1)
            
        if print_table == True: print (df3)
    
        ax = df3.plot.bar(stacked=True,legend=False,figsize={16,5},colormap = 'RdYlGn',xticks=None)
        ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in ax.get_yticks()])
        if show_title == True: plt.title('Percentage of Survivors by Passenger Type (green=Survived)');
        w = 0.5
        if ticks==False:
            ax.xaxis.set_ticks([])
            w = 1
        for container in ax.containers:
            plt.setp(container, width=w)


# Let's take a look at the stacked bar chart for the Sex variable:

# In[ ]:


PredictorPlot(df = train,var ='Sex',freqvar = 'Survived', freqvalues = [0,1])


# Now, let's encode the Sex variable as a numerical variable so we can use it in our model:

# In[ ]:


def processSex(df,dropSex=True):
    df["Female"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
    if dropSex == True: df = df.drop('Sex',axis=1)  
    return df

Xy_train_df = processSex(train)
X_test_df = processSex(test)  
Xy_train_df.head()


# ## 1.2 Feature Selection
# In this basic model, we'll only use the new "Female" variable.  Let's convert our DataFrames to NumPy arrays so that we can use them in ScikitLearn:

# In[ ]:


gnb_vars = ['Female']
X_train = Xy_train_df.drop('Survived',axis=1)[gnb_vars].values
X_test = X_test_df[gnb_vars].values
y = train['Survived'].ravel() 


# Note on naming conventions: 
# * Xy_train_df = dataframe with features and target variable
# * X_train_df = dataframe with features
# * X_train = array with features

# ## 1.3 Running the classification model
# Now, we'll fit our data to a Naive Bayes model and predict the survivors:

# In[ ]:


gnb = GaussianNB()
gnb.fit(X_train, y)
gnb_scores = cross_val_score(gnb,X_train,y,cv=5,scoring='accuracy')
gnb_survivors = gnb.predict(X_test)


# Next, let's take a look at the results:

# In[ ]:


def print_classification_results(a,b,filename):
    print('Survivors: = {0:0.0f}'.format(a.sum()))
    print('Accuracy:  = {0:0.3f}'.format(b))
    test['Survived'] = a
    test['Survived'].to_csv(filename,header=True)
print_classification_results(gnb_survivors,gnb_scores.mean(),filename='Model 1 - GNB.csv')


# 78.7% accuracy - not bad as a baseline. The number of survivors is another good indicator that we're on the right track. Assuming the survival rates are similar between the training and test data, we would expect roughly 160 survivors out of 418 in the test data.

# # PART 2 - FULL MODEL
# ## 2.1 Feature Engineering
# We'll look at each of the remaining variables in turn. We've already processed the Sex variable - let's move on to passenger fare class:
# ### 2.1.1 Passenger Class

# In[ ]:


PredictorPlot(df = Xy_train_df,var ='Pclass',freqvar = 'Survived', freqvalues = [0,1])


# Clearly, passenger class is another important variable - 1st class passengers survived at more than twice the rate of 3rd class passengers. Now, let's look at passenger fares...

# ### 2.1.2 Passenger Fare
# There are quite a few zero values which, assuming none of the passengers traveled for free, we should treat as missing data:

# In[ ]:


train.loc[(train['Fare'].isnull()) | (train['Fare']==0) ]


# All of the missing fare values in the train and test data sets are for Southampton passengers, so we'll replace the zeros with the median fare for Southampton passengers in each passenger class:

# In[ ]:


def fillFare(df):
    
    df.loc[((df['Fare'].isnull()) | (df['Fare']==0)) & (df['Pclass']==3),'Fare'] = df[
        'Fare'].loc[(df['Pclass']==3) & (df['Embarked']=='S')].median()
    df.loc[((df['Fare'].isnull()) | (df['Fare']==0)) & (df['Pclass']==2),'Fare'] = df[
        'Fare'].loc[(df['Pclass']==2) & (df['Embarked']=='S')].median()
    df.loc[((df['Fare'].isnull()) | (df['Fare']==0)) & (df['Pclass']==1),'Fare'] = df[
        'Fare'].loc[(df['Pclass']==1) & (df['Embarked']=='S')].median()
    
    return df

Xy_train_df = fillFare(Xy_train_df)


# We can also see that the Fare data is heavily skewed:

# In[ ]:


plt.figure(figsize=(16,5))
plt.hist(Xy_train_df['Fare'],bins=50);
plt.title('Distribution of Passenger Fares');


# Let's even out the distribution using a log transformation:

# In[ ]:


plt.figure(figsize=(16,5))
plt.hist(Xy_train_df['Fare'].apply(np.log),bins=50);
plt.title('Distribution of Log-transformed Passenger Fares');


# Finally, we'll bucket the data to reduce the number of discrete values. (If we don't, our RandomForest model will overweight the fare data, leading to overfitting.)

# In[ ]:


def processFare(df):
    
    df = fillFare(df)
    df['LogFare'] = df['Fare'].apply(np.log).round().clip_upper(5)
    df['LogFare'] = df['LogFare'].astype(int)
    
    return df

Xy_train_df = processFare(Xy_train_df)


# Now, let's look at the impact on survival rates:

# In[ ]:


PredictorPlot(df = Xy_train_df,var ='LogFare',freqvar = 'Survived', freqvalues = [0,1])


# Fare is another good predictor of survival likelihood. Let's look at the relationship between fares and passenger class:

# In[ ]:


plt.figure(figsize=(16,5))
sns.violinplot(x="Pclass", y="LogFare", data=Xy_train_df);
plt.title('Relationship between Fares and Passenger Class');


# The correlation is high, but there's a wide range of fares within each passenger class. To see if this additional information is useful, let's look at the survival rate by fare group for first-class passengers:

# In[ ]:


PredictorPlot(df = Xy_train_df.loc[Xy_train_df['Pclass']==1],var ='LogFare',
              freqvar = 'Survived', freqvalues = [0,1])


# The combination of fares and passenger allows to identify the 'VIP passengers'. We saw that ~60% of first-class passengers survived. Among the highest fare group within first class, the survival rate is almost 80%.

# ### 2.1.3 Parent-Child and Sibling-Spouse
# We'll take a look at the Parent-Child (Parch) and SibSp (Sibling-Spouse) variables together:

# In[ ]:


PredictorPlot(df = Xy_train_df,var ='Parch',freqvar = 'Survived', freqvalues = [0,1])
PredictorPlot(df = Xy_train_df,var ='SibSp',freqvar = 'Survived', freqvalues = [0,1])


# Large families clearly faced worse odds of survivial. One can imagine that families wanted to leave the ship together  - it probably took large families longer to find each other and they may have missed their opportunity to get on the limited number of lifeboats. Let's create a new variable for family size and look at the distribution:

# In[ ]:


train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
plt.figure(figsize=(16,5))
plt.hist(train['FamilySize']);
plt.title('Distribution of Family Size');


# There aren't many large families, so we'll create a version of the Family Size variable that groups all families above a certain size together. We'll also create a variable that captures whether someone is traveling alone:

# In[ ]:


def processParch_SibSp(df,LargeFamilySize=5,dropFamilySize=True):
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    df['Alone'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    df['CapFamSize'] = df['FamilySize'].clip_upper(LargeFamilySize)
    if dropFamilySize==True: df = df.drop('FamilySize',axis=1) 
    return df

Xy_train_df = processParch_SibSp(Xy_train_df)
PredictorPlot(df = Xy_train_df,var ='CapFamSize',freqvar = 'Survived', freqvalues = [0,1])


# ### 2.1.4 Embarked
# The embarkation port is missing for some passengers:

# In[ ]:


train.loc[train['Embarked'].isnull()]


# Given the small amount of missing data, we can just use the most common depature point for these passengers:

# In[ ]:


train['Embarked'].groupby(train['Embarked']).count()


# It looks as if most passengers boarded in Southampton. Let's fill in the missing data accordingly and encode 'Embarked' as a series of dummy variables:

# In[ ]:


def processEmbarked(df,dropEmbarked=True):
    df.loc[df['Embarked'].isnull(),'Embarked' ] ="S"
    df_embarked = pd.get_dummies(df['Embarked'])
    df = df.join(df_embarked)
    if dropEmbarked==True: df = df.drop('Embarked',axis=1) 
    return df

Xy_train_df = processEmbarked(Xy_train_df)


# ### 2.1.5 Cabin
# The cabin variable is missing for most passengers:

# In[ ]:


Xy_train_df.isnull().sum()[Xy_train_df.isnull().sum()>0]


# Given the sparse natures of the data,  we'll drop the cabin variable for now:

# In[ ]:


def processCabin(df,dropCabin=True):
    if dropCabin==True: df = df.drop('Cabin',axis=1) 
    return df

Xy_train_df = processCabin(Xy_train_df)


# ### 2.1.6 Ticket
# The Ticket variable contains a numerical component. Let's extract it and see if there's any relationship with survival rates:

# In[ ]:


train["TicketNum"] = train["Ticket"].str.extract('(\d{2,})', expand=True)
train["TicketNum"] = train["TicketNum"].apply(pd.to_numeric)
train.loc[train['TicketNum'].isnull(),'TicketNum'] = -1
PredictorPlot(df = train,var ='TicketNum',freqvar = 'Survived', freqvalues = [0,1],ticks=False)


# The distribution doens't look completely random. Let's look at how tickets are distributed by passenger type...  In the view below, red / yellow / green represents tickets held by passengers from classes 1 / 2 / 3 respectively:

# In[ ]:


PredictorPlot(df = train,var ='TicketNum',freqvar = 'Pclass', 
              freqvalues = [1,2,3],ticks=False,show_title=False)


# We can see below that blocks of tickets are held by passengers of the same passenger class, so this  at least partially explain the lack of randomness in the survival rates. 
# 
# There are also prefixes for a small number of tickets. The only intelligible prefixes seem to capture the departure point - e.g., SOTON for Southampton. 
# 
# Overall, it's not clear that we can extract any additional information from the Ticket variable, so we'll drop it for now.

# In[ ]:


def processTicket(df,dropTicket=True):
    if dropTicket==True: df = df.drop('Ticket',axis=1) 
    return df

Xy_train_df = processTicket(Xy_train_df)


# ### 2.1.7 Age and Title
# So far, we've transformed all the variables except Age and Title. Before going on, let's take a look at the current state of the data:

# In[ ]:


Xy_train_df.head()


# We can see that every name contains a title. We can extract it fairly easily as it always seem to follow the last name:

# In[ ]:


train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
pd.crosstab(train['Title'], train['Female'])


# Let's group some of the unusual titles together:

# In[ ]:


def processName(df, dropName = True,dropTitle = False ): 
    
    Title_Dictionary = {"Capt": "TitleX",
                        "Col":"TitleX",
                        "Don":"TitleX",
                        "Dona":"TitleX",
                        "Dr":"TitleX",
                        "Jonkheer": "TitleX",
                        "Lady":"TitleX",
                        "Major":"TitleX",
                        "Master" :"Master",
                        "Miss" :"Miss",
                        "Mlle":"Miss",
                        "Mr" :"Mr",
                        "Mrs":"Mrs",
                        "Mme":"Mrs",
                        "Ms":"Mrs",
                        "Rev":"TitleX",
                        "Sir" :"TitleX",
                        "the Countess":"TitleX"}

    df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    df['Title'] = df.Title.map(Title_Dictionary) # use the Title_Dictionary to map the titles
    df_title = pd.get_dummies(df['Title'])
    df = pd.concat([df, df_title], axis=1)
    if dropName==True: df = df.drop('Name',axis=1)
    if dropTitle==True: df = df.drop('Title',axis=1)
            
    return df
    
Xy_train_df = processName(Xy_train_df)
pd.crosstab(Xy_train_df['Title'], Xy_train_df['Female'])


# Now, let's look at the survival rates by title:

# In[ ]:


PredictorPlot(df = Xy_train_df,var = 'Title',freqvar = 'Survived',freqvalues = [0,1])


# This is the first indication that the survival rate for children was higher than for adults - there's a big difference between the titles 'Master' and 'Mr'. At what age did a 'Master' become a 'Mr'? Let's look at the maximum age for each title:

# In[ ]:


pd.pivot_table(Xy_train_df, values='Age', index=['Title'],columns=[], aggfunc=np.max)


# It looks as if age 12 was the cutoff. 
# 
# The 'Master' title helps us identify male children whose age is missing. It would be useful if we had a way to identify female children with missing ages. One hypothesis is that we could use the Title 'Miss' and lack of a parent as a good indicator. Let's test the hypothesis:

# In[ ]:


pd.pivot_table(Xy_train_df, values='Age', index=['Title'],columns=['Parch'], aggfunc=np.median)


# It looks as if this works well, so let's create our variables for Male and Female children:

# In[ ]:


def processAge(df,dropAge = True):
    
    df['MaleCh'] = 0
    df['FemaleCh'] = 0
    
    df.loc[( (df['Female']==0) & (df['Age']<=12) ) | (df['Master']==1),'MaleCh'] = 1  
    
    df.loc[( ( (df['Female']==1) & (df['Age']<=12) ) | 
             ( (df['Female']==1) & (df['Age'].isnull()) & (df['Miss']==1)& (df['Parch']>0) )  ),
           'FemaleCh' ] = 1
    # Female logic - A female with the title Miss and Parents onboard is likely to be a child    
    
    if dropAge==True: df = df.drop('Age',axis=1)
    if 'Title' in df.columns: df = df.drop('Title',axis=1)
            
    return df

Xy_train_df = processAge(Xy_train_df)


# Now we've finished going through all the variables, let's take a look at the transformed data:

# In[ ]:


Xy_train_df.drop('Survived',axis=1).head()


# ### 2.1.8 Processing the test data
# Let's end this section by creating a function that allows us to easily process the test data:

# In[ ]:


def process_data(filename):
    df = read_data(filename)
    df = processSex(df)
    df = processFare(df)
    df = processParch_SibSp(df)
    df = processEmbarked(df)
    df = processCabin(df)
    df = processTicket(df)
    df = processName(df)
    df = processAge(df)
    return df


# Let's just test that it's working as expected:

# In[ ]:


Xy_train_df_new = process_data('../input/train.csv')
Xy_train_df_new.equals(Xy_train_df)


# Now, we'll process the test data:

# In[ ]:


X_test_df = process_data('../input/test.csv')


# ## 2.2 Feature Selection
# To better understand some of the relatioships between the variables, we'll look at the correlation matrix:

# In[ ]:


fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(Xy_train_df.corr(), linewidths=0.1,cbar=True, annot=True, square=True, fmt='.1f')
plt.title('Correlation between Variables');


# A combination of the correlation matric and our prior analysis suggest that we should at least consider the following variables: 
# * Pclass
# * Fare
# * Female / Mrs / Miss / Mr
# * Fare / LogFare
# * Alone
# * CapFamSize

# ## 2.3 Running and Tuning the Classification Models
# ### 2.3.1 Support Vector Classification
# Based on what we already know and a little bit of trial and error, the following six variables seem to give a good result: Female, Pclass, CapFamSize, Alone, Master, LogFare. Let's run the model:

# In[ ]:


svm_params = {'kernel':'rbf','random_state' : RANDOM_STATE}
select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'LogFare'])
X_train_df = Xy_train_df.drop('Survived',axis=1)[select_vars]
X_train = X_train_df.values
X_test = X_test_df[select_vars].values 

svm = SVC(**svm_params)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
svm.fit(X_train_scaled, y)
svm_scores = cross_val_score(svm,X_train_scaled,y,cv=5,scoring='accuracy')
svm_survivors = svm.predict(X_test_scaled)
print_classification_results(svm_survivors,svm_scores.mean(),'Model 2 - SVC.csv')


# This represents a decent improvement over the Naive Bayes model. Next we'll look for opportunities to tune the model. To do this, we'll create a function that graphs the error rate for different parameter values:

# In[ ]:


def ParamChart(X,y,param,min,max,step,clf_params,UseOOB=False):
    error_rate = []
    
    for i in np.arange(min, max+1, step):
        
        new_param = {param:i}
        if UseOOB == True: 
            clf = RandomForestClassifier(**clf_params)
            clf.set_params(**new_param)
            clf.fit(X, y)
            
        if UseOOB==True:
            error_rate.append((i, 1 - clf.oob_score_))
        else:
            clf = SVC(**clf_params)
            clf.set_params(**new_param)
            scores = cross_val_score(clf,X,y,cv=5,scoring='accuracy')
            error_rate.append((i, 1 - scores.mean()))

        
    plt.figure(figsize=(16,5))
    xs, ys = zip(*error_rate)
    plt.plot(xs, ys)
    plt.xlim(min, max)
    plt.xlabel("Parameter")
    plt.ylabel("Error rate")
    plt.title('Error Rate by Parameter Value');
    plt.show()


# Now let's look at the impact of different values of the parameter 'C' on the error rate:

# In[ ]:


ParamChart(X_train_scaled,y,param='C',min=0.1,max=10,step=0.1,clf_params=svm_params)


# The default for SciKitLearn is C=1, so it's not obvious that adjusting the value of C will improve the accuracy. Let's try another algorithm...

# ### 2.3.2 RandomForest Classification
# We'll use the same variables for RandomForest with one exception... We'll start off using the raw fare values to illustrate the issues with using a high cardinality variable (i.e., one with many levels).

# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x) 
def run_rf(X1,X2,params,filename='output.csv'):
    X_train = X1.values 
    X_test = X2.values 
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y)
    rf_survivors = rf.predict(X_test)

    print_classification_results(rf_survivors,rf.oob_score_,filename)
    feature_importance = pd.DataFrame(data=rf.feature_importances_,
                                      index=X_train_df.columns.values,columns=['FeatureScore'] )
    print (feature_importance.sort_values(ascending=False,by=['FeatureScore']))

rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':50}
select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'Fare'])
X_train_df = Xy_train_df.drop('Survived',axis=1)[select_vars]
run_rf(X_train_df,X_test_df[select_vars],rf_params,'Model 3 - RF1.csv')


# Although the accuracy looks good, we can see a couple of warning signs... Fare is the most important variable in the model. Our analysis suggests that male / female is the primary driver of survival, so this suggests that the Fare is being overweighted. We can also see that the number of survivors looks low - another indication of potential overfitting.
# 
# The best way to confirm our hunch is to look at the learning curve... 

# In[ ]:


def LearningCurve(X,clf_params):
    warnings.simplefilter(action='ignore')
    error_rate = []

    for i in range(10,len(train),50): 
        X_LC = X[:i].values
        y_LC = train[:i]['Survived']
        clf = RandomForestClassifier(**clf_params)
        clf.fit(X_LC, y_LC)
        oob_error = 1 - clf.oob_score_
        training_error = 1 - clf.score(X_LC,y_LC)
        error_rate.append((i, oob_error, training_error))

    plt.figure(figsize=(16,5))
    xs, ys, zs = zip(*error_rate)
    plt.plot(xs, ys)
    plt.plot(xs, zs)
    plt.xlim(0, len(train))
    plt.xlabel("Training Examples")
    plt.ylabel("Error rate")
    plt.title('Error Rate by Sammple Size (green=Training error, blue = Out-of-bag error)');
    plt.show()

rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':50}
LearningCurve (X_train_df,rf_params)


# As we add more training examples, we would expect the error rate on the training set to rise, and the out-of-bag error to fall. If they converge, we're likely to see similar results on the test data and training data. If they don't converge (as above) it's likely that we're overfitting. Let's switch out the Fare variable for the LogFare variables to see if it makes a difference... 

# In[ ]:


select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'LogFare'])
X_train_df = Xy_train_df.drop('Survived',axis=1)[select_vars]
run_rf(X_train_df,X_test_df[select_vars],rf_params,'Model 4 - RF2.csv')
LearningCurve (X_train_df,rf_params)


# This looks a lot better. Still some room for improvement, so we'll turn our attention to tuning the model. Let's start by looking at whether we're generating enough trees:

# In[ ]:


ParamChart(X_train_df,y,param='n_estimators',min=10,max=500,step=10,clf_params=rf_params,UseOOB=True)


# We used n_estimators=50 in the first model. Clearly, the error rate hasn't stabilized at this point. We should probably use at least 200. Let's go with 500 to be safe - apart from computation time, there's no downside to a larger value.

# In[ ]:


rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':500}
rf = RandomForestClassifier(**rf_params)
ParamChart(X_train_df,y,param='min_samples_leaf',min=1,max=30,step=1,clf_params=rf_params,UseOOB=True)


# We can increase the minimum leaf size without negatively impacting the error rate. Let's increase the minium leaf size to 5. Let's turn now to the tree depth:

# In[ ]:


rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':500,
            'min_samples_leaf':5}
ParamChart(X_train_df,y,param='max_depth',min=1,max=10,step=1,clf_params=rf_params,UseOOB=True)


# We'll cap the tree depth at 5. Now, let's rerun the model and see if we've reduced the overfitting...

# In[ ]:


rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':500,
            'min_samples_leaf':5,'max_depth':5}
select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'LogFare'])
X_train_df = Xy_train_df.drop('Survived',axis=1)[select_vars]
run_rf(X_train_df,X_test_df[select_vars],rf_params,'Model 5 - RF3.csv')
LearningCurve (X_train_df,rf_params)


# We can see that the accuracy is very simlar to the previous model, but the training error rate and out-of-bag error rate converge as the number of training examples increase. This suggests that we'll have less overfitting on the test data. On the Kaggle public leaderboard, this final model achieved 0.79425.
# 
# Let's wrap up the analysis with a short block of code that we can use to run the model from start to finish:

# In[ ]:


# Define RandomForest parameters
rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':500,
            'min_samples_leaf':5,'max_depth':5}
# Select variables for RandomForest model
select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'LogFare'])
# Process data
X_train_df = process_data('../input/train.csv').drop('Survived',axis=1)[select_vars]
X_test_df = process_data('../input/test.csv')[select_vars]
# Run the model
run_rf(X_train_df,X_test_df,rf_params)


# 
