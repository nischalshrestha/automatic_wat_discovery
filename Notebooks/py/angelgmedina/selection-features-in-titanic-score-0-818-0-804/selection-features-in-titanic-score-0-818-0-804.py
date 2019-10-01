#!/usr/bin/env python
# coding: utf-8

# Hi, this is my first kernel, I have been learned a lot from others kernels, and here I present some new ideas, like trying to find groups in the passengers and selecting features, because some of them make noise and don't help to improve the acuracy.
# 
# With this procedure I have obtained a score that oscillates between 0.81818 and 0.8038, due to the randomness of the cross validation.
# 
# If you find them interesting, **I will appreciate your votes as well as your comments**. Thank you very much

# ### Table of Contents
# 
# ### 1. Loading data
# ### 2. Feature analisis and missing values
# 2.1. Embarked
# 
# 2.2. Fare
# 
# 2.3. Sex
# 
# 2.4. Pclass
# 
# 2.5. Cabin
# 
# 2.6. Title
# 
# 2.7. Family Size
# 
# 3.8. Groups
# 
# 3.9. Tickets
# 
# 3.10 Age
# 
# ### 3. Moleling
# 3.1. Preprocesing
# 
# 3.2. Quick look
# 
# 3.3. Tuning Algorithms to Look for the Best
# 
# 3.4. Feature selection
# 

# # 1. Loading data

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold    


# preprocessiong
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel, RFECV

# machine learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# import files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# preview the data
train.head()


# In[ ]:


IDtest = test["PassengerId"]
train_len = len(train)

# preview the data
total =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
total.info()


# Here we can see the missing values in Age, Cabin, Embarked and Fare, so first we have to solve this missing values.

# # 2. Analisis and missing values

# ## 2.1. Embarked

# In[ ]:


# Embarked (two missing values)
total[total[["Embarked"]].isnull().any(axis=1)]


# I think this persons travel together because they have the same Cabin and the same Ticket. So they are not family ... but they travel together.
# 
# **This gives me an idea, that people travel in groups, even if they are not family**

# In[ ]:


embark_perc = total[["Embarked", "Pclass"]].groupby(["Embarked"]).count()
a = total.groupby(["Embarked", "Pclass"]).size().reset_index(name='count')
a
b = total[["Embarked", "Pclass", "Fare"]].groupby(["Embarked", "Pclass"]).mean()
print(b)
c = total[["Embarked", "Pclass", "Fare"]].groupby(["Embarked", "Pclass"]).count()
print(c)


# In[ ]:


sns.set(style="whitegrid")
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=total, palette="husl")
plt.ylim(0,150)
plt.show()


# In[ ]:


sns.countplot(x="Embarked", data=total, hue="Pclass", palette="Greens_d");


# It's not easy. Some people think the missing value for Embarked is "C" because Fare = 80 is close to the average for Pclass=1, but i don't think so, because the Embarked "S" is also to the average (although not as much as C) and we have more people in Embarked"S" than "C" for Pclass=1.
# 
# So I choose Embarked "S" for the missing values. 

# In[ ]:


total["Embarked"] = total["Embarked"].fillna("S")


# In[ ]:


g = sns.catplot(x="Embarked", y="Survived",  data=train,  hue="Pclass", height=5, kind="bar", 
                   palette="Greens_d")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# The passengers of the "C" board are more likely to survive, although they are also the ones that are also first class, which is probably an important variable.

# Embarked is a categorical variable, so I create dummies.

# In[ ]:


total = pd.get_dummies(total, columns = ["Embarked"], prefix="Em")


# ## 2.2. Fare
# 
# We have only one missing value for Fare.
# 

# In[ ]:


total[total[["Fare"]].isnull().any(axis=1)]


# The most paid fare is the one corresponding to mode

# In[ ]:


from scipy import stats
total[["Pclass", "Fare"]].groupby(['Pclass']).agg(lambda x: stats.mode(x)[0][0])


# In[ ]:


total["Fare"] = total["Fare"].fillna(8.05)


# In[ ]:


sns.distplot(total["Fare"], color="g")
plt.show()


# ## 2.3. Sex

# In[ ]:


g = sns.barplot(x="Sex",y="Survived",data=total, palette="Greens_d")
g = g.set_ylabel("Survival Probability")


# Wow! It's clear, sex is very important

# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
total["Sex"] = total["Sex"].map({"male": 0, "female":1})


# ## 2.4. Pclass

# In[ ]:


g = sns.catplot(x="Pclass",y="Survived",data=train, kind="bar", height = 5 , palette = "Purples")
g = g.set_ylabels("survival probability")


# In this ordinal characteristic we can see that Pclass 1 is better than 2 that 3

# In[ ]:


g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   height=5, kind="bar", palette="Purples")
g = g.set_ylabels("survival probability")


# As we already know, sex is important and does not distinguish between classes

# ## 2.5. Cabin
# 
# We have only 295 rows with Cabin, maybe, the nan values is because the passenger don't have a cabin.
# I will create a new feature with de "type" of cabin and if the passeneger don't have a cabin, I will match with "X".
# I will use the original "Cabin" later.

# I will create a new feature: Has a cabin or not?

# In[ ]:


total["CabinYN"] = pd.Series([1 if not pd.isnull(i) else 0 for i in total['Cabin'] ])


# In[ ]:


g = sns.catplot(x="CabinYN",y="Survived",data=total, kind="bar", height = 5 , palette = "Oranges")
g = g.set_ylabels("survival probability")


# Have a Cabin is better than not have it

# I've been investigating, and I've found that booths A are on top of the ship and then the "B's" and so on.
# 
# So I'm going to create a new ordinal feature for the the type of cabin:

# In[ ]:


total["Cabin1"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in total['Cabin'] ])
ordinal = {"A": 1, "B": 2, "C": 3, "D":4, "E": 5, "F": 6, "G": 7, "T": 8, "X": 10}
total["Cabin_ord"] = total.Cabin1.map(ordinal)
total.drop(labels = ["Cabin1"], axis = 1, inplace = True)


# In[ ]:


total[["CabinYN", "Cabin_ord"]].head(6)


# In[ ]:


g = sns.catplot(x="Cabin_ord",y="Survived",data=total, kind="bar", height = 5 , palette = "Oranges")
g = g.set_ylabels("survival probability")


# I'm not sure about the usefulness of this variable, we'll see

# ## 2.6. Title

# In the feature "Name" we can get the title, that looks like very interesting:

# In[ ]:


# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in total["Name"]]
total["Title"] = pd.Series(dataset_title)

# Convert to categorical values Title 
total["Title"] = total["Title"].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev'], 'Officer')
total["Title"] = total["Title"].replace(['Mme'], 'Mrs')
total["Title"] = total["Title"].replace(['Ms', 'Mlle'], 'Miss')
total["Title"] = total["Title"].replace(['Dona', 'Lady', 'the Countess','Sir', 'Jonkheer'], 'Royalty')


# In[ ]:


g = sns.catplot(x="Title", y="Survived", data=total, kind="bar", palette="Blues_d")
g = g.set_ylabels("survival probability")


# Here we can clearly see the famous sentence "**women and children first**"

# In[ ]:


g = sns.countplot(total["Title"], palette="Blues_d")


# 

# In[ ]:


total[["Title", "Age"]].groupby(["Title"]).mean()


# In[ ]:


# Dummies fot Title
total = pd.get_dummies(total, columns = ["Title"])
# Drop Name variable
total.drop(labels = ["Name"], axis = 1, inplace = True)


# ## 2.7. Family size
# 
# Create a new feature for family size from SibSp and Parch
# 

# In[ ]:


total["Fsize"] = total["SibSp"] + total["Parch"] + 1

#data = total[:train.shape[0]]
g = sns.catplot(x="Fsize",y="Survived", data = total, kind='point')
g = g.set_ylabels("Survival Probability")


# In[ ]:


# Create new feature of family size
total['Single'] = total['Fsize'].map(lambda s: 1 if s == 1 else 0)
total['SmallF'] = total['Fsize'].map(lambda s: 1 if 2 <= s <= 4  else 0)
total['LargeF'] = total['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


g = sns.catplot(x="Single",y="Survived",data=total,kind="bar", height = 3, palette="YlOrBr")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="SmallF",y="Survived",data=total,kind="bar", height = 3, palette="YlOrBr")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="LargeF",y="Survived",data=total,kind="bar", height = 3, palette="YlOrBr")
g = g.set_ylabels("Survival Probability")


# Travel alone is not  good for survival, is better travel with a small family

# ## 2.8. Groups
# 
# I think there are groups of people traveling together, not just family, as we saw earlier with the two people who had the missing values of Embarked.
# 
# To create this new features, I imagine they will go in nearby booths and have tickets nearby, so I will create these two variables also to help me create the clusters

# In[ ]:


# Tiquets together and near
Torder = total["Ticket"].unique()
Torder = np.sort(Torder)
Index_T  = np.arange(len(Torder))
mapear = pd.DataFrame(data={"Torder": Torder, "index_T":Index_T})
total["Ticket_tog"] = total["Ticket"].map(mapear.set_index("Torder")["index_T"])

# the same for cabins
total["Cabin"] = total["Cabin"].fillna("X")
Corder = total["Cabin"].unique()
Corder = np.sort(Corder)
Index_C  = np.arange(len(Corder))
mapear = pd.DataFrame(data={"Corder": Corder, "index_C":Index_C})
total["Cabin_tog"] = total["Cabin"].map(mapear.set_index("Corder")["index_C"])


# I'm going to create a new feature "Groups". I guess that people who travel together were affected by the next features, so I'm going to group them by clusters

# In[ ]:


# for the cluster i will use the following features
df = total[["Cabin_tog", "Ticket_tog", "Em_C", "Em_S", "Em_Q", "Fare", 
              "Pclass", "Fsize"]]


# In[ ]:


# help me to select a n_cluster
range_n_clusters = np.arange(200, 600, 50)
for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(df)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


# In[ ]:


kmeans = KMeans(n_clusters=450)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
total["Groups"] = labels


# In[ ]:


df = total[["Sex", "Age", "Cabin_tog", "Ticket_tog", "Em_C", "Em_S", "Em_Q", "Fare", 
              "Pclass", "Fsize", "Groups", "Survived"]]
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
plt.figure(figsize = (10,5))
g = sns.heatmap(df.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# The new feature "Groups" is correlated with Cabins, Fare, Pclass, Fsize
# Surprise with the negative correlation between Groups and Survived, but interesting.
# 
# The most important feature is "Sex"

# ## 2.9 Ticket

# We can create a new feature of the first character of the "Ticket"

# In[ ]:


Ticket = []
for i in list(total.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")

total["Ticket"] = Ticket
total = pd.get_dummies(total, columns = ["Ticket"], prefix="T")


# ## 2.10. Age
# 

# We have a lot of missing values for Age, and I guess that is an important feature, so I'm going to use Linear regression for missing values.

# In[ ]:


# Explore Age vs Survived. Initial
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")


# In[ ]:


X_age = total[["Fare", "Parch", "SibSp", "Sex", "Pclass",  
        "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Officer", "Title_Royalty"]]

Y_age = total[["Age"]]
index = Y_age.Age.isnull()
X_age_train = X_age[~index]
Y_age_train = Y_age[~index]
X_age_test = X_age[index]
Y_age_test = Y_age[index]

clf = LinearRegression()
clf.fit(X_age_train, Y_age_train)

p_age = clf.predict(X_age_test).round()
total.loc[index, "Age"] = p_age


# 

# In[ ]:


# Explore Age vs Survived. Final
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")


# Good. The graphics looks like similar!

# In[ ]:


# Drop useless variables 
total.drop(labels = ["PassengerId"], axis = 1, inplace = True)
total.drop(labels = ["Cabin"], axis = 1, inplace = True)


# In[ ]:


total= total.astype(float)


# # 3. Modeling
# 

# ## 3.1. Preprocessing
# 
# Some features are not in the same scale, so I will transform all the features with StandardScaler

# In[ ]:


y = total["Survived"].copy()
X = total.drop(labels=["Survived"], axis=1)

T = preprocessing.StandardScaler().fit_transform(X)
 
X_train = T[:train_len]
X_test  = T[train_len:]
y_train = y[:train_len]


# ## 3.2. Quick look.

# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10, random_state=1)


# In[ ]:


# Modeling step Test differents algorithms 
random_state = 1
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", 
                                      cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
                       "Algorithm":["SVC", "RandomForest","ExtraTrees","GradientBoosting",
                       "MultipleLayerPerceptron", "KNeighboors","LogisticRegression"]})
cv_res = cv_res.sort_values(["CrossValMeans"], ascending=False)
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="RdYlBu",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# ## 3.3 Tuning Algorithms to Look for the Best

# In[ ]:


# Gradient Boosting
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(X_train,y_train)
print ("Score: ", gsGBC.best_score_)
# Score:  0.833894500561


# In[ ]:


# Random Forest
RFC = RandomForestClassifier()
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,y_train)
print ("Score: ", gsRFC.best_score_)
# Score:  0.833894500561


# In[ ]:


# Extra Trees Classifier
ExtC = ExtraTreesClassifier()
ex_param_grid = {"max_features": [3, 5, 8, 'auto'],
              "min_samples_split": [2, 6, 10],
              "min_samples_leaf": [1, 3, 10],
              "n_estimators" :[100, 150, 200]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(X_train, y_train)
print ("Score: ", gsExtC.best_score_)
# Score:  Score:  0.838383838384


# In[ ]:


# Logistic Regression
LG = LogisticRegression()
lg_param_grid = {'C': [1, 10, 50, 100,200,300, 1000]}
gsLG = GridSearchCV(LG,param_grid = lg_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsLG.fit(X_train,y_train)
print ("Score: ", gsLG.best_score_)
# Score:  0.820426487093


# In[ ]:


# Support Vector Machine.
SVMC = SVC()
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1, 3, 10],
                  'C': [1, 10, 50, 100,200,300]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(X_train,y_train)
print ("Score: ", gsSVMC.best_score_)
# Score:  0.832772166105


# ## 3.4. Feature selection

# In[ ]:


gsExtC.best_estimator_


# I choose the best estimators: min_samples_split=10, n_estimators=100 which were the best estimators from Extra Tree Classifier.
# 
# *Sometimes the result does not match the results that I have been given because of the randomness of the cross validation*

# In[ ]:


ExtC = ExtraTreesClassifier(min_samples_split=10, n_estimators=100 , random_state=1)
ExtC.fit(X_train, y_train)
importances = ExtC.feature_importances_
indices = np.argsort(importances)[::-1]
col = X.columns[indices]

# Top 25 features
plt.figure(figsize = (10,5))
g = sns.barplot(y=col[:25], x = importances[indices][:25] , orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("Feature importance")
plt.show()


# In[ ]:


X_train.shape[1]


# Now we have 61 features. Above we can see the top 25 and I guess many of them dont't improve the acuracy

# In[ ]:


best_score=0
var=[]
scor=[]
for i in np.arange(0.1, 1.9, 0.1):

    str_t = np.str(i) + "*mean"
    ExtC = ExtraTreesClassifier(min_samples_split=10, n_estimators=100 , random_state=1)
    
    model = SelectFromModel(ExtC, threshold = str_t )
    model.fit(X_train, y_train)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
     
    cvs = cross_val_score(ExtC, X_train_new, y = y_train, scoring = "accuracy", cv = kfold,
                          n_jobs=4)
    var.append(X_train_new.shape[1])
    scor.append(cvs.mean())
    score=round(cvs.mean(),3)
    print ("The cv accuracy is: ", round(score,4), " - i = ", i, 
           " - features = ", X_train_new.shape[1])
    if score>best_score:
        best_score = score
        print("*** BEST : Score:", score, " features selected: ", X_train_new.shape[1])
        X_train_best = X_train_new
        X_test_best = X_test_new


# In[ ]:


plt.plot(var, scor, lw=6, c="blue")
plt.plot(var, scor, "ro", ms=10, alpha=0.8)
plt.xlabel('Number of features')
plt.ylabel('Score')
plt.title('Selection Features')
plt.show()


# Wow! We can see that we have a lot of  features for nothing! Basicly the Tickets Features.
# These features generate noise

# In[ ]:


#Selected features
g = sns.barplot(y=col[:X_test_best.shape[1]], x = importances[indices][:X_test_best.shape[1]] , orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("Feature importance")
plt.show()


# So, my new features "Groups", "Ticket_tog" and "CabinYN" are some of the important features !!
# They are not among the most important features such as: Embarked and Tickets.

# The best algorithm is Extra Tree Classifier.
# Create the results file.

# In[ ]:


ExtC.fit(X_train_best, y_train)
test_Survived = pd.Series(ExtC.predict(X_test_best), name="Survived")
r = pd.DataFrame(test_Survived, dtype="int64")
results = pd.concat([IDtest,r],axis=1)
results.to_csv("result.csv",index=False)


# 
