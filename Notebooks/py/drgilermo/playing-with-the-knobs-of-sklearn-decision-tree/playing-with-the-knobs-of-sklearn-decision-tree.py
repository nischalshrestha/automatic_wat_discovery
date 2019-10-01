#!/usr/bin/env python
# coding: utf-8

# Decision Tree is a basic but common and useful classification algorithm. It is also the basis of slightly more advanced classification techniques such as Random Forest and (some implementations of) Gradient Boosting. 
# 
# The Titanic data set, comprised of both categorical and numerical features, is a great use-case for tree-based algorithms, and indeed many kagglers used them in this competition.
# 
# Sklearn is one of the most commonly used machine learning libraries in python, and in this script I will explore the effect of the different hyper-parameters of the [sklearn decision tree classifier.][1]  
# 
# 3 parameters would be exclude from this analysis: 
# 
#  1. min_weight_fraction_leaf 
#  2. random_state 
#  3. min_impurity_split
# 
#   [1]: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from subprocess import check_output
from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re
import matplotlib.pyplot as plt

print(check_output(["ls", "../input"]).decode("utf8"))


# Read the data

# In[ ]:


train = pd.read_csv("../input/train.csv")
testset = pd.read_csv("../input/test.csv")


# Encode the sex feature to be a binary one

# In[ ]:


train['Sex'][train.Sex == 'female'] = 1
train['Sex'][train.Sex == 'male'] = 0
train.head(5)


# Using the new [Imputer][1] sklearn class to modify non-existent entries:
# 
# 
#   [1]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html

# In[ ]:


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
columns = ['Pclass','Sex','Age','Fare','Parch','SibSp']
for col in columns:
    train[col] = imp.fit_transform(train[col].reshape(-1,1))


# Split the data to train and test sets:

# In[ ]:


X = train[columns]
y = train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# #Decision Tree Classifier Hyper-Parameters
# 
# 

# ## Criterion ##
# Copy-pasting from the  [sklearn documentation][1]:
# 
# **criterion : string, optional (default=”gini”)
# The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.**
# 
# The split in each node is done based on a criterion. that is, we need to find a feature and a value of the feature that would partition the data into two groups for an optimal classification.
# 
# This is usually done based on two metrics: [Gini Impurity and Entropy (or information gain][2]
# 
# This choice is not considered to be very critical. In most cases both metrics would perform similarly.
# 
# 
#   [1]: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#   [2]: http://haohanw.blogspot.co.il/2014/08/ml-decision-tree-rule-selection.html

# In[ ]:


clf = DecisionTreeClassifier(max_depth = 3)
clf.fit(X_train,y_train)
print('Accuracy using the defualt gini impurity criterion...',clf.score(X_test,y_test))

clf = DecisionTreeClassifier(max_depth = 3, criterion = "entropy")
clf.fit(X_train,y_train)
print('Accuracy using the entropy criterion...',clf.score(X_test,y_test))


# ## Splitter
# 
# from the documentation:
# 
# **The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.**
# 
# Usually, the decision tree chooses to split each node in the optimal point (based on the gini impurity or entropy information gain). However, it would be faster, and possibly not much worse, to use a random split.
# 

# We indeed see that our case is no different. 

# In[ ]:


t = time.time()
clf = DecisionTreeClassifier(max_depth = 3, splitter = 'best')
clf.fit(X_train,y_train)
print('Best Split running time...',time.time() - t)
print('Best Split accuracy...',clf.score(X_test,y_test))

t = time.time()
clf = DecisionTreeClassifier(max_depth = 3, splitter = 'random')
clf.fit(X_train,y_train)
print('Random Split running time...',time.time() - t)
print('Random Split accuracy...',clf.score(X_test,y_test))


# In[ ]:


clf = DecisionTreeClassifier(max_depth = 3)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# In[ ]:


clf = DecisionTreeClassifier(max_depth = 3 ,splitter = 'random')
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# We can see that the random splitter chooses rather odd splitting values (for instance, split on 0.2 for the sex where 1 is female and 0 is male doesn't seem to make much sense logically). Since the split is random, this means that the selected feature can be different and the structure of the trees would be different.
# 
# As expected, the random splitter trees is faster than the best-splitter tree, but performs worse.
# 

# ## Max Features
# 
#  - The number of features to consider when looking for the best split:
#     2. If int, then consider max_features features at each split.
#     3. If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
#     4. If “auto”, then max_features=sqrt(n_features).
#     5. If “sqrt”, then max_features=sqrt(n_features).
#     6. If “log2”, then max_features=log2(n_features).
#     7. If None, then max_features=n_features.
#     8. Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
# 
# As we are going to see, most hyper-parameters of the decision tree are concerned with over-fitting (and decision tree are prone definitely prone to over-fitting).
# 
# At every node, the algorithm is looking for that feature and partition that would yield the best outcome. Using the max_feature knob, we can limit the number of features to be considered. the algorithm would randomly choose the number of features (based on the limit) and only then pick the best partition from the new cohort.
# 
# This is done in order to increase the stability of the tree and reduce variance and over-fitting.

# In[ ]:


test_score = []
train_score = []
max_features = range(len(columns)-1)
for feat in max_features:
    clf = DecisionTreeClassifier(max_features = feat + 1)
    clf.fit(X_train,y_train)
    train_score.append(clf.score(X_train,y_train))
    test_score.append(clf.score(X_test,y_test))

plt.figure(figsize = (8,8))
plt.plot(max_features,train_score)
plt.plot(max_features, test_score)
plt.xlabel('Max Features')
plt.ylabel('Accuracy')
plt.legend(['Training set','Test set'])


# We see that the training score is constant and close to 100%, while the validation score is much lower. this is obviously a case of over-fitting. This is due to the fact that we haven't limited the tree depth. therefore, it keeps creating new knobs until all the leaves are "pure" (that is, only populated by samples either labelled as 1 or 0, but not both).
# 
# Let's repeat that with a limit on the tree depth
# 

# In[ ]:


test_score = []
train_score = []
max_features = range(len(columns)-1)
for feat in max_features:
    clf = DecisionTreeClassifier(max_features = feat + 1, max_depth = 5)
    clf.fit(X_train,y_train)
    train_score.append(clf.score(X_train,y_train))
    test_score.append(clf.score(X_test,y_test))
    
plt.figure(figsize = (8,8))   
plt.plot(max_features,train_score)
plt.plot(max_features, test_score)
plt.xlabel('Max Features')
plt.ylabel('Accuracy')
plt.legend(['Training set','Test set'])


# There is no clear trend. Clearly when the number of features is too low the tree is under fitting. but even when we use all our features at every node we don't get to the point of over fitting.
# 
# This is probably due to the fact that we have a relatively small number of features which were chosen carefully based on their apparent relevance. 

# ## Max Depth
# 
# **The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.**
# 
# I find this parameter to be the most useful in reducing over-fitting. 

# In[ ]:


test_score = []
train_score = []
for depth in range(20):
    clf = DecisionTreeClassifier(max_depth = depth + 1)
    clf.fit(X_train,y_train)
    train_score.append(clf.score(X_train,y_train))
    test_score.append(clf.score(X_test,y_test))

plt.figure(figsize = (8,8))
plt.plot(range(20),train_score)
plt.plot(range(20), test_score)
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.legend(['Training set','Test set'])


# The deeper the tree is, the better it fits the training data. however, if it is deep enough, if fails to generalize and starts to over-fit. more complex paths are created and the number of samples in every split is getting smaller and therefor less statistically meaningful. 
# 
# Let's visualize a shallow and a deep tree:

# In[ ]:


clf = DecisionTreeClassifier(max_depth = 6)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# In[ ]:


clf = DecisionTreeClassifier(max_depth = 3)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# When the tree is deep, we get nodes and leaves with a very small number of samples, which are therefore not very informative. we can also see that in the shallower tree, though not as common. 
# 
# The shallow tree is more simple and straight forward: 
# 
#  - if you're a man, you are unlikely to survive unless you're young.
#  - If you're a woman, you're likely to survive unless you paid a low
#    fare and bought a low-class ticket.
# 
# The deeper tree has more convulated rules: if you're younger than this, but older than this, but younger than this. etc.
# 
# That's a good opportunity to look at the feature importance (based on the shallow tree):

# In[ ]:


plt.barh(range(len(columns)),clf.feature_importances_)
plt.yticks(range(len(columns)),columns)
plt.xlabel('Feature Importance')


# ## Min Sample Split
# 
# **The minimum number of samples required to split an internal node:
# If int, then consider min_samples_split as the minimum number.
# If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.**
# 
# This limitation is similar to the tree depth, bit instead of constraining the depth, it constrains the number of samples per split. we have seen in the previous example how a low number of samples in a split may lead to over-fitting

# In[ ]:


test_score = []
train_score = []
min_sample_split = np.arange(5,100,5)
for split in min_sample_split:
    clf = DecisionTreeClassifier(min_samples_split = split)
    clf.fit(X_train,y_train)
    train_score.append(clf.score(X_train,y_train))
    test_score.append(clf.score(X_test,y_test))
    
plt.figure(figsize = (8,8))   
plt.plot(min_sample_split,train_score)
plt.plot(min_sample_split, test_score)
plt.xlabel('Min Sample Split')
plt.ylabel('Accuracy')
plt.legend(['Training set','Test set'])


# In[ ]:


clf = DecisionTreeClassifier(min_samples_split = 5)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# In[ ]:


clf = DecisionTreeClassifier(min_samples_split = 80)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# ## Min sample leaf
# 
# **The minimum number of samples required to be at a leaf node:
# If int, then consider min_samples_leaf as the minimum number.
# If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.**
# 
# This is similar to the previous parameter, but concerns with the leaf nodes. this make it a stronger limitation (for the same limit value). If this limit is too tight, our model would under-fit:

# In[ ]:


test_score = []
train_score = []
min_sample_leaf = np.arange(5,100,5)
for leaf in min_sample_leaf:
    clf = DecisionTreeClassifier(min_samples_leaf = leaf)
    clf.fit(X_train,y_train)
    train_score.append(clf.score(X_train,y_train))
    test_score.append(clf.score(X_test,y_test))

plt.figure(figsize = (8,8))
plt.plot(min_sample_split,train_score)
plt.plot(min_sample_split, test_score)
plt.xlabel('Min Sample Leaf')
plt.ylabel('Accuracy')
plt.legend(['Training set','Test set'])


# In[ ]:


clf = DecisionTreeClassifier(min_samples_leaf = 5)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# In[ ]:


clf = DecisionTreeClassifier(min_samples_leaf = 45)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# ## Min leaf nodes
# 
# **Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.**

# In[ ]:


test_score = []
train_score = []
max_leaf_nodes  = np.arange(5,100,5)
for leaf in max_leaf_nodes :
    clf = DecisionTreeClassifier(max_leaf_nodes  = leaf)
    clf.fit(X_train,y_train)
    train_score.append(clf.score(X_train,y_train))
    test_score.append(clf.score(X_test,y_test))
    
plt.figure(figsize = (8,8))
plt.plot(min_sample_split,train_score)
plt.plot(min_sample_split, test_score)
plt.xlabel('Min Leaf Nodes')
plt.ylabel('Accuracy')
plt.legend(['Training set','Test set'])


# In[ ]:


clf = DecisionTreeClassifier(max_leaf_nodes = 10)
clf.fit(X_train,y_train)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 5,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# ## Class Weight
# 
#  - Weights associated with classes in the form {class_label: weight}. If
#    not given, all classes are supposed to have weight one. For
#    multi-output problems, a list of dicts can be provided in the same
#    order as the columns of y.
#  - The “balanced” mode uses the values of y to automatically adjust
#    weights inversely proportional to class frequencies in the input data
#    as n_samples / (n_classes * np.bincount(y)) For multi-output, the
#    weights of each column of y will be multiplied.
#  - Note that these weights will be multiplied with sample_weight (passed
#    through the fit method) if sample_weight is specified.
# 
# Unlike the last parameters, this one (usually) deals with class imbalance.
# 
# We have a mild imbalance in our problem. 62% of all passangers died in the titanic disaster. In the training data the number is only 59%. but this parameter can come very handy where the data is more skewed. 
# 
# Let's say that for some reason we are more interested in the people who survived. that is, we don't mind our classifiers to have a higher rate of false positives (people who were predicted to survive but actually did not) as long as we get a lower false negative rate. that is, if someone survived, we are very likely to predict it (this is of course "Recall").
# 
# Let's see the effect:

# In[ ]:


clf = DecisionTreeClassifier(max_depth = 3)
clf.fit(X_train,y_train)
print('Class Weight is normal...')
print(confusion_matrix(y_test,clf.predict(X_test)))

clf = DecisionTreeClassifier(max_depth = 3, class_weight = 'balanced')
clf.fit(X_train,y_train)
print('Class weight is balanced to compensate for class imbalance...')
print(confusion_matrix(y_test,clf.predict(X_test)))


# In the first confusion matrix, we can see that our classifier predicted that 192 people would drown, and 103 would survive where in fact 175 have died and 120 survived. so while the average accuracy is slightly above 82%' our accuracy is 90% when it comes to those who died, and 70% with those who survived.
# 
# This makes sense since more people died than didn't and this pushes the classifier towards pessimism.
# 
# In the second case, the accuracy goes down to 80%, but the difference between the positive and negative accuracy is now smaller: 82% accuracy when predicting someone would not survive, and 76% when predicting the opposite. 

# ## Presort
# 
# **Whether to presort the data to speed up the finding of best splits in fitting. For the default settings of a decision tree on large data sets, setting this to true may slow down the training process. When using either a smaller data set or a restricted depth, this may speed up the training.**
# 
# In our case, the data set is small and our tree depth is restricted. therefore, if we can trust the documentation, we expect the algorithm to be faster when this option is enabled:

# In[ ]:


clf = DecisionTreeClassifier(max_depth = 3)
t = time.time()
clf.fit(X_train,y_train)
print('Without presot accuracy', clf.score(X_test,y_test))
print('Without presort runtime...',time.time() - t)

clf = DecisionTreeClassifier(max_depth = 3, presort = True)
t = time.time()
clf.fit(X_train,y_train)
print('With presot accuracy', clf.score(X_test,y_test))
print('With Presort runtime...',time.time() - t)


# There is some difference, although I was not able to repeat it very consistently...
