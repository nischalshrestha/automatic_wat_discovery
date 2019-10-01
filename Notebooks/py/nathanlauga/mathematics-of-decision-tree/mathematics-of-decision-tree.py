#!/usr/bin/env python
# coding: utf-8

# # The mathematics of Decision Tree
# *November 2018*
# 
# ***
# 
# Hello world !
# 
# Since last month I began a series of Kernel that aim to understand the math behind the models in Machine Learning. 
# 
# If you want more informations I invite you to take a look on this topic : [Understand the math behind models](https://www.kaggle.com/general/69885)
# 
# Today, it's the turn of the **Decision Tree**.
# 
# The goals of this kernel for me is :
# * to learn how to write a model in python just with numpy linear algebra library
# * to improve my writing skills (in English and with Markdown)
# * to have a better sense of the notebooks
# 
# I hope you will enjoy it and if you have any suggestions : I will be glad to hear them to improve my skills !
# 
# ![](https://www.sciencemag.org/sites/default/files/styles/article_main_large/public/cc_iStock-478639870_16x9.jpg?itok=1-jMc4Xv)
# 
# ## Table of content
# 
# * [1. Load libraries and read data](#1)
#     * [1.1 Load libraries](#1.1)
#     * [1.2 Read the data](#1.2)
#     * [1.3 A quick look on the data](#1.3)
#     
# * [2. Decision Tree](#2)
#     * [2.1 Entropy](#2.1)
#     * [2.2 Gini impurity](#2.2)
#     * [2.3 Information gain](#2.3)
#     * [2.4 Decision Tree building - algorithm](#2.4)
#     * [2.5 F1 Score / Accuracy](#2.5)
#     
# ## <a id="1">1. Load libraries and read data</a>
# 
# ### <a id="1.1">1.1 Load libraries</a>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.tools import FigureFactory as ff
from time import time
from IPython.display import display

warnings.filterwarnings('ignore')


# ### <a id="1.2">1.2 Read the data</a>

# In[ ]:


data = pd.read_csv('../input/train.csv')


# ### <a id="1.3">1.3 A quick look on the data</a>

# In[ ]:


data.head()


# For this model I choose to handle 2 types of columns : categorical columns and continuous columns.
# 
# In this dataset, I prefer to remove 3 columns because I think it's Useless in this kernel to keep them : Name, Ticket and Cabin.

# In[ ]:


data = data.drop(columns=['Name','Ticket','Cabin'])
len(data)


# Then since the goal of this kernel is to understand the math I don't handle NA values.

# In[ ]:


data = data.dropna()
len(data)


# ## <a id="2"> 2. Decision Tree </a>
# 
# Now let's dive into the math that make the **Decision Tree model** !
# 
# To be honest at the beginning I had some difficulties to really understand how it works because it's really different from linear regression for example. Nevertheless this model has a big **advantage** it's really easy for a human to understand how it gets the predicted result. So to start I search "Decision Tree" on Google Image and just took a look on the image to understand. 
# 
# Here is an example of a Tree :
# 
# ![](https://annalyzin.files.wordpress.com/2016/07/decision-trees-titanic-tutorial.png)
# 
# So now the question is : **How to know where to split ?** 
# 
# On many ressources that I checked they just choose a feature and a value just by visualizing on a plot. But it's not the math that automate the model ! And then I find different notions that define the model : 
# * Entropy 
# * Gini impurity
# * Information gain
# 
# And just with that we can build our model !! First let's define those terms.
# 
# ### <a id="2.1">Entropy</a>
# 
# The entropy is define by this formula : 
# $$ H(T) = I_E(p_1,p_2, ... , p_J) = - \sum_{i=1}^J p_i log_2 p_i$$
# 
# where $p_1, p_2, ...$ are fractions that add up to 1 and represent the percentage of each class present in the child node (node for a specific value of the feature). And $J$ is the number of classes for this feature.
# 
# Let's take an example : we have 10 rows of a feature $X$. Here is the data : $X = [A, A, A, A, B, B, B, C, C, C]$
# 
# To obtain $p$ for each class we just count the number of value and divise it by the number of elements in $X$ :
# 
# $$p_A = \frac{(number\space of\space A)}{(length\space of\space X)} = \frac{4}{10} = 0.4 $$
# 
# $$p_B = \frac{(number\space of\space B)}{(length\space of\space X)} = \frac{3}{10} = 0.3 $$
# 
# $$p_C = \frac{(number\space of\space C)}{(length\space of\space X)} = \frac{3}{10} = 0.3 $$
# 
# And then we can calculate the entropy for this feature :
# 
# $= -(p_A log_2 p_A + p_B log_2 p_B + p_C log_2 p_C )\\
# = -(0.4 \times (-1.3219) + 0.3 \times  (-1.737) + 0.3 \times  (-1.737) )\\
# =-( -0.529 -0.5211 -0.5211 )\\
# = 1.571$
# 
# Let's code it !

# In[ ]:


def entropy(data, y_col):
    # Get all the values for the Y column
    y_val = data[y_col].value_counts().index.values
    # Get the vector with the number of element for each Y class
    tmp = np.array([len(data[data[y_col] == y_val[i]]) for i in range(0, len(y_val))])
    return sum(-tmp / len(data) * np.log2(tmp / len(data)))


# ### <a id="2.2">2.2 Gini impurity</a>
# 
# This formula is very similar to the entropy formula :
# 
# $$ H(T) =  I_G(p_1,p_2, ... , p_J) = 1 - \sum_{i=1}^J p_i^2$$
# 
# So let's code it 👍

# In[ ]:


def gini_impurity(data, y_col):
    # Get all the values for the Y column
    y_val = data[y_col].value_counts().index.values
    # Get the vector with the number of element for each Y class
    tmp = np.array([len(data[data[y_col] == y_val[i]]) for i in range(0, len(y_val))])
    return 1 - sum((tmp / len(data))**2)


# ### <a id="2.3">2.3 Information gain</a>
# 
# The information gain is really important because it allows us to get the gain for a specific feature on the split. For that it's necessary to introduce the notion of parent and children. The parent is where the data is full and children represent the data filtered by the value of the feature (if we sum the length of each children it gives us the length of the parent).
# 
# Let's see an example to understand. Here is our dataset : 

# In[ ]:


test = pd.DataFrame(data=[['A',1],['A',1],['A',1],['B',1],['B',0],['B',0]]
                    , columns=['letter','bit'])
test


# So the entropy or gini impurity of the parent is based on all the rows of this dataset but if we want the children based on the feature *letter* here is what our children will be :

# In[ ]:


display(test[test['letter'] == 'A'])
display(test[test['letter'] == 'B'])


# In general information gain is used with entropy but here I choosed to use it with both entropy and gini impurity.
# 
# So here is the formula :
# 
# $$
#   \overbrace{IG(T, a)}^\text{Information Gain} = \overbrace{H(T)}^\text{Entropy / Gini (parent)} - \overbrace{H(T\space|\space a)}^\text{Weighted Sum of Entropy / Gini (Children)}
#  $$
#  
#  So if we take the entropy metric we get this :
#  $$
#  IG(T, a) = Entropy(T) - \sum_a{p(a) Entropy(T\space | \space a)}
#  $$
#  
#  And now we can do it in python ! 😉

# In[ ]:


def info_gain(data, feature_col, y_col, criterion='entropy'):
    # Get all the values for this feature
    feature_val = data[feature_col].value_counts().index.values
    # Get the vector of the number of element for each class
    len_feat = np.array([len(data[data[feature_col] == feature_val[i]]) for i in range(0, len(feature_val))])
    # Get the vector of the criterion for each class
    if criterion == 'entropy':
        crit_feat = np.array([entropy(data[data[feature_col] == feature_val[i]], y_col) for i in range(0, len(feature_val))])
        gain = entropy(data, y_col) - sum((len_feat / len(data)) * crit_feat)
    elif criterion == 'gini':
        crit_feat = np.array([gini_impurity(data[data[feature_col] == feature_val[i]], y_col) for i in range(0, len(feature_val))])
        gain = gini_impurity(data, y_col) - sum((len_feat / len(data)) * crit_feat)
    return gain


# ### <a id="2.4">2.4 Decision Tree building - algorithm</a>
# 
# Here is the most difficult part for the Decision tree : build the algorithm to predict the output. 
# 
# To start this we need to go step by step : first the question we need to ask ourselves is **how to know where to split ? **
# 
# We have the information gain formula which gives us the ability to quantify the gain of a feature. Now the logic is just to compare the gain of each feature and then keep the best ! It's easy on the paper but for categorical feature because we have a limited number of classes. For example for the *Sex* feature it's just *male* and *female*. When it's a continuous feature it's really different for juste a dataset with 1 000 rows we can have like 800 differents values so if we split like it's a categorical value the model is overfitting and it's not really usefull.. 
# 
# So how can we deal with continuous feature ? The answer is just to transform it in a categorical value more precisely into a boolean feature (True or False) we can take an example : the *Fare* feature we can choose to split it at the value 50 --> *Fare <= 50*
# 
# We still have a problem for continuous feature : how to choose where to split this feature ? I used an algorithm for that, it has a advantage : you are sure to take the best split and a BIG inconvinient : more data you have slower is your model. For this kernel I will use it because my goal is to understand the model not necessary to optimize it (but if you have suggestion about it do not hesitate).
# 
# The algorithm is simple :
# 
# **Step 1 :** Sort the values of the feature : $[v_1, v_2, ..., v_n]$
# 
# **Step 2 :** For i = 1 ... n- 1 : get the entropy / gini impurity for a new feature that is : 
# $$ data <= \frac{(v_i + v_{i+1})}{2} $$
# Then we choose the best split.
# 
# **Okay, now we have our best feature to split !**
# 
# Then we create n leaves where n is the number of class of the feature after that we repeat the operation on the children leaves.
# 
# Here is our algorithm to build the Decision Tree model ! It's Great now let's code it (I choose to do it with a DecisionTree class and Leaf class)
# 
# To start we need some informations :
# * The data
# * The name of the column that we want to predict
# * The name of all the categorical feature
# * The name of all the continuous feature
# * Which metric to evaluate our model : gini or entropy
# * The maximum depth of the tree

# In[ ]:


class DecisionTree:
    def __init__(self, data, y_col, cat_cols=[], cont_cols=[], criterion='entropy', max_depth=5):
        self.data = data
        self.y_col = y_col
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.criterion = criterion
        self.leaves = list()
        self.max_depth = max_depth if len(cat_cols) > max_depth or len(cont_cols) > 0 else len(cat_cols)
        
    def get_best_split_continuous(self, feature_col, data):
        # Init best gain and best split
        best_gain, best_split = -1, -1
        # Get all the values for this feature
        feat_val = data[feature_col].drop_duplicates().sort_values().reset_index(drop=True).dropna()
        # Get the information gain for each feature and keep the best
        for i in range(1, len(feat_val)):
            split = (feat_val[i - 1] + feat_val[i]) / 2
            data[feature_col + '_tmp'] = data[feature_col] <= split
            gain = info_gain(data, feature_col + '_tmp', self.y_col, criterion=self.criterion)
            best_gain, best_split = (gain, split) if best_gain < gain else (best_gain, best_split)
        return best_split, best_gain
    
    def get_best_feat_leaf(self, data, leaf=None):
        cat_cols = [c for c in self.cat_cols if c not in leaf.get_feat_parent()] if leaf is not None else self.cat_cols
        all_gains = [info_gain(data, c, self.y_col, criterion=self.criterion) for c in cat_cols]
        continuous = [(c, self.get_best_split_continuous(c, data)) for c in self.cont_cols]
        cont_gains = [c[1][1] for c in continuous]

        all_gains = all_gains + cont_gains if len(continuous) > 0 and len(all_gains) > 0 else all_gains if len(
            all_gains) > 0 else cont_gains
        all_cols = cat_cols + self.cont_cols if len(cat_cols) > 0 and len(self.cont_cols) > 0 else cat_cols if len(
            cat_cols) > 0 else cont_cols
        
        best_feat = pd.Series(data=all_gains, index=all_cols).idxmax()
        
        return best_feat if best_feat not in cont_cols else [c for c in continuous if c[0] == best_feat][0]
        
    def learn(self):
        t0 = time()
        print('----- START LEARNING -----')
        # Get the first feature where to split
        feat = self.get_best_feat_leaf(self.data)
        split = None
        
        # If the type is not a string then it's a continuous feature 
        # and we get the best value to split
        if (type(feat) != type(str())):
            split = feat[1][0]
            feat = feat[0]    
        
        # Add it to the Tree
        self.leaves.append(Leaf(None
                                , None
                                , self.data
                                , feat
                                , self.data[self.y_col]
                                , split))
        
        for i in range (1, self.max_depth):
            print('----- BEGIN DEPTH '+str(i)+' at %0.4f s -----' % (time() - t0))
            # Get all the leaves that are in the upper depth
            leaves_parent = [l for l in self.leaves if l.depth == i-1]
            
            # If there is 0 parent we can stop the learning algorithm
            if(len(leaves_parent) == 0):
                break
            else:
                for leaf in leaves_parent:
                    # If there is only one value that means it's useless to split
                    # because we already have our prediction
                    if(len(leaf.values) == 1):
                        continue
                    # Get all values for the current feature
                    feature_val = leaf.data[leaf.feature] <= leaf.split if leaf.split is not None else leaf.data[leaf.feature]
                    feature_val = feature_val.value_counts().index.values
                    
                    # Add all possibilities to the Tree
                    for k in range(0, len(feature_val)): 
                        if leaf.split is None:
                            data = leaf.data[leaf.data[leaf.feature] == feature_val[k]]
                        else:
                            split_cond = leaf.data[leaf.feature] <= leaf.split
                            data = leaf.data[split_cond == feature_val[k]]
                        
                        if len(data) > 0:
                            # Get the best feature for the split
                            next_feat = self.get_best_feat_leaf(data, leaf)

                            split = None

                            # If the type is not a string then it's a continuous feature 
                            # and we get the best value to split
                            if (type(next_feat) != type(str())):
                                split = next_feat[1][0]
                                next_feat = next_feat[0]
                            
                            self.leaves.append(Leaf(prev_leaf=leaf
                                                , condition=feature_val[k]
                                                , data=data
                                                , feature=next_feat
                                                , values=data[self.y_col]
                                                , split=split))
        print('Number of leaves : '+str(len(self.leaves)))
        print('----- END LEARNING : %0.4f s-----' % (time() - t0))
        print()
        
    def display_final_leaves(self):
        leaves = [l for l in self.leaves if len(l.values) == 1 or l.depth == self.max_depth]
        for l in leaves:
            l.display()
                        
    def predict(self, data):
        pred = list()
        for i in range(0, len(data)):
            row = data.iloc[i,:]
            leaf = self.leaves[0]
            while(len(leaf.values) > 1 and leaf.depth < self.max_depth):
                if leaf.split is None:
                    tmp_leaf = [l for l in self.leaves if (l.prev_leaf == leaf and l.condition == row[leaf.feature])]
                else:
                    tmp_leaf = [l for l in self.leaves if (l.prev_leaf == leaf and l.condition == (row[leaf.feature] <= leaf.split))]
                if (len(tmp_leaf) > 0):
                    leaf = tmp_leaf[0]                    
                else:
                    break
            pred.append(leaf.pred_class)
        return pred
    
class Leaf:
    def __init__(self, prev_leaf, condition, data, feature, values, split=None):
        self.prev_leaf = prev_leaf
        self.depth = 0 if prev_leaf is None else prev_leaf.depth+1
        self.condition = condition
        self.data = data
        self.feature = feature
        self.values = values.value_counts(sort=False)
        self.pred_class = self.set_predict_class()
        self.split = split
        
    def set_predict_class(self):
        return self.values.idxmax()
    
    def get_feat_parent(self):
        cols = [self.feature]
        leaf = self
        while(leaf.prev_leaf is not None):
            cols.append(leaf.prev_leaf.feature)
            leaf = leaf.prev_leaf
        return cols

    def display(self):
        cond = ''
        leaf = self
        while(leaf.prev_leaf is not None):
            if leaf.prev_leaf.split is None:
                cond = str(leaf.prev_leaf.feature)+' : '+str(leaf.condition)+' --> '+cond
            else:
                cond = str(leaf.prev_leaf.feature)+' <= '+str(round(leaf.prev_leaf.split,2))+' : '+str(leaf.condition)+' --> '+cond
            leaf = leaf.prev_leaf
        print(cond+' prediction : '+str(self.pred_class))


# 
# 
# So the main function of the Decision tree is the *learn* function that execute the building algorithm. I also added a function that display all the path that give us a prediction.
# 

# In[ ]:


cat_cols = ['Sex', 'Embarked', 'SibSp', 'Parch']
cont_cols = ['Age', 'Fare']

tree_gini = DecisionTree(data, 'Survived', cat_cols=cat_cols, cont_cols=cont_cols, criterion='gini', max_depth=5)
tree_entr = DecisionTree(data, 'Survived', cat_cols=cat_cols, cont_cols=cont_cols, criterion='entropy', max_depth=5)
tree_gini.learn()
tree_entr.learn()


# I'm sorry that the visualisation of the tree is not really good but I didn't code something too complicated 😉

# In[ ]:


print('----- GINI TREE -----')
tree_gini.display_final_leaves()
print()
print('----- ENTROPY TREE -----')
tree_entr.display_final_leaves()


#    ### <a id="2.5">2.5 F1 Score / Accuracy</a>
#    
#    Now that we have two trees one with entropy and one with gini impurity we can see if they have a great performance !

# In[ ]:


def plot_confusion_matrix(y_true, y_pred, name):
    trace = go.Heatmap(z=confusion_matrix(y_true, y_pred),
                       x=['Died', 'Survived'],
                       y=['Died', 'Survived'],
                       colorscale='Reds')
    
    layout = go.Layout(title='Confusion Matrix '+name,
                            xaxis=dict(
                                title='Prediction'
                            ),
                            yaxis=dict(
                                title='Real'
                            )
                        )
    fig = go.Figure(data=[trace], layout=layout)
    
    py.iplot(fig)


# In[ ]:


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
pred_gini = tree_gini.predict(data)
pred_entr = tree_entr.predict(data)

print('----- GINI TREE -----')
print('F1 Score : '+str(f1_score(data['Survived'], pred_gini)))
print('Accuracy : '+str(accuracy_score(data['Survived'], pred_gini)))
print('----- ENTROPY TREE -----')
print('F1 Score : '+str(f1_score(data['Survived'], pred_entr)))
print('Accuracy : '+str(accuracy_score(data['Survived'], pred_entr)))

plot_confusion_matrix(data['Survived'], pred_gini, 'Gini impurity')
plot_confusion_matrix(data['Survived'], pred_entr, 'Entropy')


# So I hope that you enjoyed this kernel, please feel free to say what you though about my work !
# 
# Thanks for reading,
# 
# $Nathan.$
# 
# Sources :
# * [Decision Trees in Machine Learning](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)
# * [Wikipedia : Decision Tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
# * [Introduction to Decision Tree Learning](https://heartbeat.fritz.ai/introduction-to-decision-tree-learning-cd604f85e236)
# * [Coursera : Picking the best threshold to split on](https://www.coursera.org/lecture/ml-classification/optional-picking-the-best-threshold-to-split-on-sKrGp)
