#!/usr/bin/env python
# coding: utf-8

# ## Bayesian network approach using libpgm
# 
# In this tutorial I show how to implement a Bayesian network on the Titanic dataset. I employ the python Libpgm library for modeling the network in three different and independent ways: 
#     1. the structure and the parameters (CPD) at each node are defined and calculated manually, the library is thus used to encode this information. 
#     2. the library is applied to calculate the structure of the network
#     3. the library calculates both the strucuture and the parameters of the network

# In[ ]:


import pandas as pd
import graphviz as gv
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# ## Data Dictionary
# 
# * Variable	Definition	Key
# * survival 	Survival 	0 = No, 1 = Yes
# * pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
# * sex 	Sex 	
# * Age 	Age in years 	
# * sibsp 	# of siblings / spouses aboard the Titanic 	
# * parch 	# of parents / children aboard the Titanic 	
# * ticket 	Ticket number 	
# * fare 	Passenger fare 	
# * cabin 	Cabin number 	
# * embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# ### Variable Notes
# 
# * pclass: A proxy for socio-economic status (SES)
#  - 1st = Upper
#  - 2nd = Middle
#  - 3rd = Lower
# 
# * age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * sibsp: The dataset defines family relations in this way...
#     Sibling = brother, sister, stepbrother, stepsister
#     Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# * parch: The dataset defines family relations in this way...
#     Parent = mother, father
#     Child = daughter, son, stepdaughter, stepson
#     Some children travelled only with a nanny, therefore parch=0 for them.

# **Bayesian network** is a special case of graphical Models. Graphical models solve the problem of defining the joint probability distribution, which is is difficult and requires a huge number of parameters.
# 
# In particular, Bayesian Networks present the following properties: 
#  * The independence assumptions made in a BN allows to avoid specifying the joint distributions. 
#  * A BN is a directed acyclic graph, and it allows a compact and modular representation of the joint probability distribution (probabilities at vertices are the local probability models). It allows to observe the conditional independence.
#  * In general, each variable X in the model is associated with a conditional probability distribution (CPD) that specifies a distribution over the values of X for each possible joint assignment of its parents in the model (local distributions). For a node with no parents, the CPD is conditioned on the empty set of variables, and can be seen as a marginal distribution (or prior).
#  * A BN consist of a structure and the CPDs. The chain rule for BN allows to express the joint distributions as a product of CPDs <br\>
#  => P(X1 , X2 , ... , Xn ) = Product_i  P(X_i | Parents( X_i )). 

# ### Libpgm library
# 
# The library reads the network information (nodes, edges, and CPD probabilities) from a JSON-formatted
# file with a specific format. This JSON file is read into
# * NodeData 
# * GraphSkeleton

# In[ ]:


from libpgm.graphskeleton import GraphSkeleton
from libpgm.nodedata import NodeData


# The TableCPDFactorization object wraps the discrete Bayesian network and allows us to query the CPDs in the network.
# 
# To make queries: use 
# > table_cpd=getTableCPD() <br\>
# > table_cpd.specificquery(dict(Offer='1'),dict(Grades='0'))
# 
# In BN parlance, the first argument is the "query" whilst the second corresponds to the "evidence". 

# In[ ]:


from libpgm.tablecpdfactorization import TableCPDFactorization


# In this tutorial I'm using a discrete Bayesian network (variables take on only discrete
# values).

# In[ ]:


from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork


# ## Prepare training data

# In[ ]:


df_input = pd.read_csv('../input/train.csv', sep=',')
df_test  = pd.read_csv('../input/test.csv', sep=',')


# For the sole purpose of illustrating how to use BN on a dataset, I'll keep only a few features in what follows

# In[ ]:


df_train            = df_input[['Survived', 'Pclass','Sex', 'Fare']][df_input.Fare!=0].dropna()
df_train.dropna(inplace=True)
df_train.loc[:,'Sex']  = df_train.Sex.map({'female':0 , 'male':1})

# Fare is arbitrary divided in two categories: cheap and expensive
df_train.loc[:,'Fare'] = pd.cut(df_train.Fare, [df_input.Fare.min(),15, df_input.Fare.max()], labels =[0,1])
df_train = df_train.rename(columns = {'Survived' : 'Surv'})
df_train_target        = df_train['Surv']


# In[ ]:


def get_probs_surv_cond(df, df_target, Surv, Pclass, Sex):
    # Return survival probability conditioned on Class and Sex
    # P(Surv | Sex, Pclass)
    return (df[ (df.Surv==Surv) & (df.Pclass==Pclass) & (df.Sex==Sex)].shape[0]
            /(1.0*df[(df.Pclass==Pclass) & (df.Sex==Sex)].shape[0]))

def format_data(df):
    result = []
    for row in df.itertuples():
        #print(row.Pclass)
        result.append(dict(Surv= row.Surv, Class=row.Pclass , Sex=row.Sex, Fare=row.Fare ))
    return result

def calc_BNprob(df_test):
    
    result = pd.Series()
    
    for row in df_test.itertuples():
        tablecpd=TableCPDFactorization(bn)
        prob_surv = tablecpd.specificquery(dict(Surv='1'), dict(Fare=str(row.Fare) , Sex=str(row.Sex) , Class=str(row.Pclass) ))

        if prob_surv >= 0.5:
            surv_class = 1
        else:
            surv_class  = 0        
        result = result.append(pd.Series([surv_class]), ignore_index = True )
    return result

def calc_accuracy(dff_train, dff_train_target, nb_iterations):
    
    result = np.zeros(nb_iterations)

    for itera in range(nb_iterations):
        XX_train, XX_test, yy_train, yy_test = train_test_split(dff_train, dff_train_target, test_size=0.33)
        data4bn = format_data(XX_train)
        learner = PGMLearner()
        # estimate parameters
        result_bn = learner.discrete_mle_estimateparams(skel, data4bn)
        #result_bn.Vdata
        result_predict = calc_BNprob(XX_test)
        BN_test_probs = pd.DataFrame()
        BN_test_probs['ground_truth'] = yy_test
        Test_prob = pd.concat([yy_test.reset_index().Surv, result_predict],  axis = 1, ignore_index = True)                    .rename(columns = {0:'ground_truth' , 1:'class_resu'})
        accuracy = Test_prob[Test_prob.ground_truth == Test_prob.class_resu].shape[0]/(1.0*Test_prob.shape[0])
        #print("Accuracy is {}").format(accuracy)
        result[itera] = accuracy
        
    return result


# ## Structure and parameters  defined
Now I will manually calculate the CPD's. I'm assuming that the Fare is directly related to the Class, and hence the BN looks like this:

    Fare
      |
    Class   Sex
        \   /
         \ /
        Surv
# ### Calculating the CPDs

# In[ ]:


print ("P(Class|Fare)")
print ("Class= 1, 2 ,3  ; Fare = 0 (cheap):")
print(df_train[(df_train.Fare==0) & (df_train.Pclass==1)].shape[0] /(1.0*df_train[df_train.Fare==0].shape[0]),
df_train[(df_train.Fare==0) & (df_train.Pclass==2)].shape[0] /(1.0*df_train[df_train.Fare==0].shape[0]),
df_train[(df_train.Fare==0) & (df_train.Pclass==3)].shape[0]/(1.0*df_train[df_train.Fare==0].shape[0]))
print ("Class= 1, 2 ,3  ; Fare = 1 (expensive):")
print(df_train[(df_train.Fare==1) & (df_train.Pclass==1)].shape[0] /(1.0*df_train[df_train.Fare==1].shape[0]),
df_train[(df_train.Fare==1) & (df_train.Pclass==2)].shape[0] /(1.0*df_train[df_train.Fare==1].shape[0]),
df_train[(df_train.Fare==1) & (df_train.Pclass==3)].shape[0]/(1.0*df_train[df_train.Fare==1].shape[0]))

#Sex: Prior probability
print("------------")
print ("P(Sex)")
print ("Sex = 0 (female), 1 (male)")
print (df_train[df_train.Sex==0].shape[0]/float(df_train.Sex.shape[0]) , 
       df_train[df_train.Sex==1].shape[0]/float(df_train.Sex.shape[0]))

# Surv Probability
print("------------")
print("P(Surv|Class,Sex)")
print("Surv = 0 ,1 , Class = 1 , Sex = 0")
print(get_probs_surv_cond(df_train, df_train_target, 0, 1, 0),
get_probs_surv_cond(df_train, df_train_target, 1, 1, 0))
print("Surv = 0 ,1 , Class = 2 , Sex = 0")
print(get_probs_surv_cond(df_train, df_train_target, 0, 2, 0),
get_probs_surv_cond(df_train, df_train_target, 1, 2, 0))
print("Surv = 0 ,1 , Class = 3 , Sex = 0")
print(get_probs_surv_cond(df_train, df_train_target, 0, 3, 0),
get_probs_surv_cond(df_train, df_train_target, 1, 3, 0))
print("Surv = 0 ,1 , Class = 1 , Sex = 1")
print(get_probs_surv_cond(df_train, df_train_target, 0, 1, 1),
get_probs_surv_cond(df_train, df_train_target, 1, 1, 1))
print("Surv = 0 ,1 , Class = 2 , Sex = 1")
print(get_probs_surv_cond(df_train, df_train_target, 0, 2, 1),
get_probs_surv_cond(df_train, df_train_target, 1, 2, 1))
print("Surv = 0 ,1 , Class = 3 , Sex = 1")
print(get_probs_surv_cond(df_train, df_train_target, 0, 3, 1),
get_probs_surv_cond(df_train, df_train_target, 1, 3, 1))


# All these distributions (tables) are then written into a json file with the following format:
{
  "V": [
    "Surv",
    "Class",
    "Sex",
    "Fare"
  ],
  "E": [
    [
      "Fare",
      "Class"
    ],
    [
      "Class",
      "Surv"
    ],
    [
      "Sex",
      "Surv"
    ]
  ],
  "Vdata": {
    "Surv": {
      "ord": 3,
      "numoutcomes": 2,
      "vals": [
        "0",
        "1"
      ],
      "parents": [
        "Class",
        "Sex"
      ],
      "children": ,
           ,
        "['2' , '0']": [
          0.079,
          0.921
        ],
        "['3' , '0']": [
          0.5,
          0.5
        ],
        "['1' , '1']": [
          0.631,
          0.368
        ],
        "['2' , '1']": [
          0.842,
          0.157
        ],
        "['3' , '1']": [
          0.864,
          0.135
        ]
      }
    },
    "Class": {
      "ord": 1,
      "numoutcomes": 3,
      "vals": [
        "1",
        "2",
        "3"
      ],
      "parents": [
        "Fare"
      ],
      "children": [
        "Surv"
      ],
      "cprob": {
        "['0']": [
          0.002,
          0.2,
          0.797
        ],
        "['1']": [
          0.485,
          0.205,
          0.31
        ]
      }
    },
    "Sex": {
      "ord": 2,
      "numoutcomes": 2,
      "vals": [
        "0",
        "1"
      ],
      "parents": ,
      "children": [
        "Surv"
      ],
      "cprob": [
        0.352,
        0.647
      ]
    },
    "Fare": {
      "ord": 0,
      "numoutcomes": 2,
      "vals": [
        "0",
        "1"
      ],
      "parents": None,
      "children": [
        "Class"
      ],
      "cprob": [
        0.505,
        0.49
      ]
    }
  }
}
# In this file we define the probabilities at each node. "cprob" contains a dictionary if the node has at least one parent node. In this case, the keys of the dictionary are the values assigned to the parent nodes, whilst the values correspond to the probabilities of the nodes. 
#     For example, in the Surv node, we find
#     
#     "cprob": {
#         "['1' , '0']": [
#           0.032,
#           0.968
#         ], (...)
#         
# This means that the survival probability given Class=1 and Sex = 0 is 0.968; the prob of not survival given the same conditions is 0.032.       

# I now create a bayesian network in order to run queries on it, given 
# some evidence. In this case, we're not learning any parameters, 
# we've calculated them previously and we use them to define the net.

# In[ ]:


nd       = NodeData()
skel     = GraphSkeleton()
jsonpath_skel ="titanic_skel.json"
jsonpath_node ="titanic_nodes.json"
nd.load(jsonpath_node)
skel.load(jsonpath_skel)

# load bayesian network
bn       = DiscreteBayesianNetwork(skel, nd)

print (skel.getchildren("Class"),skel.getchildren("Sex"),skel.getchildren("Fare"),skel.getchildren("Surv"))
([u'Surv'], [u'Surv'], [u'Class'], [])
# In[ ]:


# We can now start querying our network. We provide a query (first dictionary in the arguments)
# and an evidence (second dictionary in the args))

tablecpd=TableCPDFactorization(bn)
print ("P(Surv=0) = {}".format(tablecpd.specificquery(dict(Surv='0'),dict())))


# In[ ]:


tablecpd=TableCPDFactorization(bn)
print("P(Surv = 1) = {}".format(tablecpd.specificquery(dict(Surv='1'),dict())))

tablecpd=TableCPDFactorization(bn)
print("P(Surv = 1 | Fare = 0) = {}".format(tablecpd.specificquery(dict(Surv='1'),dict(Fare='0'))))
tablecpd=TableCPDFactorization(bn)
print("P(Surv = 1 | Fare = 1) = {}".format(tablecpd.specificquery(dict(Surv='1'),dict(Fare='1'))))
tablecpd=TableCPDFactorization(bn)
print("P(Surv = 1 | Fare = 1, Sex = 0) = {}".format(tablecpd.specificquery(dict(Surv='1'),dict(Fare='1' , Sex='0'))))
tablecpd=TableCPDFactorization(bn)
print("P(Surv = 1 | Fare = 1, Sex = 1, Class=3) = {}".format(tablecpd.specificquery(dict(Surv='1'),dict(Fare='1' , Sex='1' , Class='3'))))
tablecpd=TableCPDFactorization(bn)
print("P(Surv = 1 | Fare = 1, Sex = 1) = {}".format(tablecpd.specificquery(dict(Surv='1'),dict(Fare='1' , Sex='1'))))


# ## Learning Parameters
# 
# Our aim now is to calculate the parameters of the network. We provide the structure of the network 
# and then let the algorithm learn the parameters.
# 

# In[ ]:


from libpgm.pgmlearner import PGMLearner


# In[ ]:


training_data = format_data(df_train)

data has the following format:
[{'Class': 3, 'Fare': 0, 'Sex': 1, 'Surv': 0},
 {'Class': 1, 'Fare': 1, 'Sex': 0, 'Surv': 1},
 {'Class': 3, 'Fare': 0, 'Sex': 0, 'Surv': 1},
 {'Class': 1, 'Fare': 1, 'Sex': 0, 'Surv': 1},...]
# In[ ]:


nd       = NodeData()
skel     = GraphSkeleton()

#The structure is defined in the file titanic_skel
jsonpath ="titanic_skel.json"
skel.load(jsonpath)

#instatiate the learner
learner = PGMLearner()

# The methos estimates the parameters for a discrete Bayesian network with
# a structure given by graphskeleton in order to maximize the probability 
# of data given by data
result_params = learner.discrete_mle_estimateparams(skel, training_data)

result_params.Vdata['Class']# to inspect the network


# Check the prediction accuracy

# In[ ]:


#results = calc_accuracy(dff_train, dff_train_target, 100)
#plt.hist(results, bins='auto')
calc_accuracy(df_train, df_train_target, 1)


# ## Learning the structure

# In[ ]:


#instatiate learner
learner_struc = PGMLearner()

#load data and tranform it to a list of dictionaries
data = format_data(df_train)

# This method learns a Bayesian network structure from discrete data given
# by data, using constraint-based approaches. The function calls discrete_condind 
# (voir ci-dessous) to determine the dependencies between variables.
# Possible params are:
# * pvalparam is te value of the p-value used to determine whether two variables 
# are conditionally indep.(This is obviously necessary to find the net structure).
# * indegree = is used to determine the size of the set of variables used to find dependencies
# (basically the "witness" variables, this will determine the size of the array passed in the
# third argument of the discrete_condind call). 

result_structure = learner_struc.discrete_constraint_estimatestruct(data, indegree=1,pvalparam=0.05)

# The result if always the same for any value of indegree
# result is stable for smaller values of 0.05

#The resulting structure is the identical
result_structure.getchildren('Fare'), result_structure.getchildren('Class')
result_structure.E


# In[ ]:


# We can thus use the skeleton defined before in jsonpath_skel to learn params
# 

skel     = GraphSkeleton()
skel.load(jsonpath_skel)
bn_params2 = learner_struc.discrete_mle_estimateparams(skel, data)

#use result_params2.Vdata to inspect the network
# By looking at result_params2.Vdata we'll notice that the probabilities correpond 
# to the probabilities we calculated (manually) in the beginning.  


# ## Learning both the structure and the parameters

# In[ ]:


#instatiate the learner
learner_full = PGMLearner()

# Learn structure and parameters. This method fully learns a BN from
# discrete data given by data. This function combines the 
# discrete_constraint_estimatestruct method (where it passes in the 
# pvalparam and indegree arguments) with the discrete_mle_estimateparams method.
# It returns a complete DiscreteBayesianNetwork class instance learned from the data
result_full_bn = learner_full.discrete_estimatebn(training_data)

#result_full_bn.E


# In[ ]:


# We can also manually test and verify how independent two varaibles are


# In[ ]:


learner_indep = PGMLearner()
learner_indep.discrete_condind(training_data,'Surv', 'Fare', ['Class'])
# In this case, the result is chi, pval et variable U

learner_indep = PGMLearner()
print("Chi, pval, U: {}{}".format(learner_indep.discrete_condind(training_data,'Surv', 'Fare', ['Class']),
      "(Ho can't be rejected since Surv and Fare are cond independent)"))
print("Chi, pval, U: {}{}".format(learner_indep.discrete_condind(training_data,'Surv', 'Class', ['Fare']),
                               "(Ho is rejected: Surv and Class are not indep)"))
print("Chi, pval, U: {}{}".format(learner_indep.discrete_condind(training_data,'Sex', 'Class', ['Surv']),
                               "(Ho is rejected: Sex and Class are not indep)"))
print("Chi, pval, U: {}".format(learner_indep.discrete_condind(training_data,'Fare', 'Class', ['Sex'])))
print("Chi, pval, U: {}".format(learner_indep.discrete_condind(training_data,'Fare', 'Sex', ['Sex'])))

chi – The result of the chi-squared test on the data (compare the actual and the expected distri of X and Y given U). The expected distribution is P(X,Y,U)=P(U)P(X|U)P(Y|U). The Chi squared is a measure of the deviance between two distributions.  
    
pval – The p-value of the test, meaning the probability of attaining a chi-square result as extreme as or more extreme than the one found, assuming that the null hypothesis is true. (e.g., a p-value of .05 means that if X and Y were independent given U, the chance of getting a chi-squared result this high or higher are .05). The Null H (independence) is rejected if the p-value is smaller than 0.05.

U – The ‘witness’ of X and Y’s independence. This is the variable that, when it is known, leaves X and Y independent.

Recall: conditional independence means: P(X|Y,Z) = P(X|Z) in this case X and Y are independent given Z. 
# # Random Notes

# 1. It is important to note another advantage of representing the joint distribution on a network: modularity.
# When you add a new variable G, the joint distribution changes entirely. Had we used the
# explicit representation of the joint, we would have had to write down twelve new numbers. In
# the factored representation, we could reuse our local probability models for the variables I and
# S, and specify only the probability model for G — the CPD P (G | I). This property will turn
# out to be invaluable in modeling real-world systems.
# 
# 2. Bayesian networks build on the same intuitions as the naive Bayes model by exploiting con-
# ditional independence properties of the distribution in order to allow a compact and natural
# representation. However, they are not restricted to representing distributions satisfying the
# strong independence assumptions implicit in the naive Bayes model. They allow us the flexibil-
# ity to tailor our representation of the distribution to the independence properties that appear
# reasonable in the current setting.
# 
# 3. The core of the Bayesian network representation is a directed acyclic graph (DAG) G, whose
# nodes are the random variables in our domain and whose edges correspond, intuitively, to direct
# influence of one node on another.This graph G can be viewed in two very different ways:
# • as a data structure that provides the skeleton for representing a joint distribution
# compactly in a factorized way;as a compact representation for a set of conditional independence assumptions about
# a distribution.
# 
# 4. Other librairies are: 
#     * BNFinder: a lib for identification of optimal BN, fast and efficient (cross validations and ROC curves included) 

# I provide below the content of 
# 
# * titanic_nodes.json:
# <pre>
#     <code>
# {
# 	"Vdata": {
# 		"Surv": {
# 			"ord": 3,
# 			"numoutcomes": 2,
# 			"vals": ["0", "1"],
# 			"parents": ["Class", "Sex"],
# 			"children": None,
# 			"cprob": {
# 				"['1' , '0']": [.032, .968],
# 				"['2' , '0']": [.079, .921],
# 				"['3' , '0']": [.5, .5],
# 				"['1' , '1']": [.631, .368],
# 				"['2' , '1']": [.842, .157],
# 				"['3' , '1']": [.864, .135]
# 			}
# 		},
# 		"Class": {
# 			"ord": 1,
# 			"numoutcomes": 3,
# 			"vals": ["1", "2", "3"],
# 			"parents": ["Fare"],
# 			"children": ["Surv"],
# 			"cprob": {
# 				"['0']": [.002, .2, .797],
# 				"['1']": [.485, .205, .31]
# 			}
# 		},
# 		"Sex": {
# 			"ord": 2,
# 			"numoutcomes": 2,
# 			"vals": ["0", "1"],
# 			"parents": None,
# 			"children": ["Surv"],
# 			"cprob": [.352, .647]
# 		},
# 		"Fare": {
# 			"ord": 0,
# 			"numoutcomes": 2,
# 			"vals": ["0", "1"],
# 			"parents": None,
# 			"children": ["Class"],
# 			"cprob": [.505, .49]
# 		}
# 	}
# }
# </code>
# </pre>    
# ********
# 
# * titanic_skel.json
# <pre>
#     <code>
# {
# 	"V": ["Surv", "Class", "Sex", "Fare"],
# 	"E": [
# 		["Fare", "Class"],
# 		["Class", "Surv"],
# 		["Sex", "Surv"]
# 	]
# }
# </code>
# </pre>
# ***********
# 
# There is a **None** value in the titanic_nodes.json (thus not a valid json in principle), but the load function is able to open the file.
# 
# 
# 

# 

# In[ ]:




