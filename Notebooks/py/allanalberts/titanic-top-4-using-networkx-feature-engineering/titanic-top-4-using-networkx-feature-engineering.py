#!/usr/bin/env python
# coding: utf-8

# # Titanic top 4% utilizing NetworkX Feature Engineering
# 
# ### Allan Alberts
# #### September 12, 2018
# 
# *  **1 Introduction**
# *  **2 Load Data**
# *  **3 Separate Passengers into categories for Adult Men and Women/Children**
#      * 3.1 Extract Title from Name field
#      * 3.2 Identify male children
#      * 3.3 Feature PType_AdultM
# *  **4 Assign Passengers to Networks**
#      * 4.1 Create supporting features using Name and Ticket fields
#      * 4.2 Define supporting functions
#      * 4.3 Create Graph object and add Nodes for each passenger
#      * 4.4 Connect passengers using graph Edges
#      * 4.5 Remove unconnected nodes from graph
#      * 4.6 Assign a Network Number to each social group 
# *  **5 Create Predictive Features**
#      * 5.1 Feature Nstatus_F_Ch & Nstatus_AdultM
#      * 5.2 Visually inspect new features
# *  **6 Decision Tree Model** 
#      * 6.1 Split data back into training and test sets
#      * 6.2 Tune and measure model accuracy
#      * 6.3 Predict and submit results

# ###### 1. Introduction 
# This notebook makes use of NetworkX to engineer three new features which, used by themselves in a simple DecisionTree, result in a Kaggle prediction in the top 4% (0.81818). 
# 
# With limited lifeboat seats, the Captain of the Titanic ordered that Women and Children board life boats first. However, these chivalrous actions were not always followed and this model will predict when that behavior occurs. For example, passengers traveling with family members may have made different life or death decisions than those traveling alone. 
# 
# Using the initial hypothesis that women and children will generally survive, while men will perish results in a Kaggle score of 0.76076. To improve upon this score we need to predict when the exception occurs, specifically when women/children die or when men live. We can speculate that women and children may be more likely to die if other women or children that they are traveling with die, and likewise, adult men may be more likely to live if other adult men that they are traveling with live.
# 
# To test this hypothesis, we must first determine which passengers are traveling together. This model utilizes NetworkX to assign passengers together into social networks. Passengers are assigned to the graph as nodes. A four step process then adds edges to the graph to connect the passenger nodes to each other using shared ticket numbers and family names. This results in 53% of the passengers being joined together in connected component subgraphs (passenger groups/networks). 
# 
# Next, we need to determine the known survival rates of women/children and adult men traveling in these networks. Because we don't know everyones survival status, we need to create a new field that is slightly different from the existing Survival field. Passengers that we know lived are assigned a status of +1 and those that died are assigned a status of -1. We assign a status of 0 for passenger where we don't know their fate.  This system allows us to engineer an average known survival rate feature for each passenger network using a value falling somewhere between -1 (all died) and +1 (all lived). This feature is individually calculated for each passenger so that we don't incur data leakage from the current passenger (similar to how Parch and SibSp fields don't include the current passenger in their calculation). 
# 
# That's it! Three features: Ptype_AdultM (Adult Men), Nstatus_F_Ch (known survival status of Women/Children in a network), and Nstatus_AdultM (known survival status of Adult Men in a network). Feel free to fork this Kernel and incorporate these features in your model. I'd love to hear how you have improved upon this prediction. Best of luck!

# In[ ]:


import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import networkx as nx
from itertools import combinations

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Visualize the Decision Tree Classifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

import warnings
warnings.filterwarnings('ignore')


# ## 2. Load Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test =  pd.read_csv('../input/test.csv')

# join train and test datasets for feature engineering:
data = pd.concat([train, test])

# save the length of original training dataset to use later for splitting back into training and test datasets
train_len = len(train)

# Check for missing values
print('Missing Values:')
data = data.fillna(np.nan)
data.isnull().sum()


# ## 3. Separate Passengers into categories for Adult Men & Women/Children

# ### 3.1 Extract Title from the Name field

# In[ ]:


data['Title'] = data.Name.str.extract(' ([A-Z][a-z]+)\.', expand=False)


# ### 3.2 Identify Male Children
# Since there is no finite definition of the age at which a child becomes an adult, I will assume that parents consider their son an adult when they give them the title of 'Mr'. In terms of passenger behavior within a family network, a family perceiving a young man as an adult will be more relevant than their precise age. The plot below shows that the known survival of young adults also appears consistent with this definition with almost all young adult males with the title of 'Mr' not surviving.
#     
# Therefore, I will categorize Adult Males as those that do not have 'Master' as their title and all other passengers will be
#     classified as Females/Children.

# In[ ]:


sns.swarmplot(y='Age', x='Title', hue='Survived', data=data[(data.Sex=='male')&(data.Age<17)]);
plt.title('Survival of Young Men');
plt.xlabel('Title (from Name field)');
plt.legend(['Survived', 'Died'],loc='lower right');
sns.despine()


# ### 3.3 Create Features Ptype_AdultM (Adult Male) & Ptype_F_Ch (Female & Childern)

# In[ ]:


data['Ptype_AdultM'] = np.where((data.Sex=='male') & (data.Title!='Master'), 1, 0)
data['Ptype_F_Ch'] = np.where(data.Ptype_AdultM==0, 1, 0)


# ## 4. Assign Passengers to Networks

# ### 4.1. Create Supporting Features 
# These features will assist with assigning passengers to social groups/networks. 

# #### 4.1.1 LastName - if hyphenated, picks up later part of LastName

# In[ ]:


data['LastName'] = data.Name.str.extract('([A-Za-z]+),', expand=False)


# #### 4.1.2 HyphenName -  picks up the first part of a hyphenated LastName

# In[ ]:


data['HyphenName'] = data.Name.str.extract('([A-Za-z]+)-', expand=False)


# #### 4.1.3 MaidenName1 -  The majority of maiden names are within parenthesis at the end of the name string

# In[ ]:


data['MaidenName1'] = np.where(data.Title=='Mrs', data.Name.str.extract('([A-Za-z]+)\)', expand=False), np.NaN)


# #### 4.1.4 MaidenName2 - A few of the maiden names are at the end of the name string but not enclosed in parenthesis

# In[ ]:


data['MaidenName2'] = np.where(data.Title=='Mrs', data.Name.str.extract('([A-Za-z]+)$', expand=False), np.NaN)


# ##### 4.1.5  TicketNum - The numeric portion of a ticket
# This will be used to support sequential comparison of different tickets to identify ones purchased near each other.

# In[ ]:


# process tickets that end in a numeric sequence:
data['TicketNum'] = data.Ticket.str.extract('\s([0-9]+)', expand=False)

# process tickets that do not end in a numeric sequence:
data['TicketNum'] = np.where(pd.isnull(data.TicketNum), data.Ticket.str.extract('([0-9]+)', expand=False), data.TicketNum)

# assign zero where the ticket does not contain numbers:
data['TicketNum'] = data.TicketNum.fillna(0).astype(int)


# ### 4.2 Define Supporting functions

# In[ ]:


def family_names(Passenger):
    '''Returns a list of potential family names for a given passenger'''
    NameList = []
    NameList.append(G.node[Passenger]['LastName'])
    if pd.notnull(G.node[Passenger]['HyphenName']):
        NameList.append(G.node[Passenger]['HyphenName'])
    if pd.notnull(G.node[Passenger]['MaidenName1']):
        NameList.append(G.node[Passenger]['MaidenName1'])
    elif pd.notnull(G.node[Passenger]['MaidenName2']):
            NameList.append(G.node[Passenger]['MaidenName2'])
    return NameList


# In[ ]:


def name_match(Passenger1, Passenger2):
    '''Compares two passengers and returns TRUE if they share
       the same class and family name'''
    NameList_p1 = family_names(Passenger1)
    NameList_p2 = family_names(Passenger2)
    Pclass_p1 = G.node[Passenger1]['Pclass']
    Pclass_p2 = G.node[Passenger2]['Pclass']
    if Pclass_p1==Pclass_p2:
        return bool(set(NameList_p1).intersection(set(NameList_p2)))
    else:
        return False


# In[ ]:


def similar_ticket(Passenger1, Passenger2, TicketProximity):
    '''Compares two passenger tickets and returns TRUE if they
       are sequentially within the value defined by TicketProximity'''
    TicketNum_p1 = G.node[Passenger1]['TicketNum']
    TicketNum_p2 = G.node[Passenger2]['TicketNum']
    SequentialLimit = abs(TicketNum_p1-TicketNum_p2)
    return bool((SequentialLimit!=0)&(SequentialLimit<=TicketProximity))


# In[ ]:


def find_match(PassengerGroup, TargetGroup, TicketProximity, NameCheck):
    '''Finds passengers in PassengerGroup list that have a matching family member in the TargetGroup list.
       NameCheck=True indicates a family name comparison should be used and TicketProximity indicates how close the ticket
       number sequence must be for a passenger match. TicketProximity=0 indicates no ticket comparison should be used.
       Returns a list of passengers that still need to be matched to additional passengers.'''
    NodeList = []
    Append = False
    for Passenger in PassengerGroup:   
        try:  #remove passenger if in target group
            TargetGroup.remove(Passenger)
        except ValueError:
            pass  # do nothing!

        for Passenger2 in TargetGroup:
            if NameCheck:
                if name_match(Passenger,Passenger2):
                    Append = True
                    if TicketProximity > 0:
                        if not similar_ticket(Passenger, Passenger2, TicketProximity):
                            Append = False
            else:
                if TicketProximity > 0:
                    if similar_ticket(Passenger, Passenger2, TicketProximity):
                        Append = True

            if Append:
                NodeList.append(Passenger)
                NodeList.append(Passenger2)
                Append = False
        
    # make a list of tuples by creating an interator and zipping it with itself
    it = iter(NodeList) 
    EdgeList = list(zip(it, it))
    
    if pd.notnull(EdgeList).any():  # add edges to graph for a passenger match
        for u, v in EdgeList:
            G.add_edge(u,v)
    
    # Print updated Networked stats
    PassengersInNetworks = [n for n in G if G.degree(n)>0]
    print('Passengers in Networks: {}'.format(len(PassengersInNetworks)))
    print('Percentage in Networks: {:.0%}'.format(len(PassengersInNetworks)/len(G.nodes)))

    # Find passengers whose family size excedes the size of their network (still need to find edges to family members)
    PassengerNetTooSmall = [n for n, d in G.nodes(data=True) if (G.degree(n)<(d['Parch']+d['SibSp']))]
    print('\nPassengers in a network that is still smaller than their family size: {}'.format(len(PassengerNetTooSmall)))
    return PassengerNetTooSmall


# ###     4.3 Create Graph object and add Nodes for each passenger

# In[ ]:


G = nx.Graph()
all_nodes = data.PassengerId.values

# set the index equal to the node key(PassengerId)
data.set_index('PassengerId', inplace=True)

# define individual node attributes and add node to graph
for n in all_nodes:
    G.add_node(n,                TicketNum=data.loc[n].TicketNum,                Pclass=data.loc[n].Pclass,                Parch=data.loc[n].Parch,                SibSp=data.loc[n].SibSp,                LastName=data.loc[n].LastName,                MaidenName1=data.loc[n].MaidenName1,                MaidenName2=data.loc[n].MaidenName2,                HyphenName=data.loc[n].HyphenName)
data.reset_index(inplace=True) 


# ###     4.4 Connect passengers using graph edges

# In[ ]:


# Step1: Assignment based solely on shared ticket numbers

TicketList = list(data.Ticket.unique())
for T in TicketList:
    data2 = list(data[data.Ticket==T].PassengerId)
    for u, v in combinations(data2, 2):
        G.add_edge(u,v)    
        
print('Total Passengers: {}'.format(len(G.nodes)))
PassengersInNetworks = [n for n in G if G.degree(n)>0]
print('Passengers in Networks: {}'.format(len(PassengersInNetworks)))
print('Percentage in Networks: {:.0%}'.format(len(PassengersInNetworks)/len(G.nodes)))

# Find passengers whose family size excedes the size of their network (still need to find edges to family members)
PassengerNetTooSmall = [n for n, d in G.nodes(data=True) if (G.degree(n)<(d['Parch']+d['SibSp']))]
print('\nPassengers in a network that is still smaller than their family size: {}'.format(len(PassengerNetTooSmall)))


# We see that after assigning passengers to networks based on shared ticket numbers, there are 84 passengers that appear to be in a network that does not yet include all of their family members (Parch + SibSp).
# 
# Inspecting the data shows that some passengers share a family name and have ticket numbers that are in close proxiity to each other. These passengers are likely traveling together with extended family or are adult siblings, but they have purchase their tickets seperately. This scenario would occur if these passengers were in a ticket purchase line together. These passengers will be assigned to the same network in steps 2-4.

# In[ ]:


# Step2: check for matching family name AND ticket sequentially within 10 tickets of each other 

PassengerGroup = list(set(G.nodes()).copy())
TargetGroup = PassengerGroup.copy()
TicketProximity = 10
PassengerNetTooSmall = find_match(PassengerGroup, TargetGroup, TicketProximity, NameCheck=True)


# In[ ]:


# Step3: check for matching family name (irregardless of ticket sequence) for Passengers needing Family Network assginments.

PassengerGroup = PassengerNetTooSmall.copy()
TargetGroup = list(set(G.nodes()).copy() )
TicketProximity = 0
PassengerNetTooSmall = find_match(PassengerGroup, TargetGroup, TicketProximity, NameCheck=True)


# In[ ]:


# Step4: check for sequentially issued tickets that each have corresponding missing family members (name mispelling issues)
PassengerGroup = PassengerNetTooSmall.copy()
TargetGroup = PassengerGroup.copy()
TicketProximity = 1
PassengerNetTooSmall = find_match(PassengerGroup, TargetGroup, TicketProximity, NameCheck=False)
print(PassengerNetTooSmall)


# ###     4.5 Remove passengers traveling by themselves from the graph
# 47% of the passengers are not connected to other nodes including three unconnected passengers that have family members (signified by either Parch>0 or SibSp>0). The inability to connect these three passengers is likely due to incomplete data in the Parch and SibSp fields. Both of these fields are numeric so missing data may be represented by a zero value. For example, this appears to have occurred in the SibSp field for the husband and wife traveling on ticket 364498. So a word of caution when using these fields in your own predictive models. 

# In[ ]:


print('Married Couple with missing values in SibSp field:')
data[data.Ticket=='364498'][['Ticket','Name','Age','Parch','SibSp']]


# In[ ]:


# Remove unconnected passenger nodes from graph

nodelist = []
for n in G.nodes:
    if (G.degree(n)>0):
        nodelist.append(n)
G.remove_nodes_from([n for n in G if n not in nodelist])


# ###     4.6 Create new features, NetworkNum and NetSize, for passengers that are traveling in groups.

# In[ ]:


# create a dictionary of networks with the key being the new Network number and the value being a list with the passengers
# as the first element and the size of the Network as the second element.
network_members = dict()
n = 1
for g in list(nx.connected_component_subgraphs(G)):
    network_members[n] = [g.nodes(), len(g.nodes())]
    n += 1

# initialize all passengers to default for passengers not in traveling in groups (Network=0)
data['NetworkNum'] = 0 
data['NetSize'] = 1

# loop through the dictionary and assign the Network number and size to each Passenger:
data.set_index('PassengerId', inplace=True)
for key, value in network_members.items():
    for p in value[0]:
        data.loc[p, 'NetworkNum'] = key
        data.loc[p, 'NetSize'] = value[1]
data.reset_index(inplace=True) 

# Create feature InNetwork=1 if passenger is traveling with other people.
data['InNetwork'] = np.where(data.NetworkNum==0, 0, 1)


# ### 5. Create Predictive Features
# ####      5.1 Status of other passengers in the network: Nstatus_F_Ch, Nstatus_AdultM 
# 
# I have hypothesised that women and children may die if other women and children that they are traveling with die. Likewise, adult men may live of other adult men that they are traveling with live. To test this, we will create fields to measure the survival rates of other women/children in each passengers network. The same will be done of the survival rates of adult men.
# 
# - The known survival status for an individual passenger is defined as: survived=1, perished=-1, unknown=0
# - The network status for each passenger represents the average survival status of the other passengers they are traveling with in their network.
# - **Nstatus_F_Ch** is the average known survival status of the Women and Children in their network
# - **Nstatus_AdultM** is the average known survival status of men in their network.
# - Data leakage is avoided by not including the passenger in the calculation of their networks's overall survival status.
# - The network status features (Nstatus_F_Ch & Nstatus_AdultM) are defined on a continuum from -1 (all perished) to +1 (all survived). 
# - Non-networked passengers who are traveling alone will have network status values of zero.

# In[ ]:


# Create supporting passenger feature, Status, with values: -1=status dead, 0=unknown status, 1=status survived
data['Status'] = data.Survived.fillna(0)  # unknown survival status=0
data['Status'] = np.where(data.Survived==0, -1, data.Status)


# In[ ]:


# For each passenger that is in a Network (NetworkNum!=0), update Network average status for the passengers
# that are of this type, exclusive of the current passenger.

def update_passenger_network_status(data, Type):
    NstatusField = 'Nstatus' + Type
    PtypeField = 'Ptype' + Type
    
    data[NstatusField] = 0
    data.set_index('PassengerId', inplace=True)
    
    # step through each passenger network
    for n in data[data.NetworkNum!=0].NetworkNum.unique():

        # calculate the Network totals for this passenger type inclusive of the all qaulified passengers
        NetworkSum = data[(data.NetworkNum==n)&(data[PtypeField]==1)].Status.sum()
        NetworkSize = data[data.NetworkNum==n][PtypeField].sum()

        # step though each passenger in the network and update their network status
        for index, row in data[(data.NetworkNum==n)].iterrows():
            PNetworkSum = NetworkSum
            PNetworkSize = NetworkSize 
            
            # If passenger type = Type then reduce PNetworkSum and PNetworkSize by one. This step avoids data 
            # leakage from current passenger similar to how Parch and SibSp are calcualted in the original dataset. 
            if data.loc[index, PtypeField]==1:
                PNetworkSum = PNetworkSum - row['Status']
                PNetworkSize = PNetworkSize - 1

            if PNetworkSize > 0: 
                data.loc[index,NstatusField] = PNetworkSum / PNetworkSize
    data.reset_index(inplace=True)     


# In[ ]:


TypeList = ['_F_Ch', '_AdultM']
for Type in TypeList:
    update_passenger_network_status(data, Type)


# ### 5.2 Visual inspect the new features
# * The hypothese that Women & Children will die when other Women/Children in their network die (network average status < 0) appears to be extremly accurate and is true 95% (58/61) of the time in the training data as shown by the plot below. 
# * The hypothese that Adult Males will survive when other Adult Males in their network survive appears to be true 69% (9/14) of the time in the training data. 

# In[ ]:


plt.figure(figsize=(12,5))
ax=sns.swarmplot(y='Nstatus_F_Ch', x='Ptype_F_Ch', hue='Survived', data=data[(data.Ptype_F_Ch==1)&(data.InNetwork==1)]);
plt.ylabel('Network Status (Other Women & Children)');
plt.title('Survival of Networked Women & Children');
ax.xaxis.set_visible(False);


# In[ ]:


plt.figure(figsize=(12,5))
ax=sns.swarmplot(y='Nstatus_AdultM', x='Ptype_AdultM', hue='Survived', data=data[(data.Ptype_AdultM==1)&(data.InNetwork==1)]);
plt.ylabel('Network Status (Other Adult Men)');
plt.title('Survival of Networked Adult Men');
ax.xaxis.set_visible(False);


# ## 6. Decision Tree model
# ### 6.1 Split data (with new features) back into Training and Test sets

# In[ ]:


# split back into training and test datasets for Kaggle competition:
data.set_index('PassengerId', inplace=True)
data_train = data[:train_len]
data_test = data[train_len:]

model_features = ['Ptype_AdultM','Nstatus_F_Ch','Nstatus_AdultM']
X = data_train[model_features].values
y = data_train.Survived.values


# ### 6.2 Tune hyperparameters and measure model accuracy

# In[ ]:


SEED=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

steps = [('tree', DecisionTreeClassifier())]
parameters = {'tree__max_depth': [2],
              'tree__min_samples_leaf': [3,4,5,6,7,8],
              'tree__criterion': ['gini','entropy']}
pipeline = Pipeline(steps)
model = GridSearchCV(pipeline, param_grid=parameters, cv=10)

model.fit(X_train, y_train)
print('Tuned Model Parameters: {}'.format(model.best_params_))
print("Test Set Accuracy: {:.2%}".format(model.score(X_test, y_test)))


# In[ ]:


# Fit the model
dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, random_state=SEED)
dt = dt.fit(X, y)

# Display Feature Importance
importances = pd.Series(dt.feature_importances_, index=model_features)
sorted_importances = importances.sort_values()
sorted_importances.plot(kind='barh', color='lightgreen', figsize=(6,2));
plt.title('Feature Importance');plt.show()

# Visualize the Decision Tree Classifier
Target_names = ['Died','Survived'] 
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,
                 feature_names=model_features, 
                 class_names=Target_names,
                 filled=True, rounded=True,
                 special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ### 6.3 Predict and Submit results

# In[ ]:


results = data_test[model_features].copy()
results['Survived'] = dt.predict(results).astype(int)
results = results[['Survived']]
results.to_csv('DecisionTreeClassifier_engineered_features.csv')


# If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated 
