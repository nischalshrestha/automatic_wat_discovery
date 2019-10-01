#!/usr/bin/env python
# coding: utf-8

# # Plan
# 1. Load Data
# 1. Data Insights
# 1. Preprocess Data
# 1. Build Models
# 1. Train Models
# 1. Test Models
# 1. Evaluate Models

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization
from sklearn.model_selection import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load Data
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")
gs_raw = pd.read_csv("../input/gender_submission.csv")


# In[ ]:


# Data Insights - Training
train_raw.describe(include="all")


# In[ ]:


print(np.nan_to_num(train_raw['Cabin']))


# In[ ]:


train_raw.dtypes


# # Training Data Insigts
# * $m = 891$
# * $n_x = 10$ explicit features
# 
# Data Types
# 
# Field              |   Datatype 
# ------------------|---------------
# PassengerId |    int64
# Survived       |  int64
# Pclass          | int64
# Name           | object
# Sex              | object
# Age              |float64
# SibSp           | int64
# Parch           | int64
# Ticket          |object
# Fare           |float64
# Cabin          | object
# Embarked    |    object
# 

# In[ ]:



print(train_raw['Pclass'].T)
print(train_raw['Cabin'].T)


# In[ ]:


# Data Insigts (Testing)
test_raw.describe(include="all")


# In[ ]:


# Data Insigts (Gender Submission)
gs_raw.describe(include="all")


# # Preprocess Data
# Find optimal feature mapping function $\mathcal{M} : \phi \mapsto \phi'$ such that $\forall \phi'_i \in \phi', \phi'_i \in \mathbb{R}$, and $\mathbb{E}(\phi'_i) \approx 0$
# 
# with loose constraint of $ \phi'_i \in [-1, 1]$ 
# 
# Of course this will depend on feature data type, moreover, some NaN values when found will be substituted by 0 
# 
# So I propose defining the following mapping function
# 
# ## Pclass
# Since this is an integer value $\phi_{Pclass} = \phi_0 \in  \{1,2,3  \} \in \mathbb{Z}$, this mapping function will be ~~identity mapping ~~ normalized
# i.e. $ \mathcal{M}_0 \colon \phi_0 \mapsto \phi_0 - 2$ 
# 
# ## Name            
# Although passenger name is a rich feature, containing multidimensional data, including ethnicity, sex, and social class, I will bypass it in this iteration of work and have our mapping function map to a 0
# 
# $ \mathcal{M}_1 \colon \phi_1 \mapsto 0$ 
# 
# ## Sex             
# I will arbitrarily assign 1 and -1 to either sex, say $ \mathcal{M}_2  \colon female \mapsto 1$ and $ \mathcal{M}_2  \colon male \mapsto -1$   
# 
# ## Age            
# Age $\phi_{age} = \phi_3 \in \mathbb{R}^{+} \sim \mathcal{N}(30,14)$ 
# 
# hence $ \mathcal{M}_3  \colon \phi_3 \mapsto \frac{\phi_3 - 30}{14}$
# 
# both $\mu$ and $\sigma$ are approximate values
# 
# ## SibSp            
# Similar to Pclass, this feature will ~~have identity mapping~~ be normalized
# 
# i.e. $ \mathcal{M}_4 \colon \phi_4 \mapsto \phi_4 - 3$ 
# ## Parch   
# Similar to Pclass, this feature will ~~have identity mapping~~ be normalized
# 
# i.e. $ \mathcal{M}_5 \colon \phi_5 \mapsto \phi_5 - 2$ 
# ## Ticket          
# Similar to name, ticket might hide some useful implicit features, but I will ignore it for simplicity 
# $ \mathcal{M}_6 \colon \phi_6 \mapsto 0$ 
# 
# ## Fare           
# Similar to age, Fare $\phi_{fare} = \phi_7 \in \mathbb{R}^{+} \sim \mathcal{N}(33,52)$ 
# hence $ \mathcal{M}_7  \colon \phi_7 \mapsto \frac{\phi_7 - 33}{52}$
# both $\mu$ and $\sigma$ are approximate values
# 
# ## Cabin           
# Cabin will be defferred, hence $ \mathcal{M}_8 \colon \phi_8 \mapsto 0$ 
# Experiment: I will create a binary feature vector of  8 features, each one corresponds to the presence of one of the following letters in cabin $\{ A, B, C, D, E, F, G, T\}$
# 
# 
# ## Embarked        
# Embarked $\phi_9 \in \{C, Q, S\}$, I will use a simple trinary mapping function $ \mathcal{M}_9 \colon C \mapsto -1$, $ \mathcal{M}_9 \colon Q \mapsto 0$, and  $ \mathcal{M}_9 \colon S \mapsto 1$ 
# 

# In[ ]:


# Mapping function Extracting PassengerID, Survival, Feature vector phi_prime 
def get_features(X):
    # PassengerId
    PassID = X['PassengerId']
    m = PassID.shape[0]
    nx = 20 # Number of features, this is hardcoded for the timebeing
    # Add two more columns for embarked, and one more for sex
    # Add 7 more for cabin
    # Features 
    X_prime = np.zeros((m,nx))
    # 0 Pclass
    X_prime[:,0] = X['Pclass'] - 2
    # 1 Name           
    # 2,3 Sex
    X_prime[:,2] = 1* (X['Sex'] == 'female')
    X_prime[:,3] = 1* (X['Sex'] == 'male')
    
    # 4 Age           
    mu = 30
    sigma = 14 
    X_prime[:,4] = (X['Age'] - mu ) / sigma
    # 5 SibSp          
    X_prime[:,5] = X['SibSp'] - 3
    # 6 Parch     
    X_prime[:,6] = X['Parch'] - 2
    # 7 Ticket         
    # 8 Fare    
    mu = 33
    sigma = 52 
    X_prime[:,8] = (X['Fare'] - mu ) / sigma
    # 9 Cabin   
    X_prime[:,9] = X['Cabin'].str.contains('A')*1
    X_prime[:,10] = X['Cabin'].str.contains('B')*1
    X_prime[:,11] = X['Cabin'].str.contains('C')*1
    X_prime[:,12] = X['Cabin'].str.contains('D')*1
    X_prime[:,13] = X['Cabin'].str.contains('E')*1
    X_prime[:,14] = X['Cabin'].str.contains('F')*1
    X_prime[:,15] = X['Cabin'].str.contains('G')*1
    X_prime[:,16] = X['Cabin'].str.contains('T')*1
    # Extend Cabin to pick all possible letters, and the numeric value 
    # 17,18,19 Embarked  {C,Q,S}
    selector = X['Embarked'] == 'C'
    X_prime[:,17] = selector * 1 

    selector = X['Embarked'] == 'Q'
    X_prime[:,18] = selector * 1 

    selector = X['Embarked'] == 'S'
    X_prime[:,19] = selector *1
    return PassID, np.nan_to_num(X_prime)


# Attempt to create a kernel of features $\kappa(\phi')$ to find deeper insights, one idea is to create a new feature vector $\kappa_r(\phi') = (\phi' \phi'^2 ...\phi'^r) $
# 

# In[ ]:


def kernel(X):
    K_x = np.concatenate((X, np.exp(-X**2)), axis=1)
    return K_x


# ## Neural Network
# it seems that Logistic regression is stuck at 77.99%, despite non-linearity simulated by my kernel function.
# Let me try a new method, using neural network with different activation functions in the hidden layers.
# 
# ### Method
# I will simulate each neuron behaviour as follows
# $$ \nu_\psi(\mathbf{x})  = \psi(\mathbf{\omega}^T \mathbf{x} + b)$$
# where $\mathbf{x}$ is input feature vector, $\psi$ is an activation function, $\mathbf{\omega}$ is weights vector, and $b$ is bias 
# 
# Learning will be done using normal back propagation, since this will be a shallow neural network and will be as follows 
# 
# $$  \mathbf{\omega} \gets \mathbf{\omega} - \alpha \frac{\partial}{\partial \mathbf{\omega}} \mathbf{J}_\psi(\mathbf{x},y;\mathbf{\omega},b) \\
# b \gets b - \alpha \frac{\partial}{\partial b} \mathbf{J}_\psi(\mathbf{x},y;\mathbf{\omega},b) $$
# 
# where $\mathbf{J}_\psi(\mathbf{x},y;\mathbf{\omega},b)$ is an objective function to minimize 
# 
# ### Activation functions
# I will start by three simple functions 
# #### Sigmoid 
# $$ \psi_\sigma(\mathbf{z}) = \frac{1}{1 + e^{-\mathbf{z}}}$$
# $$ \frac{\partial}{\partial \mathbf{z}} \psi_\sigma(\mathbf{z}) = \psi_\sigma(\mathbf{z})(1-\psi_\sigma(\mathbf{z}))$$
# $$ \frac{\partial}{\partial \mathbf{\omega}} \psi_\sigma(\mathbf{z}) = \psi_\sigma(\mathbf{z})(1-\psi_\sigma(\mathbf{z})) \mathbf{x}^T \\
# \frac{\partial}{\partial b} \psi_\sigma(\mathbf{z}) = \psi_\sigma(\mathbf{z})(1-\psi_\sigma(\mathbf{z}))$$
# #### Hyperbolic TAN (tanh)
# $$ \psi_{\tanh}(\mathbf{z}) = \frac{2}{1 + e^{-2\mathbf{z}}} - 1 $$
# $$ \frac{\partial}{\partial \mathbf{z}} \psi_{\tanh}(\mathbf{z}) =1-\psi_{\tanh}(\mathbf{z})^2 $$
# $$ \frac{\partial}{\partial \mathbf{w}} \psi_{\tanh}(\mathbf{z}) =1-\psi_{\tanh}(\mathbf{z})^2 \mathbf{x}^T$$
# $$ \frac{\partial}{\partial b} \psi_{\tanh}(\mathbf{z}) =1-\psi_{\tanh}(\mathbf{z})^2 $$
# #### Rectified Linear Unit (ReLU)
# $$ \psi_{ReLU}(\mathbf{z}) = \left\{
#                 \begin{array}{ll}
#                   \mathbf{z}\  |\  \mathbf{z} > 0\\
#                   0\ |\ else
#                 \end{array}
#               \right. $$
# $$ \frac{\partial}{\partial \mathbf{z}} \psi_{ReLU}(\mathbf{z}) = \left\{
#                 \begin{array}{ll}
#                   1  |\  \mathbf{z} > 0\\
#                   0\ |\ else
#                 \end{array}
#               \right. $$
#   $$ \frac{\partial}{\partial \mathbf{\omega}} \psi_{ReLU}(\mathbf{z}) = \left\{
#                 \begin{array}{ll}
#                   \mathbf{x}  |\  \mathbf{z} > 0\\
#                   0\ |\ else
#                 \end{array}
#               \right. $$
# $$ \frac{\partial}{\partial b} \psi_{ReLU}(\mathbf{z}) = \left\{
#                 \begin{array}{ll}
#                   1  |\  \mathbf{z} > 0\\
#                   0\ |\ else
#                 \end{array}
#               \right. $$
# 

# In[ ]:


# Neuron 
class Neuron:
    def __init__(self, feature_size, alpha, act=2):
        self.w = np.random.random_sample((feature_size,1))*0.1
        self.b = 0.0
        self.alpha = alpha
        self.activation_function = act
    def activate(self, z):
        # based on set activation function set make the calculation
        if self.activation_function == 2: # Sigmoid
            return 1 / (1 + np.exp(-z))
        if self.activation_function == 3: # tanh
            return 2 / (1 + np.exp(-z)) - 1
        if self.activation_function == 7: # ReLU
            return (z>0)*z
        if self.activation_function == 21: # Gaussian
            return np.exp(-z**2)
    def gradient(self, y_hat, err):
        
        m = y_hat.shape[0] # Sample size
        #da = y/y_hat + (1-y)/(1-y_hat)
        # based on set activation function set make the calculation
        if self.activation_function == 2: # Sigmoid
            return  err*y_hat*(1-y_hat)
        if self.activation_function == 3: # tanh
            return err*(1-y_hat**2)
        if self.activation_function == 7: # ReLU
            return (y_hat>0)*err
        if self.activation_function == 21: # Gaussian
            return -2*y_hat*np.exp(-y_hat**2)*err

    def forward(self, X):
        # given feature vector X, find activation response
        z = np.dot(X,self.w)+self.b
        a = self.activate(z)
        return a
    def backward(self, X, err):
        m = y.shape[0]
        y_hat = self.forward(X)
        grad = self.gradient(y_hat,err)
        self.b -= self.alpha * np.sum(grad)/m
        self.w -= self.alpha * np.dot(X.T,grad)/m
    def print_params(self):
        print("b = {}, w = {}".format(self.b,self.omega.T))


# ## Cross Validation 
# I need to create some cross validation generation function capable of pulling random samples from my training data, train on n-1 samples and test on 1
# 
# I will start first by pure randomization, then by applying t-test on sample data to avoid bias

# In[ ]:


# TODO: Cross Validation Generator, Expects input features and labels, and returns equally sized samples 
def cross_validated(X, n_samples):
    kf = KFold(n_samples, shuffle = True)
    result = [group for group in kf.split(X)]
    
    return result


# # Performance Measure
# I think I was doing something wrong measuring performance not taking into account false positives and false negatives
# i will use instead mean SSD (Sum of Squared Difference)
# 
# $$ SSD(\hat{y},y^{*}) = \frac{1}{m} \sum_i^m ( {y^{*}}^{(i)} - \hat{y}^{(i)} )^2 $$ 

# In[ ]:


def ssd(y_hat, y):
    m = y.shape[0]
    return np.dot((y_hat-y).T,y_hat-y)/m


# In[ ]:



PassID, X = get_features(train_raw)
X = X
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 3000
instances = 5

folds = 3
oinst = folds
cv_groups = cross_validated(X, folds)

n_x = X.shape[1] # Features
alpha = 0.05
alph = np.ones(oinst)*alpha + (np.random.random_sample(oinst)-0.5)*0.0001
# Setup single neurons
N_g = [Neuron(n_x,alpha,21) for j in range(oinst)]
final_cost = []

for j in range(oinst):
    cost_g = []
    N_g[j].alpha = alph[j]
    
    # Prepare Training and testing sets 
    
    X_train = X[cv_groups[j][1],:] 
    y_train = np.reshape(y[cv_groups[j][1],0],(-1,1)) 
    X_test = X[cv_groups[j][0],:] 
    y_test = np.reshape(y[cv_groups[j][0],0],(-1,1))
    
    for i in range(iterations):
        # Evaluate
        a_g = N_g[j].forward(X_train)
        
        # Learn
        N_g[j].backward(X_train,a_g - y_train)
        
        # Evaluate
        # a_g = N_g[j].forward(X_test)
        
        # Performance vote
        # cost_g.append(np.sum(np.abs(a_g-y_test))/y.shape[0])
    
    a_g = (N_g[j].forward(X_test)>0.5)*1
    m = y_test.shape[0]
    
    final_cost.append(1-ssd(a_g,y_test))

    print("Testing:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y_test)))

    #plt.plot(cost_g)
    #plt.title("Cost over time")
    #plt.show()

    #plt.scatter(a_g, y)
    #plt.show()
final_cost = np.reshape((final_cost),(-1,1)).tolist()     

plt.plot(final_cost)    
plt.show()



# In[ ]:


PassID, X = get_features(test_raw)
X2 = X

best_of_breed = final_cost.index(max(final_cost))
a_gm = N_g[best_of_breed].forward(X2)
    
y_hat = np.reshape(a_gm,(-1,1))

print(y_hat.shape)
data = y_hat > 0.5
data = data*1
s0 = pd.Series(PassID, index=PassID)
s1 = pd.Series(data[:,0], index=PassID)

df = pd.DataFrame(data = s1,index = PassID)
df.columns = ['Survived']
df.to_csv('best_of_breed_gauss_nn_3.csv', sep=',')


# # Try it with 2 layered NN

# In[ ]:



PassID, X = get_features(train_raw)
X = X
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 5000
instances = 5

folds = 3
oinst = folds
cv_groups = cross_validated(X, folds)

n_x = X.shape[1] # Features
alpha = 0.25
alph = np.ones(oinst)*alpha + (np.random.random_sample(oinst)-0.5)*0.0001
# Setup Hidden Layer
N_21 = [Neuron(n_x,alpha,3) for j in range(oinst)]
N_22 = [Neuron(n_x,alpha,3) for j in range(oinst)]
# Setup output layer
N_g = [Neuron(2,alpha,2) for j in range(oinst)]
final_gain = []
all_gain = []
for j in range(oinst):
    cost_g = []
    N_g[j].alpha = alph[j]
    N_21[j].alpha = alph[j]
    N_22[j].alpha = alph[j]
    
    # Prepare Training and testing sets 
    X_train = X[cv_groups[j][1],:] 
    y_train = np.reshape(y[cv_groups[j][1],0],(-1,1)) 
    X_test = X[cv_groups[j][0],:] 
    y_test = np.reshape(y[cv_groups[j][0],0],(-1,1))
    
    for i in range(iterations):
        # Evaluate
        a_21 = N_21[j].forward(X_train) 
        a_22 = N_22[j].forward(X_train)  
        a_2 = np.concatenate((a_21,a_22),axis=1)
        a_g = N_g[j].forward(a_2)
        
        # Learn
        grad = N_g[j].gradient(a_g,a_g - y_train)
        N_g[j].backward(a_2,a_g - y_train)
        
        N_21[j].backward(X_train,grad*N_g[j].w[0])
        N_22[j].backward(X_train,grad*N_g[j].w[1])
    
    
    # Finally Evaluate Model 
    a_21 = N_21[j].forward(X_test) 
    a_22 = N_22[j].forward(X_test)  
    a_2 = np.concatenate((a_21,a_22),axis=1)
    a_g = N_g[j].forward(a_2)
    #plt.scatter(a_g, y_test)
    #plt.show()
    a_g = (a_g>0.5)*1
    m = y_test.shape[0]
    final_gain.append(1-ssd(a_g,y_test))

    print("Testing:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y_test)))

    # Evaluate All
    a_21 = N_21[j].forward(X) 
    a_22 = N_22[j].forward(X)  
    a_2 = np.concatenate((a_21,a_22),axis=1)
    a_g = N_g[j].forward(a_2)
    #plt.scatter(a_g, y)
    #plt.show()
    a_g = (a_g>0.5)*1
    m = y.shape[0]
    all_gain.append(1-ssd(a_g,y))
    print("Testing All:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y)))
    #plt.plot(cost_g)
    #plt.title("Cost over time")
    #plt.show()


final_gain= np.reshape((final_gain),(-1,1)).tolist()         
all_gain= np.reshape((all_gain),(-1,1)).tolist()   
plt.plot(final_gain)
plt.plot(all_gain)
plt.show()


# In[ ]:


PassID, X = get_features(test_raw)
X2 = X

best_of_breed = final_gain.index(max(final_gain))
a_21 = N_21[best_of_breed].forward(X2) 
a_22 = N_22[best_of_breed].forward(X2)  
a_2 = np.concatenate((a_21,a_22),axis=1)
a_gm = N_g[best_of_breed].forward(a_2)
    
y_hat = np.reshape(a_gm,(-1,1))

print(y_hat.shape)
data = y_hat > 0.5
data = data*1
s0 = pd.Series(PassID, index=PassID)
s1 = pd.Series(data[:,0], index=PassID)

df = pd.DataFrame(data = s1,index = PassID)
df.columns = ['Survived']
df.to_csv('best_of_2tanh_1sig.csv', sep=',')


# # Search Optimal Learning Parameters 
# * Search for best $\alpha$ at 3000 Epochs
# * Search for best number of iterations at optimal $\alpha$

# In[ ]:



PassID, X = get_features(train_raw)
X = X
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 5000
instances = 4
alph = np.linspace(0.1, 0.40, instances)

folds = 3
oinst = folds
cv_groups = cross_validated(X, folds)

n_x = X.shape[1] # Features
final_gain = []
all_gain = []
alphas = []
for a in range(instances):
    alpha = alph[a]+(np.random.random_sample()-0.5)*0.0001
    # Setup Hidden Layer
    N_21 = [Neuron(n_x,alpha,3) for j in range(oinst)]
    N_22 = [Neuron(n_x,alpha,3) for j in range(oinst)]
    # Setup output layer
    N_g = [Neuron(2,alpha,2) for j in range(oinst)]

    for j in range(oinst):
        alpha = alph[a]+(np.random.random_sample()-0.5)*0.00001
        N_g[j].alpha = alpha
        N_21[j].alpha = alpha
        N_22[j].alpha = alpha
        alphas.append(alpha)
        # Prepare Training and testing sets 
        X_train = X[cv_groups[j][1],:] 
        y_train = np.reshape(y[cv_groups[j][1],0],(-1,1)) 
        X_test = X[cv_groups[j][0],:] 
        y_test = np.reshape(y[cv_groups[j][0],0],(-1,1))
    
        for i in range(iterations):
            # Evaluate
            a_21 = N_21[j].forward(X_train) 
            a_22 = N_22[j].forward(X_train)  
            a_2 = np.concatenate((a_21,a_22),axis=1)
            a_g = N_g[j].forward(a_2)
        
            # Learn
            grad = N_g[j].gradient(a_g,a_g - y_train)
            N_g[j].backward(a_2,a_g - y_train)
        
            N_21[j].backward(X_train,grad*N_g[j].w[0])
            N_22[j].backward(X_train,grad*N_g[j].w[1])
    
    
        # Finally Evaluate Model 
        a_21 = N_21[j].forward(X_test) 
        a_22 = N_22[j].forward(X_test)  
        a_2 = np.concatenate((a_21,a_22),axis=1)
        a_g = N_g[j].forward(a_2)
        print("Testing Sample:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y_test)))
        #plt.scatter(a_g, y_test)
        #plt.show()
        a_g = (a_g>0.5)*1
        m = y_test.shape[0]
        final_gain.append(1-ssd(a_g,y_test))


        # Evaluate All
        a_21 = N_21[j].forward(X) 
        a_22 = N_22[j].forward(X)  
        a_2 = np.concatenate((a_21,a_22),axis=1)
        a_g = N_g[j].forward(a_2)
        #plt.scatter(a_g, y)
        #plt.show()
        a_g = (a_g>0.5)*1
        m = y.shape[0]
        all_gain.append(1-ssd(a_g,y))
        print("Testing All:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y)))
        #plt.plot(cost_g)
        #plt.title("Cost over time")
        #plt.show()


final_gain= np.reshape((final_gain),(-1,1)).tolist()         
all_gain= np.reshape((all_gain),(-1,1)).tolist()   
plt.scatter(alphas,final_gain)
plt.scatter(alphas,all_gain)
plt.show()


# ## Results - Hidden Layer of 1 TANH and 1 ReLU and Output is Gaussian
# 
# ### 7 Fold Expirements 
# For 7 folds, training on one fold and testing on the rest 
# 
# Epochs | optimal $\alpha$ (range) | estimate score (range)
# -----------|----------------------------------|-------------------------------
# 5000     | 0.065 ~ 0.085                  |  0.76 ~ 0.82
# 3000     |  0.175 ~ 0.2                     |  0.75 ~ 0.83
# 1000     |  0.38  ~ 0.45                    |  0.77 ~ 0.83
# 500       |  0.745 ~ 0.765                 |  0.75 ~ 0.82
# 100       |  0.45 ~ 1.0                       |  0.55 ~ 0.8
# 
# 
# ## Results - Hidden Layer of 2 TANH and Output is Gaussian
# 
# ### 3 Fold Expirements 
# For 3 folds, training on one fold and testing on the rest 
# 
# Epochs | optimal $\alpha$ (range) | estimate score (range)
# -----------|----------------------------------|--------------------------
# 1000     |  0.25 ~ 0.325                   |  0.78 ~ 0.81
# 2000     |  0.055 ~ 0.07                   |  0.77 ~ 0.81
# 3000     |  0.04 ~ 0.1                       |  0.78 ~ 0.8
# 5000     |   0.02 ~ 0.04                    |  0.78 ~ 0.8

# In[ ]:




