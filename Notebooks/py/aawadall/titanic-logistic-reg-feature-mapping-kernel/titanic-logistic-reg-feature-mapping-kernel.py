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

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Load Data
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")
gs_raw = pd.read_csv("../input/gender_submission.csv")


# In[3]:


# Data Insights - Training
train_raw.describe(include="all")


# In[4]:


print(np.nan_to_num(train_raw['Cabin']))


# In[5]:


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

# In[6]:



print(train_raw['Pclass'].T)
print(train_raw['Cabin'].T)


# In[7]:


# Data Insigts (Testing)
test_raw.describe(include="all")


# In[8]:


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

# In[9]:


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


# In[10]:


PassID, X = get_features(train_raw)

# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))

print(X)
plt.plot(X)
plt.show()


# In[11]:


#print(np.char.count(np.array(train_raw['Cabin']),'A'))
#print(train_raw['Cabin'].str.contains('A')*1)


# # Build Models
# I propose the following models
# * Logistic Regression 
# * Neural Network
# * Support Vector Machine
# 

# ## Logistic Regresion 
# Using simple sigmoid activation function $\sigma(z) = \frac{1}{1+e^{-z}}$
# 
# where $z = \omega^T X + b$
# 
# Cost function used here will be $\mathcal{L}(\omega,b,x,y) = -y log(\sigma(b+\omega^TX))+(1-y)log(1-\sigma(b+\omega^TX))$
# 
# Gradient descent will conveniently use $\frac{\partial}{\partial \omega} \mathbf{J} = \mathbf{X}^T . (\hat{y} - y)$ and $\frac{\partial}{\partial b} \mathbf{J} = \sum_i^m \hat{y}^{(i)} - y^{(i)}$
# 

# In[12]:


# Initialize Model
def init(X):
    w = np.zeros((X.shape[1],1))
    b = 0.0
    return w, b

# Sigmoid function 
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Predict 
def predict(X, w, b):
    return sigmoid(np.dot(X, w)+b)

# Cost function
def lcost(X, w, b, y):
    m = y.shape[0]
    y_hat = predict(X, w, b)
    return -1/m*np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

# gradients
def gradient(X, y_hat, y):
    
    db = np.sum(y_hat - y)
    dw = np.dot(X.T,y_hat-y)
    
    return db, dw

# Learn 
def learn(X, w, b, y, alpha):
    y_hat = predict(X, w, b)
    db, dw = gradient(X, y_hat, y)
    w -= alpha * dw 
    b -= alpha * db
    return w, b


# In[13]:


# Test Model
alpha = 0.001 # Learning Rate
w, b = init(X)

y_hat = predict(X, w, b)


plt.plot(y_hat)
plt.show()

j = []

for i in range(2000):
    w, b = learn(X, w, b, y, alpha)
    j.append(lcost(X, w, b, y))
    
y_hat = predict(X, w, b)
plt.plot(j)
plt.show()


m = y.shape[0]
measure_p = (y_hat > 0.5) == y
measure_n = (y_hat <= 0.5) != y
print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))


plt.scatter( y_hat, y)
plt.show()



# Attempt to create a kernel of features $\kappa(\phi')$ to find deeper insights, one idea is to create a new feature vector $\kappa_r(\phi') = (\phi' \phi'^2 ...\phi'^r) $
# 

# In[14]:


def kernel(X):
    K_x = np.concatenate((X, X**2,X**3,  np.exp(-X)), axis=1)
    return K_x


# In[15]:


X2 = kernel(X)
# Test Model
alpha = 0.00001 # Learning Rate
w, b = init(X2)

j = []

for i in range(1500):
    w, b = learn(X2, w, b, y, alpha)
    j.append(lcost(X2, w, b, y))
    
y_hat = predict(X2, w, b)
plt.plot(j)
plt.show()

plt.scatter( y_hat, y)
plt.show()

m = y.shape[0]
measure_p = (y_hat > 0.5) == y
measure_n = (y_hat <= 0.5) != y
print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))

# Validate for test set 
PassID, X = get_features(test_raw)
X2 = kernel(X)

# Assuming sorted lists - Survived
y = np.array(gs_raw['Survived'])
y = np.reshape(y,(-1,1))

y_hat = predict(X2, w, b)
m = y.shape[0]
measure_p = (y_hat > 0.5) == y
measure_n = (y_hat <= 0.5) != y
print("Testing: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))


# ~~At this stage no need to test other methods, as it seems LR is sufficient to identify survival data at this rate 96.17%~~
# 
# Score was 77.99%
# maybe I should build a better kernel function to extract more features from selected feature vector
# also, I might work on extracting features from censored features i.e. Name, Ticket, and Cabin
# 

# In[16]:


# Prepare Data for submission
data = y_hat > 0.5
data = data*1
s0 = pd.Series(PassID, index=PassID)
s1 = pd.Series(data[:,0], index=PassID)

df = pd.DataFrame(data = s1,index = PassID)
df.columns = ['Survived']
df.to_csv('lr_submission.csv', sep=',')


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

# In[17]:


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


# ## Test a single neuron with different activation function

# In[18]:


PassID, X = get_features(train_raw)
X2 = kernel(X)
n_x = X2.shape[1] # Features
alpha = 0.0014
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))

instances = 4
# Setup single neurons
N_s = [Neuron(n_x,alpha,2) for j in range(instances)]  # Sigmoid Activation
N_t = [Neuron(n_x,alpha,3) for j in range(instances)]  # tanh Activation
N_r = [Neuron(n_x,alpha,7) for j in range(instances)]  # ReLU Activation
N_g = [Neuron(n_x,alpha,21) for j in range(instances)]  # Gaussian Activation

cost_s = []
cost_t = []
cost_r = []
cost_g = []

for i in range(1000):
    c_s = []
    c_t = []
    c_r = []
    c_g = []
    for j in range(instances):
        # Evaluate 
        a_s = N_s[j].forward(X2)
        a_t = N_t[j].forward(X2)
        a_r = N_r[j].forward(X2)
        a_g = N_g[j].forward(X2)
        
        # Learn
        N_s[j].backward(X2,a_s - y)
        N_t[j].backward(X2,a_t - y)
        N_r[j].backward(X2,a_r - y)
        N_g[j].backward(X2,a_g - y)
        
        # Evaluate
        a_s = N_s[j].forward(X2)
        a_t = N_t[j].forward(X2)
        a_r = N_r[j].forward(X2)
        a_g = N_g[j].forward(X2)
        
        # Performance vote
        c_s.append(np.sum(np.abs(a_s-y))/y.shape[0])
        c_t.append(np.sum(np.abs(a_t-y))/y.shape[0])    
        c_r.append(np.sum(np.abs(a_r-y))/y.shape[0])
        c_g.append(np.sum(np.abs(a_g-y))/y.shape[0])
    
    cost_s.append(np.mean(c_s))
    cost_t.append(np.mean(c_t))    
    cost_r.append(np.mean(c_r))
    cost_g.append(np.mean(c_g))


plt.plot(cost_s)
plt.title("Average Sigmoid Performance")
plt.show()
plt.plot(cost_t)
plt.title("Average tanh Performance")
plt.show()
plt.plot(cost_r)
plt.title("Average ReLU Performance")
plt.show()
plt.plot(cost_g)
plt.title("Average Gaussian Performance")
plt.show()


# ## Test Neurons with simple features
# test each activation type with original feature vector (no kernel)

# In[19]:


PassID, X = get_features(train_raw)
X2 = X
n_x = X2.shape[1] # Features
alpha = 0.008
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))

instances = 4
# Setup single neurons
N_s = [Neuron(n_x,alpha,2) for j in range(instances)]  # Sigmoid Activation
N_t = [Neuron(n_x,alpha,3) for j in range(instances)]  # tanh Activation
N_r = [Neuron(n_x,alpha,7) for j in range(instances)]  # ReLU Activation
N_g = [Neuron(n_x,alpha,21) for j in range(instances)]  # Gaussian Activation

cost_s = []
cost_t = []
cost_r = []
cost_g = []

for i in range(10000):
    c_s = []
    c_t = []
    c_r = []
    c_g = []
    for j in range(instances):
        # Evaluate
        a_s = N_s[j].forward(X2)
        a_t = N_t[j].forward(X2)
        a_r = N_r[j].forward(X2)
        a_g = N_g[j].forward(X2)
        
        # Learn
        N_s[j].backward(X2,a_s - y)
        N_t[j].backward(X2,a_t - y)
        N_r[j].backward(X2,a_r - y)
        N_g[j].backward(X2,a_g - y)
        
        # Evaluate
        a_s = N_s[j].forward(X2)
        a_t = N_t[j].forward(X2)
        a_r = N_r[j].forward(X2)
        a_g = N_g[j].forward(X2)
        
        # Performance vote
        c_s.append(np.sum(np.abs(a_s-y))/y.shape[0])
        c_t.append(np.sum(np.abs(a_t-y))/y.shape[0])    
        c_r.append(np.sum(np.abs(a_r-y))/y.shape[0])
        c_g.append(np.sum(np.abs(a_g-y))/y.shape[0])
    
    cost_s.append(np.mean(c_s))
    cost_t.append(np.mean(c_t))    
    cost_r.append(np.mean(c_r))
    cost_g.append(np.mean(c_g))


plt.plot(cost_s)
plt.title("Average Sigmoid Performance")
plt.show()
plt.plot(cost_t)
plt.title("Average tanh Performance")
plt.show()
plt.plot(cost_r)
plt.title("Average ReLU Performance")
plt.show()
plt.plot(cost_g)
plt.title("Average Gaussian Performance")
plt.show()


# ## Testing a network of neurons 
# Build a Simple Neural Network with one hidden layer of 2 neurons and one output layer of a single neuron
# with one of the hidden neurons with tanh activation function, while the other with ReLU

# In[20]:


PassID, X = get_features(train_raw)
X2 = X
n_x = X2.shape[1] # Features
alpha = 0.02
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 1000
# Setup Layers
N_21 = Neuron(n_x, alpha, 7)  # ReLU Activation
N_22 = Neuron(n_x, alpha, 7)  # ReLU Activation
N_23 = Neuron(n_x, alpha, 7)  # ReLU Activation
N_31 = Neuron(3 ,alpha, 21)  # Sigmoid Activation
cost= []

for i in range(iterations):
    # Forward 
    a_21 = N_21.forward(X2)
    a_22 = N_22.forward(X2)
    a_23 = N_23.forward(X2)
    a_2 = np.concatenate((a_21,a_22, a_23),axis=1)
    
    a_3 = N_31.forward(a_2)

    # Backward Learning
    grad = N_31.gradient(a_3,a_3 - y)
    N_31.backward(a_2,a_3 - y)
    
    N_21.backward(X2,grad*N_31.w[0])
    N_22.backward(X2,grad*N_31.w[1])
    N_23.backward(X2,grad*N_31.w[2])
    
    # Performance 
    cost.append(np.sum(np.abs(a_3-y))/y.shape[0])

plt.plot(cost)
plt.show()

m = y.shape[0]
measure_p = (a_3 > 0.5) == y
measure_n = (a_3 <= 0.5) != y
print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))


# In[21]:


PassID, X = get_features(test_raw)
X2 = X

a_21 = N_21.forward(X2)
a_22 = N_22.forward(X2)
a_23 = N_23.forward(X2)
a_2 = np.concatenate((a_21,a_22, a_23),axis=1)
    
a_3 = N_31.forward(a_2)


y_hat = np.reshape(a_3,(-1,1))
print(y_hat.shape)
data = y_hat > 0.5
data = data*1
s0 = pd.Series(PassID, index=PassID)
s1 = pd.Series(data[:,0], index=PassID)

df = pd.DataFrame(data = s1,index = PassID)
df.columns = ['Survived']
df.to_csv('simple_nn_2.csv', sep=',')


# Test with a bigger hidden layer 

# In[22]:


PassID, X = get_features(train_raw)
X2 = X #kernel(X)
n_x = X2.shape[1] # Features
alpha = 0.15/n_x
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 5000
# Setup Layers
# 2 - sigmoid
# 3 - tanh
# 7 - ReLU
# 21- Gaussian
N_21 = Neuron(n_x, alpha, 7)  
N_22 = Neuron(n_x, alpha, 7)   
N_23 = Neuron(n_x, alpha, 7) 
N_24 = Neuron(n_x, alpha, 7)
N_25 = Neuron(n_x, alpha, 7)   
N_26 = Neuron(n_x, alpha, 7)   
N_27 = Neuron(n_x, alpha, 7)   
N_28 = Neuron(n_x, alpha, 7)    

n_hidden = 8
N_31 = Neuron(n_hidden ,alpha, 21)  # 
cost= []

for i in range(iterations):
    # Forward 
    a_21 = N_21.forward(X2)
    a_22 = N_22.forward(X2)
    a_23 = N_23.forward(X2)
    a_24 = N_24.forward(X2)
    a_25 = N_25.forward(X2)
    a_26 = N_26.forward(X2)
    a_27 = N_27.forward(X2)
    a_28 = N_28.forward(X2)
    
    a_2 = np.concatenate((a_21,a_22, a_23, a_24, a_25,a_26, a_27, a_28),axis=1)
    a_3 = N_31.forward(a_2)

    # Backward Learning
    grad = -N_31.gradient(a_3,y)
    N_31.backward(a_2,a_3 - y)
    
    N_21.backward(X2,grad*N_31.w[0])
    N_22.backward(X2,grad*N_31.w[1])
    N_23.backward(X2,grad*N_31.w[2])
    N_24.backward(X2,grad*N_31.w[3])
    N_25.backward(X2,grad*N_31.w[4])
    N_26.backward(X2,grad*N_31.w[5])
    N_27.backward(X2,grad*N_31.w[6])
    N_28.backward(X2,grad*N_31.w[7])

    # Performance 
    cost.append(np.sum(np.abs(a_3-y))/y.shape[0])

plt.plot(cost)
plt.show()

m = y.shape[0]
measure_p = (a_3 > 0.5) == y
measure_n = (a_3 <= 0.5) != y
print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))
plt.scatter(a_3,y)
plt.show()


# # TODO
# *  Fix neural networks backpropagation to learn using the gradient of outer layers instead of pure delta
# * in case of categorical features with more than two classes, split it into a set of binary features one for each class 
# * implement neural nets with seed weights instead of random ones 

# In[23]:


# Prepare Data for submission
data = a_3 > 0.5
data = data*1
s0 = pd.Series(PassID, index=PassID)
s1 = pd.Series(data[:,0], index=PassID)

df = pd.DataFrame(data = s1,index = PassID)
df.columns = ['Survived']
df.to_csv('nn_submission.csv', sep=',')
print('Done')


# # Experiment
# in feature extraction portion, I will use single value decoposition as mapped feature vector 
# try to build a model on it, and then attempt to reproduce U from s and V to repeat the same operation on test data

# In[24]:


PassID, X = get_features(train_raw)
X2 = kernel(X)

Ux, sx, Vx = np.linalg.svd(X2, full_matrices=False)
X2 = Ux
n_x = X2.shape[1] # Features
alpha = 0.0015
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 25000
# Setup Layers
# 2 - sigmoid
# 3 - tanh
# 7 - ReLU
# 21- Gaussian
N_21 = Neuron(n_x, alpha, 2)  

cost= []

for i in range(iterations):
    # Forward 
    a_2 = N_21.forward(X2)

    # Backward Learning
    grad = N_21.gradient(a_2,a_2 - y)
    N_21.backward(X2,a_2 - y)

    # Performance 
    cost.append(np.sum(np.abs(a_2-y))/y.shape[0])

plt.plot(cost)
plt.show()

m = y.shape[0]
measure_p = (a_2 > 0.5) == y
measure_n = (a_2 <= 0.5) != y
print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))
plt.scatter(a_2,y)
plt.show()


# Test with Logistic Regression instead

# In[25]:


PassID, X = get_features(train_raw)

# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))

X2 = kernel(X)
# Test Model
alpha = 0.005 # Learning Rate
Ux, sx, Vx = np.linalg.svd(X2, full_matrices=False)
X2 = Ux
print(Ux.shape)
w, b = init(X2)
print(w.shape)
cost = []

for i in range(500):
    w, b = learn(X2, w, b, y, alpha)
    cost.append(lcost(X2, w, b, y))
    
y_hat = predict(X2, w, b)
print(w.shape)
plt.plot(cost)
plt.show()

plt.scatter( y_hat, y)
plt.show()

m = y.shape[0]
measure_p = (y_hat > 0.5) == y
measure_n = (y_hat <= 0.5) != y
print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))


# In[26]:


# Validate for test set 
PassID, X = get_features(test_raw)
X2 = kernel(X)
U2 = np.dot(np.dot(X2,Vx),np.linalg.inv(np.diag(sx)))
print(U2.shape)
print(w.shape)
y_hat = predict(U2, w, b)

# Prepare Data for submission
data = y_hat > 0.5
data = data*1
s0 = pd.Series(PassID, index=PassID)
s1 = pd.Series(data[:,0], index=PassID)

df = pd.DataFrame(data = s1,index = PassID)
df.columns = ['Survived']
df.to_csv('lr_svd_submission.csv', sep=',')


# In[27]:


PassID, X = get_features(train_raw)
X2 = kernel(X)
Ux, sx, Vx = np.linalg.svd(X2, full_matrices=False)
X2 = Ux

n_x = X2.shape[1] # Features
alpha = 0.00125/n_x
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 2500
# Setup Layers
# 2 - sigmoid
# 3 - tanh
# 7 - ReLU
# 21- Gaussian
N_21 = Neuron(n_x, alpha, 7)  
N_22 = Neuron(n_x, alpha, 7)   
N_23 = Neuron(n_x, alpha, 7) 
N_24 = Neuron(n_x, alpha, 7)
N_25 = Neuron(n_x, alpha, 7)   
N_26 = Neuron(n_x, alpha, 7)   
N_27 = Neuron(n_x, alpha, 7)   
N_28 = Neuron(n_x, alpha, 7)    

n_hidden = 8
N_31 = Neuron(n_hidden ,alpha/n_hidden, 21)  
cost= []

for i in range(iterations):
    # Forward 
    a_21 = N_21.forward(X2)
    a_22 = N_22.forward(X2)
    a_23 = N_23.forward(X2)
    a_24 = N_24.forward(X2)
    a_25 = N_25.forward(X2)
    a_26 = N_26.forward(X2)
    a_27 = N_27.forward(X2)
    a_28 = N_28.forward(X2)
    
    a_2 = np.concatenate((a_21,a_22, a_23, a_24, a_25,a_26, a_27, a_28),axis=1)
    a_3 = N_31.forward(a_2)

    # Backward Learning
    grad = N_31.gradient(a_3,a_3 - y)
    N_31.backward(a_2,a_3 - y)

    N_21.backward(X2,grad*N_31.w[0])
    N_22.backward(X2,grad*N_31.w[1])
    N_23.backward(X2,grad*N_31.w[2])
    N_24.backward(X2,grad*N_31.w[3])
    N_25.backward(X2,grad*N_31.w[4])
    N_26.backward(X2,grad*N_31.w[5])
    N_27.backward(X2,grad*N_31.w[6])
    N_28.backward(X2,grad*N_31.w[7])

    # Performance 
    cost.append(np.sum(np.abs(a_3-y))/y.shape[0])

plt.plot(cost)
plt.show()

m = y.shape[0]
measure_p = (a_3 > 0.5) == y
measure_n = (a_3 <= 0.5) != y
print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))
plt.scatter(a_3,y)
plt.show()


# In[28]:


PassID, X = get_features(train_raw)
X2 = kernel(X)
#Ux, sx, Vx = np.linalg.svd(X2, full_matrices=False)
#X2 = Ux

n_x = X2.shape[1] # Features
alpha = 0.00125/n_x
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 1500
# Setup Layers
# 2 - sigmoid
# 3 - tanh
# 7 - ReLU
# 21- Gaussian
N_21 = Neuron(n_x, alpha, 7)  
N_22 = Neuron(n_x, alpha, 7)   

n_hidden = 2
N_31 = Neuron(n_hidden ,alpha/n_hidden, 21)  # Sigmoid Activation
cost= []

for i in range(iterations):
    # Forward 
    a_21 = N_21.forward(X2)
    a_22 = N_22.forward(X2)
    
    a_2 = np.concatenate((a_21,a_22),axis=1)
    a_3 = N_31.forward(a_2)

    # Backward Learning
    
    grad = N_31.gradient(a_3,y)
    N_31.backward(a_2,a_3 - y)
    N_21.backward(X2,grad*N_31.w[0])
    N_22.backward(X2,grad*N_31.w[1])

    # Performance 
    cost.append(np.sum(np.abs(a_3-y))/y.shape[0])

plt.plot(cost)
plt.show()

m = y.shape[0]
measure_p = (np.abs(a_3) > 0.5) == y
measure_n = (np.abs(a_3) <= 0.5) != y
print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))
plt.scatter(a_3,y)
plt.show()


# In[29]:


# Prepare Data for submission

# Validate for test set 
PassID, X = get_features(test_raw)
X2 = kernel(X)

a_21 = N_21.forward(X2)
a_22 = N_22.forward(X2)
    
a_2 = np.concatenate((a_21,a_22),axis=1)
y_hat = N_31.forward(a_2)

data = y_hat > 0.5
data = data*1
s0 = pd.Series(PassID, index=PassID)
s1 = pd.Series(data[:,0], index=PassID)

df = pd.DataFrame(data = s1,index = PassID)
#print(df)
df.columns = ['Survived']
df.to_csv('nn_gauss.csv', sep=',')



# In[45]:


# Experiment, Random ReLUs without learning, with output fed to Gaussian 
PassID, X = get_features(train_raw)
X2 = X
#Ux, sx, Vx = np.linalg.svd(X2, full_matrices=False)
#X2 = Ux
n_x = X2.shape[1] # Features
alpha = 0.05
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 500
instances = 5
oinst = 300
N_r0 = Neuron(n_x,alpha,7)
N_r = [Neuron(n_x,alpha,7) for j in range(instances)]  # ReLU Activation 

a_2 = N_r0.forward(X2)

for i in range(instances):
    a_2i = N_r[i].forward(X2)
    a_2 = np.concatenate((a_2,a_2i),axis=1)

# Setup single neurons
N_g = [Neuron(n_x,alpha,21) for j in range(oinst)]
final_cost = []

alph = np.linspace(0.55,0.6,oinst)

for j in range(oinst):
    cost_g = []
    N_g[j].alpha = alph[j]
    for i in range(iterations):
        # Evaluate
        a_g = N_g[j].forward(X2)
        
        # Learn
        N_g[j].backward(X2,a_g - y)
        
        # Evaluate
        a_g = N_g[j].forward(X2)
        
        # Performance vote
        cost_g.append(np.sum(np.abs(a_g-y))/y.shape[0])
    
    m = y.shape[0]
    measure_p = (a_g > 0.5) == y
    measure_n = (a_g <= 0.5) != y
    final_cost.append(cost_g[-1])

    print("Training: Positive Success = {}, Negative Success = {}".format(np.sum(measure_p)/m,np.sum(measure_n)/m))

    #plt.plot(cost_g)
    #plt.title("Cost over time")
    #plt.show()

    #plt.scatter(a_g, y)
    #plt.show()
    
plt.plot(alph,final_cost)    
plt.show()


# In[46]:


PassID, X = get_features(test_raw)
X2 = X

a_gm = N_g[0].forward(X2)
for i in range(oinst-1):
    a_gi = N_g[i+1].forward(X2)
    a_gm = np.concatenate((a_gm,a_gi),axis=1)
    
y_hat = np.mean(a_gm, axis=1)
y_hat = np.reshape(y_hat,(-1,1))
print(y_hat.shape)
data = y_hat > 0.5
data = data*1
s0 = pd.Series(PassID, index=PassID)
s1 = pd.Series(data[:,0], index=PassID)

df = pd.DataFrame(data = s1,index = PassID)
df.columns = ['Survived']
df.to_csv('voting_gauss_nn_3.csv', sep=',')


# I think I should use some cross validation before declaring a result
# my models are overfitting to training data

# In[32]:


# Search optimal alpha
PassID, X = get_features(train_raw)
X2 = X
#Ux, sx, Vx = np.linalg.svd(X2, full_matrices=False)
#X2 = Ux
n_x = X2.shape[1] # Features
alpha = 0.24
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 3000
instances = 5
oinst = 150
N_r0 = Neuron(n_x,alpha,7)
N_r = [Neuron(n_x,alpha,7) for j in range(instances)]  # ReLU Activation 

a_2 = N_r0.forward(X2)
m = y.shape[0]


# Setup single neurons
N_g = [Neuron(n_x,alpha,21) for j in range(oinst)]
final_gain = []
alph = np.linspace(0.0,0.20,oinst)
for j in range(oinst):
    gain_g = []
    N_g[j].alpha = alph[j]
    for i in range(iterations):
        # Evaluate
        a_g = N_g[j].forward(X2)
        
        # Learn
        N_g[j].backward(X2,a_g - y)
        
        # Evaluate
        a_g = N_g[j].forward(X2)
        
        # Performance vote
        gain_g.append(np.sum(measure_p)/m)
    
    
    measure_p = (a_g > 0.5) == y
    measure_n = (a_g <= 0.5) != y

    final_gain.append(np.sum(measure_p)/m)

    #plt.plot(cost_g)
    #plt.title("Cost over time")
    #plt.show()

    #plt.scatter(a_g, y)
    #plt.show()
    
plt.scatter(alph,final_gain)    
plt.show()


# In[33]:


# Search optimal alpha
PassID, X = get_features(train_raw)
X2 = X
#Ux, sx, Vx = np.linalg.svd(X2, full_matrices=False)
#X2 = Ux
n_x = X2.shape[1] # Features
alpha = 0.051
# Survived
y = np.array(train_raw['Survived'])
y = np.reshape(y,(-1,1))
iterations = 100
instances = 5
oinst = 150

a_2 = N_r0.forward(X2)
m = y.shape[0]


# Setup single neurons
N_g = [Neuron(n_x,alpha,3) for j in range(oinst)]
final_gain = []
alph = np.linspace(0.05,0.055,oinst)
iters = [100+i for i in range(oinst)]

for j in range(oinst):
    gain_g = []
    #N_g[j].alpha = alph[j]
    for i in range(iterations):
        # Evaluate
        a_g = N_g[j].forward(X2)
        
        # Learn
        N_g[j].backward(X2,a_g - y)
        
        # Evaluate
        a_g = N_g[j].forward(X2)
        
        # Performance vote
        gain_g.append(np.sum(measure_p)/m)
    
    
    measure_p = (a_g > 0.5) == y
    measure_n = (a_g <= 0.5) != y

    final_gain.append(np.sum(measure_p)/m)

    #plt.plot(cost_g)
    #plt.title("Cost over time")
    #plt.show()

    #plt.scatter(a_g, y)
    #plt.show()
    
plt.scatter(alph,final_gain)    
plt.show()


# In[ ]:





# In[ ]:




