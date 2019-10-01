#!/usr/bin/env python
# coding: utf-8

# ## Welcome Kaggler!
# 
# With this interactive diving course I invite you to learn some machine learning basics. This course is designed as a series of kernels that guides you through different topics and hopefully you can discover some hidden treasures that push you forward on your data science road. You don't have to pick the courses in sequence if you are only interested in some topics that are covered. If you are new I would recommend you to take them step by step one after another. ;-)
# 
# Just fork the kernels and have fun! :-)
# 
# * [Prepare to start](https://www.kaggle.com/allunia/titanic-dive-through-prepare-to-start): Within this kernel we will prepare our data such that we can use it to proceed. Don't except nice feature selection or extraction techniques here because we will stay as simple as possible. Without a clear motivation we won't change any features. Consequently we are only going to explore how to deal with missing values and how to turn objects to numerical values. In the end we will store our prepared data as output such that we can continue working with it in the next kernel.
# * **MyClassifier**: Are you ready to code your own classifier? Within this kernel you will build logistic regression from scratch. By implementing the model ourselves we can understand the assumptions behind it. This knowledge will help us to make better decisions in the next kernel where we will use this model and build some diagnosis tools to improve its performance.
# * [The feature cave](https://www.kaggle.com/allunia/titanic-dive-through-feature-cave): By using our own logistic regression model we will explore how we can improve by adding a bias term and why we should encode categorical features. 
# * [Feature scaling and outliers](https://www.kaggle.com/allunia/titanic-dive-through-feature-scaling-and-outliers/edit): Why is it important to scale features and to detect outliers? By analyzing the model structure we will discover how our gradients and our model performance are influenced by these topics. 

# ## Get your equipment
# 
# Now we are ready to dive deeper! For this course we have already given the prepared data from our last tour in the input folder:

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


print(os.listdir("../input/titanicdivethrough"))


# Ok, let's load our packages and read in the data:

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set()


# In[ ]:


train = pd.read_csv("../input/titanicdivethrough/prepared_train.csv", index_col=0)
train.head()


# In[ ]:


test = pd.read_csv("../input/titanicdivethrough/prepared_test.csv", index_col=0)
test.head()


# ## MyClassifier skeleton
# 
# During this interactive course you will code all methods we need to build a simple logistic regression model. As you might be new to python programming this tutorial provides a skeleton class that you will complete with your solutions step by step. 
# 
# Everytime you have finsihed one diving station you must copy your solution and plug it into the appropriate method!

# In[ ]:


class MyClassifier:
    
    def __init__(self):
        np.random.seed(0)
        self.w = np.random.normal(loc=0, scale=0.01, size=7)
        self.losses = []
        pass
    
    def predict(self, x):
        pass
    
    def loss(self, y, t):
        pass
    
    def gradient(self, x, y, t):
        pass
    
    def update(self, eta, grad):
        pass
    
    def learn(self, x, t, eta, max_steps, tol):
        pass
    
    def score(self, x, t):
        pass


# ## Station 1: Making predictions
# 
# Ok, you have already peeked at our simplyfied data we will start with. But how to predict the survival of passengers in the test set? 
# 
# ### Deciding survival by threshold
# 
# One simple idea could be to multiply each feature $x_{d}$ with some parameter $w_{d}$ and sum up all terms to obtain a quantity $g$ that helps us to make a decision for one passenger if he or she survived or not: 
# 
# $$g = x_{1} \cdot w_{1} + ... + x_{ 7} \cdot w_{7} = \sum_{d=1}^{D=7} x_{d} \cdot w_{d}$$
# 
# $$y=
#     \begin{cases}
#       0, & \text{if}\ g > \tau \\
#       1, & \text{otherwise}
#     \end{cases} 
# $$
# 
# Now we would predict that a passenger has not survived if its $g$ is higher than some threshold $\tau$ and otherwise the passenger survived. By forcing the parameter w to be shared over all passengers in the train set we would gain a general way to calculate the survival of new passengers like those in the test set. But with this approach we are running into problems:
# 
# * How shall we set $\tau$ and the 7 parameter $w_{d}$ to obtain the best predictions in train and test?
# * Finding the best predictions with least errors between prediction and target labels in the train set sounds like an optimization task. Consequently we may need gradients with repsect to parameters w and tau. But how should we do that with our prediction function y? Using a threshold has caused some trouble as the [Heaviside-function](http://https://en.wikipedia.org/wiki/Heaviside_step_function) we used is not differentiable. 
# 
# How can we solve this problem? 
# 
# ### A smooth approximation
# 
# As you might have read in the wikipedia article we could use a smooth approximation of the heaviside function -  the logistic function. Let's go even one step further and take the [sigmoid function](http://https://en.wikipedia.org/wiki/Sigmoid_function) $\sigma$ as a special and simple case of the logistic function. Then our prediction function would turn to:
# 
# $$ y = \sigma \left( \sum_{d=1}^{D=7} x_{d} \cdot w_{d} \right) $$
# 
# $$ \sigma(x) = \frac{1}{1 + \exp(-x)}$$
# 
# Hah, Nice! We get rid of the threshold parameter $\tau$ and the only thing we are left off is to find optimal parameter w that minimize the errors between our predictions and target values of survival in the train set. Now, let's turn to practice:

# In[ ]:


# It's your turn: Delete the pass and implement the sigmoid function:
def sigmoid(x):
    pass
    #result = <your code>
    #return result


# Let's check if your solution is right. To do this uncomment the assert statements and run the cell. If nothing happens, your solution seems to be right! :-)

# In[ ]:


#assert(sigmoid(0) == 0.5)
#assert(np.round(sigmoid(-6), 2) == 0.0)
#assert(np.round(sigmoid(6), 2) == 1.0)


# Now we have a neat function to make predictions y and we have already in mind that we need to find the parameters w to minimize the error between the predictions y and targets t of survival in the train data. Ok, now it's your turn: Write code to calculate predictions of the survival of all passengers in the train set. You can use np.sum(..., axis=1) to obtain the sum over feature dimensions.  

# In[ ]:


# Your task: Delete the pass and fill in your solution. Use w in your equation as parameters.
def predict(x, w):
    pass
    #y = <your code>
    #return y


# Let's check your solution by passing the features of all passengers in the train set. For this purpose we should drop the PassengerId and the target label:

# In[ ]:


np.random.seed(0)
w = np.random.normal(loc=0, scale=0.01, size=7)

X = train.drop(["PassengerId", "Survived"], axis=1).values
# Y = predict(X, w)


# ### Your initial predictions
# 
# 

# In[ ]:


plt.figure(figsize=(20,5))
#sns.distplot(Y)
plt.xlabel("predicted values y")
plt.ylabel("frequency")
plt.title("Initial predictions of all train passengers")


# The outliers are strange, aren't they? What causes the distribution to be skewed? What do you think? 
# 
# Uncomment and run the cell to check your solution:

# In[ ]:


#assert(np.round(np.mean(Y), 2) == 0.51)
#assert(np.round(np.std(Y), 2) == 0.09)
#assert(np.round(np.median(Y), 2) == 0.54)
#assert(np.round(np.min(Y), 2) == 0.01)
#assert(np.round(np.max(Y), 2) == 0.67)


# **Great!!!** If nothing happens you are done with the prediction. Copy your solution and plug it into your MyClassifer skeleton. Don't forget to pass self to your prediction method and to use self.w instead of w! 

# ## Station 2 - How to define the errors 
# 
# Hopefully this wasn't too easy for you until now. If so, I'm glad to tell you that the water in our diving bay will become a bit more turbulent from now on. :-) We have to think about how to measure the error E between the target labels t and our predictions y of the survival of passengers in the train set. A naive approach would be the following: Just take the absolute difference of target and prediction for each passenger $n$ and take the mean of those errors....
# 
# $$ E = \frac{1} {N} \sum_{n=1}^{N} |t_{n} - y_{n}| $$
# 
# Well, ... this mean absolute error is not a good idea. We want to minimize this function with respect to our parameter w and the absolute value function is not differentiable at point 0. We can circumvent this by switching to the mean squared difference of targets and predictions. In that case we would work with the so called mean squared error function but unfortunately this is again not a good choice. Why?
# 
# $$ E = \frac{1} {N} \sum_{n=1}^{N} ||t_{n} - y_{n}||^{2}$$
# 
# The mean squared error is a commonly used error metric for problems that have targets that live in a continous space. This is not true in our case. We have only two discrete values, 0 and 1, that stands for "not survived" or "survived". Consequently we should use some error metric that is related to this discrete nature. But how to do that? We could try to find a distribution that tells us how likely it is that our predictions are close to the targets. Luckily there is a distribution out there - the Bernoulli distribution:
# 
# $$ p_{n}(t_{n}) = y_{n}^{t_{n}} \cdot (1 - y_{n})^{(1-t_{n})} $$
# 
# For each passenger we can calculate how probable it is that our prediction is close to the target. Let's do this for your prediction of the first passenger in the train set (Hint: use np.power):
# 

# In[ ]:


def are_they_close(y, t):
    pass
    #p = <your code>
    #return p


# In[ ]:


t = train.Survived.values[0]
#y = Y[0]
#probability = are_they_close(y,t)
#probability


# In[ ]:


#assert(np.round(are_they_close(y,t), 2) == 0.45)


# :-)
# 
# This time, there is nothing to plug into our MyClassifier skeleton. But don't be sad, the next station you can!

# ## Station 3 - cross entropy in the treasure trove
# 
# Now we have an expression for the probability of the match between the target t and the prediction y of one passenger. By assuming that the targets were drawn indepentently from the Bernoulli distribution we can multiply all passenger probabilities to obtain the overall probability that the prediction outputs Y matches the targets T:
# 
# $$ P(T|X,W) = \prod_{n=1}^{N} y_{n}^{t_{n}} \cdot (1 - y_{n})^{(1-t_{n})} $$
# 
# This function is also called likelihood function. Next, we can solve our problem by adjusting the parameters w such that they maximize the probability of matches! Let's try!
# 
# $$\frac{\partial P}{\partial w_{d}} = \frac{\partial}{\partial w_{d}} \prod_{n=1}^{N} y_{n}^{t_{n}} \cdot (1 - y_{n})^{(1-t_{n})} = 0$$
# 
# Oh, now we have a problem: Taking the derivative of a product with N factors. :-(
# If this would be a sum things would become nice and easy. Let's take the logarithm of our probability and maximize that:
# 
# $$\log P =  \sum_{n=1}^{N} [t_{n} \log(y_{n}) + (1-t_{n}) \log(1-y_{n})] = 0$$
# 
# This looks better! Often this log-likelihood function is multiplied by -1 as the result of that is related to the error or loss function: $E = - log P$. If we would have assumed that our targets were drawn from a normal distribution we would have obtained the sum of squared error function similar to our mean squared error above. In our case we used Bernoulli distributions and we end up with the binary cross entropy:  :-)
# 
# $$ E = - \log P = - \sum_{n=1}^{N} [t_{n} \log(y_{n}) + (1-t_{n}) \log(1-y_{n})]  $$
# 
# Ok, to minimize the cross entropy, we take the gradient with respect to w and set everything to zero. **Try it yourself! **
# You should end up with:
# 
# $$\frac{\partial{E}}{\partial w_{d}} =  \sum_{n=1}^{N} (y_{n} - t_{n})\cdot x_{n,d} $$
# 
# Even if you haven't done the derivation of this equation, it's your turn now:
# 

# In[ ]:


# Fill in the code to compute the loss of N passengers. The method arguments y and t are both vectors of length N. 
def loss(y, t):
    pass
    # E = <your code>
    # return E


# If you just plugin a prediction of $y_{n} = 0.5$ for each passenger in the train set you should obtain a loss value close to 618:

# In[ ]:


prediction = 0.5 * np.ones(train.shape[0])
target = train.Survived.values


# In[ ]:


#assert(np.round(loss(prediction, target))==618.0)


# Nice! Ok, now plugin your loss into the skeleton. 

# ## Station 4 - Play *Hit The Pot* with gradient descent
# 
# Cool! You have managed the main part of this course: We have derived the loss and the gradient of the loss with respect to parameters. Consequently we can now use this information to minimize the loss and make our model learn. We will do this by using a numerical method, namely gradient descent: 
# 
# It's a bit like playing "Hit The Pot" in a mountain landscape where the Pot is given by the deepest valley. Imagine you would stand on some hill with eyes covered. You have a spoon in your hand and every time you hit the ground someone tells you where the direction is steepest. Then, you are able to take a step of a fixed size. Afterwards you start again: hit the ground... a voice in the off tells you where to go and you take a step. Finally and hopefully you will end up close to the deepest valley and the change of your height is small each step you take from now on. Perhaps you will oscillate around it somehow as your step is fixed and you jump a bit over the deepest point. But you hit the pot and win a prize! :-) 
# 
# **It's worth to think about it** - What kind of problems can occur when you play *Hit the Pot* in a mountain landscape?  
# 
# In our case the heigth of the mointain landscape stand for the loss we want to minimize. And of course this problem is not three dimensional. Our landscape has as many dimensions as features. To start our game we chose some inital parameters w - a rondomly chosen point in that loss landscape. In MyClassifier this is already implemented in the __init__. Playing "hit the pot" now means that we make a step of fixed size $\eta$ in the direction of the steepest descent of our loss. We will do this stepwise again and again and hopefully we will end up in the minimum we were looking for. Hence to make our classifier work, we have to implement this concept of gradient descent. For this purpose you will write the update of parameters between two consequtive steps $\tau$ and $\tau + 1$ for each parameter $w_{d}$:
# 
# $$ w_{d}^{\tau + 1} = w_{d}^{\tau} - \eta \cdot \frac{\partial E}{\partial w^{\tau}_{d}}$$
# 
# **Now, it's your turn:**
# 
# Implement the gradient method that returns a vector of gradients with d elements that hold the derivatives $\frac{\partial E}{\partial w_{d}}$. Remember, they were given by:
# 
# $$\frac{\partial{E}}{\partial w_{d}} =  \sum_{n=1}^{N} (y_{n} - t_{n})\cdot x_{n,d} $$
# 
# Then, complete the update method to obtain the next parameters $w_{d}^{\tau + 1}$.
# 
# Hint: If you are not sure how to use vectors within your methods, you can use for loops if you like ;-) .

# In[ ]:


# implement the equation for the gradient of E with respect to parameters self.w
def gradient(x, y, t):
    #grad = np.zeros(w.shape[0])
    #for d in range(w.shape[0]):
        #grad[d] = <your code>
    #return grad
    pass
    
# implement the update equation for all parameters w in self.w
def update(w, eta, grad):
    # w_next = np.zeros(w.shape) 
    #for d in range(w.shape[0]):
        #w_next[d] = <your code>
    #return w_next
    pass


# In[ ]:


T = train.Survived.values
#grads = gradient(X, Y, T)
#w_next = update(w, 0.5 grads)


# In[ ]:


grads_control = np.array([429, -80, 3973, 59, 0, -5587, 10])
new_weights_control = np.array([-215, 40, -1987, -30, 0, 2793, -5])


# In[ ]:


#np.testing.assert_array_almost_equal(grads_control, grads, 0)
#np.testing.assert_array_almost_equal(new_weights_control, w_next, 0)


# Cool! If you passed this gate you can copy your code into the skeleton of MyClassifer. But be aware that you use self.w instead of w! Hence you can use update(self, eta, grads) instead of update(w, eta, grads). Let's have a look at the gradients and the change of the weights $\delta w = w^{\tau +1} - w^{\tau}$ :

# In[ ]:


#dw = w_next - w 


# ## Station 5 - The gradients magnitude

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,5))
#sns.barplot(x=train.drop(["PassengerId", "Survived"], axis=1).columns, y=grads, ax=ax[0])
#sns.barplot(x=train.drop(["PassengerId", "Survived"], axis=1).columns, y=dw, ax=ax[1])
ax[0].set_ylabel("gradients of loss")
ax[0].set_xlabel("per feature weight")
ax[0].set_title("Gradients of 1 step")
ax[1].set_ylabel("change of weights")
ax[1].set_xlabel("per feature")
ax[1].set_title("Change of weights after 1 step")


# Interestingly some gradients like for the fare and age weights are much stronger than the others. But is this really a surprise? Have a look at the equation we used to calculate the gradients:
# 
# $$\frac{\partial{E}}{\partial w_{d}} =  \sum_{n=1}^{N} (y_{n} - t_{n})\cdot x_{n,d} $$
# 
# Can you explain why some gradients are that strong? 
# 
# Of course, you can: The age and the fare exhibit only positive values. Consequently the only way to balance out the sum towards zero is by opposite signs caused by the residues $y_{n} - t_{n}$. But that's not enough. Terms that balance out should have equal oppositve values that are given by the product of the residues AND the feature values $x_{n,d}$ . To balance out or to obtain small valued gradients we need a distribution of $(y_{n} - t_{n}) \cdot x_{n,d}$ over all passengers N that is symmetric with mirror axis of zero. Let's have a look at the contributions of the age and the fare:

# In[ ]:


#contributions_age = (Y-T) * train.Age.values
#contributions_fare = (Y-T) * train.Fare.values

fig, ax = plt.subplots(1,2,figsize=(20,5))
#sns.distplot(contributions_age, ax=ax[0], color="Orange")
ax[0].set_title("Contributions to the age gradient")
ax[0].set_xlabel("contribution $(y_{n} - t_{n}) \cdot age $")
#sns.distplot(contributions_fare, ax=ax[1], color="Purple")
ax[1].set_title("Contributions to the fare gradient")
ax[1].set_xlabel("contribution $(y_{n} - t_{n}) \cdot fare $")


# Uff! Take a breather and look closely to what we have found! This is a big treasure trove! The distribution of the contributions to the gradient for the age weight is bimodal and unbalanced, as we have more positive contributions. Consequently the sum results in a high positive gradient for the age weight. In contrast the contributions of the fare seem to be very symmetric but left skewed due to some outliers in the fare distribution:

# In[ ]:


plt.figure(figsize=(10,3))
sns.distplot(train.Fare)
plt.title("Remember the fare outliers")


# Unfortunately its the same drama as with the mean: The outliers contribute strongly to the gradient! But this is of course not what we want! We don't want a single but seldom event to have a hugh impact on what our model learns. We want to make good predictions for the majority of passengers and not for some exotics. 
# 
# Now it's your turn: Activate your brain cells :-) ...
# 
# 1. Do you think it's a good idea that the features contribute to their gradients with different magnitudes caused by their range of values? Why would it be better to have equally valued ranges over all features? If you are not sure: Imagine you would play *hit the pot* and your like to take a fixed size step. In one dimension x you can make very steep steps but in the second dimension y you can only have a weak slope and your step results in low hight differences. What do you need to reach the minimum of BOTH dimensions? What would be if you could make equally hight distances per step in both dimensions?
# 2. What causes the age contributions to be unbalanced? Is this a problem at all? What do you think is the consequence for the rate of false positive and false negative predictions in the end of our learning phase?
# 3. How should we work with outliers in our feature distributions?
# 4. What about discrete categorical feature distributions with more than 2 values like Embarked or Pclass?  
# 
# We will come back to this topic in the next diving kernel, äh, course ;-)

# ## Station 6 - Bring your model to life
# 
# Yeah! We are close to finish our model. As our model learns by taking gradients and updating weights in consequtive steps we have to write a loop over steps. But before that we should define our starting point by making some initial predictions with the inital weights that are already given. Afterwards you can make as much steps as you like or until your loss change is smaller than some tolerance value: 

# In[ ]:


def learn(x, t, eta, max_steps, tol):
    losses = []
    np.random.seed(0)
    w = np.random.normal(loc=0, scale=0.01, size=7)
    y = predict(x, w)
    #for step in range(max_steps):
        #current_loss = <your code>
        #losses.append(current_loss)
        #grads = <your code>
        #w = <your code>
        #y = <your code>
        #next_loss = <your code>
        #if (current_loss - next_loss) < tol:
            #break
    #return losses, w
    pass


# In[ ]:


X = train.drop(["PassengerId", "Survived"], axis=1).values
T = train.Survived.values

#losses, w = learn(X, T, 0.000001, 100000, 0.00001)
#w


# In[ ]:


#assert(np.round(np.max(losses)) == 703)
#assert(np.round(np.mean(losses)) == 422)
#assert(np.round(np.min(losses)) == 404)
#assert(np.round(np.std(losses)) == 30)
#weights_control = np.array([-0.562, 2.58, -0.015, -0.284, -0.092, 0.008, 0.246])
#np.testing.assert_array_equal(np.round(w, 3), weights_control, 3)


# Jippie! :-) If you are right, plug your solution into MyClassifier! Use self.w and self.losses instead of w and losses. This way you can skip the return as well as the values are stored as attributes of our classifer and we can get them by just typing classifier.w or classifier.losses. In addition call the methods of your class by using self.method (for example self.predict(x) instead of predict(x)). 
# 
# Let's have a look at the loss during the iteration. You should see that it decreases and converges:

# In[ ]:


plt.figure(figsize=(15,5))
#plt.plot(losses)
plt.xlabel("Iteration steps")
plt.ylabel("cross-entropy loss")


# ## Station 7 - Good enough?
# 
# Now we have everything together: We can learn, we can compute the loss and make predictions, but how good is our trained model? Has it learned something? To find it out, we will finally implement a score metric. In this case we will use the accuracy score, namely the percentage of correct predictions. All we need is to make predictions based on the weights we have learned given some passengers and their features x. Next we have to make a decision, for example we could say a passenger died or survived by this rule:
# 
# $$\hat{y}=
#     \begin{cases}
#       1, & \text{if}\ y \geq 0.5 \\
#       0, & \text{otherwise}
#     \end{cases} 
# $$
# 
# Then we can compute the accuracy by:
# 
# $$ score = \frac{1}{N} \sum_{n=1}^{N} 1 - |t_{n} - \hat{y}_{n}| $$
# 
# If we made a wrong prediction we will obtain $|t_{n} - \hat{y}_{n}| = 1$ and no contribution to the sum. Now it's your turn: 

# In[ ]:


# Implement the accuracy score. Hint use np.sum and np.abs
def score(x, t):
    N = x.shape[0]
    y = predict(x, w)
    #accuracy = <your code>
    #return accuracy
    pass


# In[ ]:


#accuracy = score(X, T)
#accuracy


# In[ ]:


#assert(np.round(accuracy, 2) == 0.80)


# If you passed the gate, plug in your code into the score method of your classifier. But don't forget to use self.w instead of w. As you have finished your classifier, run the cell of MyClassifier to make sure everything is included. 

# Ok! Our model is not good! It has learned something but it is not able to make perfect matches for the train set. It seems to be that there are some difficulties which prevent our model to deeply understand the patterns in our data. We have already figured out that there are some problems with our features and the gradient compution. Perhaps there are even some more problems we haven't discovered so far? You you like to find it out: Take the next diving course! Don't be sad, you will need to know the stuff you have learned during this course to deeply understand why and what we should do to improve the performance of our predictions. 

# ## Final station
# 
# Very cool that you managed so far! You have build your own classifer. Ok, it's still not working as you like but with your own classifier at hand you are now able to code diagnosis tools that will help you to find out what's ill. This way you will not end up playing silly games by trying randomly this and that and staring at the score. ;-) 
# 
# Finally let's see if our classifier class works as expected. 

# In[ ]:


features = train.drop(["PassengerId", "Survived"], axis=1).columns

x_train = train[features].values
t_train = train.Survived.values

classifier = MyClassifier()
#classifier.learn(x_train, t_train, 0.0000001, 100000, 0.00001)
#score = classifier.score(x_train, t_train)
#score


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(classifier.losses)
plt.xlabel("Iteration steps")
plt.ylabel("Loss")
plt.title("My classifer losses")


# We should obtain the same losses, weights and score as before:

# In[ ]:


#assert(np.round(score, 2) == 0.80)
#assert(np.round(np.max(classifier.losses)) == 703)
#assert(np.round(np.mean(classifier.losses)) == 422)
#assert(np.round(np.min(classifier.losses)) == 404)
#assert(np.round(np.std(classifier.losses)) == 30)
#weights_control = np.array([-0.562, 2.58, -0.015, -0.284, -0.092, 0.008, 0.246])
#np.testing.assert_array_equal(np.round(classifier.w, 3), weights_control, 3)

