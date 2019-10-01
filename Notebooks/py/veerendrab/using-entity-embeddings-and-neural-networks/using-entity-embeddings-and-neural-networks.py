#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
from matplotlib.gridspec import GridSpec

get_ipython().magic(u'matplotlib inline')

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import optim
import copy
import math
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


PATH ='../input/'
train = pd.read_csv(PATH+'train.csv')
test = pd.read_csv(PATH+'test.csv')
combine = pd.concat([train.assign(DS='train') ,test.assign(DS='test',Survived='NA')],axis=0)


# In[3]:


print(combine.isna().sum(axis=0))


# In[4]:


surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]
surv_col = "blue"
nosurv_col = "red"

print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"      %(len(surv), 1.*len(surv)/len(train)*100.0,        len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train)))


# Check the age groups distribution

# In[5]:


agebins = plt.hist(combine['Age'][combine['Age'].isnull()==False],bins=10)
combine['AgeAbove12'] = np.NAN
combine['AgeAbove12'] =[ 2 if x>12 else 1 for x in combine['Age']]


# In[6]:


combine['NoPerTicket']  = combine['Ticket'].value_counts()[combine['Ticket']].values
# or 
combine['Ticket_group'] = combine.groupby('Ticket')['Name'].transform('count')
combine.drop(labels=['Ticket_group'],inplace=True,axis=1)


# In[7]:


t = combine[(combine['Parch']==0) & (combine['Age'].isnull()==True) & (combine['SibSp']==0) & (combine['NoPerTicket']==1)]['PassengerId']
for i in combine['PassengerId']:
    if i in t:
        combine.loc[i,'AgeAbove12'] =2

#single ticket but siblings or parch are there
combine[(combine['NoPerTicket']==1) & (combine['Age'].isnull()==False) & ((combine['Parch']>0) | (combine['SibSp']>0)) ]


# In[8]:


combine['Title'] = combine.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
combine['Title'] = combine['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
combine['Title'] = combine['Title'].replace('Mlle', 'Miss')
combine['Title'] = combine['Title'].replace('Ms', 'Miss')
combine['Title'] = combine['Title'].replace('Mme', 'Mrs')
print(combine['Title'].value_counts(dropna=False))
print(pd.crosstab(combine['Title'],combine['Sex']))


# In[9]:


#To check whether any male with title as Mr is having less than 10 yrs age and travelling alone without siblings or parch or col traveller( same ticket)
combine[ (combine['NoPerTicket']==1) &  (combine['Age'] <13) & (combine['Sex']=='male') &(combine['Title']=='Mr')].sort_values('Age')


# In[10]:


#consider traveller with title Mr and travelling alone as aboveag 12.
t = combine[(combine['NoPerTicket']==1) & 
            (combine['Age'].isnull()) & 
            (combine['SibSp']==0) & 
            (combine['Parch']==0) &  
            (combine['Sex']=='male') &
            (combine['Title']=='Mr') & (combine['AgeAbove12']==0) ]['PassengerId']
for i in combine['PassengerId']:
    if i in t:
        combine.loc[i,'AgeAbove12'] =1

t


# Null value handlilng

# In[11]:


#sort the ages into logical categories
combine["Age"] = combine["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
combine['AgeGroup'] = pd.cut(combine["Age"], bins, labels = labels)
combine['AgeAvailable'] =(combine['Age']==-0.5)==False


# In[12]:


combine['Family'] = combine['Parch'] + combine['SibSp']


# In[13]:


combine['Cabin'] = combine['Cabin'].fillna('Unknown')
combine['CabinPretext'] = combine.Cabin.str.extract('([A-Za-z]+)', expand=False)
combine['CabinAvialable'] = combine['Cabin']=='Unknown'


# In[14]:


combine.reset_index(drop=True,inplace=True)


# In[15]:


combine[combine['Embarked'].isnull()]


# In[16]:


combine['Embarked'].iloc[61] = "C"
combine['Embarked'].iloc[829] = "C"


# In[17]:


combine[combine['Fare'].isnull()]


# In[18]:


combine['Fare'].iloc[1043] = combine['Fare'][combine['Pclass'] == 3].dropna().median()
print(combine['Fare'].iloc[1043])


# In[19]:


combine['Fare_cat'] = pd.DataFrame(np.floor(np.log10(combine['Fare'] + 1))).astype('int')
combine['Fare_cat'].value_counts()
combine['Shared_ticket'] = np.where(combine.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
combine['Fare_eff'] = combine['Fare']/combine['NoPerTicket']
combine['Fare_eff_cat'] = np.where(combine['Fare_eff']>16.0, 2, 1)
combine['Fare_eff_cat'] = np.where(combine['Fare_eff']<8.5,0,combine['Fare_eff_cat'])
combine['Ttype'] = combine['Ticket'].str[0]
combine['Bad_ticket'] = combine['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])

combine['FareBins'] = pd.cut(combine['Fare'],[-1,0,5,8,10,20,30,40,50,100,150,200,250,300,400,550],labels=       ['0','0_5','5_8','8_10','10_20','20_30','30_40','40_50',        '50_100','100_150','150_200','200_250','250_300','300_400','400_550'])

combine['FareEffBins'] = pd.cut(combine['Fare_eff'],[-1,0,5,8,10,15,20,30,40,50,60,70,80,90,100,110,130],labels=       ['0','0_5','5_8','8_10','10_15','15_20','20_30','30_40','40_50',        '50_60','60_70','70_80','80_90','90_100','100_110','110_130'])


# In[20]:


combine.isnull().sum()


# In[21]:


colnotreq = ['Age','PassengerId','Cabin','Name','DS','Ticket','Fare','Fare_eff']
Target = ['Survived']
cols = combine.columns
cols = [x for x in cols if x not in colnotreq]
cols = [x for x in cols if x not in Target]


# In[22]:


t = combine[cols].dtypes!='category'
numcols = list(t.index[t.values])

for col in numcols:
    combine[col] = (combine[col].astype('category'))
    
combine_index = combine.copy()
combine_eEmbed = combine.copy()


# In[23]:


fig = plt.figure(figsize=(15,30))
plt.style.use('ggplot')
sns.despine(left=True)
colno = 2
rowno = (len(cols)+4)//colno
gs = GridSpec(rowno,colno)

i = 0
j = 0
for col in cols:
    if combine[col].value_counts().shape[0] > 10 :
        sns.barplot(x=col, y="Survived", data=combine[combine['DS']=='train'],ax=plt.subplot(gs[i,:])) 
        j=0
        i +=1
    else:
        sns.barplot(x=col, y="Survived", data=combine[combine['DS']=='train'],ax=plt.subplot(gs[i,j])) 
        j+=1
    if j == colno :
        j = 0
        i +=1

            
plt.tight_layout()


# In[24]:


# convert categorical values to codes to process the correlation .
for col in cols:
    combine_index[col] = (combine[col].cat.codes)


# In[25]:


combine_index

# Compute the correlation matrix
corr = combine_index[cols].corr(method='spearman')

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
plt.style.use('ggplot')

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, mask=mask,vmax=1,center=0,annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.tight_layout()


# In[26]:


cor_cols = ['Shared_ticket','Family','Pclass','Fare_eff_cat','CabinAvailable']


# In[27]:


cols = combine_index.columns
cols = [x for x in cols if x not in colnotreq]
cols = [x for x in cols if x not in Target]
cols = [x for x in cols if x not in cor_cols]
combine_index.dtypes


# In[28]:


#Create train and validation sets from the indexed combine_index dataframe.
from sklearn.model_selection import train_test_split
train= combine_index[combine_index['DS']=='train'].reset_index(drop=True)
train['Survived'] =train['Survived'].astype(np.int)
test = combine_index[combine_index['DS']=='test'].reset_index(drop=True)

x_train, x_val, y_train, y_val = train_test_split(train[cols], train['Survived'], test_size = 0.22, random_state = 0)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_gaussian)

# Logistic Regression


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_logreg)

# Support Vector Machines
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_svc)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_linear_svc)

# Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_perceptron)

#Decision Tree
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_decisiontree)

# Random Forest
randomforest = RandomForestClassifier(n_estimators=1000)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_randomforest)

# KNN or k-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_knn)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_sgd)

# Gradient Boosting Classifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_gbk)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)



# Create Entity Embeddings for Categorical values using Neural Network

# In[29]:


#Create embedding sizes.
cat_sz = [(c, len(combine_eEmbed[c].cat.categories)) for c in cols]
emb_szs = [(c, min(10, c)) for _,c in cat_sz]
emb_szs


# Create Functions required for creating Model in Pytorch and Dataloader for loading data using Batch process

# In[30]:


from torch.nn.init import kaiming_uniform, kaiming_normal

class DlTrain(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        X = self.data[index][1:]
        y = self.data[index][0]
        return torch.from_numpy(X), torch.FloatTensor(np.array([y]))
    
    def __len__(self):
        return len(self.data)

def emb_init(x):
    x = x.weight.data
    sc = 2/(x.size(1)+1)
    x.uniform_(-sc,sc)
    

class simplenet(nn.Module):
    def __init__(self,emb_sz,emb_drop,hl_sz,out_sz,drops):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c,s) for c,s in emb_sz])
        for emb in self.embs: emb_init(emb)
        self.n_emb = sum(e.embedding_dim for e in self.embs)

        hl_sz = [self.n_emb]+hl_sz
        self.lins =nn.ModuleList([nn.Linear(hl_sz[i],hl_sz[i+1]) for i in range(len(hl_sz)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(sz,momentum=0.1) for sz in hl_sz[1:]])
        for o in self.lins: kaiming_normal(o.weight.data)
        
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        
        self.outl = nn.Linear(hl_sz[-1],out_sz)
    def forward(self, x):
        if self.n_emb!=0:
#             print(np.unique((x[:,i].data).numpy()), ": ",i)
            x = [e(x[:,i]) for i,e in enumerate(self.embs)]
            x = torch.cat(x,1)
            x = self.emb_drop(x)
        for l,d,b in zip(self.lins,self.drops,self.bns):
            x = F.relu(l(x))
            x = b(x)
            x = d(x)
        x = self.outl(x)
        return x


# In[31]:


data = DlTrain(np.array(pd.concat([y_train,x_train],axis=1)))
data_val = DlTrain(np.array(pd.concat([y_val,x_val],axis=1)))

data_dl = {'train':DataLoader(data, batch_size= 54, shuffle=True, num_workers=0),
          'val':DataLoader(data_val, batch_size=54, shuffle=True, num_workers=0)}
dataset_sizes = {x: len(data_dl[x]) for x in ['train', 'val']}
dataset_sizes = {'train':x_train.shape[0], 'val':x_val.shape[0]}


# Initiate model and train

# In[33]:


epochs = 500 # set epohcs
criterion = nn.BCEWithLogitsLoss()# define loss function

m = simplenet(emb_szs,0.5,[100,10],1,[0.5,0.5])


### keep track of training loss
losses = []

###params
lr = 1e-2

optimizer = optim.Adam(lr = lr, params=m.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)

best_model_wts = copy.deepcopy(m.state_dict())
best_acc = 0.0

### Training 
since = time.time()
for epoch in range(epochs):
    # Each epoch has a training and validation phase
    for phase in ['train','val']:
        if phase == 'train':
            scheduler.step()
            m.train(True)  # Set model to training mode
        else:
            m.train(False)  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
#         for data1 in data_dl[phase]:
        for i, batch in enumerate(data_dl[phase]):
            inputs, Survived = batch
            inputs, Survived = Variable(inputs), Variable(Survived,requires_grad=False)
            optimizer.zero_grad()
            outputs = m(inputs) 
            # compute loss and gradients
            loss = criterion(outputs, Survived)
#             losses.append(loss)
            if phase == 'train':
                loss.backward()
                # update weights 
                optimizer.step()

            # statistics
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(torch.round(F.sigmoid(outputs.data)) == Survived.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]
        
        if epoch and np.mod(epoch+1,100) == 0:
#             print(f'epoch {epoch}')
            print('epoch:{} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(m.state_dict())

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))


# Get the Best Model embedding layers weights and use weights for regression analysis

# In[34]:


i= 0
ee_list = list()
for col in cols:
    w_col = list(best_model_wts.keys())[i]
    ee_list.append((best_model_wts[w_col].numpy()[combine_index[col],:]))
    i +=1
ee_combine  = np.hstack(ee_list)


# Get the first two dimensions of weights using PCA and plot them to see the relation between them.

# In[35]:


fig = plt.figure(figsize=(15,25))
plt.style.use('ggplot')
sns.despine(left=True)
colno = 2
rowno = (len(cols)+1)//colno
gs = GridSpec(rowno,colno)

for i in range(len(cols)):
    ax = fig.add_subplot(gs[i//2,np.mod(i,2)])
    w = m.embs[i].weight.data
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(w)
    ax.scatter(result[:, 0], result[:, 1],linewidths=2)
    ax.set_title(cols[i])
    combine_eEmbed[cols[i]]
    for j, txt in enumerate(combine_eEmbed[cols[i]].cat.categories):
        ax.annotate(txt,(result[j, 0], result[j, 1]))
plt.tight_layout()


# In[36]:


#Create train and val sets.
train['Survived'] =train['Survived'].astype(np.int)
target = train['Survived'].astype(np.int)

EEtrain = ee_combine[combine_eEmbed['DS']=='train']
EEtest = ee_combine[combine_eEmbed['DS']=='test']
x_train, x_val, y_train, y_val = train_test_split(EEtrain, target, test_size = 0.22, random_state = 0)


gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
ee_acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_gaussian)

# Logistic Regression


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
ee_acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_logreg)

# Support Vector Machines
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
ee_acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_svc)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
ee_acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_linear_svc)

# Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
ee_acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_perceptron)

#Decision Tree
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
ee_acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_decisiontree)

# Random Forest
randomforest = RandomForestClassifier(n_estimators=1000)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
ee_acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_randomforest)

# KNN or k-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
ee_acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_knn)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
ee_acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_sgd)

# Gradient Boosting Classifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
ee_acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_gbk)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk],
    'Entity Embedding Score': [ee_acc_svc, ee_acc_knn, ee_acc_logreg, 
              ee_acc_randomforest, ee_acc_gaussian, ee_acc_perceptron,ee_acc_linear_svc, ee_acc_decisiontree,
              ee_acc_sgd, ee_acc_gbk]},columns = ['Model','Score','Entity Embedding Score'])
models.sort_values(by='Entity Embedding Score', ascending=False)




# References
# *  https://www.kaggle.com/headsortails/pytanic
# * http://course.fast.ai/lessons/lesson4.html

# In[ ]:




