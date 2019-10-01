#!/usr/bin/env python
# coding: utf-8

# # Titanic Kaggle

# Worked example for the Titanic Kaggle competition: https://www.kaggle.com/c/titanic

# 0. Import libraries & create helper functions
# 1. Import & look at data
# 2. Prepare categorical data
# 3. Prepare numerical data
# 4. Create model
# 5. Train model
# 6. ???
# 7. Profit

# ## Imports & methods for plotting graphs

# In[ ]:


import matplotlib.pyplot as plt

import pandas as pd

import math

from keras import backend
from keras import Input

from keras.layers import concatenate, Dense, BatchNormalization
from keras.models import Model, Sequential

# for generating a png image of our model
from keras.utils import plot_model
from IPython.display import Image


# In[ ]:


# disable the `SettingWithCopy` because we are pretty sure we know what we are doing
pd.set_option('mode.chained_assignment', None) 


# In[ ]:


# when plotting, smooth out the points by some factor (0.5 = rough, 0.99 = smooth)
# method taken from `Deep Learning with Python` by François Chollet

def smooth_curve(points, factor=0.75):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# In[ ]:


# Plot model history more easily

def set_plot_history_data(ax, history, which_graph):

    if which_graph == 'acc':
        train = smooth_curve(history.history['acc'])
        valid = smooth_curve(history.history['val_acc'])

    if which_graph == 'loss':
        train = smooth_curve(history.history['loss'])
        valid = smooth_curve(history.history['val_loss'])

    plt.xkcd() # make plots look like xkcd
        
    epochs = range(1, len(train) + 1)
        
    trim = 5 # remove first 5 epochs
    # when graphing loss the first few epochs may skew the (loss) graph
    
    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', label=('Training'))
    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', linewidth=15, alpha=0.1)
    
    ax.plot(epochs[trim:], valid[trim:], 'g', label=('Validation'))
    ax.plot(epochs[trim:], valid[trim:], 'g', linewidth=15, alpha=0.1)

    
def get_max_validation_accuracy(history):
    validation = smooth_curve(history.history['val_acc'])
    ymax = max(validation)
    return 'Max validation accuracy ≈ ' + str(round(ymax, 3)*100) + '%'


# In[ ]:


def plot_history(history):    
    
    fig, (ax1, ax2) = plt.subplots(nrows=2,
                                   ncols=1,
                                   figsize=(10, 6),
                                   sharex=True,
                                   gridspec_kw = {'height_ratios':[5, 2]})

    set_plot_history_data(ax1, history, 'acc')
    
    set_plot_history_data(ax2, history, 'loss')
    
    # Accuracy graph
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(bottom=0.5, top=1)
    ax1.legend(loc="lower right")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    ax1.spines['bottom'].set_visible(False)
    
    # max accuracty text
    plt.text(0.97,
             0.97,
             get_max_validation_accuracy(history),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax1.transAxes,
             fontsize=12)

    # Loss graph
    ax2.set_ylabel('Loss')
    ax2.set_yticks([])
    ax2.plot(legend=False)
    ax2.set_xlabel('Epochs')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()


# ## Let's begin!

# In[ ]:


# labeled data
titanic_path = '../input/train.csv'

# unlabeled data -- we'll need to predict this data
titanic_unlabeled = '../input/test.csv'

df = pd.read_csv(titanic_path, quotechar='"')

predict = pd.read_csv(titanic_unlabeled, quotechar='"')


# In[ ]:


df.head(5)


# In[ ]:


labeled_rows = len(df)
labeled_rows


# In[ ]:


predict.insert(1, 'Survived', '?');


# In[ ]:


predict.head(5)


# In[ ]:


frames = [df, predict]
together = pd.concat(frames)


# In[ ]:


together[888:894]


# Keep only desired rows and rearrange for easier view:

# In[ ]:


preferredOrder = ['PassengerId', 'Sex', 'Pclass', 'Cabin', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare', 'Survived']
together = together[preferredOrder]


# In[ ]:


together.head(5)


# ## Create two new categories: alone & child

# In[ ]:


together['isalone'] = '?'

def set_is_alone(row):
    if row.SibSp >= 1 or row.Parch >= 1:
        return '0'
    else: 
        return '1'

together['isalone'] = together.apply(set_is_alone, axis=1)


# In[ ]:


together['ischild'] = '?'

def set_is_child(row):
    if row.Age < 18:
        return '1'
    else:
        return '0'

together['ischild'] = together.apply(set_is_child, axis=1)


# In[ ]:


# confirm we did what we wanted
together[['SibSp', 'Parch', 'isalone', 'Age', 'ischild']].head(10)


# In[ ]:


together.head(5)


# In[ ]:


together.shape


# ## Prepare categorical data

# We will use 6 categories. The four given:
#  - Sex
#  - Pclass
#  - Cabin
#  - Embarked
#  
# as well as the two we created
#  - is_child
#  - is_alone

# In[ ]:


# first four columns
categorical_data = preferredOrder[1:5] # ignore `PassengerId`

# and the two we created
categorical_data.append('ischild')
categorical_data.append('isalone')

categorical_data


# `Cabin` values need to be placed into categories.

# In[ ]:


together['Cabin'].values[:20]


# In[ ]:


def cleanCabin(el):
    if isinstance(el, str):
        return el[:1]
    else:
        return '0'
together['Cabin'] = together['Cabin'].apply(cleanCabin)
together['Cabin'].values[:20]


# Keras has `to_categorical` method for **one hot encoding** of data, but it requires natural number input.
# 
# We will create our own conversion

# In[ ]:


def convert_to_natural_number(x, temp_dict):
    if x in temp_dict:
        return temp_dict[x]
    else:
        temp_dict[x] = temp_dict['curr_count']
        temp_dict['curr_count'] += 1
        return temp_dict[x]


# In[ ]:


def categorical_column_to_number(col):
    temp_dict = temp_dict = {'curr_count': 0}
    together[col] = together[col].apply(convert_to_natural_number, args=(temp_dict,))


# In[ ]:


together[categorical_data].head(10)


# In[ ]:


for col in categorical_data:
    categorical_column_to_number(col)


# In[ ]:


together[categorical_data].head(10)


# Now we are ready to convert data to _one hot encoding_

# In[ ]:


together.shape


# In[ ]:


newDF = pd.DataFrame()


# In[ ]:


for col in categorical_data:
    one_hot = pd.get_dummies(together[col])
    one_hot = one_hot.add_prefix(col)
    for new_name in list(one_hot):
      newDF[new_name] = one_hot[new_name]


# In[ ]:


newDF.shape


# In[ ]:


newDF.head(5)


# In[ ]:


newDF.shape


# In[ ]:


print(list(newDF))


# Some columns have too-few elements and we'll drop them - you can't learn from too-few examples (they may be flukes and not statistically representative)

# In[ ]:


print('Column   \ttotal entrties')
print('-------------------------------')
for col in list(newDF):
    total = newDF[col].sum()
    print(col,':    ', '\t', total, sep='', end='')
    if (total < 10):
        newDF = newDF.drop([col], axis=1)
        print('\t<-- dropped', end='')
    print()


# In[ ]:


newDF.head(5)


# In[ ]:


one_hot_columns = list(newDF)
print(one_hot_columns)


# In[ ]:


result = pd.concat([newDF, together], axis=1, join_axes=[newDF.index])


# In[ ]:


result = result.drop(categorical_data, axis=1)


# In[ ]:


result[one_hot_columns].head(5)


# ## Prepare numerical data

# We will use these 4 as numbers:
#  - SibSp (# of **sib**lings / # of **sp**ouses)
#  - Parch (# of **par**ent / # of **ch**ildren)
#  - Age
#  - Fare

# In[ ]:


numerical_data = preferredOrder[5:9]
numerical_data


# In[ ]:


def normalize(x, colMax, colMean):
    if math.isnan(x):
        return 0
        # I have seen this approach instead but don't think it yields better results
        # return colMean
    if isinstance(x, float):
        return x / colMax
    elif isinstance(x, int):
        return float(x) / colMax
    else:
        return 0


# In[ ]:


def applyNormalize(col):
    column_max = result[col].max()
    column_mean = result[col].mean()
    result[col] = result[col].apply(normalize, args=(column_max, column_mean))


# In[ ]:


for col in numerical_data:
    applyNormalize(col)


# In[ ]:


result[numerical_data].head(3)


# In[ ]:


result[numerical_data].describe()


# Notice that max is now 1

# ## Split data for training

# In[ ]:


result[one_hot_columns].head(5)


# In[ ]:


result[numerical_data].head(5)


# **Randomize the order of all the rows**
# 
# _this is important because the original data may not be (and in this case is not) random_

# In[ ]:


rows_to_predict = result[labeled_rows:]


# In[ ]:


rows_to_predict.head(5)


# In[ ]:


result = result[:labeled_rows].sample(frac=1)


# In[ ]:


result.shape


# In[ ]:


x_cat_all = result[one_hot_columns]
x_num_all = result[numerical_data]

y_data_all = result['Survived']


# In[ ]:


# Here is how you can manually split the data into training & validation
# later we will use Keras's native method (making this process very simple)

# 80% for training, the rest for validation
cutoff = round(0.8 * len(x_cat_all)) 

# training
x_cat = x_cat_all[:cutoff]
x_num = x_num_all[:cutoff]
y_train = y_data_all[:cutoff]

# validation
x_cat_val = x_cat_all[cutoff:]
x_num_val = x_num_all[cutoff:]
y_validation = y_data_all[cutoff:]


# In[ ]:


print('left input:', len(one_hot_columns))
print('right input:', len(numerical_data))


# In[ ]:


x_cat.shape


# In[ ]:


x_num.shape


# ## Build two-input model

# In[ ]:


# backend.clear_session()

# categorical branch -- 'left'
left_in = Input(shape=(19,))

left1  = Dense(64, activation='relu')(left_in)
left1n = BatchNormalization()(left1)
left2  = Dense(32, activation='relu')(left1n)
left2n = BatchNormalization()(left2)
left3  = Dense(16, activation='relu')(left2n)
left3n = BatchNormalization()(left3)
left4  = Dense(8,  activation='relu')(left3n)
left4n = BatchNormalization()(left4)
left5  = Dense(4,  activation='relu')(left4n)

left_out  = Dense(1, activation='sigmoid')(left5)

# numerical branch -- 'right'
right_in = Input(shape=(4,))

right1  = Dense(64, activation='relu')(right_in)
right1a = BatchNormalization()(right1)
right2  = Dense(32, activation='relu')(right1a)
right2a = BatchNormalization()(right2)
right3  = Dense(16, activation='relu')(right2a)
right3a = BatchNormalization()(right3)
right4  = Dense(8,  activation='relu')(right3a)
right4a = BatchNormalization()(right4)
right5  = Dense(4,  activation='relu')(right4a)

right_out = Dense(1, activation='sigmoid')(right5)

# merge two branches
merge_in = concatenate([left_out, right_out])

dense1  = Dense(16, activation='relu')(merge_in)
dense2  = Dense(8,  activation='relu')(dense1)
dense2a = BatchNormalization()(dense2)
dense3  = Dense(8,  activation='relu')(dense2a)

output  = Dense(1, activation='sigmoid')(dense3)

model_new = Model(inputs=[left_in, right_in], outputs=output)

model_new.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# model_new.summary()


# In[ ]:


plot_model(model_new, show_shapes=True, to_file='model.png', show_layer_names=False)
Image('model.png', width=200)


# In[ ]:


history_final = model_new.fit([x_cat, x_num],
                              y_train,
                              epochs=150,
                              verbose=0,
                              batch_size=64,
                              validation_data=([x_cat_val, x_num_val], y_validation))


# In[ ]:


plot_history(history_final)


# Can be as high as 83% accurate

# In[ ]:


# model_new.save('model_new.h5')


# # Alternative approach
# Build two models, train them, and then build a third to fuse the two

# ## Built categorical model

# In[ ]:


x_data = result[one_hot_columns]
y_data = result['Survived']


# In[ ]:


print('Number of inputs:', len(one_hot_columns))


# In[ ]:


result[one_hot_columns].shape


# In[ ]:


# backend.clear_session()

model_cat = Sequential()

model_cat.add(Dense(64, activation='relu', input_shape=(19,)))
model_cat.add(Dense(32, activation='relu'))
model_cat.add(BatchNormalization())
model_cat.add(Dense(16, activation='relu'))
model_cat.add(BatchNormalization())
model_cat.add(Dense(8, activation='relu'))
model_cat.add(BatchNormalization())
model_cat.add(Dense(4, activation='relu'))
model_cat.add(BatchNormalization())
# model_cat.add(Dense(8, activation='relu'))
# model_cat.add(Dense(4, activation='relu'))
# model_cat.add(Dense(8, activation='relu'))
# model_cat.add(Dense(32, activation='relu'))
model_cat.add(Dense(1, activation='sigmoid'))

model_cat.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# model_cat.summary()


# ## Train categorical

# Note that unlike in the previous example, we are using Keras's method 
# ```
# validation_split=0.2
# ```
# instead of explicit validation
# ```
# validation_data=(x_validation, y_validation)
# ```

# In[ ]:


history_cat = model_cat.fit(x_data,
                            y_data,
                            validation_split=0.2,
                            epochs=150,
                            batch_size=64,
                            verbose=0)


# In[ ]:


plot_history(history_cat)


# Can be as high as 82%

# ## Build numerical

# In[ ]:


result[numerical_data].head(5)


# Split validation & training data

# In[ ]:


x_data = result[numerical_data]
y_data = result['Survived']


# In[ ]:


# backend.clear_session()

from keras.layers import Dropout

model_num = Sequential()

model_num.add(Dense(64, activation='relu', input_shape=(4,)))
model_num.add(Dropout(0.1))
model_num.add(Dense(32, activation='relu'))
model_num.add(Dropout(0.1))
model_num.add(Dense(16, activation='relu'))
model_num.add(Dropout(0.1))
model_num.add(Dense(8, activation='relu'))
model_num.add(Dense(4, activation='relu'))
# model_num.add(Dense(8, activation='relu'))
# model_num.add(Dense(4, activation='relu'))
# model_num.add(Dense(32, activation='relu'))
model_num.add(Dense(1, activation='sigmoid'))

model_num.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# model_num.summary()


# ## Train numerical

# In[ ]:


history_num = model_num.fit(x_data,
                            y_data,
                            epochs=300,
                            batch_size=len(x_data),
                            verbose=0,
                            validation_split=0.2)


# In[ ]:


plot_history(history_num)


# Can be as high as 77%

# ## Merge the two models

# In[ ]:


# one option is to freeze the first two models
# model_cat.trainable = False
# model_num.trainable = False
# but we will not
model_cat.trainable = True
model_num.trainable = True


# In[ ]:


merge = concatenate([model_num.output, model_cat.output])

# d1 =  Dense(8, activation='relu')(merge)
# # d1n = BatchNormalization()(d1)
# d2 =  Dense(16, activation='relu')(d1)
# # d2n = BatchNormalization()(d2)
# d3 =  Dense(8, activation='relu')(d2)
# # d3n = BatchNormalization()(d2)
# d4 =  Dense(4, activation='relu')(d3)

output = Dense(1, activation='sigmoid')(merge)

joint_model = Model(inputs=[model_num.input, model_cat.input], outputs=output)

joint_model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# joint_model.summary()


# In[ ]:


plot_model(joint_model, show_shapes=True, to_file='model2.png', show_layer_names=False)
Image('model2.png', width=200)


# Prepare checkpoint method for storing the best model

# In[ ]:


from keras.callbacks import ModelCheckpoint


# In[ ]:


filepath="{val_acc:.2f}-accuracy.h5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True,
                             mode='max')

callbacks_list = [checkpoint]


# In[ ]:


history_final = joint_model.fit([x_num, x_cat],
                                y_train,
                                epochs=300,
                                verbose=0,
                                batch_size=len(x_num),
                                callbacks=callbacks_list,
                                validation_split=0.2)


# In[ ]:


plot_history(history_final)


# Can be as high as 87%

# In[ ]:


# joint_model.save('joint_model.h5')


# ## Predict unlabeled data

# In[ ]:


from keras.models import load_model


# In[ ]:


# you would load the best saved model here (notice the custom callback that saves the best-performing model) 
# loaded_model = load_model('0.84-accuracy.h5')
# but for Kaggle kernel we'll just load the model as it is after the last epoch
loaded_model = joint_model


# In[ ]:


rows_to_predict.head(5)


# In[ ]:


rows_to_predict.shape


# In[ ]:


passenger_ids_to_predict = rows_to_predict['PassengerId']


# In[ ]:


prediction = loaded_model.predict([rows_to_predict[numerical_data], rows_to_predict[one_hot_columns]])


# In[ ]:


prediction.shape


# In[ ]:


prediction[:10]


# In[ ]:


prediction = (prediction > 0.5).astype(int).reshape(-1)


# In[ ]:


prediction[:10]


# In[ ]:


submission = pd.DataFrame({"PassengerId": passenger_ids_to_predict, "Survived": prediction})
submission.to_csv('submission.csv', index=False)


# # Archive

# ## Archive: Keras one-hot method

# In[ ]:


from keras.utils import to_categorical


# In[ ]:


for col in categorical_data:
    print(col)
    lol = (to_categorical(together[col]))
    print(lol[:5])


# ## Archive: another way to visually see your model
# 
# This method does not create a .png file and will render directly to Jupyter (but result will not be as detailed or pretty)
# 
# *note*: you'll need to install `graphviz` externally for this diagram to be generated

# In[ ]:


# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# SVG(model_to_dot(model_new).create(prog='dot', format='svg'))


# ## Archive: another way to plot
# simpler code but less pretty

# In[ ]:


def plot_this(training, validation, label):
    
    epochs = range(1, len(training) + 1)
    
    plt.clf() # clear out old
    
    plt.xkcd() # make look like xkcd
    
    # plt.figure(figsize=(8, 3)) # make wider
    
    trim = 10 # remove first 10 data points

    plt.plot(epochs[trim:], training[trim:], 'b', label=('Training '+label))
    plt.plot(epochs[trim:], validation[trim:], 'g', label=('Validation '+label))
    plt.title('Model ' + label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    
    if label == 'Loss':
        plt.yticks([])
        
    if label == 'Accuracy':
        plt.ylim(ymin=0.5)
        plt.ylim(ymax=1)
        ymax = max(validation[trim:])
        best = 'Max validation accuracy ≈ ' + str(round(ymax, 3)*100) + '%'
        plt.text(0, 0.35, best, fontsize=12)
        
    plt.legend()

    return plt


# In[ ]:


def plot_history_old(history):

    label1 = 'Loss'
    train1 = smooth_curve(history.history['loss'])
    valid1 = smooth_curve(history.history['val_loss'])

    plot_this(train1, valid1, label1).show()
    
    label2 = 'Accuracy'
    train2 = smooth_curve(history.history['acc'])
    valid2 = smooth_curve(history.history['val_acc'])
    
    plot_this(train2, valid2, label2).show()


# In[ ]:


plot_history_old(history_final)


# ## Archive: plot with seaborn

# In[ ]:


import seaborn as sns


# In[ ]:


facet = sns.FacetGrid(df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df['Age'].max()))
facet.add_legend()

