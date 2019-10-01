#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import tree


# ## Step 1: Get the raw data
# Di kaggle prediction titanic survival, terdapat dua data yang akan digunakan: test.csv dan train.csv

# In[ ]:


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


# # Kenali datanya
# kita pahami dulu definisi dan kata kunci dari tiap kolomnya:
# 
# | Variable | Definition | Key |
# |----------|------------|-----|
# | Survival | Survival | 0 = No, 1 = Yes |
# | pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |
# | sex | Gender | |
# | Age | Age in years | |
# | sibsp | # of siblings / spouses aboard the Titanic | |
# | parch | # of parents / children aboard the Titanic | |
# | ticket | Ticket number | |
# | fare | Passenger fare | |
# | cabin | Cabin number | |
# | embarked | Port of embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

# In[ ]:


# Visual tabel dari train, 8 baris teratas
train.head(8)


# In[ ]:


# Info jumlah data terisi dan type data dari masing-masing kolom di dataset train
train.info()


# Pada dataset train, ditemukan pada kolom Age hanya terisi 714/891 data, yang lainnya 0 atau NaN, begitu pula dengan cabin dan Embarked, 204/891 dan 889/891. Hal ini akan sering ditemui oleh Data Scientist dan sebelum masuk ke model, hal ini harus diselesaikan terlebih dahulu.
# 
# ## Step 2: Data Cleaning
# Imputing data NaN dengan mode, median atau mean.

# In[ ]:


# Impute the Embarked variable
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Impute the Age variable
train['Age'] = train['Age'].fillna(train['Age'].mean())

# Impute the cabin variable
train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mode()[0])


# In[ ]:


# Cek perubahannya
train.info()


# Terlihat disini semua kolom telah terisi denngan data, tanpa NaN (non-null).
# 
# ## Step 3: Featuring Engineering
# Melakukan konversi kategori ke numerik, menambahkan predictor (extract), mengurangi feature/kolom yang tidak berpengaruh pada prediksi.
# Mengapa data categorycal perlu kita konversi ke numerik, ya namanya komputer itu sebenernya kalkulator dia akan lebih mudah mencerna data yang berupa angka, mungkin gampangnya begitu. Definisi ini masih sebatas pemahaman saya yang saat ini saya tangkap ya, cmiiw.

# In[ ]:


# convert the category of male and female to integer form
train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1


# In[ ]:


# Convert the Embarked classes to integer form
# S = 0
# C = 1
# Q = 2
train['Embarked'][train['Embarked'] == 'S'] = 0
train['Embarked'][train['Embarked'] == 'C'] = 1
train['Embarked'][train['Embarked'] == 'Q'] = 2

train.head(8)


# In[ ]:


# Analisa apakah gender berpengaruh terhadap prediksi survival?
# Penumpang yang hidup vs mati
print(train['Survived'].value_counts(normalize = True))

# Penumpang kelas 1 yang hidup vs mati
print(train['Survived'][train['Pclass'] == 1].value_counts(normalize = True))

# Penumpang kelas 2 yang hidup vs mati
print(train['Survived'][train['Pclass'] == 2].value_counts(normalize = True))

# Penumpang kelas 3 yang hidup vs mati
print(train['Survived'][train['Pclass'] == 3].value_counts(normalize = True))


# Secara total, ditemukan fakta pada data train.csv, penumpang yang berhasil selamat hanya 38%.
# Jika dibreakdown berdasarkan kelas tiket, sbb:
# 
# | Kelas | Selamat | Meninggal |
# |-------|---------|-----------|
# | Kelas 1 | 63% | 37% |
# | Kelas 2 | 47% | 53% |
# | Kelas 3 | 24% | 76% |
# 
# Karena tiap kategori kelas memiliki proporsi yang tidak seragam, disimpulkan sementara kategori kelas ini akan dapat menjadi predictor.
# 
# Analisa kolom nama belum dapat di extract informasi apa yang dapat membantu. Nama-nama disini juga tidak dapat disederhanakan menjadi category. SKIP DULU.

# In[ ]:


# Penumpang laki-laki yang hidup vs mati
print(train['Survived'][train['Sex'] == 0].value_counts(normalize = True))

# Penumpang perempuan yang hidup vs mati
print(train['Survived'][train['Sex'] == 1].value_counts(normalize = True))


# Jika dibreakdown berdasarkan gender akan nampak seperti tabel dibawah:
# 
# | gender | Selamat | Meninggal |
# |--------|---------|-----------|
# | Lelaki | 20% | 80% |
# | Wanita | 74% | 26% |
# 
# Dengan perbandingan seperti tabel diatas, dapat disimpulkan sementara bahwa gender dapat dijadikan sebagai predictor yang cukup bagus. Artinya dapat memberikan akurasi yang cukup tinggi (apapun model yang dipilih nantinya).

# ## Does age play a role?
# Dalam beberapa blog yang membahas dataset titanic ini ada yang membahas kemungkinan umur berperan penting dalam prediksi ini. Ada kemungkinan pada masanya dulu, anak-anak diselamatkan terlebih dahulu daripada penumpang dewasa. Hal ini dapat dilakukan test seperti diatas. Tetapi sebelumnya kita harus melakukan konversi umur kedalam kategori anak-anak (1) dan dewasa (0). Akan masuk ke kategori anak-anak jika umurnya kurang dari 18 tahun, dewasa lebih atau sama dengan 18 tahun.

# In[ ]:


# Buat kolom baru dengan nama Child, type data float
train['Child'] = float('NaN')

# masukan kategori sesuai penjelasan diatas
train['Child'][train['Age'] < 18] = 1
train['Child'][train['Age'] >= 18] = 0

train.head(8)


# In[ ]:


# Penumpang anak-anak hidup vs mati
print(train['Survived'][train['Child'] == 1].value_counts(normalize = True))

# Penumpang dewasa hidup vs mati
print(train['Survived'][train['Child'] == 0].value_counts(normalize = True))


# Seperti sebelumnya, kita rangkum dalam tabel sbb:
# 
# | Kategori | Selamat | Meninggal |
# |----------|---------|-----------|
# | Anak2 | 54% | 46% |
# | Dewasa | 36% | 64% |
# 
# Sebenarnya, kayaknya sih ya, anak-anak tidak signifikan, hampir imbang antara yang selamat ama yang meninggal. Tetapi untuk penumpang dewasa cukup berdampak. Ya kita coba masukan dahulu ke dalam predictor.

# ## Step 4: Bikin model prediksinyaaa...
# ### Kita coba model decision tree
# 
# Kita akan menggunakan module scikit-learn untuk memanggil algoritma decision tree, dan akan dibantu pula dengan modul numpy library. Kita tinggal panggil class DecisionTreeClassifier dari scikit. Dalam algoritma ini kita akan butuh (paling tidak) dua argumen:
# - target: Sebuah one-dimensional numpy array yang berisikan target prediksi/response dari data training kita, dalam kasus kita adalah prediksi kolom survived.
# - features: Sebuah multidimensional numpy array yang berisikan feature/predictor yang telah kita siapkan di data training.
# 
# Dalam model machine learning, atau untuk melihat performa dari model, kita dapat melakukan cek tinggi pengaruhnya feature yang telah kita masukkan pada model terhadap hasil prediksi. Bisa jadi, kolom Pclass, Sex, Age dan Fare yang kita jadikan predictor di awal ternyata tidak berpengaruh signifikan. Fungsi lainnya, dari keempat predictor, kita dapat lakukan ranking, predictor mana yang paling penting. Hal ini dapat dilakukan dengan .feature_importance_ pada algoritma kita nantinya.

# In[ ]:


target = train['Survived'].values
features_one = train[['Pclass', 'Sex', 'Age', 'Fare']].values # dari artikel lain ada yang memasukan fare tetapi belum nemu penjelasan logisnya
# (atau saya aja yg belum paham) dan fare ini dimasukan tanpa kategori seperti pada kolom umur. kita lihat aja dulu.


# In[ ]:


# My first decision tree
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)


# In[ ]:


# Cek feature importance dari 4 predictor, dengan urutan: Pclass, Sex, Age lalu Fare
print(my_tree_one.feature_importances_)


# Ternyata feature Sex adalah predictor yang paling berpengaruh dan paling penting, disusul dengan kolom Fare (cukup mengejutkan, nanti coba dipahami dulu kenapa ini jadi penting).

# In[ ]:


# Cek score dari data training pada train menggunakan model Decision Tree
my_tree_one.score(features_one, target)


# Score yang cukup menjanjikan: 98%

# ## Start predict
# 
# Dengan model decision tree yang sudah jadi tadi, kita dapat melakukan "generate" prediksi kita pada data test tanpa harus input masing-masing data (Alhamdulillah).
# 
# Kita akan gunakan .predict() method. Dari model decision tree yang sudah dibuat (my_tree_one) pada data test dengan feature yang sama dengan train.

# In[ ]:


# Persiapkan data test nya, seperti pada data train
test['Sex'][test['Sex'] == 'female'] = 1
test['Sex'][test['Sex'] == 'male'] = 0
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode()[0])
test.Fare[152] = test['Fare'].median()
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values # Features yang sama dengan training


# In[ ]:


test.info()


# In[ ]:


# Prediksi, hasil dari training diterapkan pada data test. my_tree_one adalah hasil dari training, menggunakan atribut predict dengan argumen test_features
# yang berisi data dari test.csv
my_prediction = my_tree_one.predict(test_features)


# In[ ]:


# Karena kaggle menentukan yang di upload hanya dua kolom: PassengerId dan Survived, maka kita bikin dataframe baru yang berisikan hasil prediksi
PassengerId = np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns= ['Survived'])


# In[ ]:


# lakukan konversi ke format .csv
my_solution.to_csv('my_solution_one.csv', index_label=['PassengerId'])


# ### Selesaii...
# ### Maaf kalau kayak anak Jaksel yang campur bahasa dan english :*
