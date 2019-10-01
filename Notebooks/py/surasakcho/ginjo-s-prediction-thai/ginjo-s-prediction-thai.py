#!/usr/bin/env python
# coding: utf-8

# **จากจิงโจถึงลูกหลานชาวดาว Data**
# พื้นที่นี้ คือ พื้นที่่ส่วนตัวที่จิงโจใช้ในการเรียนรู้ เพื่อหาคำตอบว่าภรรยาและลูกน้อยของเขาจะยังรอดชีวิตอยู่รึไม่
# 
# ########################################################
# 
# จิงโจ คือ ชาวดาว Data ที่อาศัยอยู่บนโลกในช่วงต้นศตวรรษที่ 20 ซึ่งเป็นช่วงเดียวกันที่เรือ Titanic เกิดการอัปปาง
# เป็นความโชคร้ายของเขา ที่ภรรยาและลูกน้อยของเขาขึ้นเรือ Titanic และต้องประสบกับเหตุร้ายครั้งนั้น
# เขาพยายามใช้ข้อมูลในการ predict ว่าภรรยาและลูกน้อยที่หายสาบสูญไปของเขานั้น แท้จริงแล้วจะยังมีชีวิตอยู่รึไม่
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #import graph library
import seaborn as sns #import graph library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.y


# In[ ]:


#อ่านข้อมูลจาก train.csv มาเก็บไว้ใน dataframe
df_train = pd.read_csv('../input/train.csv') 

#ลองดูจำนวน record, data type, NaN 
df_train.info()


# In[ ]:


#นับrecordที่เป็น NaN ในแต่ละ record
df_train.isnull().sum()


# In[ ]:


#ลองดูสถิติคร่าวๆขอข้อมูลก่อน
df_train.describe(include='all')


# In[ ]:


#ลองดูหน้าตาของข้อมูลคร่าวๆ
df_train.head().T


# **ข้อสังเกต**
# - ข้อมูลมีทั้งแบบตัวเลข และข้อความ, result ที่เราอยาก predict 'SURVIVED' น่าจะเป็น binary (0 / 1)
# - พอเดาได้ว่ามี NaN (blank) ปนอยู่ในหลายๆ column
# - 'Name' มีคำนำหน้าปนอยู่ในชื่อด้วย อาจจะแยกออกมาเป็นอีก column ได้
# - 'Age','Cabin','Embarked' มี NaN ด้วย อาจจะต้อง clean
# - 'Ticket' ดูมั่วๆ อาจจะต้องลองหา pattern
# - 'Cabin' เหมือนจะแบ่งเป็นกรุ๊ปใหญ่ๆ A,B,C,... ได้ด้วยตัวหนังสือตัวแรก
# 
# 

# In[ ]:


#ลองไอ้ที่น่าจะง่ายสุดก่อน นั่นคือ แยก group ให้ cabin โดย substring ตัวแรกออกมาเป็นอีก column
df_train['CabinGrp'] = df_train.loc[:,'Cabin'].str[:1]
df_train


# In[ ]:





# In[ ]:


#สร้างfunction สำหรับดึง CabinGrp ออกมา เผื่อใช้ซ้ำตอน test
def CreateCabinGrp(df):
    df['CabinGrp'] = df.loc[:,'Cabin'].str[:1]
    return df


# In[ ]:


#ลอง plot histogram ข้อมูลทั้งหมดที่ plot ได้ดู
df_train.hist(figsize=(10,10))


# **ข้อสังเกต**
# - คนส่วนใหญ่อายุประมาณ 20-40
# - Fare ถูกๆ เยอะสุด
# - Pclass 3 เยอะสุด  => สันนิษฐานได้ว่า Pclass 3 น่า Fare จะถูกสุด 
# - คนส่วนใหญ่มาคนเดียว

# In[ ]:


#ลอง plot histogram แยก column เทียบกันระหว่างรอด(ฟ้า)กับไม่รอด (แดง)
#สร้าง DF แยกชุดกันระหว่าง รอดกับไม่รอด
surv = df_train[df_train['Survived']==1]
nosurv = df_train[df_train['Survived']==0]
surv_col = "blue"
nosurv_col = "red"

#plot Age
binsrange = range(-10,90,2)
plt.figure(figsize=[10,10])
#fill NaN ด้วย -10 จะได้เห็นภาพว่าพวกที่ Age ว่างๆรอดกันเยอะไหม
sns.distplot(surv['Age'].fillna(-10).values, bins=binsrange, kde=True,color=surv_col,axlabel='Age')
sns.distplot(nosurv['Age'].fillna(-10).values,bins=binsrange, kde=True, color=nosurv_col)




# **ข้อสังเกต**
# - เด็กๆอายุไม่เกิน 15 ส่วนใหญ่จะรอด (histogram และ เส้น kde - kernel density estimation สีแดงอยู่ต่ำกว่าสีน้ำเงิน)
# - พวก NaN (ในกราฟคือ -10) ส่วนใหญ่จะไม่รอด แต่อาจจะต้องหาทางจำแนกอีกทีนึง 
# - ที่เหลือดูไม่ต่างกันมาก
# - ถ้าหาทางระบุอายุให้พวกที่ NaN ได้ ก็น่าจะช่วยได้เยอะ โดยเฉพาะถ้าอายุจริงๆน้อยกว่า 15 
# 
# **Idea**
# - คุ้นๆว่าในชื่อจะมีคำนำหน้าอยู่ เมื่อกี้แอบแวบไปดูมา นอกจาก Mr., Miss, Mrs. แล้ว  พวกเด็กๆจะมีคำนำหน้าเหมือนกัน เช่น Master (ตัวอย่างเช่น เด็กอายุ 2 ขวบชื่อ "Palsson, Master. Gosta Leonard") ลองหาทาง extract คำนำหน้าออกมา อาจจะใช้บอกช่วงอายุให้พวกที่เป็น NaN ได้  เท่าที่สังเกต pattern ของชื่อ จะเป็น "<นามสกุล>, <คำนำหน้า>. <ชื่อต้น> <ชื่อกลาง>" ถ้าเราอยากแยกคำนำหน้าออกมา ก็น่าจะพอใช้ , กับ . เป็นตัวแยกได้ 

# In[ ]:


#ทดสอบการแยกคำนำหน้า
tmpName = 'Palsson, Master. Gosta Leonard'
tmpName.split(',')[1].split('.')[0].strip()


# In[ ]:


#ใช้ apply กะ lambda มาช่วยในการแยกคำนำหน้า
df_train['Title'] = df_train['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
df_train['Title'].head()


# In[ ]:


#แอบเช็คว่า คำนำหน้ามีไรบ้าง รอดต่างกันยังไง
df_train.loc[:,('PassengerId','Title','Survived')].groupby(['Title','Survived']).agg(['count'])


# **ข้อสังเกต**
# - โห เจอแปลกๆเพียบ 

# In[ ]:


#พวกที่ Age เป็น NaN มี title อะไรบ้าง
df_train.loc[:,('PassengerId','Title')][df_train['Age'].isnull()].groupby(['Title']).agg(['count'])


# **ข้อสังเกต**
# - มาทั้ง Master, Miss, Mr, Mrs แล้วก็มี Dr ติดมาด้วยแฮะ
# 

# In[ ]:


#ก่อนจะตัดสินใจทำอะไรกับ Age ที่ว่างๆไป กลับไปเช็คอีกรอบ ก่อนว่าคำนำหน้าแต่ละอันมีช่วงอายุเท่าไหร่กันบ้าง
df_train.loc[:,('Age','Title')].groupby(['Title']).agg(['min','max','count','mean','median'])


# **ข้อสังเกต**
# - Master น่าจะหมายถึงเด็กชัวร์ เพราะ max ยังไม่ถึง 12 เลย
# - ถ้าจะลองหาค่าไปใส่ให้พวกที่อายุNaN ดูแล้ว median ก็น่าจะใช้เป็นตัวแทนของอายุได้

# In[ ]:


#สร้าง DF ขึ้นมาเก็บ Median ของ Age ของแต่ละ Title 
AgeTitleMed_tmp = df_train.loc[:,('Age','Title')].groupby(['Title']).agg(['median'])
#ใช้ squeeze ช่วยลด dimension ของ aggregate result และสร้างเป็น series ใหม่
AgeTitleMedian = pd.Series(data=np.squeeze(AgeTitleMed_tmp.values),index=AgeTitleMed_tmp.index)
print(AgeTitleMedian)




# In[ ]:


#Python มีความสามารถในการ fill na แบบแยกตาม index ได้ เราใช้มันให้เป็นประโยชน์ โดยเปลี่ยน index ให้เป็น Title 
#งั้นเรามาทดลองการ fill na Age ตาม Title กันดีกว่า
df_tmp = df_train.copy() #สร้าง DF อันใหม่ขึ้นมาอีกอัน กันพัง 555
df_tmp = df_tmp.set_index('Title') #set Title ให้เป็น index
df_tmp['AgeAdj'] = df_tmp['Age']
df_tmp['AgeAdj'].fillna(AgeTitleMedian)[df_tmp['Age'].isnull()] #ทดลอง fill na ด้วย median ที่เราหามา


# **ข้อสังเกต**
# - ผลลัพธ์ดูดี งั้นลุยโลด

# In[ ]:





# In[ ]:


#สร้าง function สำหรับ fill age ขึ้นมา จะได้เรียกใช้ซ้ำได้ง่ายๆตอน test
def AgeFill(df,AgeTitle):    
    df = df.reset_index()
    df = df.set_index('Title')
    df['Age'] = df['Age'].fillna(AgeTitle)
    df = df.reset_index()
    return df


# In[ ]:





# In[ ]:


df_train = AgeFill(df_train,AgeTitleMedian) #fill na ของ age ด้วย median ของแต่ละ title
df_train


# In[ ]:


df_train.isnull().sum() #ลองเช็คอีกทีว่า NaN ยังเหลือตรงไหนบ้าง


# **ข้อสังเกต**
# - Age ไม่มี NaN แล้ว
# - CabinGrp derive มาจาก Cabin ซึ่ง NaN ซะเยอะเลย ยัง no idea แฮะ
# - Embarked (ท่าเรือที่ผู้โดยสารขึ้น) นี่ดูน่าสนใจ  ลองดู embarked กันต่อดีกว่า

# In[ ]:


#ลองมา pivot embark กะ survived ดู
df_train.loc[:,('PassengerId','Survived','Embarked')].groupby(['Embarked','Survived']).agg('count')


# **ข้อสังเกต**
# - เหมือน C จะรอดเยอะกว่าเพื่อนแฮะ
# - S ประมาณ 2 ใน 3 ไม่รอด
# - Embarked อาจจะมีผลต่อการรอดจริงๆ ฉะนั้นไอ้ที่ข้อมูลหายไป 2 คน ควรจะต้องหาทาง clean
# - แอบเดาว่า Embark อาจจะเกี่ยวกับ Cabin ก็ได้ อาจจะมีการกำหนด Cabin สำหรับคนที่ขึ้นเรือในแต่ละ port จะได้ไม่วุ่นวายเดินมั่วไปทั้งเรือตอนที่ขึ้น

# In[ ]:


#ลอง factor plot ของ embarked
sns.factorplot(x = 'Embarked',y="Survived", data = df_train,color="r");


# **ข้อสังเกต**
# - ค่อนข้างชัดเจนทีเดียวว่า C จะรอดเยอะกว่าเพื่อนเลย 
# - S นี่หนักสุด

# In[ ]:


#ลองมา pivot embark กะ cabin ดู
df_train.loc[:,('PassengerId','CabinGrp','Embarked')].groupby(['Embarked','CabinGrp']).agg('count')


# **ข้อสังเกต**
# - Hopeless... หา pattern อะไรไม่ได้เลย  สงสัยต้องทำใจ
# - อ๊ะ นึกได้อีกอย่าง CabinGrp อาจจะเกี่ยวกับ pclass ก็ได้ ลองหาความสัมพันธ์กันดีกว่า

# In[ ]:


#ลองมา pivot pclass กะ cabingrp ดู
df_train.loc[:,('PassengerId','CabinGrp','Pclass')].groupby(['Pclass','CabinGrp']).agg('count')


# **ข้อสังเกต**
# - Pclass 2 กะ 3 ไม่มี CabinGrp เป็น A,B,C เลยแฮะ
# - แต่ถ้าคิดจะ clean CabinGrp โดยดูจาก Pclass ก็ยังแอบยาก สงสัยต้องทำใจปล่อยไป

# In[ ]:


#กลับมาดู column อื่นกันบ้าง เริ่มที่ Fare
#ลอง plot graph ดูดีกว่า
plt.figure(figsize=[10,10])
sns.distplot(surv['Fare'].fillna(-20).values, kde=False, color=surv_col,axlabel='Fare')
sns.distplot(nosurv['Fare'].fillna(-20).values, kde=False, color=nosurv_col)


# **ข้อสังเกต**
# - ข้อมูลแอบกระจุกอยู่ที่ตั๋วราคาถูก ลองเปลี่ยน scale เป็น log อาจจะดูสวยขึ้น

# In[ ]:


#
plt.figure(figsize=[10,10])
#sns.distplot(np.log(surv['Fare'].values), kde=False, color=surv_col,axlabel='Fare') Not working because some records contain 0 which log of 0 = -infinity
#sns.distplot(np.log(nosurv['Fare'].values), kde=False, color=nosurv_col) Not working because some records contain 0 which log of 0 = -infinity

sns.distplot(np.log(surv['Fare'].values + 1), kde=False, color=surv_col,axlabel='Fare') #add a small number to prevent -in
sns.distplot(np.log(nosurv['Fare'].values + 1), kde=False, color=nosurv_col)


# **ข้อสังเกต**
# - ดูกระจายดีขึ้นหน่อย แต่เหมือนยังกระจุกกันตรง Fare น้อยๆ  เด๋วลอง log10 ดูเผื่อจะกระจายดีขึ้น
# - ตรงแถวๆ 0 นี่น่าจะเป็น NaN แล้วเจอเรา treat ด้วย + ไป
# - ขนาดของ bin ดูมั่วๆด้วย คงต้องปรับขนาด bin

# 

# In[ ]:


#
binsrange = np.arange(0,3,0.2)
plt.figure(figsize=[10,10])
sns.distplot(np.log10(surv['Fare'].values + 1),bins = binsrange, kde=False, color=surv_col,axlabel='Fare') #add a small number to prevent -in
sns.distplot(np.log10(nosurv['Fare'].values + 1),bins = binsrange, kde=False, color=nosurv_col)


# **ข้อสังเกต**
# - ดูๆไปเหมือน log เฉยๆจะดีกว่านะนี่ งั้นใช้ log เฉยๆก็พอ

# In[ ]:


df_train['LogFare'] = np.log(df_train['Fare'] + 1) #สร้าง column ใหม่ LogFare


# In[ ]:


#มีอีกอย่างที่อยากลอง คือ Sex กับ Embarked จะสัมพันธ์กับตัวแปรไหนเป็นพิเศษไหม
#แต่เนื่องจากทั้ง 2 column เป็น string ต้องหาทางแปลงให้เป็นตัวเลขก่อน 
#ลองมาใช้ pd.factorize() เพื่อแปลง string เป็นตัวเลขกัน
df_temp = df_train.copy()
gender_id,gender_desc  = pd.factorize(df_temp.loc[0:10,'Sex'])
print(df_temp.loc[0:10,'Sex'])
print(gender_id)
print(gender_desc.values)


# **ข้อสังเกต**
# - ผลลัพธ์ใช้ได้ งั้นจัดไป

# In[ ]:


df_train['GenderId'], gender_desc = pd.factorize(df_train['Sex'])
df_train['EmbarkedId'], embarked_desc = pd.factorize(df_train['Embarked'])
df_train.loc[:,('GenderId','Sex','EmbarkedId','Embarked')].head(20)


# In[ ]:


#นึกได้ว่าจริงๆก็น่าจะลองหาความสัมพันธ์ระหว่างแต่ละ column ด้วยกันเองดูด้วย
#ใช้ correlation matrix ละกัน
corr = df_train.corr()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(corr)
plt.legend()
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns);


# In[ ]:


#ลองใช้ heatmap ดูบ้าง
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


#ไปเจอของน่าสนใจมาจาก stackoverflow
#https://stackoverflow.com/questions/34556180/how-can-i-plot-a-correlation-matrix-as-a-set-of-ellipses-similar-to-the-r-open
#เป็น chart อีกแบบเรียกว่า EclipseMatrix

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection

def plot_corr_ellipses(data, ax=None, **kwargs):

    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec


# In[ ]:


data = df_train.corr()
fig, ax = plt.subplots(1, 1,figsize=[12,10])
m = plot_corr_ellipses(data, ax=ax, cmap='seismic')
cb = fig.colorbar(m)
cb.set_label('Correlation coefficient')
ax.margins(0.1)


# **ข้อสังเกต**
# - อันที่สะดุดตาอันแรกๆ คือ Fare กับ Pclass ดูสัมพันธ์กันชัดเจน ซึ่งก็ make sense นะ 
# - LogFare กับ Fare ละไว้ในฐานที่เข้าใจนะว่ามันต้องสัมพันธ์กันอยู่แล้ว
# - นั่นไง Gender ก็ดูสัมพันธ์กับ Survived ระดับหนึ่ง ก็ make sense เช่นกัน ผู้หญิงอาจจะได้รับความช่วยเหลือมากกว่าผู้ชายเลยรอดเยอะกว่า
# - รองลงมาเป็น Pclass กับ Survived ก็ดูสัมพันธ์กันแบบสวนทางกัน Pclass เลขต่ำๆ => เป็นพวกชั้นสูง => รอดเยอะ
# (โลกช่างโหดร้ายกับคนจน...)
# - ถัดมาก็เป็น Age กับ Survived และ Age กับ Pclass ก็พอเข้าใจได้พวกอายุเยอะกว่าก็น่าจะมีตังค์ซื้อตั๋วชั้นสูงๆ
# - SibSp ก็สัมพันธ์กับอายุ 
# - ดูไปดูมา Parch ก็เหมือนจะสัมพันธ์กับ Sibsp แต่ก็คงไม่แปลกอะไร อาจจะมากันเป็นครอบครัว พ่อแม่ลูก ไรงี้
# -  ส่วน Embarked นี่ดูไม่เกี่ยวไรเลย แต่ก่อนหน้านี้เราเคยดูมาแล้วว่า Embarked มีผลกับ Survived อาจจะเพราะข้อมูล EmbarkedId ไม่ได้เรียงแบบเหมาะส

# In[ ]:


#ยังคาใจกับ Embarked ที่ขาดหายไป 2 record อยู่
#ลองหาความสัมพันธ์ของ Embarked กะ field อื่นๆดูดีกว่า
#ลอง factor plot ของ embarked เทียบกับ field อื่นๆที่เป็น numberical
sns.factorplot(x = 'Embarked',y="Pclass", data = df_train,color="r");
sns.factorplot(x = 'Embarked',y="GenderId", data = df_train,color="r");
sns.factorplot(x = 'Embarked',y="Age", data = df_train,color="r");
sns.factorplot(x = 'Embarked',y="SibSp", data = df_train,color="r");
sns.factorplot(x = 'Embarked',y="Parch", data = df_train,color="r");
sns.factorplot(x = 'Embarked',y="Age", data = df_train,color="r");
sns.factorplot(x = 'Embarked',y="Fare", data = df_train,color="r");




# In[ ]:


**ข้อสังเกต**
- Pclass กับ Fare น่าสนใจมาก! ค่อนข้างชัดเจนว่าใกล้ชิดกับ Embarked 


# In[ ]:


#ลอง boxplot แบบดู Embarked คู่กับ Fare และ Pclass กันดูบ้าง
plt.figure(figsize=[10,10])
sns.boxplot(x = 'Embarked',y="Fare",hue='Pclass', data = df_train);
#sns.boxplot(x = 'Embarked',y="GenderId", data = df_train,color="r");
#sns.boxplot(x = 'Embarked',y="Age", data = df_train,color="r");
#sns.boxplot(x = 'Embarked',y="SibSp", data = df_train,color="r");
#sns.boxplot(x = 'Embarked',y="Parch", data = df_train,color="r");
#sns.boxplot(x = 'Embarked',y="Age", data = df_train,color="r");
#sns.boxplot(x = 'Embarked',y="Fare", data = df_train,color="r");


# In[ ]:


**ข้อสังเกต**
- เห็นกลุ่มชัดเจนระหว่างแต่ละ Pclass
- น่าสงสัยอีกอย่างคือ เหมือนมี outlier Fare กระเด็นออกมา จาก Pclass 1 ลองดูหน่อยว่ามีอะไรแปลกๆที่ record พวกนี้ไหม


# In[ ]:


df_train.sort_values(by='Fare',ascending=False)


# **ข้อสังเกต**
# - งานเข้า พอมาดูดีๆกลายเป็นว่า Fare ไม่ใช่ราคาตั๋วต่อคน แต่เป็นราคาตั๋วแบบรวมกลุ่ม (ดูหมายเลข Ticket จะเจอว่า เลขเดียวกันเพียบเลย
# - ถ้าจะเอาราคาตั๋วมาคิด อาจจะต้องหาราคาตั๋วต่อคนเพิ่มซะแล้ว

# In[ ]:


fig, axes = plt.subplots(nrows=3,ncols=3,figsize=(10,10))
axes[0,0].violinplot(df_train['Pclass'], pos, points=20, widths=0.3,
                      showmeans=True, showextrema=True, showmedians=True)


# 
# 
# 

# In[ ]:





# In[ ]:




