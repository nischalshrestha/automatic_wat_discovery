#!/usr/bin/env python
# coding: utf-8

# I thought it might be interesting to see what (if any) impact nationality/ethnicity had on survival. The training/test data supplied doesn't contain any explicit nationality info. Let's see if we can infer it from passengers' names. 
# 
# # Getting Nationalities from Names
# To start, let's extract first and last name. We'll ignore titles and nicknames and for married women, we'll use maiden names:

# In[ ]:


import re

def get_real_name(full_name):
    real_name = ''
    surname = full_name.split(', ')[0]
    first_name = ' '.join(full_name.split(', ')[1].split(' ')[1:])
    title = full_name.split(', ')[1].split(' ')[0]
    first_name = re.sub(r'\"[^"]*\"', '', first_name).replace('()', '')
    real_name = (first_name + ' ' + surname)
    if title == 'Mrs.' or title == 'Mme.' or title == 'Lady.' or title == 'the':
        p_start = first_name.find('(')
        p_end = first_name.find(')')
        if p_start != -1 and p_end != -1:
            real_name = first_name[(p_start+1):p_end]
            if len(real_name.split()) == 1:
                real_name = (real_name + ' ' + surname)
    real_name = ' '.join(real_name.split())
    return real_name


# <sup>\* *There were a handful of names with double quotes or parentheses in random places that I manually fixed.*</sup>
# 
# We can use [NamePrism](http://www.name-prism.com)'s api to estimate a passengers' ethnicity based on their name:

# In[ ]:


import requests
from requests.utils import quote

def get_nat_from_name_prism(name):
    request_url = 'http://www.name-prism.com/api/json/' + quote(name)
    r = requests.get(request_url)
    try:
        data = r.json()
        # data = {"European,SouthSlavs": 2.3147843795616316e-06, "Muslim,Pakistanis,Bangladesh": 1.0081305531161525e-05, "European,Italian,Italy": 3.856736678433249e-05, "European,Baltics": 7.713378348172508e-07, ...}
    except:
        print('ERROR')
        print(r)
        return None
    max_match = 0.0
    nat = ''
    for k,v in data.items():
        if data[k] > max_match:
            max_match = data[k]
            nat = k
    return nat


# NamePrism returns a list of nationalitiy probabilities for a given name. We'll just return the one with the highest probability. 
# 
# Next, let's tie everything together and write our results to a CSV:

# In[ ]:


import csv

file_name = 'pass_nationalities.csv'

def write_nationality_data(nat_data):
    with open(file_name, 'w') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(['PassengerId','Nationality'])
        for k,v in nat_data.items():
            writer.writerow([k,v])
    
         
# Useful for restarting where we left off in case of an exception
def read_nationality_data():
    res = {}    
    try:
        with open(file_name) as f1:
            reader = csv.DictReader(f1)
            for row in reader:
                res[row['PassengerId']] = row['Nationality']
    except:
        print('no file to start')
    return res


def extract_nat_data():
    known_nat_data = read_nationality_data() 
    files = ['train.csv', 'test.csv']
    for fname in files:
        with open(fname) as f1:
            reader = csv.DictReader(f1)
            for row in reader:
                pid = row['PassengerId']
                real_name = get_real_name(row['Name'])
                if not pid in known_nat_data:
                    nat_data = get_nat_from_name_prism(real_name)
                    if nat_data != None:
                        known_nat_data[pid] = nat_data
                write_nationality_data(known_nat_data) # write after each request in case program crashes


# \***I included a CSV of passenger id's to nationalities for the training set in the data section of this notebook. (see: *pass_nationalities.csv*) Feel free to use it in your own models. =)**
# 
# # Exploring Passenger Nationality
# Here's the results of our passenger nationality/ethnicity estimate based on the train and test data:
# 
# | Nationality                  | Pass. Count | % pop.      |
# |------------------------------|-------------|-------------|
# | CelticEnglish                | 747         | 0.570664629 |
# | European,French              | 106         | 0.080977846 |
# | Nordic,Scandinavian,Sweden   | 101         | 0.077158136 |
# | European,German              | 96          | 0.073338426 |
# | Hispanic,Spanish             | 55          | 0.042016807 |
# | EastAsian,Malay,Indonesia    | 25          | 0.019098549 |
# | Nordic,Finland               | 25          | 0.019098549 |
# | Nordic,Scandinavian,Norway   | 24          | 0.018334607 |
# | Hispanic,Portuguese          | 22          | 0.016806723 |
# | Muslim,Nubian                | 20          | 0.015278839 |
# | European,SouthSlavs          | 20          | 0.015278839 |
# | European,Italian,Italy       | 15          | 0.011459129 |
# | Nordic,Scandinavian,Denmark  | 15          | 0.011459129 |
# | Hispanic,Philippines         | 9           | 0.006875477 |
# | African,EastAfrican          | 5           | 0.00381971  |
# | EastAsian,Chinese            | 4           | 0.003055768 |
# | Muslim,Persian               | 3           | 0.002291826 |
# | Greek                        | 3           | 0.002291826 |
# | European,Italian,Romania     | 3           | 0.002291826 |
# | EastAsian,Indochina,Vietnam  | 2           | 0.001527884 |
# | SouthAsian                   | 2           | 0.001527884 |
# | EastAsian,South Korea        | 1           | 0.000763942 |
# | Muslim,Pakistanis,Bangladesh | 1           | 0.000763942 |
# | African,WestAfrican          | 1           | 0.000763942 |
# | Muslim,Turkic,Turkey         | 1           | 0.000763942 |
# | EastAsian,Malay,Malaysia     | 1           | 0.000763942 |
# | EastAsian,Japan              | 1           | 0.000763942 |
# | European,Russian             | 1           | 0.000763942 |
# 
# Considering the ship was traveling from England to New York, these seem like pretty reasonable results. 
# 
# Here's a more simplified version using Name Prism's taxonomy:
# <img src="https://plot.ly/~reppic/1.png?share_key=IKNDrw1herg1bBX6RLqrQt" alt="Fine Grain Titanic Nats" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" />
# <sup>*I included the 3 Geek passengers as part of the European block and the 2 SouthAsian passengers as part of the Asian block.</sup>
# <hr>
# 
# Now let's just examine the survival rate and average class of the passengers from the train.csv file: 
# 
# | Nationality                  | Pass. Count | Survived | Survival Rate | Avg. Pass. Class  |
# |------------------------------|-------------|----------|---------------|-------------|
# | CelticEnglish                | 490         | 204      | 0.416326531   | 2.104081633 |
# | European,French              | 78          | 38       | 0.487179487   | 2.282051282 |
# | European,German              | 70          | 27       | 0.385714286   | 2.285714286 |
# | Nordic,Scandinavian,Sweden   | 69          | 18       | 0.260869565   | 2.927536232 |
# | Hispanic,Spanish             | 39          | 19       | 0.487179487   | 2.256410256 |
# | Nordic,Finland               | 22          | 8        | 0.363636364   | 2.863636364 |
# | Nordic,Scandinavian,Norway   | 17          | 5        | 0.294117647   | 3           |
# | EastAsian,Malay,Indonesia    | 17          | 4        | 0.235294118   | 2.529411765 |
# | Hispanic,Portuguese          | 16          | 6        | 0.375         | 2.3125      |
# | European,SouthSlavs          | 16          | 1        | 0.0625        | 3           |
# | Nordic,Scandinavian,Denmark  | 12          | 0        | 0             | 3           |
# | Muslim,Nubian                | 12          | 2        | 0.166666667   | 2.833333333 |
# | European,Italian,Italy       | 9           | 4        | 0.444444444   | 2.333333333 |
# | Hispanic,Philippines         | 3           | 1        | 0.333333333   | 1.666666667 |
# | EastAsian,Chinese            | 3           | 2        | 0.666666667   | 3           |
# | African,EastAfrican          | 3           | 1        | 0.333333333   | 2           |
# | European,Italian,Romania     | 2           | 0        | 0             | 3           |
# | SouthAsian                   | 2           | 0        | 0             | 3           |
# | EastAsian,Indochina,Vietnam  | 2           | 1        | 0.5           | 3           |
# | Muslim,Persian               | 2           | 0        | 0             | 3           |
# | EastAsian,Malay,Malaysia     | 1           | 0        | 0             | 3           |
# | Greek                        | 1           | 0        | 0             | 3           |
# | Muslim,Turkic,Turkey         | 1           | 0        | 0             | 3           |
# | African,WestAfrican          | 1           | 0        | 0             | 3           |
# | Muslim,Pakistanis,Bangladesh | 1           | 1        | 1             | 3           |
# | European,Russian             | 1           | 0        | 0             | 3           |
# | EastAsian,South Korea        | 1           | 0        | 0             | 3           |
# <br>
# Again, we'll group passengers based on Name Prism's taxonomy for simplicity:
# <img src="https://plot.ly/~reppic/6.png?share_key=auUDOD6oV08bBSdzMeSmDc" alt="Survive Nat" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" />
# <img src="https://plot.ly/~reppic/8.png?share_key=ER8MZmDrglizKATCF9Rhfi" alt="Class by Nat 2" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" />
# 
# 

# # Conclusions
# It seems that nationality/ethnicity is predictive of survival rate but it does appear (somewhat) correllated with passenger class. 
# 
# **Potential Issues:**<br>
# In case it wasn't already obvious, these are *really* rough estimates. 
# * The sample sizes for a majority of the groups were very small.
# * Name Prism returns probabilities and often the top few results are really close together.
# * From what I can gather Name Prism was trained with data collected recently. It may not be accurate for names >100 years old.
# * Even if we could assume Name Prism was 100% accurate, trying to infer someone's ethnicity from their name is definetely not an exact science. (Ex: "Anne McGowan" was labeled "CelticEnglish" but really, she could have been from anywhere.)
# 
# **Acknowlegenemts:**<br>
# *HUGE* thanks to Junting Ye, Shuchu Han, Yifan Hu, Baris Coskun, Meizhu Liu, Hong Qin and Steven Skiena for creating Name Prism, which is what's really doing all of the heavy lifting here.
# 
# **Thanks for reading!** <br>
# I'm pretty new to this whole field so **let me know what I got wrong and what I can do better**! Feel free to use the data in *pass_nationalities.csv* for your own analysis and models.
