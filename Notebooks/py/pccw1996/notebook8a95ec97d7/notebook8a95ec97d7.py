#!/usr/bin/env python
# coding: utf-8

# learning

# In[ ]:


{"cells":[{"cell_type":"markdown","metadata":{"_cell_guid":"c9c0744f-9e2d-223c-acce-38c9ef8235d2"},"source":["# Test 1"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"5767a33c-8f18-4034-e52d-bf7a8f7d8ab8"},"outputs":[],"source":["# data analysis and wrangling\n","import pandas as pd\n","import numpy as np\n","import random as rnd\n","\n","# visualization\n","import seaborn as sns\n","import matplotlib.pyplot as plt\n","%matplotlib inline\n","\n","# machine learning\n","from sklearn.linear_model import LogisticRegression\n","from sklearn.svm import SVC, LinearSVC\n","from sklearn.ensemble import RandomForestClassifier\n","from sklearn.neighbors import KNeighborsClassifier\n","from sklearn.naive_bayes import GaussianNB\n","from sklearn.linear_model import Perceptron\n","from sklearn.linear_model import SGDClassifier\n","from sklearn.tree import DecisionTreeClassifier"]},{"cell_type":"markdown","metadata":{"_cell_guid":"6b5dc743-15b1-aac6-405e-081def6ecca1"},"source":["## Acquire data\n","\n","The Python Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. We also combine these datasets to run certain operations on both datasets together."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"e7319668-86fe-8adc-438d-0eef3fd0a982"},"outputs":[],"source":["train_df = pd.read_csv('../input/train.csv')\n","test_df = pd.read_csv('../input/test.csv')\n","combine = [train_df, test_df]"]},{"cell_type":"markdown","metadata":{"_cell_guid":"3d6188f3-dc82-8ae6-dabd-83e28fcbf10d"},"source":["## Analyze by describing data\n","\n","Pandas also helps describe the datasets answering following questions early in our project.\n","\n","**Which features are available in the dataset?**\n","\n","Noting the feature names for directly manipulating or analyzing these. These feature names are described on the [Kaggle data page here](https://www.kaggle.com/c/titanic/data)."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"ce473d29-8d19-76b8-24a4-48c217286e42"},"outputs":[],"source":["print(train_df.columns.values)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"cd19a6f6-347f-be19-607b-dca950590b37"},"source":["**Which features are categorical?**\n","\n","These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.\n","\n","- Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.\n","\n","**Which features are numerical?**\n","\n","Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.\n","\n","- Continous: Age, Fare. Discrete: SibSp, Parch."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"8d7ac195-ac1a-30a4-3f3f-80b8cf2c1c0f"},"outputs":[],"source":["# preview the data\n","train_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"97f4e6f8-2fea-46c4-e4e8-b69062ee3d46"},"source":["**Which features are mixed data types?**\n","\n","Numerical, alphanumeric data within same feature. These are candidates for correcting goal.\n","\n","- Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.\n","\n","**Which features may contain errors or typos?**\n","\n","This is harder to review for a large dataset, however reviewing a few samples from a smaller dataset may just tell us outright, which features may require correcting.\n","\n","- Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"f6e761c2-e2ff-d300-164c-af257083bb46"},"outputs":[],"source":["train_df.tail()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"8bfe9610-689a-29b2-26ee-f67cd4719079"},"source":["**Which features contain blank, null or empty values?**\n","\n","These will require correcting.\n","\n","- Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.\n","- Cabin > Age are incomplete in case of test dataset.\n","\n","**What are the data types for various features?**\n","\n","Helping us during converting goal.\n","\n","- Seven features are integer or floats. Six in case of test dataset.\n","- Five features are strings (object)."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"9b805f69-665a-2b2e-f31d-50d87d52865d"},"outputs":[],"source":["train_df.info()\n","print('_'*40)\n","test_df.info()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"859102e1-10df-d451-2649-2d4571e5f082"},"source":["**What is the distribution of numerical feature values across the samples?**\n","\n","This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.\n","\n","- Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).\n","- Survived is a categorical feature with 0 or 1 values.\n","- Around 38% samples survived representative of the actual survival rate at 32%.\n","- Most passengers (> 75%) did not travel with parents or children.\n","- Nearly 30% of the passengers had siblings and/or spouse aboard.\n","- Fares varied significantly with few passengers (<1%) paying as high as $512.\n","- Few elderly passengers (<1%) within age range 65-80."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"58e387fe-86e4-e068-8307-70e37fe3f37b"},"outputs":[],"source":["train_df.describe()\n","# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.\n","# Review Parch distribution using `percentiles=[.75, .8]`\n","# SibSp distribution `[.68, .69]`\n","# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`"]},{"cell_type":"markdown","metadata":{"_cell_guid":"5462bc60-258c-76bf-0a73-9adc00a2f493"},"source":["**What is the distribution of categorical features?**\n","\n","- Names are unique across the dataset (count=unique=891)\n","- Sex variable as two possible values with 65% male (top=male, freq=577/count=891).\n","- Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.\n","- Embarked takes three possible values. S port used by most passengers (top=S)\n","- Ticket feature has high ratio (22%) of duplicate values (unique=681)."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"8066b378-1964-92e8-1352-dcac934c6af3"},"outputs":[],"source":["train_df.describe(include=['O'])"]},{"cell_type":"markdown","metadata":{"_cell_guid":"2cb22b88-937d-6f14-8b06-ea3361357889"},"source":["### Assumtions based on data analysis\n","\n","We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.\n","\n","**Correlating.**\n","\n","We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.\n","\n","**Completing.**\n","\n","1. We may want to complete Age feature as it is definitely correlated to survival.\n","2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.\n","\n","**Correcting.**\n","\n","1. Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.\n","2. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.\n","3. PassengerId may be dropped from training dataset as it does not contribute to survival.\n","4. Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.\n","\n","**Creating.**\n","\n","1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.\n","2. We may want to engineer the Name feature to extract Title as a new feature.\n","3. We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.\n","4. We may also want to create a Fare range feature if it helps our analysis.\n","\n","**Classifying.**\n","\n","We may also add to our assumptions based on the problem description noted earlier.\n","\n","1. Women (Sex=female) were more likely to have survived.\n","2. Children (Age<?) were more likely to have survived. \n","3. The upper-class passengers (Pclass=1) were more likely to have survived."]},{"cell_type":"markdown","metadata":{"_cell_guid":"6db63a30-1d86-266e-2799-dded03c45816"},"source":["## Analyze by pivoting features\n","\n","To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.\n","\n","- **Pclass** We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.\n","- **Sex** We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).\n","- **SibSp and Parch** These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1)."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"0964832a-a4be-2d6f-a89e-63526389cee9"},"outputs":[],"source":["train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"68908ba6-bfe9-5b31-cfde-6987fc0fbe9a"},"outputs":[],"source":["train_df[[\"Sex\", \"Survived\"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"01c06927-c5a6-342a-5aa8-2e486ec3fd7c"},"outputs":[],"source":["train_df[[\"SibSp\", \"Survived\"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"e686f98b-a8c9-68f8-36a4-d4598638bbd5"},"outputs":[],"source":["train_df[[\"Parch\", \"Survived\"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"0d43550e-9eff-3859-3568-8856570eff76"},"source":["## Analyze by visualizing data\n","\n","Now we can continue confirming some of our assumptions using visualizations for analyzing the data.\n","\n","### Correlating numerical features\n","\n","Let us start by understanding correlations between numerical features and our solution goal (Survived).\n","\n","A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)\n","\n","Note that x-axis in historgram visualizations represents the count of samples or passengers.\n","\n","**Observations.**\n","\n","- Infants (Age <=4) had high survival rate.\n","- Oldest passengers (Age = 80) survived.\n","- Large number of 15-25 year olds did not survive.\n","- Most passengers are in 15-35 age range.\n","\n","**Decisions.**\n","\n","This simple analysis confirms our assumptions as decisions for subsequent workflow stages.\n","\n","- We should consider Age (our assumption classifying #2) in our model training.\n","- Complete the Age feature for null values (completing #1).\n","- We should band age groups (creating #3)."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"50294eac-263a-af78-cb7e-3778eb9ad41f"},"outputs":[],"source":["g = sns.FacetGrid(train_df, col='Survived')\n","g.map(plt.hist, 'Age', bins=20)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"87096158-4017-9213-7225-a19aea67a800"},"source":["### Correlating numerical and ordinal features\n","\n","We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values.\n","\n","**Observations.**\n","\n","- Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.\n","- Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.\n","- Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.\n","- Pclass varies in terms of Age distribution of passengers.\n","\n","**Decisions.**\n","\n","- Consider Pclass for model training."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"916fdc6b-0190-9267-1ea9-907a3d87330d"},"outputs":[],"source":["# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')\n","grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)\n","grid.map(plt.hist, 'Age', alpha=.5, bins=20)\n","grid.add_legend();"]},{"cell_type":"markdown","metadata":{"_cell_guid":"36f5a7c0-c55c-f76f-fdf8-945a32a68cb0"},"source":["### Correlating categorical features\n","\n","Now we can correlate categorical features with our solution goal.\n","\n","**Observations.**\n","\n","- Female passengers had much better survival rate than males. Confirms classifying (#1).\n","- Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.\n","- Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).\n","- Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).\n","\n","**Decisions.**\n","\n","- Add Sex feature to model training.\n","- Complete and add Embarked feature to model training."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"db57aabd-0e26-9ff9-9ebd-56d401cdf6e8"},"outputs":[],"source":["# grid = sns.FacetGrid(train_df, col='Embarked')\n","grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)\n","grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')\n","grid.add_legend()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"6b3f73f4-4600-c1ce-34e0-bd7d9eeb074a"},"source":["### Correlating categorical and numerical features\n","\n","We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).\n","\n","**Observations.**\n","\n","- Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.\n","- Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).\n","\n","**Decisions.**\n","\n","- Consider banding Fare feature."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"a21f66ac-c30d-f429-cc64-1da5460d16a9"},"outputs":[],"source":["# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})\n","grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)\n","grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)\n","grid.add_legend()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"cfac6291-33cc-506e-e548-6cad9408623d"},"source":["## Wrangle data\n","\n","We have collected several assumptions and decisions regarding our datasets and solution requirements. So far we did not have to change a single feature or value to arrive at these. Let us now execute our decisions and assumptions for correcting, creating, and completing goals.\n","\n","### Correcting by dropping features\n","\n","This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.\n","\n","Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.\n","\n","Note that where applicable we perform operations on both training and testing datasets together to stay consistent."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"da057efe-88f0-bf49-917b-bb2fec418ed9"},"outputs":[],"source":["print(\"Before\", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)\n","\n","train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)\n","test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)\n","combine = [train_df, test_df]\n","\n","\"After\", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape"]},{"cell_type":"markdown","metadata":{"_cell_guid":"6b3a1216-64b6-7fe2-50bc-e89cc964a41c"},"source":["### Creating new feature extracting from existing\n","\n","We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.\n","\n","In the following code we extract Title feature using regular expressions. The RegEx pattern `(\\w+\\.)` matches the first word which ends with a dot character within Name feature. The `expand=False` flag returns a DataFrame.\n","\n","**Observations.**\n","\n","When we plot Title, Age, and Survived, we note the following observations.\n","\n","- Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.\n","- Survival among Title Age bands varies slightly.\n","- Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).\n","\n","**Decision.**\n","\n","- We decide to retain the new Title feature for model training."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"df7f0cd4-992c-4a79-fb19-bf6f0c024d4b"},"outputs":[],"source":["for dataset in combine:\n","    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)\n","\n","pd.crosstab(train_df['Title'], train_df['Sex'])"]},{"cell_type":"markdown","metadata":{"_cell_guid":"908c08a6-3395-19a5-0cd7-13341054012a"},"source":["We can replace many titles with a more common name or classify them as `Rare`."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"553f56d7-002a-ee63-21a4-c0efad10cfe9"},"outputs":[],"source":["for dataset in combine:\n","    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\\\n"," \t'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')\n","\n","    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')\n","    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')\n","    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')\n","    \n","train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"6d46be9a-812a-f334-73b9-56ed912c9eca"},"source":["We can convert the categorical titles to ordinal."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"67444ebc-4d11-bac1-74a6-059133b6e2e8"},"outputs":[],"source":["title_mapping = {\"Mr\": 1, \"Miss\": 2, \"Mrs\": 3, \"Master\": 4, \"Rare\": 5}\n","for dataset in combine:\n","    dataset['Title'] = dataset['Title'].map(title_mapping)\n","    dataset['Title'] = dataset['Title'].fillna(0)\n","\n","train_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"f27bb974-a3d7-07a1-f7e4-876f6da87e62"},"source":["Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"9d61dded-5ff0-5018-7580-aecb4ea17506"},"outputs":[],"source":["train_df = train_df.drop(['Name', 'PassengerId'], axis=1)\n","test_df = test_df.drop(['Name'], axis=1)\n","combine = [train_df, test_df]\n","train_df.shape, test_df.shape"]},{"cell_type":"markdown","metadata":{"_cell_guid":"2c8e84bb-196d-bd4a-4df9-f5213561b5d3"},"source":["### Converting a categorical feature\n","\n","Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.\n","\n","Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"c20c1df2-157c-e5a0-3e24-15a828095c96"},"outputs":[],"source":["for dataset in combine:\n","    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)\n","\n","train_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"d72cb29e-5034-1597-b459-83a9640d3d3a"},"source":["### Completing a numerical continuous feature\n","\n","Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.\n","\n","We can consider three methods to complete a numerical continuous feature.\n","\n","1. A simple way is to generate random numbers between mean and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation).\n","\n","2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using [median](https://en.wikipedia.org/wiki/Median) values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...\n","\n","3. Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.\n","\n","Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"c311c43d-6554-3b52-8ef8-533ca08b2f68"},"outputs":[],"source":["# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')\n","grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)\n","grid.map(plt.hist, 'Age', alpha=.5, bins=20)\n","grid.add_legend()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"a4f166f9-f5f9-1819-66c3-d89dd5b0d8ff"},"source":["Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"9299523c-dcf1-fb00-e52f-e2fb860a3920"},"outputs":[],"source":["guess_ages = np.zeros((2,3))\n","guess_ages"]},{"cell_type":"markdown","metadata":{"_cell_guid":"ec9fed37-16b1-5518-4fa8-0a7f579dbc82"},"source":["Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"a4015dfa-a0ab-65bc-0cbe-efecf1eb2569"},"outputs":[],"source":["for dataset in combine:\n","    for i in range(0, 2):\n","        for j in range(0, 3):\n","            guess_df = dataset[(dataset['Sex'] == i) & \\\n","                                  (dataset['Pclass'] == j+1)]['Age'].dropna()\n","\n","            # age_mean = guess_df.mean()\n","            # age_std = guess_df.std()\n","            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)\n","\n","            age_guess = guess_df.median()\n","\n","            # Convert random age float to nearest .5 age\n","            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5\n","            \n","    for i in range(0, 2):\n","        for j in range(0, 3):\n","            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\\\n","                    'Age'] = guess_ages[i,j]\n","\n","    dataset['Age'] = dataset['Age'].astype(int)\n","\n","train_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"dbe0a8bf-40bc-c581-e10e-76f07b3b71d4"},"source":["Let us create Age bands and determine correlations with Survived."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"725d1c84-6323-9d70-5812-baf9994d3aa1"},"outputs":[],"source":["train_df['AgeBand'] = pd.cut(train_df['Age'], 5)\n","train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"ba4be3a0-e524-9c57-fbec-c8ecc5cde5c6"},"source":["Let us replace Age with ordinals based on these bands."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"797b986d-2c45-a9ee-e5b5-088de817c8b2"},"outputs":[],"source":["for dataset in combine:    \n","    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0\n","    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1\n","    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2\n","    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3\n","    dataset.loc[ dataset['Age'] > 64, 'Age']\n","train_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"004568b6-dd9a-ff89-43d5-13d4e9370b1d"},"source":["We can not remove the AgeBand feature."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"875e55d4-51b0-5061-b72c-8a23946133a3"},"outputs":[],"source":["train_df = train_df.drop(['AgeBand'], axis=1)\n","combine = [train_df, test_df]\n","train_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"1c237b76-d7ac-098f-0156-480a838a64a9"},"source":["### Create new feature combining existing features\n","\n","We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"7e6c04ed-cfaa-3139-4378-574fd095d6ba"},"outputs":[],"source":["for dataset in combine:\n","    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1\n","\n","train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"842188e6-acf8-2476-ccec-9e3451e4fa86"},"source":["We can create another feature called IsAlone."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"5c778c69-a9ae-1b6b-44fe-a0898d07be7a"},"outputs":[],"source":["for dataset in combine:\n","    dataset['IsAlone'] = 0\n","    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1\n","\n","train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"e6b87c09-e7b2-f098-5b04-4360080d26bc"},"source":["Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"74ee56a6-7357-f3bc-b605-6c41f8aa6566"},"outputs":[],"source":["train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)\n","test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)\n","combine = [train_df, test_df]\n","\n","train_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"f890b730-b1fe-919e-fb07-352fbd7edd44"},"source":["We can also create an artificial feature combining Pclass and Age."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"305402aa-1ea1-c245-c367-056eef8fe453"},"outputs":[],"source":["for dataset in combine:\n","    dataset['Age*Class'] = dataset.Age * dataset.Pclass\n","\n","train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"13292c1b-020d-d9aa-525c-941331bb996a"},"source":["### Completing a categorical feature\n","\n","Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"bf351113-9b7f-ef56-7211-e8dd00665b18"},"outputs":[],"source":["freq_port = train_df.Embarked.dropna().mode()[0]\n","freq_port"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"51c21fcc-f066-cd80-18c8-3d140be6cbae"},"outputs":[],"source":["for dataset in combine:\n","    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)\n","    \n","train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"f6acf7b2-0db3-e583-de50-7e14b495de34"},"source":["### Converting categorical feature to numeric\n","\n","We can now convert the EmbarkedFill feature by creating a new numeric Port feature."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"89a91d76-2cc0-9bbb-c5c5-3c9ecae33c66"},"outputs":[],"source":["for dataset in combine:\n","    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)\n","\n","train_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"e3dfc817-e1c1-a274-a111-62c1c814cecf"},"source":["### Quick completing and converting a numeric feature\n","\n","We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.\n","\n","Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.\n","\n","We may also want round off the fare to two decimals as it represents currency."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"3600cb86-cf5f-d87b-1b33-638dc8db1564"},"outputs":[],"source":["test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)\n","test_df.head()"]},{"cell_type":"markdown","metadata":{"_cell_guid":"4b816bc7-d1fb-c02b-ed1d-ee34b819497d"},"source":["We can not create FareBand."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"0e9018b1-ced5-9999-8ce1-258a0952cbf2"},"outputs":[],"source":["train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)\n","train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"d65901a5-3684-6869-e904-5f1a7cce8a6d"},"source":["Convert the Fare feature to ordinal values based on the FareBand."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"385f217a-4e00-76dc-1570-1de4eec0c29c"},"outputs":[],"source":["for dataset in combine:\n","    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0\n","    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1\n","    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2\n","    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3\n","    dataset['Fare'] = dataset['Fare'].astype(int)\n","\n","train_df = train_df.drop(['FareBand'], axis=1)\n","combine = [train_df, test_df]\n","    \n","train_df.head(10)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"27272bb9-3c64-4f9a-4a3b-54f02e1c8289"},"source":["And the test dataset."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"d2334d33-4fe5-964d-beac-6aa620066e15"},"outputs":[],"source":["test_df.head(10)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"69783c08-c8cc-a6ca-2a9a-5e75581c6d31"},"source":["## Model, predict and solve\n","\n","Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:\n","\n","- Logistic Regression\n","- KNN or k-Nearest Neighbors\n","- Support Vector Machines\n","- Naive Bayes classifier\n","- Decision Tree\n","- Random Forrest\n","- Perceptron\n","- Artificial neural network\n","- RVM or Relevance Vector Machine"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"0acf54f9-6cf5-24b5-72d9-29b30052823a"},"outputs":[],"source":["X_train = train_df.drop(\"Survived\", axis=1)\n","Y_train = train_df[\"Survived\"]\n","X_test  = test_df.drop(\"PassengerId\", axis=1).copy()\n","X_train.shape, Y_train.shape, X_test.shape"]},{"cell_type":"markdown","metadata":{"_cell_guid":"579bc004-926a-bcfe-e9bb-c8df83356876"},"source":["Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression).\n","\n","Note the confidence score generated by the model based on our training dataset."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"0edd9322-db0b-9c37-172d-a3a4f8dec229"},"outputs":[],"source":["# Logistic Regression\n","\n","logreg = LogisticRegression()\n","logreg.fit(X_train, Y_train)\n","Y_pred = logreg.predict(X_test)\n","acc_log = round(logreg.score(X_train, Y_train) * 100, 2)\n","acc_log"]},{"cell_type":"markdown","metadata":{"_cell_guid":"3af439ae-1f04-9236-cdc2-ec8170a0d4ee"},"source":["We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.\n","\n","Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).\n","\n","- Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.\n","- Inversely as Pclass increases, probability of Survived=1 decreases the most.\n","- This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.\n","- So is Title as second highest positive correlation."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"e545d5aa-4767-7a41-5799-a4c5e529ce72"},"outputs":[],"source":["coeff_df = pd.DataFrame(train_df.columns.delete(0))\n","coeff_df.columns = ['Feature']\n","coeff_df[\"Correlation\"] = pd.Series(logreg.coef_[0])\n","\n","coeff_df.sort_values(by='Correlation', ascending=False)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"ac041064-1693-8584-156b-66674117e4d0"},"source":["Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of **two categories**, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).\n","\n","Note that the model generates a confidence score which is higher than Logistics Regression model."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"7a63bf04-a410-9c81-5310-bdef7963298f"},"outputs":[],"source":["# Support Vector Machines\n","\n","svc = SVC()\n","svc.fit(X_train, Y_train)\n","Y_pred = svc.predict(X_test)\n","acc_svc = round(svc.score(X_train, Y_train) * 100, 2)\n","acc_svc"]},{"cell_type":"markdown","metadata":{"_cell_guid":"172a6286-d495-5ac4-1a9c-5b77b74ca6d2"},"source":["In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. Reference [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).\n","\n","KNN confidence score is better than Logistics Regression but worse than SVM."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"ca14ae53-f05e-eb73-201c-064d7c3ed610"},"outputs":[],"source":["knn = KNeighborsClassifier(n_neighbors = 3)\n","knn.fit(X_train, Y_train)\n","Y_pred = knn.predict(X_test)\n","acc_knn = round(knn.score(X_train, Y_train) * 100, 2)\n","acc_knn"]},{"cell_type":"markdown","metadata":{"_cell_guid":"810f723d-2313-8dfd-e3e2-26673b9caa90"},"source":["In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. Reference [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).\n","\n","The model generated confidence score is the lowest among the models evaluated so far."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"50378071-7043-ed8d-a782-70c947520dae"},"outputs":[],"source":["# Gaussian Naive Bayes\n","\n","gaussian = GaussianNB()\n","gaussian.fit(X_train, Y_train)\n","Y_pred = gaussian.predict(X_test)\n","acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)\n","acc_gaussian"]},{"cell_type":"markdown","metadata":{"_cell_guid":"1e286e19-b714-385a-fcfa-8cf5ec19956a"},"source":["The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. Reference [Wikipedia](https://en.wikipedia.org/wiki/Perceptron)."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"ccc22a86-b7cb-c2dd-74bd-53b218d6ed0d"},"outputs":[],"source":["# Perceptron\n","\n","perceptron = Perceptron()\n","perceptron.fit(X_train, Y_train)\n","Y_pred = perceptron.predict(X_test)\n","acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)\n","acc_perceptron"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"a4d56857-9432-55bb-14c0-52ebeb64d198"},"outputs":[],"source":["# Linear SVC\n","\n","linear_svc = LinearSVC()\n","linear_svc.fit(X_train, Y_train)\n","Y_pred = linear_svc.predict(X_test)\n","acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)\n","acc_linear_svc"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"dc98ed72-3aeb-861f-804d-b6e3d178bf4b"},"outputs":[],"source":["# Stochastic Gradient Descent\n","\n","sgd = SGDClassifier()\n","sgd.fit(X_train, Y_train)\n","Y_pred = sgd.predict(X_test)\n","acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)\n","acc_sgd"]},{"cell_type":"markdown","metadata":{"_cell_guid":"bae7f8d7-9da0-f4fd-bdb1-d97e719a18d7"},"source":["This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).\n","\n","The model confidence score is the highest among models evaluated so far."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"dd85f2b7-ace2-0306-b4ec-79c68cd3fea0"},"outputs":[],"source":["# Decision Tree\n","\n","decision_tree = DecisionTreeClassifier()\n","decision_tree.fit(X_train, Y_train)\n","Y_pred = decision_tree.predict(X_test)\n","acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)\n","acc_decision_tree"]},{"cell_type":"markdown","metadata":{"_cell_guid":"85693668-0cd5-4319-7768-eddb62d2b7d0"},"source":["The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).\n","\n","The model confidence score is the highest among models evaluated so far. We decide to use this model's output (Y_pred) for creating our competition submission of results."]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"f0694a8e-b618-8ed9-6f0d-8c6fba2c4567"},"outputs":[],"source":["# Random Forest\n","\n","random_forest = RandomForestClassifier(n_estimators=100)\n","random_forest.fit(X_train, Y_train)\n","Y_pred = random_forest.predict(X_test)\n","random_forest.score(X_train, Y_train)\n","acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)\n","acc_random_forest"]},{"cell_type":"markdown","metadata":{"_cell_guid":"f6c9eef8-83dd-581c-2d8e-ce932fe3a44d"},"source":["### Model evaluation\n","\n","We can now rank our evaluation of all the models to choose the best one for our problem. While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set. "]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"1f3cebe0-31af-70b2-1ce4-0fd406bcdfc6"},"outputs":[],"source":["models = pd.DataFrame({\n","    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', \n","              'Random Forest', 'Naive Bayes', 'Perceptron', \n","              'Stochastic Gradient Decent', 'Linear SVC', \n","              'Decision Tree'],\n","    'Score': [acc_svc, acc_knn, acc_log, \n","              acc_random_forest, acc_gaussian, acc_perceptron, \n","              acc_sgd, acc_linear_svc, acc_decision_tree]})\n","models.sort_values(by='Score', ascending=False)"]},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"28854d36-051f-3ef0-5535-fa5ba6a9bef7"},"outputs":[],"source":["submission = pd.DataFrame({\n","        \"PassengerId\": test_df[\"PassengerId\"],\n","        \"Survived\": Y_pred\n","    })\n","# submission.to_csv('../output/submission.csv', index=False)"]},{"cell_type":"markdown","metadata":{"_cell_guid":"fcfc8d9f-e955-cf70-5843-1fb764c54699"},"source":["Our submission to the competition site Kaggle results in scoring 3,883 of 6,082 competition entries. This result is indicative while the competition is running. This result only accounts for part of the submission dataset. Not bad for our first attempt. Any suggestions to improve our score are most welcome."]},{"cell_type":"markdown","metadata":{"_cell_guid":"aeec9210-f9d8-cd7c-c4cf-a87376d5f693"},"source":["## References\n","\n","This notebook has been created based on great work done solving the Titanic competition and other sources.\n","\n","- [A journey through Titanic](https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic)\n","- [Getting Started with Pandas: Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests)\n","- [Titanic Best Working Classifier](https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier)"]}],"metadata":{"_change_revision":0,"_is_fork":false,"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"codemirror_mode":{"name":"ipython","version":3},"file_extension":".py","mimetype":"text/x-python","name":"python","nbconvert_exporter":"python","pygments_lexer":"ipython3","version":"3.6.0"}},"nbformat":4,"nbformat_minor":0}


# In[ ]:


import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;

# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId']


# In[ ]:





# In[ ]:


train.head(3)


# In[ ]:


full_data = [train, test]

# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] ;


# In[ ]:


# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)


# In[ ]:


train.head(3)


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


# In[ ]:


# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer


# In[ ]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# In[ ]:


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# In[ ]:


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data


# In[ ]:


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")


# In[ ]:


rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


# In[ ]:


rf_features = [0.10474135,  0.21837029,  0.04432652,  0.02249159,  0.05432591,  0.02854371
  ,0.07570305,  0.01088129 , 0.24247496,  0.13685733 , 0.06128402]
et_features = [ 0.12165657,  0.37098307  ,0.03129623 , 0.01591611 , 0.05525811 , 0.028157
  ,0.04589793 , 0.02030357 , 0.17289562 , 0.04853517,  0.08910063]
ada_features = [0.028 ,   0.008  ,      0.012   ,     0.05866667,   0.032 ,       0.008
  ,0.04666667 ,  0.     ,      0.05733333,   0.73866667,   0.01066667]
gb_features = [ 0.06796144 , 0.03889349 , 0.07237845 , 0.02628645 , 0.11194395,  0.04778854
  ,0.05965792 , 0.02774745,  0.07462718,  0.4593142 ,  0.01340093]


# In[ ]:


cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })


# In[ ]:


# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# In[ ]:


# Create the new column containing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(3)


# In[ ]:


y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')


# In[ ]:


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()


# In[ ]:


data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Portland',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')


# In[ ]:


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


# In[ ]:


gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# In[ ]:


# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

