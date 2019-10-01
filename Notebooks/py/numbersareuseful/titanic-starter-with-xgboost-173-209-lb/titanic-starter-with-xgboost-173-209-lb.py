#!/usr/bin/env python
# coding: utf-8

# # Table of Contents

# # 1. Introduction

# # 2. Preamble

# ## Jupyter Magic

# In[1]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')


# ## Imports

# In[2]:


import datetime
import math
import warnings
from functools import partial
from typing import Iterable, Union

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from matplotlib import pyplot as plt
from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin

from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedShuffleSplit,
    cross_val_score, train_test_split
)

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


# ## Library Settings

# In[3]:


plt.rcParams['figure.figsize'] = (15,4)
sns.set(
    style='whitegrid',
    color_codes=True,
    font_scale=1.5)
np.set_printoptions(
    suppress=True,
    linewidth=200)
pd.set_option(
    'display.max_rows', 1000,
    'display.max_columns', None,
)


# ## Helpers

# 
# ### Transformers

# In[4]:


class Apply(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(fn)


class AsType(TransformerMixin):
    def __init__(self, t):
        self.t = t

    def fit(self, X, y=None):
        if self.t == 'category':
            self.dtype = pd.Categorical(X.unique())
        else:
            self.dtype = self.t
        return self

    def transform(self, X):
        return X.astype(self.dtype)


class ColMap(TransformerMixin):
    def __init__(self, trf):
        self.trf = trf

    def fit(self, X, y=None):
        self.trf_list = [self.trf().fit(col) for _, col in X.iteritems()]
        return self
    
    def transform(self, X):
        cols = [t.transform(X.iloc[:,i]) for i, t in enumerate(self.trf_list)]
        return pd.concat(cols, axis=1)


class ColProduct(TransformerMixin):
    def __init__(self, trf):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.product(axis=1)


class ColSum(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.sum(axis=1)


class DataFrameUnion(TransformerMixin):
    def __init__(self, trf_list):
        self.trf_list = trf_list

    def fit(self, X, y=None):
        for t in self.trf_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        return pd.concat([t.transform(X) for t in self.trf_list], axis=1)


class FillNA(TransformerMixin):
    def __init__(self, val):
        self.val = val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna(self.val)


class GetDummies(TransformerMixin):
    def __init__(self, drop_first=False):
        self.drop = drop_first

    def fit(self, X, y=None):
        self.name = X.name
        self.cat = pd.Categorical(X.unique())
        return self

    def transform(self, X):
        return pd.get_dummies(X.astype(self.cat), prefix=self.name, drop_first=self.drop)


class Identity(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class NADummies(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.isna().astype(np.uint8).rename('na_' + X.name)


class PdStandardScaler(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mean = X.mean()
        self.std = X.std(ddof=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


class PdTransform(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.fn(X)


class QCut(TransformerMixin):
    def __init__(self, q, interp=None):
        self.q = q
        self.interp = interp

    def fit(self, X, y=None):
        self.bins, self.bin_vals = QCut.params(X, self.q, self.interp)
        return self

    def transform(self, X):
        return (
            X
            .apply(lambda x: bin_val(x, self.bins, self.bin_vals))
            .rename(f'{X.name}_qcut{self.q}')
        )

    def params(X, q, interp=None):
        _, bins = pd.qcut(X, q, retbins=True)
        idx = X.apply(lambda x: bin_val(x, bins))

        if interp == 'median':
            v = X.groupby(idx).median()
        elif interp == 'mean':
            v = X.groupby(idx).mean()
        elif interp == 'min':
            v = X.groupby(idx).min()
        elif interp == 'max':
            v = X.groupby(idx).max()
        else:
            return bins, seq(0, len(bins))

        v = list(v)
        bin_vals = [v[0]] + v + [v[-1]]

        return bins, bin_vals


class Rename(TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.rename(self.name)


class Map(TransformerMixin):
    def __init__(self, d):
        self.d = d

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.map(self.d)


class SelectColumns(TransformerMixin):
    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.include:
            X = X[self.include]
        if self.exclude:
            return X.drop(columns=self.exclude)
        return X


class SelectDtypes(TransformerMixin):
    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=self.include, exclude=self.exclude)


# ### Misc

# In[5]:


Numeric = Union[int, float, np.number]


class Path:
    """
    Path management.
    """
    def __init__(self, prefix='', suffix=''):
        self.prefix = prefix
        self.suffix = suffix

    def add(self, *names):
        for x in names:
            setattr(self, x, self.prefix + x + self.suffix)
    
    def rename(self, attr, name):
        setattr(self, attr, self.prefix + name + self.suffix)


def suppress(fn):
    """
    Suppress xgb warnings.
    """
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fn(*args, **kwargs)
    return wrapper


def is_int(
        x: Numeric) \
        -> bool:
    """
    Whether `x` is int.
    """
    return isinstance(x, (int, np.integer))


def n_dec(
        x: Numeric) \
        -> int:
    """
    Number of decimal places, using `str` conversion.
    """
    if x == 0:
        return 0
    _, _, dec = str(x).partition('.')
    return len(dec)


def seq(
        start: Numeric,
        stop: Numeric,
        step: Numeric = None) \
        -> np.ndarray:
    """
    Inclusive sequence.  Requires `is_int` and `n_dec`.
    """
    if step is None:
        if start < stop:
            step = 1
        else:
            step = -1

    if is_int(start) and is_int(step):
        dtype = 'int'
    else:
        dtype = None

    d = max(n_dec(step), n_dec(start))
    n_step = math.floor(round(round(stop - start, d + 1) / step, d + 1)) + 1
    delta = np.arange(n_step) * step
    return np.round(start + delta, decimals=d).astype(dtype)


def bin_val(x, bins, vals=None, side='left', nan=np.nan):
    """
    Map `x` to bin value.
    """
    if vals is None:
        vals = seq(0, len(bins))
    else:
        assert len(vals) == len(bins) + 1, 'len(vals) must equal len(bins) + 1'

    if np.isnan(x):
        return nan
    elif x < bins[0]:
        index = 0
    elif x == bins[0]:
        index = 1
    elif x > bins[-1]:
        index = len(bins)
    else:
        index = np.searchsorted(bins, x, side=side)

    return vals[index]


def find_title(s, df):
    """
    Rows that contain title `s`.
    """
    if isinstance(s, str):
        s = ', ' + s
    else:
        s = '|'.join([f', {x}' for x in s])
    return df[(
        df.name
        .str.lower()
        .str.contains(s)
    )]


def na(X):
    """
    Returns features with na values.
    """
    count = X.isna().sum()
    if len(X.shape) < 2:
        return count
    else:
        return count[lambda x: x > 0]


def perc(x, n_dec=1):
    """
    Convert decimal to whole number percentage.
    """
    return np.round(x*100, n_dec)


def reorder(df, order=None):
    """
    Sort `df` columns by dtype and name.
    """
    def sort(df):
        return df.dtypes.reset_index().sort_values([0, 'index'])['index']
    if order is None:
        order = [np.floating, np.integer, 'category', 'object']
    names = [sort(df.select_dtypes(s)) for s in order]
    return df[[x for ls in names for x in ls]]


# ### Preprocessing

# In[6]:


def preprocess(pip, csv_train, csv_test):
    raw_X, y, raw_test_X = load_titanic(csv_train, csv_test)

    X = pip.fit_transform(raw_X)
    test_X = pip.transform(raw_test_X)

    return reorder(X), y, reorder(test_X)


def load_titanic(train, test):
    X, y = load(train)
    test_X = load(test)
    return X, y, test_X


def load(csv, ycol='Survived'):
    col_names = {
        'Survived': 'survived',
        'Pclass': 'ticket_class',
        'Name': 'name',
        'Sex': 'sex',
        'Age': 'age',
        'SibSp': 'n_sib_sp',
        'Parch': 'n_par_ch',
        'Ticket': 'ticket',
        'Fare': 'fare',
        'Cabin': 'cabin',
        'Embarked': 'port',
    }

    exclude = [
        'PassengerId'
    ]

    d = {
        'Survived': np.uint8,
        'Pclass': np.uint8,
        'Name': 'object',
        'Sex': 'object',
        'Age': np.float32,
        'SibSp': np.uint8,
        'Parch': np.uint8,
        'Ticket': 'object',
        'Fare': np.float64,
        'Cabin': 'object',
        'Embarked': 'object'
    }

    df = pd.read_csv(
        csv,
        dtype=d,
        usecols=lambda x: x not in exclude,
    )

    df = df.rename(columns=col_names)
    df = reorder(df)

    ycol = str.lower(ycol)

    if ycol in df.columns:
        return df.drop(columns=ycol), df[ycol]
    else:
        return df


# ### XGBoost

# In[7]:


@suppress
def submit(model, name, X, y, test_X, folder='./'):
    """
    Save predictions.
    """
    y_hat = clone(model).fit(X, y).predict(test_X)
    df = (
        pd.read_csv(csv.sample)
        .assign(Survived=y_hat)
    )
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')
    path = f'{folder}{timestamp}_{name}.csv'
    df.to_csv(path, index=False)


@suppress
def xgb_fit(
        model,
        X, y,
        early_stopping_rounds=None,
        test_size=0.1,
        eval_metric=('error', 'logloss')):
    """
    Fit XGBoost model, with optional early stopping.
    """
    if early_stopping_rounds is None:
        return model.fit(X, y)
    
    if isinstance(eval_metric, tuple):
        eval_metric = list(eval_metric)

    X, test_X, y, test_y =         train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=SEED)

    fit_params = {
        'eval_metric': eval_metric,
        'eval_set': [(X, y), (test_X, test_y)],
        'early_stopping_rounds': early_stopping_rounds,
        'verbose': False,
    }

    model.fit(X, y, **fit_params)

    if hasattr(model, 'best_ntree_limit'):
        return (
            model
            .set_params(n_estimators=model.best_ntree_limit)
            .fit(X, y)
        )
    else:
        return model


def xgb_fi(model, X):
    """
    XGBoost feature importance plot.
    """
    fi = model.feature_importances_
    names = X.columns
    df = (
        pd.DataFrame(data=fi, index=names)
        [lambda df: df[0] > 0]
        .sort_values(0, ascending=False)
    )
    sns.barplot(x=df.values.ravel(), y=df.index, data=df)


def xgb_fi_grid(model_dict, X, y, names=None, ncols=3, n=9):
    """
    XGBoost feature importance plots.
    """
    nrows = math.ceil(len(model_dict) / ncols)
    for i, (k, m) in enumerate(model_dict.items()):
        if i >= n:
            break
        plt.subplot(nrows, ncols, 1 + i)
        plt.xlim(0, 0.5)
        xgb_fi(m, X)
        plt.title(k)


# ### Cross Validation

# In[8]:


@suppress
def diff(one, *rest, X, y, test_X):
    return [np.nonzero(clone(one).fit(X, y).predict(test_X) != clone(m).fit(X, y).predict(test_X)) for m in rest]


@suppress
def cv(model, X, y, n=10, t_list=tuple(seq(0.1, 0.9, 0.1)), kf=True, scoring='accuracy'):
    m = clone(model)

    row_names = list(t_list)
    cv_list = [
        StratifiedShuffleSplit(
            n_splits=n,
            test_size=i,
            random_state=SEED)
        for i in t_list
    ]

    if kf:
        row_names += ['kf5', 'kf10']
        cv_list += [RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=n//5,
            random_state=SEED
        )]
        cv_list += [RepeatedStratifiedKFold(
            n_splits=10,
            n_repeats=n//10,
            random_state=SEED
        )]

    results = [cross_val_score(m, X, y, cv=x, scoring=scoring, n_jobs=1) for x in cv_list]

    return pd.DataFrame(
        data=[perc(np.mean(x), 2) for x in results] + [perc(np.mean(results))],
        index=row_names + ['avg'],
        columns=['score'],
    )


def cv_grid(model_dict, X, y, **kwargs):
    df = pd.concat(
        [cv(m, X, y, **kwargs) for m in model_dict.values()],
        axis=1,
    )
    df.columns = model_dict.keys()
    return df


def read_evals(model, name):
    return pd.DataFrame({
        'Test': model.evals_result_['validation_1'][name],
        'Train': model.evals_result_['validation_0'][name],
    })


def early_stop_grid(
        model,
        X, y,
        eval_metric='logloss',
        r_list=(1, 3, 5, 10, 25, 50, 100),
        t_list=tuple(seq(0.1, 0.9, 0.1)),
        n=100):
    """
    Early stopping across grid values.
    """
    m = clone(model)
    d = dict()
    for t in t_list:
        for r in r_list:
            m.set_params(n_estimators=n)
            xgb_fit(m, X, y, early_stopping_rounds=r, test_size=t, eval_metric=eval_metric)
            d[(t, r)] = m.best_ntree_limit
    return pd.DataFrame(d, index=[0]).stack().reset_index(level=0, drop=True).T


def plotgrid_evals(
        model,
        t_list=tuple(seq(0.1, 0.5, 0.1)),
        n=100,
        ylim_top=(0, 0.7),
        ylim_bot=(0, 0.3)):
    """
    Plotgrid of learning curves.
    """
    m = clone(model)

    def plot(name):
        df = read_evals(m, name)
        plt.plot(df)
        plt.xticks((n * np.array([0, 25, 50, 75, 100]) / 100).astype(np.int32))

    ncols = len(t_list)

    for i, x in enumerate(t_list):
        m.set_params(n_estimators=n)
        xgb_fit(m, X, y, early_stopping_rounds=n, test_size=x)

        plt.subplot(2, ncols, 1 + i)
        plt.title(f'Test Size: {x}')
        plot('logloss')
        plt.ylim(*ylim_top)
        plt.gca().set_xticklabels([])
        if i == 0:
            plt.ylabel('log loss')
        else:
            plt.gca().set_yticklabels([])

        plt.subplot(2, ncols, 1 + i + ncols)
        plt.xlabel('n_estimators')
        plot('error')
        plt.ylim(*ylim_bot)
        if i == 0:
            plt.ylabel('error rate')
            plt.legend(
                labels=['Test', 'Train'],
                loc='lower left',
                prop={'size': 12},
                frameon=True,
            )
        else:
            plt.gca().set_yticklabels([])


# ### Parameter Search

# In[9]:


class Uniform:
    def __init__(self, support):
        self.support = support

    def rvs(self, random_state):
        return random_state.choice(self.support)


def search(
        param_dict,
        cv_obj,
        X, y,
        n_iter=1_000,
        skeleton=None,
        scoring='neg_log_loss',
        **kwargs):
    if skeleton is None:
        skeleton = XGBClassifier(n_jobs=1, random_state=SEED)
        if 'early_stopping_rounds' in kwargs:
            skeleton.set_params(n_estimators=1_000)

    dist = {k: Uniform(v) for k, v in param_dict.items()}

    optim = RandomizedSearchCV(
        estimator=skeleton,
        param_distributions=dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_obj,
        return_train_score=True,
        verbose=1,
        n_jobs=4,
        random_state=SEED,
    )

    return xgb_fit(optim, X, y, **kwargs)


def search_results(optim, n=10):
    names = [
        '^n$',
        'mean_test',
        'mean_train',
        'std_test',
        'std_train',
        'param_'
    ]
    df = pd.DataFrame(optim.cv_results_)
    return (
        df
        .assign(n = df.index)
        .set_index('rank_test_score')
        .sort_index()
        .head(n)
        .round(3)
        .filter(regex='|'.join(names))
        .T
    )


def search_params(optim, index):
    return (
        pd.DataFrame(optim.cv_results_)
        .set_index('rank_test_score')
        .sort_index()
        ['params']
        .iloc[index]
    )


def search_info(optim, param_dict, n_list=(10, 20, 30, 50, 100)):
    df = (
        pd.concat([search_results(optim, n=x).T.median() for x in n_list], axis=1)
        .round(3)
        .T
        .filter(like='param_')
        .rename(columns=lambda x: x.replace('param_', ''))
        .set_axis(n_list, inplace=False)
        .T
        .assign(ev_mean=[np.mean(v) for v in param_dict.values()])
    )
    return df


def plot_params(optim, n_list=None):
    """
    Requires `Uniform` class inside `param_distributions`.
    """
    if n_list is None:
        n_list = (optim.n_iter * np.array([1, 2, 3, 5, 10]) / 100).astype(np.int32)

    ncols = len(n_list)
    for i, n in enumerate(n_list):
        df = (
            search_results(optim, n)
            .filter(like='param_', axis=0)
        )
        nrows = len(df)
        for j, row in enumerate(df.values):
            name = df.index[j].replace('param_', '')
            plt.subplot(nrows, ncols, 1 + i + j*ncols)
            plt.title(name)
            sns.violinplot(
                data=row,
                width=0.7,
                cut=0,
                inner='point',
                orient='h',
            )

            support = optim.param_distributions[name].support
            a = np.min(support)
            b = np.max(support)
            if a == b:
                a -= (max(0.01, a*0.05))
                b += (max(0.01, b*0.05))
            plt.xlim(a, b)
            if j in (0, nrows - 1):
                plt.ylabel(n_list[i])


# ### Model Helpers

# In[10]:


def get_model(optim, rank, X, y, early_stopping_rounds=50, n_jobs=1):
    assert rank >= 1, 'rank must be >= 1'
    if early_stopping_rounds:
        m = XGBClassifier(
            **search_params(optim, rank - 1),
            n_estimators=1_000,
            n_jobs=n_jobs,
            random_state=SEED,
        )
        return xgb_fit(m, X, y, early_stopping_rounds)
    else:
        m = XGBClassifier(
            **search_params(optim, rank - 1),
            n_jobs=n_jobs,
            random_state=SEED,
        )
        return xgb_fit(m, X, y)


def get_models(optim, rank_list):
    """
    Requires partial `get_model`.
    """
    return {f'rank{r}': get_model(optim, r, n_jobs=4) for r in rank_list}


def voter(optim, rank_list, voting='soft'):
    return VotingClassifier(
        [(f'rank{r}', get_model(optim, r, n_jobs=4)) for r in rank_list],
        voting=voting,
    )


def voter_dict(optim, rank_list):
    return {f'top{r}': voter_top(optim, r) for r in rank_list}


def voter_top(optim, rank):
    return voter(optim, seq(1, rank))


# ### Feature Engineering

# In[11]:


def deck(X):
    return (
        X
        .cabin
        .str.extract(r'([A-Z]\d|[A-Z]$)', expand=False)
        .str.extract(r'([A-Z])', expand=False)
        .rename('deck')
    )


def cabin_factor(X):
    return (
        (
            deck(X)
            .map({
                'A': 1,
                'B': 2,
                'C': 3,
                'D': 4,
                'E': 5,
                'F': 6,
                'G': 7})
            * 1000
            + cabin_no(X)
        )
        .astype(np.float32)
        .rename('cabin_factor')
    )


def cabin_no(X):
    return (
        X
        .cabin
        .str.extract(r'(\d+)', expand=False)
        .astype(np.float32)
        .rename('cabin_no')
    )


def starboard(X):
    return (
        cabin_no(X)
        .apply(lambda x: np.nan if np.isnan(float(x)) else int(x) % 2 == 1)
        .astype(np.float32)
        .rename('starboard')
    )


def ticket_cut(X, n=1):
    return (
        X
        .ticket
        .str.extract(rf'(\d{{{n}}})\d*$', expand=False)
        .fillna(0)
        .astype(np.uint32)
        .rename(f'ticket_cut_{n}')
    )


def ticket_no(X):
    return (
        X
        .ticket
        .str.extract(r'(\d+)$', expand=False)
        .fillna(0)
        .astype(np.uint32)
        .rename('ticket_no')
    )


def title(X, raw=False):
    return (
        X.name
        .str.lower()
        .str.extract(r', (\w+)', expand=False)
        .rename('title')
    )


# ## Globals

# In[12]:


SEED = 0

csv = Path('../input/', '.csv')
csv.add(
    'leaderboard',
    'train',
    'test',
)
csv.rename('sample', 'gender_submission')

folder = Path()
folder.submit = './'


# # &sect; CSV Preprocessing

# In[13]:


a, b, c = load_titanic(csv.train, csv.test)


# In[14]:


# peek


# In[15]:


# raw


# # &sect; Baseline

# In[16]:


# Zeros


# In[17]:


# Sex
class Benchmark(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X.sex.values

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# # &sect; Feature Engineering

# In[18]:


# X.groupby(title(X)).age.median()


# In[19]:


# test_X.groupby(title(test_X)).age.median()


# # &sect; Pipeline Preprocessing

# In[20]:


X_pipeline = DataFrameUnion([
    # age
    SelectColumns('age'),
    make_pipeline(
        SelectColumns('age'),
        NADummies(),
    ),

    # fare
    SelectColumns('fare'),

    # n_par_ch + n_sib_sp
    SelectColumns('n_par_ch'),
    SelectColumns('n_sib_sp'),
    make_pipeline(
        SelectColumns(['n_par_ch', 'n_sib_sp']),
        ColSum(),
        AsType(np.uint8),
        Rename('n_fam'),
    ),

    # ticket
    SelectColumns('ticket_class'),
    PdTransform(ticket_no),

    # cabin
    PdTransform(cabin_factor),
    PdTransform(cabin_no),
    PdTransform(starboard),
    make_pipeline(
        PdTransform(deck),
        GetDummies(),
    ),
    make_pipeline(
        SelectColumns('cabin'),
        NADummies(),
    ),

    # name -> title -> dummies
    make_pipeline(
        PdTransform(title),
        GetDummies(),
    ),

    # port -> dummies
    make_pipeline(
        SelectColumns('port'),
        GetDummies(),
    ),

    # sex -> 0/1
    make_pipeline(
        SelectColumns('sex'),
        Map({
            'female': 1,
            'male': 0,
        }),
        AsType(np.uint8),
    ),
])


# In[21]:


X, y, test_X = preprocess(X_pipeline, csv.train, csv.test)


# In[22]:


X.head()


# In[23]:


# X.head()
# na(X)
# na(test_X)
# X.max().astype(str)
# test_X.max().astype(str)
# X.dtypes
# (X.columns == test_X.columns).all()
# (X.dtypes == test_X.dtypes).all()


# # &sect; Partials

# In[24]:


# X, y, test_X
submit = partial(submit, X=X, y=y, test_X=test_X)
diff = partial(diff, X=X, y=y, test_X=test_X)

# X, y
early_stop_grid = partial(early_stop_grid, X=X, y=y)
search = partial(search, X=X, y=y)
cv_grid = partial(cv_grid, X=X, y=y)
get_model = partial(get_model, X=X, y=y)
xgb_fi_grid = partial(xgb_fi_grid, X=X, y=y)


# # &sect; Starter

# In[25]:


starter = xgb_fit(XGBClassifier(n_jobs=4, random_state=SEED), X, y)


# In[26]:


# starter


# In[27]:


# plt.figure(figsize=(15,4))
# plt.xlim(0, 0.5)
# xgb_fi(starter, X)


# In[28]:


# xgboost.plot_tree(starter, rankdir='LR')
# plt.gcf().set_size_inches(18, 16)


# In[29]:


submit(starter, 'starter')


# ## Early Stop

# In[30]:


# early_stop_grid(starter)


# In[31]:


# plt.figure(figsize=(26, 5))
# plotgrid_evals(starter)


# In[32]:


starter_earlystop = xgb_fit(clone(starter), X, y, early_stopping_rounds=10)


# In[33]:


# starter_earlystop


# In[34]:


# %%time
# cv_grid({
#     'bench': Benchmark(),
#     'starter': starter,
#     'early': starter_earlystop,
# })


# In[35]:


# plt.figure(figsize=(15,5))
# plt.xlim(0, 0.5)
# xgb_fi(starter_earlystop, X)


# In[36]:


# xgboost.plot_tree(starter_earlystop, rankdir='LR')
# plt.gcf().set_size_inches(18, 16)


# In[37]:


# diff(starter, starter_earlystop)


# In[38]:


submit(starter, 'starter_earlystop')


# # &sect; Prototype

# In[39]:


# asdf = XGBClassifier(
#     colsample_bylevel=1,
#     colsample_bytree=1,
#     gamma=0,
#     learning_rate=0.01,
#     max_depth=36,
#     n_estimators=1000,
#     reg_lambda=1,
#     subsample=0.1,
#     n_jobs=4,
#     random_state=SEED,
# )


# In[40]:


# cv(asdf, X, y, kf=False)
# early_stop_grid(asdf, t_list=[0.1, 0.5, 0.7, 0.8, 0.9], n=1000)

# plt.figure(figsize=(26, 5))
# plotgrid_evals(asdf, n=1000, t_list=[0.1, 0.5, 0.7, 0.8, 0.9])

# plt.figure(figsize=(10,6))
# xgb_fi(asdf.fit(X, y), X)

# xgboost.plot_tree(asdf, rankdir='LR')
# plt.gcf().set_size_inches(26, 26)

# submit(asdf, 'asdf')


# # &sect; Random Search

# In[ ]:


cv_obj = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=SEED)
param_dict = {
    'colsample_bylevel': seq(0.025, 1.0, 0.025),
    'colsample_bytree': seq(0.025, 1.0, 0.025),
    'gamma': seq(2, 6, 0.1),
    'max_depth': np.array([3, 4, 5, 6, 16]),
    'reg_lambda': seq(1, 10, 0.01),
    'scale_pos_weight': seq(1.0, 1.5, 0.05),
    'subsample': seq(0.05, 1.00, 0.05),
}


# In[ ]:


optim = search(param_dict, cv_obj, n_iter=10_000, early_stopping_rounds=10, test_size=0.3)


# In[ ]:


search_results(optim, n=20)


# In[ ]:


plt.figure(figsize=(26,15))
plt.subplots_adjust(hspace=2)
plot_params(optim, n_list=[10, 25, 50, 100, 200])


# In[ ]:


# search_info(optim, param_dict)


# # &sect; Models

# In[ ]:


tops = get_models(optim, seq(1, 10))
voters = voter_dict(optim, [2, 3, 4, 5, 10])


# In[ ]:


early_stop_grid(tops['rank1'], r_list=[1, 5, 10, 20, 30, 50, 100], n=500)


# In[ ]:


plt.figure(figsize=(26, 5))
plotgrid_evals(tops['rank1'], t_list=seq(0.1, 0.9, 0.1), n=100, ylim_top=(0.2, 0.5), ylim_bot=(0.0, 0.3))


# In[ ]:


plt.figure(figsize=(26, 30))
plt.subplots_adjust(wspace=0.3)
xgb_fi_grid(tops)


# In[ ]:


xgboost.plot_tree(tops['rank1'], rankdir='LR')
plt.gcf().set_size_inches(40, 40)


# In[ ]:


xgboost.plot_tree(tops['rank2'], rankdir='LR')
plt.gcf().set_size_inches(40, 40)


# In[ ]:


xgboost.plot_tree(tops['rank3'], rankdir='LR')
plt.gcf().set_size_inches(40, 40)


# In[ ]:


xgboost.plot_tree(tops['rank4'], rankdir='LR')
plt.gcf().set_size_inches(40, 40)


# In[ ]:


diff(tops['rank1'], tops['rank2'], voters['top2'], voters['top3'])


# In[ ]:


diff(voters['top2'], voters['top3'], voters['top5'], voters['top10'])


# In[ ]:


diff(tops['rank2'], voters['top2'], tops['rank3'])


# # &sect; Cross Validation

# In[ ]:


# %%time
# cv_grid({
#     'bench': Benchmark(),
#     'starter': starter,
#     'early': starter_earlystop}) \
# .assign(delta = lambda x: x.early - x.starter)


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'cv_grid(tops, t_list=seq(0.5, 0.9, 0.1), kf=False)')


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'cv_grid(voters, t_list=seq(0.5, 0.9, 0.1), kf=False)')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


for k, m in tops.items():
    submit(m, k)


# In[ ]:


for k, m in voters.items():
    submit(m, k)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




