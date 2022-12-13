#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:52:50 2022

@author: malika
"""

import pandas as pd
import numpy as np
from scipy import stats
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm


#To define the datasets to be substracted from one another
path0 = "/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/Datasets & Data cleaning/df0_clean.csv"
df0 = pd.read_csv(path0)
df0 = df0.drop('Unnamed: 0', axis=1)
df0 = df0.set_index('participant_id')
df0['A0male'] = pd.get_dummies(df0['A0sex'], drop_first=True)

path1 = "/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/Datasets & Data cleaning/df1_clean.csv"
df1 = pd.read_csv(path1)
df1 = df1.drop('Unnamed: 0', axis=1)
df1 = df1.set_index('participant_id')
df1['A1male'] = pd.get_dummies(df1['A1sex'], drop_first=True)


path2 = "/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/Datasets & Data cleaning/df2_clean.csv"
df2 = pd.read_csv(path2)
df2 = df2.drop('Unnamed: 0', axis=1)
df2 = df2.set_index('participant_id')
df2['A2male'] = pd.get_dummies(df2['A2sex'], drop_first=True)


# join datasets

df = df0.join(df1)
df = df.join(df2)


df.to_csv('/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/Datasets & Data cleaning/df_all_clean.csv', index=True)


