#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:15:40 2022

@author: malika
"""

### This file will create one dataset from Lifestyle Data A0 (df0_clean) and cognitive performances 
import pandas as pd
import numpy as np
from scipy import stats
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

#Output path for the final file
output_path = '/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/HierarchicalClusteringAnalysis/'

#Import latest version of the df0 cleaned
path0 = "/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/Datasets & Data cleaning/df0_clean.csv"
df0 = pd.read_csv(path0)
df0 = df0.drop('Unnamed: 0', axis=1)
df0 = df0.set_index('participant_id')
df0['A0male'] = pd.get_dummies(df0['A0sex'], drop_first=True)

# Import the latest version of cognitive changes
cog = '/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/Datasets & Data cleaning/'
cast = pd.read_csv(cog + 'cast_performance.csv')
cast = cast.set_index('user_id')

snapC = pd.read_csv(cog + 'snap_category_performance.csv')
snapC = snapC.set_index('user_id')

snapI = pd.read_csv(cog + 'snap_initials_performance.csv')
snapI = snapI.set_index('user_id')

swift = pd.read_csv(cog + 'swift_performance.csv')
swift = swift.set_index('user_id')

twist = pd.read_csv(cog + 'twist_performance.csv')
twist = twist.set_index('user_id')

#Columns to be kept in the lifestyle data
lifestyleA0 = ['A0_age', 'A0male','A0PA_duration',
 'A0PA_intensity',
 'A0PA_score',
 'A0smoking_situation',
 'A0smoking_amount',
 'A0socialI',
 'A0loneliness',
 'A0sleep_problems',
 'A0fruit_veg',
 'A0alcohol_unit',
 'A0glasses_water',
 'A0time_reading',
 'A0time_mental_stimulation',
 'A0time_instrument',
 'A0time_skill',
 'A0relaxed_scale',
 'A0mood_rate',
 'A0enjoyment_activities']

#create df as a subset of df0 keeping only lifestylee + age and gender for control
df = df0[lifestyleA0]


#Add cast data
not_in_cast_index = []
for i in df.index:
    if i in cast.index:
        df.loc[i, 'A0cast_performance'] = cast.loc[i, 'Assessment 0']
    elif i not in cast.index:
        not_in_cast_index.append(i)
        
## Add snapC data
not_in_snapC_index = []
for i in df.index:
    if i in snapC.index:
        df.loc[i, 'A0snap_category_performance'] = snapC['Assessment_0'][i]
    elif i not in snapC.index:
        not_in_snapC_index.append(i)
## snapI data
not_in_snapI_index = []
for i in df.index:
    if i in snapI.index:
        df.loc[i, 'A0snap_initials_performance'] = snapI['Assessment_0'][i]
    elif i not in snapI.index:
        not_in_snapI_index.append(i)
        
## Swift
## Swift needs to be split in two categories: congruent and incongruent
swift_inc = swift[swift['match'] == 'Incongruent colour-word pair']
swift_cong = swift[swift['match'] == 'Congruent colour-word pair']

## Swift Incongruent
not_in_swiftI_index = []
for i in df.index:
    if i in swift_inc.index:
        df.loc[i, 'A0swift_incongruent_performance'] = swift_inc['Assessment 0'][i]
    elif i not in swift_inc.index:
        not_in_swiftI_index.append(i)

## Swift Congruent
not_in_swiftC_index = []
for i in df.index:
    if i in swift_cong.index:
        df.loc[i, 'A0swift_congruent_performance'] = swift_cong['Assessment 0'][i]
    elif i not in swift_cong.index:
        not_in_swiftC_index.append(i)

## Twist Congruent
not_in_twist_index = []
for i in df.index:
    if i in twist.index:
        df.loc[i, 'A0twist_performance'] = twist['Assessment 0'][i]
    elif i not in twist.index:
        not_in_twist_index.append(i)
        
df.to_csv('/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/Datasets & Data cleaning/DataframeCogPerfA0.csv', index=True)

        

