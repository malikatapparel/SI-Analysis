#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:51:41 2022

@author: malika
"""

# Improvers non-improvers lifestyle
import pandas as pd
import numpy as np
from scipy import stats
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

#To define the datasets to be substracted from one another
path0 = "/Users/malika/Documents/Five Lives/Analysis/data/SI_Datasets/df0_clean.csv"
df0 = pd.read_csv(path0)
df0 = df0.drop('Unnamed: 0', axis=1)
df0 = df0.set_index('participant_id')
df0['A0male'] = pd.get_dummies(df0['A0sex'], drop_first=True)


path2 = "/Users/malika/Documents/Five Lives/Analysis/data/SI_Datasets/df2_clean.csv"
df2 = pd.read_csv(path2)
df2 = df2.drop('Unnamed: 0', axis=1)
df2 = df2.set_index('participant_id')
df2['A2male'] = pd.get_dummies(df2['A2sex'], drop_first=True)

# to define the assessment index

A0 = 'A0'
A2 = 'A2'

# Define output plot path
plot_path = '/Users/malika/Documents/Five Lives/Analysis/Results SI/Result Screen/'

#Variables to be added in the dataset : age, gender + lifestyle

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


lifestyleA2 =  ['A2_age', 'A2male','A2PA_duration',
 'A2PA_intensity',
 'A2PA_score',
 'A2smoking_situation',
 'A2smoking_amount', 'A2socialI',
 'A2loneliness',
 'A2sleep_problems',
 'A2fruit_veg',
 'A2alcohol_unit',
 'A2glasses_water',
 'A2time_reading',
 'A2time_mental_stimulation',
 'A2time_instrument',
 'A2time_skill',
 'A2relaxed_scale',
 'A2mood_rate',
 'A2enjoyment_activities']

##Joins dataset into a bigger dfL dataset 
dfL0 = df0[lifestyleA0]
dfL2 = df2[lifestyleA2]
dfL = dfL0.join(dfL2)

## Create list of lifestyle factors without the A0 or A2 in the front
LS_A0 = ['A0PA_duration',
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
LS_all = []
for i in LS_A0:
    x = i[2:]
    LS_all.append(x)
# loop for all substractions

for l in LS_all:
    dfL[A0 + A2 + l] = np.nan
    for i in dfL.index:
        if np.isnan(dfL[A0 + l][i]) or np.isnan(dfL[A2 +l][i]) == True:
            dfL.loc[i,A0 + A2 + l] = np.nan
        elif np.isnan(dfL[A0 + l][i]) & np.isnan(dfL[A2 +l][i]) == False:
            dfL.loc[i,A0 + A2 + l] = dfL[A2 +l][i] - dfL[A0 + l][i]
        else:
            raise ValueError('Value could not be attributed, as it is not in the list')

## create variables (A0A2IMP...)showing improvers and non-improvers
for l in LS_all:
    dfL[A0 + A2 + 'IMP' + l] = np.nan
    for i in dfL.index:
        if dfL[A0 + A2 + l][i] <= 0:
            dfL.loc[i,A0 + A2 +'IMP' + l] = 0
        elif dfL[A0 + A2 + l][i] > 0:
            dfL.loc[i,A0 + A2 +'IMP' + l] = 1
            
## Save the dataset
dfL.to_csv('/Users/malika/Documents/Five Lives/Analysis/data/DataframeA2-A0.csv', index=True)


## Distribution graphs to see how people improve + save to plot_path
for l in LS_all:
    toplot = A0 + A2 + l
    plt.figure('single feature', clear=True)
    g= sns.histplot(x=dfL[toplot], hue = dfL['A0male'], multiple="dodge", stat = 'density', shrink = 0.8, common_norm=False)
    g.axvline(dfL[toplot].median(), color='k', ls='--', lw=2)
    g.text(-0.5, 0.90,'Median = '+ str(dfL[toplot].median()), fontsize=10)
    plt.savefig(plot_path + toplot + '.png')
    
    
for l in LS_all:
    toplot = A0 + A2 + 'IMP' + l
    plt.figure('single feature', clear=True)
    g= sns.histplot(x=dfL[toplot], hue = dfL['A0male'], multiple="dodge", stat = 'density', shrink = 0.8, common_norm=False)
    g.axvline(dfL[toplot].median(), color='k', ls='--', lw=2)
    g.text(-0.5, 0.90,'Median = '+ str(dfL[toplot].median()), fontsize=10)
    plt.savefig(plot_path + toplot + '.png')
    
    
    
    
## Create Dataset improvers non-improvers


