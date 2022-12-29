#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:02:03 2022

@author: malika
"""

#Plots and models
import pandas as pd
import numpy as np
from scipy import stats
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import statsmodels.formula.api as smf



def ReplaceImprNonImpr(DataFrame, column):
    for i in DataFrame.index:
        if DataFrame.loc[i, column] == 'Improver':
            DataFrame.loc[i, column] = 1
        elif DataFrame.loc[i, column] == 'Non-improver':
            DataFrame.loc[i, column] = 0
        else:
            DataFrame.loc[i, column] = np.nan

# Import paths
path2 = "/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/LSA0A2_Impr_NonImpr/Datasets/DataframeA2-A0.csv"
df1 = pd.read_csv(path2)
df1 = df1.set_index('participant_id')

path3 = "/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/LSA0A2_Impr_NonImpr/Datasets/improvers_per_game.csv"
df_cog = pd.read_csv(path3)
df_cog = df_cog.set_index('user_id')

##Simplify LS dataset to reflect only Lifestyle change + ange and gender
LSchange = ['A0_age',
 'A0male','A0A2IMPPA_duration',
 'A0A2IMPPA_intensity',
 'A0A2IMPPA_score',
 'A0A2IMPsmoking_situation',
 'A0A2IMPsmoking_amount',
 'A0A2IMPsocialI',
 'A0A2IMPloneliness',
 'A0A2IMPsleep_problems',
 'A0A2IMPfruit_veg',
 'A0A2IMPalcohol_unit',
 'A0A2IMPglasses_water',
 'A0A2IMPtime_reading',
 'A0A2IMPtime_mental_stimulation',
 'A0A2IMPtime_instrument',
 'A0A2IMPtime_skill',
 'A0A2IMPrelaxed_scale',
 'A0A2IMPmood_rate',
 'A0A2IMPenjoyment_activities']

df1 = df1[LSchange]

CogChange = ['snap_words_change_A0_A2_category',
 'snap_words_change_A0_A2_initial',
 'swift_acc_change_A0_A2',
 'swift_RT_change_A0_A2',
 'cast_acc_change',
 'cast_RT_change',
 'trail_0_twist_change_A0_A2',
 'trail_1_twist_change_A0_A2',
 'trail_2_twist_change_A0_A2']

df_cog = df_cog[CogChange]

df2 = df1.join(df_cog)

## Many IDs still didn't complete the third assessment so need to first remove the nan variables
### by usiong one of the change variable from lifestyle

df2 = df2.dropna(axis=0, subset = 'A0A2IMPPA_duration')

## Transform cognitive variables as a) 1, 0 instead of improver non-improver / b) float instead of strings
for i in CogChange:
    ReplaceImprNonImpr(df2, i)
    df2[i] = df2[i].astype('float')
    
    
# Model
## c will be the name of the variables to be conrolled for
c = 'A0_age + A0male'

#Get LS change list without age and male
LSchange.remove('A0male')
LSchange.remove('A0_age')
#Choose predicted variable and drop nan in this specific column (to change for each prediction)


### prepare list for failures
fail_sum = []
### For list for cognitive change variables
for to_predict in CogChange:
    # drop nan in each subset to predict to avoid errors due to too many nan
    df = df2.dropna(axis=0, subset = to_predict)
    model_sum = [to_predict]
    for predictor in LSchange:
        try:
            #Model
            m = to_predict + ' ~ ' + predictor + ' + ' + c
            mdf = smf.logit(m , data=df).fit()
            #Create a list to save the models
            model_sum.append(mdf.summary())
            file_name = to_predict
            path = '/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/LSA0A2_Impr_NonImpr/Discrete LS/LSA2A0b_ImprNonimpr/Individual/'+file_name+'.txt'
            fp= open(path, 'w')
            with open(path, 'w') as fp:
                for item in model_sum:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                    print('Done')
        except:
            fail = to_predict + '_by_' + predictor
            fail_sum.append(fail)
            path = '/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/LSA0A2_Impr_NonImpr/Discrete LS/LSA2A0b_ImprNonimpr/Individual/ErrorRaised.txt'
            fp= open(path, 'w')
            with open(path, 'w') as fp:
                for item in fail_sum:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                    print('Done')
                    
# Repeat models but on different pillars
physical = 'A0A2IMPPA_duration + A0A2IMPPA_intensity + A0A2IMPPA_score'
sleep = 'A0A2IMPsleep_problems'
diet =  'A0A2IMPfruit_veg + A0A2IMPalcohol_unit + A0A2IMPglasses_water '
mood =  'A0A2IMPrelaxed_scale + A0A2IMPmood_rate + A0A2IMPenjoyment_activities'
mental_stim =  'A0A2IMPsocialI + A0A2IMPloneliness + A0A2IMPtime_mental_stimulation + A0A2IMPtime_instrument + A0A2IMPtime_skill+ A0A2IMPtime_reading'
smoking =  'A0A2IMPsmoking_situation + A0A2IMPsmoking_amount'
FL_pillars = []

## Model reproeduced on FL pillars
FL_pillars = [physical, sleep, diet, mood, mental_stim, smoking ]
fail_sum = []
### For list for cognitive change variables
for to_predict in CogChange:
    # drop nan in each subset to predict to avoid errors due to too many nan
    df = df2.dropna(axis=0, subset = to_predict)
    model_sum = [to_predict]
    for predictor in FL_pillars:
        try:
            #Model
            m = to_predict + ' ~ ' + predictor + ' + ' + c
            mdf = smf.logit(m , data=df).fit()
            #Create a list to save the models
            model_sum.append(mdf.summary())
            file_name = to_predict
            path = '/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/LSA0A2_Impr_NonImpr/Discrete LSLSA2A0b_ImprNonimpr/Pillar/'+file_name+'.txt'
            fp= open(path, 'w')
            with open(path, 'w') as fp:
                for item in model_sum:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                    print('Done')
        except:
            fail = to_predict + '_by_' + predictor
            fail_sum.append(fail)
            path = '/Users/malika/Documents/Five Lives/SI Data Analysis/LifestyleCognition/LSA0A2_Impr_NonImpr/Discrete LS/LSA2A0b_ImprNonimpr/Pillar/ErrorRaised.txt'
            fp= open(path, 'w')
            with open(path, 'w') as fp:
                for item in fail_sum:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                    print('Done')
