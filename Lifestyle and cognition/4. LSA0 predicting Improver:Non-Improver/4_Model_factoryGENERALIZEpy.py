#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:36:24 2022

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

## Functions
def ReplaceImprNonImpr(DataFrame, column):
    for i in DataFrame.index:
        if DataFrame.loc[i, column] == 'Improver':
            DataFrame.loc[i, column] = 1
        elif DataFrame.loc[i, column] == 'Non-improver':
            DataFrame.loc[i, column] = 0
        else:
            DataFrame.loc[i, column] = np.nan
## Dataset of modified lifestyle data + swift RT
path2 = "/Users/malika/Documents/Five Lives/Analysis/data/DataframeLifestyle1.csv"
A = 'A0'
df1 = pd.read_csv(path2)
df1 = df1.set_index('participant_id')

# Import improver dataset
path3 = "/Users/malika/Documents/Five Lives/Analysis/data/Cognition/improvers_per_game.csv"
df_cog = pd.read_csv(path3)
df_cog = df_cog.set_index('user_id')
# df_cog = df_cog[~df_cog.index.duplicated(keep='first')]

plot_path = '/Users/malika/Documents/Five Lives/Analysis/Results SI/Result Screen/Improvers/'

# Join datasets into lifestyle + cognitive data
df1 = df1.join(df_cog)

## TO MODIFY IF NEEDED
###  columns names to change from cetegorical to numerical
cog = ['snap_words_change_A0_A1_category', 'snap_words_change_A0_A1_initial',
       'snap_words_change_A1_A2_category', 'snap_words_change_A1_A2_initial',
       'snap_words_change_A0_A2_category', 'snap_words_change_A0_A2_initial',
       'swift_RT_change_A0_A1', 'swift_RT_change_A1_A2',
       'swift_RT_change_A0_A2', 'cast_acc_change', 'cast_RT_change',
       'trail_0_twist_change_A0_A1', 'trail_0_twist_change_A1_A2',
       'trail_0_twist_change_A0_A2', 'trail_1_twist_change_A0_A1',
       'trail_1_twist_change_A1_A2', 'trail_1_twist_change_A0_A2',
       'trail_2_twist_change_A0_A1', 'trail_2_twist_change_A1_A2',
       'trail_2_twist_change_A0_A2']

### replace 'improver' and 'non-improver' by 1 and 0
error = []
for i in cog:
    to_predict = i
    df = df1.dropna(axis=0, subset = to_predict)
    ReplaceImprNonImpr(df, i)
    df[to_predict] = df[to_predict].astype('float')
    try:
        ## Model Physical activity
        model = '~ PA_score + Age + male'
        mdf = smf.logit(to_predict + model , data=df).fit()
        mdf.summary()
        print(mdf.summary())
        
        plt.figure('single feature', clear=True)
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(mdf.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path + to_predict + 'PA_model.png')
        
        
        #Model Smoking score (1-5)
        
        df[['dum_current', 'dum_never', 'dum_past']] = pd.get_dummies(df['smoking_situation'], drop_first=False)
        model = '~ ' + A + 'smoking_situation + ' + A + 'smoking_amount' + A +'Age + ' + A + 'male'
        md_smoke = OLS.from_formula(to_predict +'~ dum_current + dum_never + dum_past + smoke_score + Age + male', df)
        mdf_smoke = md_smoke.fit()
        print(mdf_smoke.summary())
        
        # plt.figure('single feature', clear=True)
        # plt.rc('figure', figsize=(12, 7))
        # plt.text(0.01, 0.05, str(mdf_smoke.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(plot_path + to_predict + 'Smoke_model.png')
        
        # Social Isolation and loneliness
        
        md_social = smf.logit(to_predict + ' ~ socialI + lonely+ Age + male', df)
        mdf_social = md_social.fit()
        print(mdf_social.summary())
        
        plt.figure('single feature', clear=True)
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(mdf_social.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path +  to_predict + 'Social_model.png')
        
        # Sleep
        
        md_sleep = smf.logit(to_predict + ' ~ sleep_problem + Age + male', df)
        mdf_sleep = md_sleep.fit()
        print(mdf_sleep.summary())
        
        plt.figure('single feature', clear=True)
        plt.rc('figure', figsize=(16, 9))
        plt.text(0.01, 0.05, str(mdf_sleep.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path +  to_predict + 'Sleep_model.png')
        
        # Diet
        
        # md_diet = smf.logit(to_predict + '~ fruit_veg + portion_fish + portion_nuts + Age + male', df)
        # mdf_diet = md_diet.fit()
        # print(mdf_diet.summary())
        
        # plt.figure('single feature', clear=True)
        # plt.rc('figure', figsize=(16, 9))
        # plt.text(0.01, 0.05, str(mdf_diet.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(plot_path +  to_predict + 'Diet_model.png')
        
        # Alcohol
        md_alcohol = smf.logit(to_predict + ' ~ alcohol_unit + Age + male', df)
        mdf_alcohol = md_alcohol.fit()
        print(mdf_alcohol.summary())
        
        plt.figure('single feature', clear=True)
        plt.rc('figure', figsize=(16, 9))
        plt.text(0.01, 0.05, str(mdf_alcohol.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path +  to_predict + 'Alcohol_model.png')
        
        # Water
        md_water = smf.logit(to_predict + '~ glasses_water + Age + male', df)
        mdf_water = md_water.fit()
        print(mdf_water.summary())
        
        plt.figure('single feature', clear=True)
        plt.rc('figure', figsize=(16, 79))
        plt.text(0.01, 0.05, str(mdf_water.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path +  to_predict + 'glasseswater_model.png')
        
        ##Mental stimulation
        md_mentalstim = smf.logit(to_predict + '~ time_reading + time_mental_stimulation + time_instrument + time_skill + Age + male', df)
        mdf_mentalstim = md_mentalstim.fit()
        print(mdf_mentalstim.summary())
        
        plt.figure('single feature', clear=True)
        plt.rc('figure', figsize=(16, 9))
        plt.text(0.01, 0.05, str(mdf_mentalstim.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path +  to_predict + 'mentalstim_model.png')
        
        ## Stress and mood
        md_mood = smf.logit(to_predict + '~ relaxed_scale + mood_rate + enjoyment_activities + Age + male', df)
        mdf_mood = md_mood.fit()
        print(mdf_mood.summary())
        
        plt.figure('single feature', clear=True)
        plt.rc('figure', figsize=(16,9))
        plt.text(0.01, 0.05, str(mdf_mood.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path +  to_predict + 'mood_model.png') 
    except:
        error.append(i)
        print(i +"Exception was thrown")





