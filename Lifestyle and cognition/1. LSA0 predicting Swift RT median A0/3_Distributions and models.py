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


path2 = "/Users/malika/Documents/Five Lives/Analysis/data/DataframeLifestyle1.csv"
df = pd.read_csv(path2)
df = df.set_index('participant_id')
plot_path = '/Users/malika/Documents/Five Lives/Analysis/Results SI/Result Screen/'


#All correlations
plt.subplots(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,linewidths=.5)
plt.savefig(plot_path + 'all_correlations.png')


# Physical activity
plt.figure('single feature', clear=True)
sns.histplot(x=df["PA_score"].astype('str'), 
             hue=df["male"], multiple="dodge", stat = 'density', shrink = 0.8, common_norm=False)
PA_mean = df['PA_score'].mean()
plt.axvline(x=df['PA_score'].mean(), color = 'black')
plt.title('Physical Activity score')
plt.xlabel('PA score (Intensity (1-3)* Duration (0-4))')
plt.ylabel('Density')
plt.legend(loc='upper left', labels = ['Total mean' ,'Male', 'Female'])
PA_mean = df['PA_score'].mean()
print('total mean: ',df['PA_score'].mean(), ', total median: ', df['PA_score'].median())
plt.savefig(plot_path + 'PA_countplot_all.png')

## Model Physical activity
md = OLS.from_formula('Swift_medianRT ~ PA_score + Age + male', df)
mdf = md.fit()
print(mdf.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(mdf.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'PA_model.png')

#Smoking
## Smoking score (1-5)

plt.figure('single feature', clear=True)
fig, axs = plt.subplots(ncols=2, figsize=(16,6))
sns.histplot(data=df, x=str("smoke_score"), hue="male",multiple="dodge", stat = 'density', shrink = 0.8, common_norm=False, ax=axs[0])
sns.histplot(data=df, x=str("smoking_situation"), hue="male", multiple="dodge", stat = 'density', shrink = 0.8, common_norm=False, ax=axs[1])
plt.savefig(plot_path + 'smoke_all.png')


##Model
df[['dum_current', 'dum_never', 'dum_past']] = pd.get_dummies(df['smoking_situation'], drop_first=False)
md_smoke = OLS.from_formula('Swift_medianRT ~ dum_current + dum_never + dum_past + smoke_score + Age + male', df)
mdf_smoke = md_smoke.fit()
print(mdf_smoke.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(mdf_smoke.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'Smoke_model.png')

# Social Isolation and loneliness
## Plot
plt.figure('single feature', clear=True)
fig, axs = plt.subplots(ncols=3, figsize=(16,6))
sns.countplot(data=df, x="socialI", hue="male", ax=axs[0])
sns.countplot(data=df, x="lonely", hue="male", ax=axs[1])
sns.heatmap(df[['socialI', 'lonely']].corr(),annot=True,linewidths=.5, ax=axs[2])
plt.savefig(plot_path + 'SocialIsolation_Loneliness.png')

md_social = OLS.from_formula('Swift_medianRT ~ socialI + lonely+ Age + male', df)
mdf_social = md_social.fit()
print(mdf_social.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(mdf_social.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'Social_model.png')

# Sleep
plt.figure('single feature', clear=True)
sns.countplot(data=df, x="sleep_problem", hue="male")
plt.axvline(x=df['sleep_problem'].mean(), color = 'black')
plt.title('Sleep Problems')
plt.suptitle('Sum of all sleep problems (e.g Insomnia, Difficulites falling asleep, etc.), range 0-6')
plt.xlabel('Sleep problem')
plt.ylabel('Count')
plt.legend(loc='upper right', labels = ['Total mean' ,'Male', 'Female'])
plt.savefig(plot_path + 'Sleep_countplot_all.png')

md_sleep = OLS.from_formula('Swift_medianRT ~ sleep_problem + Age + male', df)
mdf_sleep = md_sleep.fit()
print(mdf_sleep.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(16, 9))
plt.text(0.01, 0.05, str(mdf_sleep.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'Sleep_model.png')

# Diet
plt.figure('single feature', clear=True)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16,6))

sns.countplot(data=df, x="fruit_veg", hue="male", ax=axs[0,0])
sns.countplot(data=df, x="portion_fish", hue="male", ax=axs[0,1])
sns.countplot(data=df, x="portion_nuts", hue="male", ax=axs[1,0])
sns.heatmap(df[['fruit_veg', 'portion_fish','portion_nuts' ]].corr(),annot=True,linewidths=.5, ax=axs[1,1])
plt.savefig(plot_path + 'Diet_all.png')

md_diet = OLS.from_formula('Swift_medianRT ~ fruit_veg + portion_fish + portion_nuts + Age + male', df)
mdf_diet = md_diet.fit()
print(mdf_diet.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(16, 9))
plt.text(0.01, 0.05, str(mdf_diet.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'Diet_model.png')

# Alcohol
plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(16, 9))
sns.countplot(data=df, x="alcohol_unit", hue="male")
plt.axvline(x=df['alcohol_unit'].mean(), color = 'black')
plt.xlabel('AlcoholUnit')
plt.ylabel('Count')
plt.legend(loc='upper right', labels = ['Total mean' ,'Male', 'Female'])
plt.savefig(plot_path + 'AlcoholUnit_all.png')

md_alcohol = OLS.from_formula('Swift_medianRT ~ alcohol_unit + Age + male', df)
mdf_alcohol = md_alcohol.fit()
print(mdf_alcohol.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(16, 9))
plt.text(0.01, 0.05, str(mdf_alcohol.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'Alcohol_model.png')

# Water
plt.figure('single feature', clear=True)
sns.countplot(data=df, x="glasses_water", hue="male")
plt.axvline(x=df['glasses_water'].mean(), color = 'black')
plt.xlabel('glasses_water')
plt.ylabel('Count')
plt.legend(loc='upper right', labels = ['Total mean' ,'Male', 'Female'])
plt.savefig(plot_path + 'glasseswater_all.png')

md_water = OLS.from_formula('Swift_medianRT ~ glasses_water + Age + male', df)
mdf_water = md_water.fit()
print(mdf_water.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(16, 79))
plt.text(0.01, 0.05, str(mdf_water.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'glasseswater_model.png')

# Mental stimulation
plt.figure('single feature', clear=True)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16,6))

plt.figure('single feature', clear=True)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16,6))

sns.histplot(x= df['time_reading'].astype('str'), 
             hue=df["male"], multiple="dodge", stat = 'density', shrink = 0.8, 
             ax=axs[0,0], common_norm=False)
sns.histplot(x= df['time_mental_stimulation'].astype('str'), 
              hue=df["male"], multiple="dodge", stat = 'density', shrink = 0.8,
              ax=axs[0,1], common_norm=False)
sns.histplot(x= df['time_instrument'].astype('str'), 
             hue=df["male"], multiple="dodge", stat = 'density', shrink = 0.8,
             ax=axs[1,0], common_norm=False)
sns.histplot(x= df['time_skill'].astype('str'), 
                hue=df["male"], multiple="dodge", stat = 'density', shrink = 0.8, 
                ax=axs[1, 1], common_norm=False)
plt.savefig(plot_path + 'mentalstim_all.png')

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(12, 7))
sns.heatmap(df[['time_reading', 'time_mental_stimulation','time_instrument', 'time_skill', 'Swift_medianRT'  ]].corr(),annot=True,linewidths=6.5)
plt.savefig(plot_path + 'mentalstim_corr.png')


md_mentalstim = OLS.from_formula('Swift_medianRT ~ time_reading + time_mental_stimulation + time_instrument + time_skill + Age + male', df)
mdf_mentalstim = md_mentalstim.fit()
print(mdf_mentalstim.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(16, 9))
plt.text(0.01, 0.05, str(mdf_mentalstim.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'mentalstim_model.png')

## Stress and mood
plt.figure('single feature', clear=True)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16,6))

sns.countplot(data=df, x="relaxed_scale", hue="male", ax=axs[0,0])
sns.countplot(data=df, x="mood_rate", hue="male", ax=axs[0,1])
sns.countplot(data=df, x="enjoyment_activities", hue="male", ax=axs[1,0])
sns.heatmap(df[['relaxed_scale', 'mood_rate','enjoyment_activities']].corr(),annot=True,linewidths=.5, ax=axs[1,1])
plt.savefig(plot_path + 'mood_all.png')

md_mood = OLS.from_formula('Swift_medianRT ~ relaxed_scale + mood_rate + enjoyment_activities + Age + male', df)
mdf_mood = md_mood.fit()
print(mdf_mood.summary())

plt.figure('single feature', clear=True)
plt.rc('figure', figsize=(16,9))
plt.text(0.01, 0.05, str(mdf_mood.summary()), {'fontsize': 11}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_path + 'mood_model.png')

## Correlation with cognitive change
#Plots and models

path2 = "/Users/malika/Documents/Five Lives/Analysis/data/DataframeLifestyle1.csv"
df = pd.read_csv(path2)
df = df.set_index('participant_id')
plot_path = '/Users/malika/Documents/Five Lives/Analysis/Results SI/Result Screen/'
path3 = '/Users/malika/Documents/Five Lives/Analysis/data/Cognition/rate_of_change_per_game.csv'
df1 = pd.read_csv(path3)
df1 = df1.set_index('user_id')

#merge dataset and drop lines with nan
df_all = df.join(df1)
df_all = df_all.drop(['cast_acc_change', 'cast_RT_change'], axis=1 )
dfAll = df_all[df_all['snap_change_category'].notna()]

#plot
plt.subplots(figsize=(20,15))
sns.heatmap(dfAll.corr(),annot=True,linewidths=.5)
plt.savefig(plot_path + 'all_correlations2.png')




