#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:28:09 2022

@author: malika
"""
#Lifestyle data cleaning
#Cleans all variable into interpretable data
import pandas as pd
import numpy as np
from scipy import stats
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS

path2 = "/Users/malika/Documents/Five Lives/Analysis/data/DFModifiable.csv"
## WARNING: Original code was done with the DFModifiable_wSwiftRT.csv / code 3_ will work only with that initial code in place
# path2 = "/Users/malika/Documents/Five Lives/Analysis/data/DFModifiable_wSwiftRT.csv"
df = pd.read_csv(path2)
df = df.set_index('participant_id')

# create fancy df
df1 = pd.DataFrame()
df1.index= df.index
df1['Age'] = df['A0_age']
df1['male'] = pd.get_dummies(df['A0sex'], drop_first=True)
## Sex in 0 (female) or 1 (male)


#Physical acitivity score
## duration
df1['PA_duration'] = np.nan

for i in df.index:
    if df['A0physical_activity_duration'][i] == 'none':
        df1.loc[i, 'PA_duration'] = 0
    elif df['A0physical_activity_duration'][i] == 'less_than_1h':
        df1.loc[i, 'PA_duration'] = 1
    elif df['A0physical_activity_duration'][i] == '1h_to_2h':
        df1.loc[i, 'PA_duration'] = 2
    elif df['A0physical_activity_duration'][i] == '2h_to_3h':
        df1.loc[i, 'PA_duration'] = 3
    elif df['A0physical_activity_duration'][i] == 'more_than_3h':
        df1.loc[i, 'PA_duration'] = 4
        
## intensity
df1['PA_intensity'] = np.nan

for i in df.index:
    if df['A0physical_activity_intensity'][i] == 'moderate':
        df1.loc[i, 'PA_intensity'] = 1
    elif df['A0physical_activity_intensity'][i] == 'mixed':
        df1.loc[i, 'PA_intensity'] = 2
    elif df['A0physical_activity_intensity'][i] == 'intense':
        df1.loc[i, 'PA_intensity'] = 3
    else:
        df1.loc[i, 'PA_intensity'] = 0
        

df1['PA_score'] = np.nan
for i in df1.index:
    df1.loc[i, 'PA_score'] = (df1.loc[i, 'PA_duration']*df1.loc[i, 'PA_intensity'])

df1 = df1.drop(['PA_duration', 'PA_intensity'], axis=1)

# Smoking
## Smoking or not
df1['smoking_situation'] = df['A0smoking']
## build score
## smoking score is smoking/increasing values: lowest score indicates a heavier smoker 
df1['smoke_score'] = np.nan
for i in df.index:
    if df['A0smoking_amount'][i] == 'less_than_5':
        df1.loc[i, 'smoke_score'] = 1
    elif df['A0smoking_amount'][i] == '5_to_9':
        df1.loc[i, 'smoke_score'] = 2
    elif df['A0smoking_amount'][i] == '10_to_14':
        df1.loc[i, 'smoke_score'] = 3
    elif df['A0smoking_amount'][i] == '15_to_24':
        df1.loc[i, 'smoke_score'] = 4
    elif df['A0smoking_amount'][i] == '25_or_more':
        df1.loc[i, 'smoke_score'] = 5
    elif np.isnan(df['A0smoking_amount'][i]) == True:
        df1.loc[i, 'smoke_score'] = np.nan
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')

# social interactions
df1['socialI'] = np.nan
for i in df.index:
    if df['A0social_interactions'][i] == 'none':
        df1.loc[i, 'socialI'] = 0
    elif df['A0social_interactions'][i] == '1_to_3_times':
        df1.loc[i, 'socialI'] = 1
    elif df['A0social_interactions'][i] == '4_to_6_times':
        df1.loc[i, 'socialI'] = 2
    elif df['A0social_interactions'][i] == '7_to_10_times':
        df1.loc[i, 'socialI'] = 3
    elif df['A0social_interactions'][i] == 'more_than_10_times':
        df1.loc[i, 'socialI'] = 4
    else:
        df1.loc[i, 'socialI'] = np.nan
        
## loneliness
df1['lonely'] = df['A0loneliness']

## sleep problems
df2 = pd.DataFrame(df['A0sleep_problems'])
df2 = df['A0sleep_problems'].str.split(',', expand=True)
missing_data_sleep = df2.isnull()
for i in range(len(df2)):
    for j in range(6):
        if missing_data_sleep[j][i] == True:
            df2[j][i] = 0
        elif missing_data_sleep[j][i] == False:
            df2[j][i] = 1
            
df1['sleep_problem'] = df2.sum(axis=1)



# Diet
## Fruit and veggies
df1['fruit_veg'] = np.nan
for i in df.index:
    if df['A0portions_fruit_veg'][i] == 'one':
        df1.loc[i, 'fruit_veg'] = 1
    elif df['A0portions_fruit_veg'][i] == 'two':
        df1.loc[i, 'fruit_veg'] = 2
    elif df['A0portions_fruit_veg'][i] == 'three':
        df1.loc[i, 'fruit_veg'] = 3
    elif df['A0portions_fruit_veg'][i] == 'four':
        df1.loc[i, 'fruit_veg'] = 4
    elif df['A0portions_fruit_veg'][i] == 'five':
        df1.loc[i, 'fruit_veg'] = 5
    elif df['A0portions_fruit_veg'][i] == 'more_than_5':
        df1.loc[i, 'fruit_veg'] = 6
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')

## Portions fish and nuts
df_fish = pd.get_dummies(df['A0portions_fish'])
df_nuts = pd.get_dummies(df['A0portion_nuts'])

df1['portion_fish'] = df_fish[True].astype('float')
df1['portion_nuts'] = df_nuts[True].astype('float')

##Alcohol
df1['alcohol_unit'] = df['A0alcohol_total_units']

## Water
df1['glasses_water'] = np.nan
for i in df.index:
    if df['A0glasses_water'][i] == 'less_than_3':
        df1.loc[i, 'glasses_water'] = 0
    elif df['A0glasses_water'][i] == '4to6':
        df1.loc[i, 'glasses_water'] = 1
    elif df['A0glasses_water'][i] == '7to9':
        df1.loc[i, 'glasses_water'] = 2
    elif df['A0glasses_water'][i] == '10to12':
        df1.loc[i, 'glasses_water'] = 3
    elif df['A0glasses_water'][i] == '13to15':
        df1.loc[i, 'glasses_water'] = 4
    elif df['A0glasses_water'][i] == 'more_than_16':
        df1.loc[i, 'glasses_water'] = 5
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')
        
# Mental stimulation
## Reading
df1['time_reading'] = np.nan
for i in df.index:
    if df['A0time_reading'][i] == 'no_time_at_all':
        df1.loc[i, 'time_reading'] = 0
    elif df['A0time_reading'][i] == 'less_than_1_hour':
        df1.loc[i, 'time_reading'] = 1
    elif df['A0time_reading'][i] == '1to3_hours':
        df1.loc[i, 'time_reading'] = 2
    elif df['A0time_reading'][i] == 'more_than_3':
        df1.loc[i, 'time_reading'] = 3
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')
        
## Mental stimulation
df1['time_mental_stimulation'] = np.nan
for i in df.index:
    if df['A0time_mental_stimulation'][i] == 'no_time_at_all':
        df1.loc[i, 'time_mental_stimulation'] = 0
    elif df['A0time_mental_stimulation'][i] == 'less_than_1_hour':
        df1.loc[i, 'time_mental_stimulation'] = 1
    elif df['A0time_mental_stimulation'][i] == '1to3_hours':
        df1.loc[i, 'time_mental_stimulation'] = 2
    elif df['A0time_mental_stimulation'][i] == 'more_than_3':
        df1.loc[i, 'time_mental_stimulation'] = 3
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')
        
## Instrument
df1['time_instrument'] = np.nan

for i in df.index:
    if df['A0time_instrument'][i] == 'no_time_at_all':
        df1.loc[i, 'time_instrument'] = 0
    elif df['A0time_instrument'][i] == 'less_than_1_hour':
        df1.loc[i, 'time_instrument'] = 1
    elif df['A0time_instrument'][i] == '1to3_hours':
        df1.loc[i, 'time_instrument'] = 2
    elif df['A0time_instrument'][i] == 'more_than_3':
        df1.loc[i, 'time_instrument'] = 3
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')
        
## Skill
df1['time_skill'] = np.nan

for i in df.index:
    if df['A0time_skill'][i] == 'no_time_at_all':
        df1.loc[i, 'time_skill'] = 0
    elif df['A0time_skill'][i] == 'less_than_1_hour':
        df1.loc[i, 'time_skill'] = 1
    elif df['A0time_skill'][i] == '1to3_hours':
        df1.loc[i, 'time_skill'] = 2
    elif df['A0time_skill'][i] == 'more_than_3':
        df1.loc[i, 'time_skill'] = 3
    elif np.isnan(df['A0time_skill'][i]) == True:
        df1.loc[i, 'time_skill'] = 6
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')

## Mood & relaxed
df1[['relaxed_scale', 'mood_rate',
       'enjoyment_activities']] = df[['A0relaxed_scale', 'A0mood_rate',
       'A0enjoyment_activities']]
                                      
                                      
# Model
##Cognitive score
#df1['Swift_medianRT']= df['Swift_medianRT']

df1.to_csv('/Users/malika/Documents/Five Lives/Analysis/data/DataframeLifestyle1.csv', index=True)

