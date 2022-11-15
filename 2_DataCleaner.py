#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:17:38 2022

@author: malika
"""


import pandas as pd
import numpy as np
from scipy import stats
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

path = '/Users/malika/Documents/Five Lives/Analysis/data/SI_Datasets/'

Data = pd.read_csv(path + 'df2.csv')
A = 'A2'
OutputDataset = 'df2_clean.csv'

#Data to group
A0Assist = ['A0assistance_checks', 'A0assistance_bills', 'A0assistance_checkbook', 'A0assistance_events', 'A0assistance_memory_appointments', 'A0assistance_memory_occasions', 'A0assistance_memory_holidays', 'A0assistance_memory_medications', 'A0assistance_memory_recent', 'A0assistance_memory_names']
A0heart = ['A0heart_attack', 'A0cardiac_arrest', 'A0congestive_heart_failure', 'A0hypertension', 'A0atrial_fibrillation', 'A0peripheral_vascular_disease', 'A0angina', 'A0heart_issues_other']
A0heart_treatment = ['A0heart_procedures', 'A0cardiac_bypass', 'A0angioplasty', 'A0endarterectomy', 'A0stent', 'A0heart_procedures_other', 'A0heart_medications', 'A0anticoagulants']
A0mental_problem = ['A0low_mood', 'A0diagnosis_depression', 'A0schizophrenia', 'A0psychosis', 'A0bipolar_disorder', 'A0anxiety', 'A0adhd', 'A0ptsd', 'A0ocd']
A0typing_issues = ['A0typing_issue_arthritis_injury', 'A0typing_issue_tremor', 'A0typing_issue_eyesight', 'A0typing_issue_dexterity', 'A0typing_issue_neurological_disorders', 'A0typing_issue_other']
DFLifestyle = ['A0physical_activity_duration', 'A0physical_activity_intensity', 'A0smoking', 'A0smoking_amount', 'A0social_interactions', 'A0loneliness', 'A0hours_sleep', 'A0tiredness', 'A0sleep_problems', 'A0diff_falling_asleep', 'A0diff_staying_asleep', 'A0waking_early', 'A0moving_during_sleep', 'A0insomnia', 'A0sleep_apnea', 'A0sleep_issue_other', 'A0portions_fruit_veg', 'A0portions_fish', 'A0portion_nuts', 'A0small_wine_units', 'A0standard_wine_units', 'A0large_wine_units', 'A0beer_bottle_units', 'A0beer_can_units', 'A0pint_lowStrength_units', 'A0pint_highStrength_units', 'A0small_shot_units', 'A0large_measure_units', 'A0alcopop_units', 'A0alcohol_total_units', 'A0glasses_water', 'A0mixing_herbicides', 'A0time_reading', 'A0time_mental_stimulation', 'A0time_instrument', 'A0time_skill', 'A0relaxed', 'A0relaxed_scale', 'A0mood_rate', 'A0enjoyment_activities', 'A0discussion_with_doctor_likelihood_scale', 'A0family_member_repeat_statements_frequency', 'A0family_member_date_frequency', 'A0family_member_check_date_frequency', 'A0family_member_finance_frequency']
DFLifestyle_todelete= ['A0lifestyle_improvement_likelihood_scale', 'A0moving_during_sleep', 'A0insomnia', 'A0sleep_apnea', 'A0sleep_issue_other', 'A0discussion_with_doctor_likelihood_scale', 'A0family_member_repeat_statements_frequency', 'A0family_member_date_frequency', 'A0family_member_check_date_frequency', 'A0family_member_finance_frequency', 'A0family_member_direction_frequency']




#Age
Age_col = A + '_age'
Date_col = A + 'date_of_birth'

Data[Age_col] = np.nan
for i in Data.index:
    if type(Data[Date_col][i]) == str:
        birth = Data[Date_col][i][6:10]
        age = (2022 - float(birth))
        Data.loc[i, Age_col] = age
    elif np.isnan(Data[Date_col][i]) == True:
        Data.loc[i, Age_col] = np.nan
    
Data= Data.drop(Date_col, axis=1)

#Date last job
DataJob_col = A + 'time_since_last_job'
TimyJob_col = A + '_years_since_last_job'

Data[TimyJob_col] = np.nan
for i in Data.index:
    if type(Data[DataJob_col][i]) == str:
        lastjob = Data[DataJob_col][i][6:10]
        age = (2022 - float(lastjob))
        Data.loc[i, TimyJob_col] = age
    elif np.isnan(Data[DataJob_col][i]) == True:
        Data.loc[i, TimyJob_col] = np.nan

Data= Data.drop(DataJob_col, axis=1)

#create a simplified dataset 'Sdf'
Sdf = Data

##Assistance

Assist = []
for i in A0Assist:
    x = i[2:]
    Assist.append(x)
    
for i in range(len(Assist)):
    Assist[i] = A + Assist[i]

Sdf[ A+ 'Ass_all'] = np.sum(Data[Assist], axis=1)
Sdf = Sdf.drop(Sdf[Assist], axis=1)

## Heart problems

Heart = []
for i in A0heart:
    x = i[2:]
    Heart.append(x)
    
for i in range(len(Heart)):
    Heart[i] = A + Heart[i]

Sdf[ A+ 'heart_problem_sum'] = np.sum(Sdf[Heart], axis=1)
Sdf = Sdf.drop(Heart, axis=1)

## Heart treatment

HeartT = []
for i in A0heart_treatment:
    x = i[2:]
    HeartT.append(x)
    
for i in range(len(HeartT)):
    HeartT[i] = A + HeartT[i]

Sdf[ A+ 'heart_treatment_sum'] = np.sum(Sdf[HeartT], axis=1)
Sdf = Sdf.drop(HeartT, axis=1)

## Mental issues
MentalI = []
for i in A0mental_problem:
    x = i[2:]
    MentalI.append(x)
    
for i in range(len(MentalI)):
    MentalI[i] = A + MentalI[i]

Sdf[ A+ 'mental_problem_sum'] = np.sum(Sdf[MentalI], axis=1)
Sdf = Sdf.drop(MentalI, axis=1)

## Typing issues
TypingI = []

for i in A0typing_issues:
    x = i[2:]
    TypingI.append(x)
    
for i in range(len(TypingI)):
    TypingI[i] = A + TypingI[i]

Sdf[ A+ 'typing_problem_sum'] = np.sum(Sdf[TypingI], axis=1)
Sdf = Sdf.drop(TypingI, axis=1)


##Lifestyle
Lifestyle = [i for i in DFLifestyle if i not in DFLifestyle_todelete]

for i in range(len(Lifestyle)):
    Lifestyle[i] = Lifestyle[i][2:]
    

for i in range(len(Lifestyle)):
    Lifestyle[i] = A + Lifestyle[i]

df1 = Sdf.drop(Lifestyle, axis =1)

### PA
y = A + 'PA_duration'

df1[y] = np.nan
x = A + 'physical_activity_duration'
for i in Sdf.index:
    if Sdf[x][i] == 'none':
        df1.loc[i, y] = 0
    elif Sdf[x][i] == 'less_than_1h':
        df1.loc[i, y] = 1
    elif Sdf[x][i] == '1h_to_2h':
        df1.loc[i, y] = 2
    elif Sdf[x][i] == '2h_to_3h':
        df1.loc[i, y] = 3
    elif Sdf[x][i] == 'more_than_3h':
        df1.loc[i, y] = 4

##PA intensity
y = A + 'PA_intensity'
df1[y] = np.nan
x = A+ 'physical_activity_intensity'
for i in Sdf.index:
    if Sdf[x][i] == 'moderate':
        df1.loc[i, y] = 1
    elif Sdf[x][i] == 'mixed':
        df1.loc[i, y] = 2
    elif Sdf[x][i] == 'intense':
        df1.loc[i, y] = 3
    else:
        df1.loc[i, y] = 0
        
df1[A +'PA_score'] = np.nan
for i in df1.index:
    df1.loc[i, A +'PA_score'] = (df1.loc[i, A+'PA_duration']*df1.loc[i, A+'PA_intensity'])
    
    
### Smoke
## smoking score is smoking/increasing values: lowest score indicates a heavier smoker 
y = A +'smoke_score'
df1[y] = np.nan
x = A + 'smoking_amount'
for i in Sdf.index:
    if Sdf[x][i] == 'less_than_5':
        df1.loc[i, y] = 1
    elif Sdf[x][i] == '5_to_9':
        df1.loc[i, y] = 2
    elif Sdf[x][i] == '10_to_14':
        df1.loc[i, y] = 3
    elif Sdf[x][i] == '15_to_24':
        df1.loc[i, y] = 4
    elif Sdf[x][i] == '25_or_more':
        df1.loc[i, y] = 5
    elif np.isnan(Sdf[x][i]) == True:
        df1.loc[i, y] = np.nan
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')
        

# social interactions
y = A + 'socialI'
df1[y] = np.nan
x = A + 'social_interactions'
for i in Sdf.index:
    if Sdf[x][i] == 'none':
        df1.loc[i, y] = 0
    elif Sdf[x][i] == '1_to_3_times':
        df1.loc[i, y] = 1
    elif Sdf[x][i] == '4_to_6_times':
        df1.loc[i, y] = 2
    elif Sdf[x][i] == '7_to_10_times':
        df1.loc[i, y] = 3
    elif Sdf[x][i] == 'more_than_10_times':
        df1.loc[i, y] = 4
    else:
        df1.loc[i, y] = np.nan
## loneliness
df1[A +'lonely'] = Sdf[A + 'loneliness']

## sleep problems
df2 = pd.DataFrame(Sdf[A +'sleep_problems'])
df2 = Sdf[A + 'sleep_problems'].str.split(',', expand=True)           
df1[A+'sleep_problems'] = df2.count(axis=1)   

# Diet
## Fruit and veggies
x = A + 'portions_fruit_veg'
y = A + 'fruit_veg'
df1[y] = np.nan
for i in Sdf.index:
    if Sdf[x][i] == 'one':
        df1.loc[i, y] = 1
    elif Sdf[x][i] == 'two':
        df1.loc[i, y] = 2
    elif Sdf[x][i] == 'three':
        df1.loc[i, y] = 3
    elif Sdf[x][i] == 'four':
        df1.loc[i, y] = 4
    elif Sdf[x][i] == 'five':
        df1.loc[i, y] = 5
    elif Sdf[x][i] == 'more_than_5':
        df1.loc[i, y] = 6
    elif np.isnan(Sdf[x][i]) == True:
        df1.loc[i, y] = np.nan
    else:
        raise ValueError('Value could not be attributed, as it is not in the list') 

## Portions fish and nuts
# df_fish = pd.get_dummies(Sdf[A + 'portions_fish'])
# df_nuts = pd.get_dummies(Sdf[A + 'portion_nuts'])

# df1[A+'portion_fish'] = df_fish[True].astype('float')
# df1[A+'portion_nuts'] = df_nuts[True].astype('float')

##Alcohol
df1[A+'alcohol_unit'] = Sdf[A + 'alcohol_total_units']
## Water

x = A + 'glasses_water'
y =  A + 'glasses_water'

df1[y] = np.nan
for i in Sdf.index:
    if Sdf[y][i] == 'less_than_3':
        df1.loc[i, y] = 0
    elif Sdf[x][i] == '4to6':
        df1.loc[i, y] = 1
    elif Sdf[y][i] == '7to9':
        df1.loc[i, y] = 2
    elif Sdf[x][i] == '10to12':
        df1.loc[i, y] = 3
    elif Sdf[x][i] == '13to15':
        df1.loc[i, y] = 4
    elif Sdf[x][i] == 'more_than_16':
        df1.loc[i, y] = 5
    elif np.isnan(Sdf[x][i]) == True:
        df1.loc[i, y] = np.nan
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')

x = A + 'time_reading'

df1[x] = np.nan
for i in Sdf.index:
    if Sdf[x][i] == 'no_time_at_all':
        df1.loc[i, x] = 0
    elif Sdf[x][i] == 'less_than_1_hour':
        df1.loc[i, x] = 1
    elif Sdf[x][i] == '1to3_hours':
        df1.loc[i, x] = 2
    elif Sdf[x][i] == 'more_than_3':
        df1.loc[i, x] = 3
    elif np.isnan(Sdf[x][i]) == True:
        df1.loc[i, x] = np.nan
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')
    
## Mental stimulation
x = A + 'time_mental_stimulation'
df1[x] = np.nan

for i in Sdf.index:
    if Sdf[x][i] == 'no_time_at_all':
        df1.loc[i, x] = 0
    elif Sdf[x][i] == 'less_than_1_hour':
        df1.loc[i, x] = 1
    elif Sdf[x][i] == '1to3_hours':
        df1.loc[i, x] = 2
    elif Sdf[x][i] == 'more_than_3':
        df1.loc[i, x] = 3
    elif np.isnan(Sdf[x][i]) == True:
        df1.loc[i, x] = np.nan
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')
        
## Instrument
x = A + 'time_instrument'
df1[x] = np.nan

for i in Sdf.index:
    if Sdf[x][i] == 'no_time_at_all':
        df1.loc[i, x] = 0
    elif Sdf[x][i] == 'less_than_1_hour':
        df1.loc[i, x] = 1
    elif Sdf[x][i] == '1to3_hours':
        df1.loc[i, x] = 2
    elif Sdf[x][i] == 'more_than_3':
        df1.loc[i, x] = 3
    elif np.isnan(Sdf[x][i]) == True:
        df1.loc[i, x] = np.nan
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')
        
## Skill
x = A + 'time_skill'
df1[x] = np.nan
for i in Sdf.index:
    if Sdf[x][i] == 'no_time_at_all':
        df1.loc[i, x] = 0
    elif Sdf[x][i] == 'less_than_1_hour':
        df1.loc[i, x] = 1
    elif Sdf[x][i] == '1to3_hours':
        df1.loc[i, x] = 2
    elif Sdf[x][i] == 'more_than_3':
        df1.loc[i, x] = 3
    elif np.isnan(Sdf[x][i]) == True:
        df1.loc[i, x] = 6
    else:
        raise ValueError('Value could not be attributed, as it is not in the list')

## Mood & relaxed
df1[[A +'relaxed_scale', A +'mood_rate',
      A + 'enjoyment_activities']] = Sdf[[A + 'relaxed_scale', A + 'mood_rate',
       A + 'enjoyment_activities']]
                                    
                                          
##Save clean dataset
df1.to_csv(path + OutputDataset, index=True)                        
                                          
