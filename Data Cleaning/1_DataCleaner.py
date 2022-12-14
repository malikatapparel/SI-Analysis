#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:04:06 2022

@author: malika
"""


import pandas as pd
import numpy as np
from scipy import stats
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

# functions
## Encode true/false variables in 0 and 1
def ReplaceTrueFalse(DataFrame, column):
    for i in DataFrame.index:
        if DataFrame.loc[i, column] == True:
            DataFrame.loc[i, column] = 1
        elif DataFrame.loc[i, column] == False:
            DataFrame.loc[i, column] = 0
        else:
            DataFrame.loc[i, column] = np.nan


## For all assessment using the above function
def ReplaceTrueFalseAll(Data, nude_column):
    AssIndex = ['A0', 'A1', 'A2']
    for i in AssIndex:
        y = i + nude_column
        ReplaceTrueFalse(Data, y)

#Read latest dataframe
path = "/Users/malika/Documents/Five Lives/Analysis/data/oxford 28102022/2022-10-28_questionaire.csv"
Data = pd.read_csv(path)

# Read excluded IDs file
pathIDS = "/Users/malika/Documents/Five Lives/Analysis/data/excludedIDs_A0A2.csv"
dfex = pd.read_csv(pathIDS)
#Exclude wrong indexes and set index with user id
Data = Data.set_index('participant_id')
id_excluded = []
for i in dfex['Excluded ID']:
    x = str(i)
    id_excluded.append(x)

## Drop wrong user ID
all_index = Data.index
Index = []
for x in Data.index:
    if str(x) in id_excluded:
        Index.append(x)
        Data = Data.drop(x, axis=0)
df = Data

##Exclude A3 data
for i in df.columns:
    if i[0:2] == 'A3':
        df = df.drop([i], axis=1)

#True/False variables

TF_values = ['A0work_shifts', 'A0assistance_checks', 'A0assistance_bills', 'A0assistance_checkbook', 'A0assistance_events','A0assistance_memory_appointments','A0assistance_memory_occasions','A0assistance_memory_holidays', 'A0assistance_memory_medications','A0assistance_memory_recent','A0assistance_memory_names','A0relative_dementia', 
             'A0relative_dementia_65', 'A0heart_attack','A0cardiac_arrest','A0congestive_heart_failure','A0hypertension','A0atrial_fibrillation', 
             'A0peripheral_vascular_disease','A0angina','A0heart_issues_other','A0heart_procedures','A0cardiac_bypass','A0angioplasty','A0endarterectomy','A0stent','A0heart_procedures_other','A0heart_medications', 'A0anticoagulants','A0blood_glucose_estimation','A0blood_glucose_exact', 'A0diagnosis_diabetes','A0diabetes_medication',
             'A0insulin', 'A0stroke','A0mini_stroke','A0b12','A0b12_replacement','A0thyroid_disease', 'A0thyroxine_replacement', 'A0renal_problem', 'A0dialysis', 'A0alcohol_disorder','A0hearing_problems','A0hearing_problems_treated', 'A0visual_problems', 'A0visual_problems_treated', 'A0colour_blind', 'A0low_mood', 'A0diagnosis_depression', 'A0schizophrenia', 'A0psychosis', 'A0bipolar_disorder',
             'A0anxiety', 'A0adhd', 'A0ptsd', 'A0ocd', 'A0personality_disorder', 'A0psychiatric_diagnosis_other', 'A0parkinsons' ,'A0epilepsy'
,'A0multiple_sclerosis'
,'A0migraines'
,'A0diagnosis_neurological_diseases_other'
,'A0head_injury'
,'A0loss_of_consciousness'
,'A0dyslexia'
,'A0bowel_disease_IBS'
,'A0diagnosis_dementia'
,'A0diagnosis_mci'
,'A0lipid_lowering_medications'
,'A0anti_inflammatory_medications'
,'A0anti_hypertensives_medications'
,'A0antidepressants'
,'A0anxiolytic'
,'A0antiparkinson_medications'
,'A0glucose_medications'
,'A0regular_medications_other'
,'A0today_date'
,'A0typing_issue_arthritis_injury'
,'A0typing_issue_tremor'
,'A0typing_issue_eyesight'
,'A0typing_issue_dexterity'
,'A0typing_issue_neurological_disorders'
,'A0typing_issue_other'
,'A0diff_falling_asleep'
,'A0diff_staying_asleep'
,'A0waking_early'
,'A0moving_during_sleep'
,'A0insomnia'
,'A0sleep_apnea'
,'A0sleep_issue_other'
,'A0portions_fish'
,'A0portion_nuts']

for i in range(len(TF_values)):
    TF_values[i] = TF_values[i][2:]

for i in TF_values:
    ReplaceTrueFalseAll(df, i)     

##Separate datasets
A0 = []
for i in df.columns:
    if i[:2]== 'A0':
        A0.append(i)
df0 = df[A0]
df0.to_csv('/Users/malika/Documents/Five Lives/Analysis/data/SI_Datasets/df0.csv', index=True)

##A1
A1 = []
for i in df.columns:
    if i[:2]== 'A1':
        A1.append(i)
df1 = df[A1]
df1.to_csv('/Users/malika/Documents/Five Lives/Analysis/data/SI_Datasets/df1.csv', index=True)

##A2
A2 = []
for i in df.columns:
    if i[:2]== 'A2':
        A2.append(i)
df2 = df[A2]
df2.to_csv('/Users/malika/Documents/Five Lives/Analysis/data/SI_Datasets/df2.csv', index=True)



# Join datasets
df12 = df0.join(df1)
df12 = df12[df12['A1date_of_birth'].notna()]

df123 = df12.join(df2)
df123 = df123[df123['A2date_of_birth'].notna()]






