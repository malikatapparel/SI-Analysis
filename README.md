# SI-Analysis
Codes for SI Data analysis
Those codes will clean the latest exported data from AWS and provide a clean dataset of the 3 assessments
:: 1_DataCleaner.py: using the latest questionnaire dataset (e.g "2022-10-28_questionaire.csv") as well as the excluded IDs list, will: 1) Exculde IDs / 2) replace True/False by 0s and 1s
:: 2_DataCleaner.py: Code to run for each Assessment number separately (to run 3x if you want all assessments data). Uses the previously generated dataset (e.g "df0.csv") and recodes several variables
:: 3_DataMerger.py: will merge df0_clean.csv, df1_clean.csv and df2_clean.csv
