import pandas as pd
import numpy as np
import csv
import functools
from models import *

# combine device, patient, and followup csvs into one
# and then separate pre- and post-implant variables
# @input:
# - patient_INTERMACS_Data_Dictionary.csv
# - device_INTERMACS_Data_Dictionary.csv
# - followup_INTERMACS_Data_Dictionary.csv
# @output
# - all (pickled df): three csvs joined
# - input (pickled df): pre-implant-related columns, match with ouput on 'OPER_ID'
# - output (pickled df): post-implant-related columns, match with input on 'OPER_ID'

blood_type_dict = {
    1: 'O',
    2: 'A',
    3: 'B',
    4: 'AB',
    998: '-',
}

nyha_dict = {
    'Class I: No Limitation or Symptoms': 'ClassI',
    'Class II: Slight Limitation': 'ClassII',
    'Class III: Marked Limitation': 'ClassIII',
    'Class IV: Unable to Carry on Minimal Physical Activity': 'ClassIV',
    'Unknown': '-',
}

num_card_hosp_dict = {'0-1': 0, '2-3': 2, '4 or more': 4,'-': np.nan,}

time_card_dgn_dict = {
    '< 1 month': 0,
    '1 month - 1 year': 1,
    '1-2 years': 12,
    '> 2 years': 24,
    '-': np.nan,
}


def feature_dict():
    """
    imports INTERMACS Data Dictionary with Format Value tables into one dict
    @returns:
    - a dictionary that combines all three data dictinoaries
    - pcols: patient table columns
    - dplus (list): device table columns that's not in patient table
    - fplus: followup talbe collumns that are not in patient OR device OR events table
    """

    # import data dictionary csv as dict
    # returns {'VARIABLE': {'TABLE': csv_name, 'TYPE': ..., 'FORMAT_VALUE', ....}}
    p = pd.read_csv('static/data/patient_INTERMACS_Data_Dictionary.csv',
                    index_col='VARIABLE', encoding="ISO-8859-1").to_dict(orient='index')
    d = pd.read_csv('static/data/device_INTERMACS_Data_Dictionary.csv',
                    index_col='VARIABLE', encoding="ISO-8859-1").to_dict(orient='index')
    f = pd.read_csv('static/data/followup_INTERMACS_Data_Dictionary.csv',
                    index_col='VARIABLE', encoding="ISO-8859-1").to_dict(orient='index')

    # print (len(d.keys() | p.keys() | e.keys() | f.keys())) # 1080 columns in total

    return {**d, **p, **f} # 1080 columns in total


def import_and_join():
    """
    import three datasets (device, patient, events) and join on "OPER_ID"
    """
    # load csvs
    # leave only pre-implant rows in follow-up data
    pdf = pd.read_csv('static/data/patientnewdata.csv', na_values=[
                      'Missing', '.', '.U', '998'], encoding="ISO-8859-1")  # 456 cols 17075 rows (patients)
    ddf = pd.read_csv('static/data/devicenewdata.csv', na_values=[
                      'Missing', '.', '.U', '998'], encoding="ISO-8859-1")  # 19207 rows (implants)
    fdf = pd.read_csv('static/data/followupnewdata.csv',
                      na_values=['Missing', '.', '.U', '998'], encoding="ISO-8859-1")
    fdf = fdf.loc[fdf['FORM_ID'] == 'Pre-Implant']  # 19207 rows (implants)

    # merge device and followup tables
    df = ddf.merge(fdf[list(fdf.columns.difference(ddf.columns)) +
                       ['OPER_ID']], on='OPER_ID', how='left', suffixes=['', ''])

    # append patient table to the joined table
    df = df.merge(pdf[list(pdf.columns.difference(df.columns)) + ['OPER_ID']],
                  on='OPER_ID', how='left', suffixes=['', ''])  # 1447 cols 19207rows (implants)

    print(len(df.columns.values), len(df))  # 693 cols 19207 rows (implants)

    """
    leave only patient_IDs with death records
    """
    # 4989 rows are death records
    # == 4989 patients have death records in the df
    dead_pts = df[df['DEAD'] == 1.0]['PATIENT_ID'].unique()
    df = df[df['PATIENT_ID'].isin(dead_pts)]  # 5820 rows left

    # convert nominal data type/ coding
    df['BLOOD_TYPE'].replace(blood_type_dict, inplace=True)
    df['TIME_CARD_DGN'].replace(time_card_dgn_dict, inplace=True)
    df['NUM_CARD_HOSP'].replace(num_card_hosp_dict, inplace=True)
    df['NYHA'].replace(nyha_dict, inplace=True)

    """
    seperate columns about pre-implant conditions and post-implant outcomes
    """

    idx_col = ['OPER_ID', 'PATIENT_ID', 'FORM_ID']
    outcome_col = ['EXPLANT_DEVICE_TY', 'EXPLANT_REASON', 'EXPL_THROM', 'EXCH_NEW_STUDY', 'TXPL', 'EXPL', 'DEAD', 'INT_DEAD', 'INT_TXPL', 'INT_EXPL', 'DISCHARGE_TO', 'DISCHARGE_STATUS', 'DIS_INT_TRANSPLANT', 'DIS_INT_INV_CARD_PROC', 'DIS_INT_0', 'DIS_INT_NONE', 'DIS_INT_SURG_PROC_DEV', 'DIS_INT_SURG_PROC_NC', 'DIS_INT_SURG_PROC_OTHER', 'DIS_INT_SURG_PROC_0', 'DIS_INT_BLEED_GT_48', 'DIS_INT_BLEED_LE_48', 'DIS_INT_DRAINAGE', 'DIS_INT_AVS_REPAIR_NC', 'DIS_INT_AVS_REPAIR_WC',
                   'DIS_INT_AVS_REPLACE_BIO', 'DIS_INT_AVS_REPLACE_MECH', 'DIS_INT_MVS_REPAIR', 'DIS_INT_MVS_REPLACE_BIO', 'DIS_INT_MVS_REPLACE_MECH', 'DIS_INT_TVS_REPAIR_DEVEGA', 'DIS_INT_TVS_REPAIR_OTHER', 'DIS_INT_TVS_REPAIR_RING', 'DIS_INT_TVS_REPLACE_BIO', 'DIS_INT_TVS_REPLACE_MECH', 'DIS_INT_PVS_REPAIR', 'DIS_INT_PVS_REPLACE_BIO', 'DIS_INT_PVS_REPLACE_MECH', 'DIS_INT_CARD_OTHER', 'DIS_INT_CARD_0', 'DIS_INT_REINTUBATION', 'DIS_INT_DIALYSIS', 'DIS_INT_BRONCHOSCOPY', 'DIS_INT_OTHER']
    input_col = list(set(df.columns.get_values()) - set(outcome_col) - set(idx_col))

    df.to_csv('static/data/all_180301.csv', encoding='utf-8')
    df.loc[:, outcome_col + ['OPER_ID']].to_csv('static/data/outcome_180301.csv', encoding='utf-8') # excludes PATIENT_ID and FORM_ID
    df.loc[:, input_col + ['OPER_ID']].to_csv('static/data/input_180301.csv', encoding='utf-8') # excludes PATIENT_ID and FORM_ID


feature_dict = feature_dict()
import_and_join()
