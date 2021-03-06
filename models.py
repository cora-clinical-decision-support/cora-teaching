#!/usr/bin/python
# -*- coding: utf-8 -*-
# All the models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import tree
from sklearn import preprocessing, metrics
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import math
import random as rand 
from point import Point
import matlab.engine 
import tensorflow as tf 
from collections import Counter
import logging
from scipy.sparse import csr_matrix

predictorNames = ['PREV_CARD_OP_CON_SUR_AP_SH', 'MED_PRE_IMP_PHOSPHODIESTERASE', 'INTERVENTION_48_HRS_IABP', 'CC_HIST_LYMPHOMA_LEUKEMIA_M', 'HIV_ART_TRUVADA', 'SELF_CARE', 'NUM_CARD_HOSP', 'THROMBIN_INHIB_DRUG_ASPIRIN', 'IV_INO_THERAPY_AGENTS_LEVOSI', 'OUTCOME_I', 'CC_HIST_HIV', 'CC2_HIST_DRUG_USE_M', 'CC_HIST_ATRIAL_ARRHYTHMIA', 'CONCOM_SURG_AVS_REP_MECH', 'CONCOM_SURG_TVS_REP_BIO', 'MED_PRE_IMP_NITRIC_OXIDE', 'CC_HISTORY_HEPATITIS_M', 'CC_CURR_SMOKING_M', 'CC_HIST_SMOKING_M', 'ALBUMIN_G_L', 'MED_PRE_IMP_AMIODARONE', 'BUN_MMOL_L', 'CC2_LARGE_BMI_M', 'EVENT_HOSP_LVAD', 'RV_CENTRIFUGAL', 'THROMBIN_INHIB_DRUG_DIPYRID', 'CC2_HIST_LYMPHOMA_LEUKEMIA_M', 'CRP', 'KCCQ12SL', 'BLOOD_TYPE', 'AORTIC_REGURG', 'MITRAL_REGURG', 'WORK_INCOME', 'CC_ALLOSENSITIZATION_M', 'MED_PRE_IMP_NESERITIDE', 'CV_PRES', 'KCCQ12PL', 'HIV_ART_MARAVIROC', 'INT_DPT', 'SEC_DGN_REST_MYO_AMYLOID', 'INTERVENTION_48_HRS_CABG', 'MODIFIER_FF_HOME', 'CONCOM_SURG_VSD_CLOSE', 'PREV_CARDIAC_OPER_PREV_TRANS', 'CONCOM_SURG_AVS_C', 'VO2_MAX', 'EVENT_HOSP_DIALYSIS', 'CONCOM_SURG_TVS_R_OTHER', 'WBC_I', 'HIV_ART_ATRIPLA', 'EQ_INDEX', 'CONSENT_WITHDRAWN', 'THERMOMETER_I', 'BNP_I', 'CC2_HIST_ALCOHOL_ABUSE_M', 'CONCOM_SURG_NONE', 'CC2_HEP_INDUCED_THROMBO_M', 'BUN_I', 'MED_PRE_IMP_WARFARIN', 'CURRENT_ICD', 'CC_FREQUENT_ICD_SHOCKS', 'TRAIL_MAKING_TIME', 'HIV_INF_PRO_PENTAMIDINE', 'INFECTION_TYPE', 'TRAIL_MAKING_STATUS', 'CC2_HISTORY_GI_ULCERS', 'AGE_GRP', 'PREV_CARDIAC_OPER_TAH', 'PILURICACID_MG_I', 'VERSION', 'REASON_ADMINISTRATIVE', 'CC_MAJOR_STROKE_M', 'DIS_INT_UNKNOWN', 'MED_PRE_IMP_ACE_INHIBITORS', 'CC_MUSCSKELETAL_LIMIT_M', 'HIV_INF_PRO_AZITHROMYCIN', 'PREV_CARDIAC_OPER_PREV_ECMO', 'CC_MALNUTRITION_CACHEXIA_M', 'HIV_ART_INDINAVIR', 'SEC_DGN_SV_HETEROTAXY', 'LDH_I', 'CONCOM_SURG_MVS_REPAIR', 'HIV_ART_RALTEGRAVIR', 'CC_HIST_SOLID_ORGAN_CNCR_M', 'EVENT_HOSP_ANEURS', 'GAITSPEED_I', 'PREV_CARDIAC_OPER_AVR', 'PREV_CARD_OP_CON_SUR_CLASSIC', 'CC_LTD_COG_UNDERSTND_M', 'EVENT_HOSP_CAR_SUR_OTHER', 'SEC_DGN_BV_TGA', 'PAIN', 'HIV_ART_DIDANOSINE', 'IV_INO_THERAPY_AGENTS_EPINEPH', 'VAD_STUDY', 'CC_OTHER_CO_MORBIDITY_M', 'DISCONTINUE_INOTROPES', 'MED_PRE_IMP_BETA_BLOCKERS', 'HIV_ART_SAQUINAVIR', 'MED_PRE_IMP_ANGIOSTENSIN', 'CC2_HIST_ATRIAL_ARRHYTHMIA', 'INTERVENTION_48_HRS_AVR', 'CC2_UNFAV_MEDIASTINAL_ANAT_M', 'PREV_CARDIAC_OPER_LVAD', 'HGT_CM', 'RV_DURABLE', 'CC2_ADVANCED_AGE_M', 'IMPL_YR', 'CC2_OTHER_CO_MORBIDITY_M', 'HIV_ART_COMBIVIR', 'PREV_CARD_OP_CON_SUR_EBSTEIN', 'OP2INTT', 'TRICUSPID_INSUFFICIENCY', 'ACUTE_CARE', 'RACE_UNKNOWN', 'GENDER', 'LVEF', 'CONCOM_SURG_AVS_REP_BIO', 'OP3EXPREA', 'PRO_BNP_NG_L', 'MED_PRE_IMP_CRT', 'CC_SEVERE_DIABETES_M', 'MED_PRE_IMP_LD_DOSE_MG_I', 'SEC_DGN_BV_CAVC', 'HIV_ART_RITONAVIR', 'SEC_DGN_COR_ART_DIS', 'PUL_DIA_PRES', 'FORMSTAT', 'CC2_MAJOR_STROKE_M', 'CONFIDENT_I', 'SEC_DGN_UNKNOWN', 'OP4INTR', 'EVENT_HOSP_VENTILAT', 'PREV_CARD_OP_CON_SUR_PA_BAND', 'CC2_FRAILTY_M', 'EVENT_HOSP_MVR', 'CC_CONTRAIN_TO_IMMSUPPRES_M', 'SGPT_ALT', 'CLOSE_FRIENDS_I', 'LVAD_CAN_OUTFLOW', 'CONCOM_SURG_PVS_REPAIR', 'AORTIC_INSUFFICIENCY', 'HGT_CM_I', 'MED_PRE_IMP_ANTEPLATELET', 'RACE_OTHER', 'CHOLESTEROL_I', 'RA_PRES', 'SYS_BP_I', 'CC_FRAILTY_M', 'KCCQ12', 'INR_I', 'CC_PULMONARY_DISEASE_M', 'CC2_HIST_SOLID_ORGAN_CNCR_M', 'HIV_ART_DELAVIRDINE', 'PUL_DIA_PRES_I', 'CPB_TIME', 'CC2_CHRONIC_INF_CONCERNS_M', 'HIV_ART_ENFUVIRTIDE', 'SEC_DGN_BV_CCT', 'HIV_INF_PRO_FLUCONAZOLE', 'OP4DEV_TY', 'ASSOC_FINDINGS_NONE', 'ASSOC_FINDINGS_TRIC_INSUF', 'CC2_CHRONIC_RENAL_DISEASE_M', 'KCCQ12QL', 'THROMBIN_INHIB_DRUG_CLOPID', 'BILI_TOTAL_MG_DL', 'MED_PRE_IMP_DIURETICTYPE_BUMET', 'CC2_LIMITED_SOCIAL_SUPPORT_M', 'MED_PRE_IMP_METALOZONE', 'ASSOC_FINDINGS_AORT_INSUF', 'INTERVENTION_48_HRS_VENTILAT', 'CC_NARCOTIC_DEPENDENCE', 'PREV_CARD_OP_CON_SUR_ART_SWCH', 'CRP_I', 'SEC_DGN_DIL_MYO_VIRAL', 'PLATELET_X10_9_L', 'CC2_NARCOTIC_DEPENDENCE', 'LV_CONT', 'EVENT_HOSP_FEED_TUBE', 'HIV_ART_RILPIVIRINE', 'SEC_DGN_DIL_MYO_ISCHEMIC', 'CC_HISTORY_GI_ULCERS', 'SEC_DGN_DIL_MYO_MYOCARDITIS', 'SEC_DGN_REST_MYO_OTHER', 'SURGICAL_APPROACH', 'POTASSIUM_I', 'HIV_ART_LAMIVUDINE', 'SEC_DGN_DIL_MYO_OTHER', 'VO2_MAX_I', 'KCCQ12SF', 'LDH_U_L', 'PREV_CARD_OP_CON_SUR_NORWOOD', 'POTASSIUM_MMOL_L', 'SGPT_ALT_I', 'CC2_OTH_CEREBROVASC_DISEASE_M', 'SEC_DGN_SV_UNSPECIFIED', 'PREV_CARD_OP_CON_SUR_DBL_SWCH', 'SEC_DGN_SV_OTHER', 'CREAT_UMOL_L', 'IV_INO_THERAPY', 'IMMEDIATE_CARE', 'PLATELET_X10_3_UL', 'SEC_DGN_DIL_MYO_ALCOHOLIC', 'REASON_NOTCOMPLETED', 'TREC_PT', 'HIV_ART_EFAVIRENZ', 'EVENT_HOSP_MAJOR_INF', 'BUN_MG_DL', 'PX_PROFILE', 'RV_CONT', 'WBC_X10_3_UL', 'VAD_STUDY_INDUSTRY', 'CC_OTH_CEREBROVASC_DISEASE_M', 'CC2_OTH_MAJOR_PSYCH_DIAG_M', 'SEC_DGN_BV_TOF', 'MED_PRE_IMP_DIURETICTYPE_TORSE', 'INTERVENTION_48_HRS_DIALYSIS', 'SEC_DGN_REST_MYO_IDIOPATHIC', 'CC_LIVER_DYSFUNCTION_M', 'KCCQ_PARENT_QUESTION', 'SEC_DGN_NONE', 'CC_LIMITED_SOCIAL_SUPPORT_M', 'LYMPH_CNT_I', 'CC2_PX_DOES_NOT_WANT_TX_M', 'WGT_LBS', 'EVENT_HOSP_POS_BLD_CULT', 'PREV_CARD_OP_CON_SUR_DKS', 'CONCOM_SURG_IABP', 'LUPUSANTICOAG', 'BNP_NG_L', 'CONCOM_SURG_AVS_NC', 'INT_TRPT', 'ACTIVITY_MAIN', 'CC_HEP_INDUCED_THROMBO_M', 'BILI_TOTAL_UMOL_L', 'EVENT_HOSP_AVR', 'HR_RATE_I', 'COPING_I', 'CC_OTH_MAJOR_PSYCH_DIAG_M', 'CONCOM_SURG_PVS_REP_MECH', 'PEAK_R', 'PRE_ALBUMIN_I', 'HEMOGLOBIN_MMOL_L', 'EVENT_HOSP_INTUB', 'MED_PRE_IMP_INOTROPE_INFUSION', 'PUL_WEDGE_PRES', 'PREV_CARDIAC_OPER_TVR', 'EVENT_HOSP_ECMO', 'INR', 'RVAD_CAN_OUTFLOW', 'MED_PRE_IMP_METALOZONEOPTIONS', 'CC_ADVANCED_AGE_M', 'PREV_CARD_OP_CON_SUR_ASD', 'CC2_CONTRAIN_TO_IMMSUPPRES_M', 'VOLUME_2015', 'CONCOM_SURG_ECMO_DECANN', 'PREV_CARD_OP_CON_SUR_OTHER', 'IV_INO_THERAPY_AGENTS_DOPA', 'HIV_ART_ATAZANAVIR', 'ECG_RHYTHM', 'EVENT_HOSP_MAJOR_MI', 'CHOLESTEROL_MMOL_L', 'CC_UNFAV_MEDIASTINAL_ANAT_M', 'ADMISSION_REASON', 'HX_MCSD', 'SEC_DGN_HYPER_CARDIOMYO', 'CC2_HIST_SMOKING_M', 'CC2_PERIPH_VASC_DISEASE_M', 'THROMBIN_INHIB_DRUG_HEPARIN', 'SEC_DGN_REST_MYO_ENDO_FIB', 'EVENT_HOSP_NONE', 'CPB_TIME_I', 'CONCOM_SURG_PVS_REP_BIO', 'MED_PRE_IMP_ALLOPURINOL', 'CREAT_MG_DL', 'CC2_ALLOSENSITIZATION_M', 'CC2_RPTD_NON_COMPLIANCE_M', 'LVEDD_I', 'EVENT_HOSP_TAH', 'CARD_BIOPSY', 'HIV_ART_ZIDOVUDINE', 'EVENT_HOSP_ULTRAFILT', 'IMMEDIATE_CARE_I', 'VOLUME_2014', 'LVAD_CAN_INFLOW', 'EVENT_HOSP_OTH_SUR_PROC', 'IV_INO_THERAPY_AGENTS_DOBUT', 'SODIUM_I', 'HR_RATE', 'CC2_MALNUTRITION_CACHEXIA_M', 'PREV_CARD_OP_CON_SUR_FONTAN', 'INTERVENTION_48_HRS_TAH', 'SEC_DGN_REST_MYO_SARCOIDOSIS', 'IV_INO_THERAPY_AGENTS_NOREPI', 'BNP_PG_ML', 'MODIFIER_TCS', 'LV_CENTRIFUGAL', 'HIV_ART_ABACAVIR', 'PREV_CARDIAC_OPER_ANEURS_DOR', 'HEMOGLOBIN_G_DL', 'PX_PRIMARY', 'MED_PRE_IMP_ALDOSTERONE', 'WBC_X10_9_L', 'SEC_DGN_DIL_MYO_FAMILIAL', 'PARENT_QUESTION', 'CIGARETTES_M', 'SEC_DGN_SV_PA_IVS', 'CC2_SEVERE_DIABETES_M', 'CC2_RCNT_PULM_EMBOLUS_M', 'CARDIAC_OUTPUT_I', 'LOST_WEIGHT', 'DEVICE_TO_MANUF', 'ALBUMIN_I', 'CC_HIST_ALCOHOL_ABUSE_M', 'CONCOM_SURG_OTHER', 'RACE_AF_AMER', 'EVENT_HOSP_IABP', 'PRE_ALBUMIN_MG_L', 'QRTR', 'LYMPH_CNT_PERCENT', 'ACUTE_CARE_I', 'WORK_YES_STATUS', 'CC_PX_DOES_NOT_WANT_TX_M', 'WORK_NO_STATUS', 'HIV_ART_KALETRA', 'HIV_ART_TDF', 'CC_CHRONIC_INF_CONCERNS_M', 'HIV_INF_PRO_DAPSONE', 'RACE_AM_IND', 'CC_PULMONARY_HYPERTENSION_M', 'CONCOM_SURG_CONG_CARD_SURG', 'MED_PRE_IMP_LD_DOSE_MG', 'SEC_DGN_BV_TA', 'PATIENT_REASON_M', 'INTERVENTION_48_HRS_MVR', 'CONCOM_SURG_CABG', 'SURGERY_TIME_I', 'CARDIAC_OUTPUT', 'INTERVENTION_48_HRS_FEED_TUBE', 'DOPPLER_OP_I', 'SEC_DGN_DIL_MYO_ADRIAMYCIN', 'SEC_DGN_BV_KD', 'LDH_L', 'BILI_TOTAL_I', 'PREV_CARDIAC_OPER_CABG', 'DEVICE_FUNC_NORM', 'CC2_CHRONIC_COAGULOPATHY', 'IV_INO_THERAPY_AGENTS_MILRI', 'CONCOM_SURG_ASD_CLOSE', 'PREV_CARD_OP_CON_SUR_VSD_REP', 'CREAT_I', 'PREV_CARDIAC_OPER_MVR', 'CONCOM_SURG_MVS_REP_MECH', 'EDUC_LEVEL', 'CC_PERIPH_VASC_DISEASE_M', 'ASSOC_FINDINGS_PFO_ASD', 'DIS_INT_SURG_PROC_UNKNOWN', 'MOBILITY', 'STRESS_I', 'CC2_LTD_COG_UNDERSTND_M', 'CC_HIST_DRUG_USE_M', 'VAD_INDICATION', 'WGT_KG', 'SEC_DGN_BV_LHV', 'INTERVENTION_48_HRS_CON_CAR_SUR', 'PREV_CARD_OP_CON_SUR_GLN_BI', 'RACE_WHITE', 'CONSOLE_CHANGE', 'INTERVENTION_48_HRS_NONE', 'PREV_CARDIAC_OPER_RVAD', 'HIV_INF_PRO_ATOVAQUONE', 'IV_INO_THERAPY_AGENTS_UNK', 'MODIFIER_A', 'THERMOMETER', 'HIV_INF_PRO_TMP_SMX', 'HIV_ART_TIPRANIVIR', 'PUL_WEDGE_PRES_I', 'THROMBIN_INHIB_DRUG_DIRECT', 'EVENT_HOSP_RVAD', 'TRICUSPID_REGURG', 'PEAK_R_I', 'SIX_MIN_WALK_I', 'HIV_ART_ETRAVIRINE', 'PRO_BNP_I', 'RA_PRES_I', 'HIV_ART_FOSAMPRENAVIR', 'HIV_ART_STRIBILD', 'DIA_BP_I', 'DEVICE_STRATEGY', 'CC_THORACIC_AORTIC_DIS_M', 'CC2_HIST_BONE_MARROW_TX', 'HIV_ART_DARUNAVIR', 'CC2_MUSCSKELETAL_LIMIT_M', 'INTERVENTION_48_HRS_ANEURS', 'REASON_ADMIN_M', 'SEC_DGN_VALV_HRT_DIS', 'HIV_ART_COMPLERA', 'INTERVENTION_48_HRS_ULTRAFILT', 'PREV_CARD_OP_CON_SUR_SEN_MSTRD', 'CONCOM_SURG_PFO_CLOSE', 'THROMBIN_INHIB_DRUG_COUMAD', 'CONCOM_SURG_RVAD_EXPLANT', 'CC2_SEVERE_DEPRESSION_M', 'ALBUMIN_G_DL', 'INT_TPT', 'CC2_PULMONARY_DISEASE_M', 'ANXIETY', 'RACE_PAC_ISLAND', 'PUL_SYS_PRES', 'SODIUM_MMOL_L', 'HIV_HX_OP_IN', 'MODIFIER_FF', 'HIV_ART_DOLUTEGRAVIR', 'SEC_DGN_SV_PA_IVS_RVDC', 'SURGERY_TIME', 'PREV_CARDIAC_OPER_CON_CAR_SUR', 'TIME_CARD_DGN', 'PREV_CARD_OP_CON_SUR_TRUNCUS', 'TRANSFER_CARE', 'INTERVENTION_48_HRS_LVAD', 'RVAD_CAN_INFLOW', 'EVENT_HOSP_CABG', 'WGT_KG_I', 'CC2_PULMONARY_HYPERTENSION_M', 'SEC_DGN_BV_EA', 'PRIMARY_DGN', 'SEC_DGN_CANCER', 'PUL_SYS_PRES_I', 'NYHA', 'CC_LARGE_BMI_M', 'HIV_ART_TRIZIVIR', 'TXPL_PT', 'PREV_CARD_OP_CON_SUR_TOV_REP', 'CC_HIST_BONE_MARROW_TX', 'POTASSIUM_MEQ_L', 'PRIMARY_COD_CANCER', 'ETHNICITY', 'INTERVENTION_48_HRS_ECMO', 'CC_RCNT_PULM_EMBOLUS_M', 'CARDIAC_INDEX', 'CC2_HISTORY_HEPATITIS_M', 'PREV_CARD_OP_CON_SUR_GLN_CL', 'CARDIAC_INDEX_I', 'EVENT_HOSP_UNKNOWN', 'HIV_CD4', 'SGOT_AST_I', 'DIS_INT_CARD_UNKNOWN', 'ACT_FLG', 'PRO_BNP_PG_ML', 'HIV_ART_NELFINAVIR', 'HIV_ART_EPZICOM', 'CC_RPTD_NON_COMPLIANCE_M', 'PRE_ALBUMIN_MG_DL', 'PREV_CARDIAC_OPER_NONE', 'SODIUM_MEQ_L', 'TEST_ADMINISTERED_QOL', 'CONCOM_SURG_TVS_REP_MECH', 'CC_CHRONIC_RENAL_DISEASE_M', 'DEVICE_TY', 'EVENT_HOSP_CON_CAR_SUR', 'IV_INO_THERAPY_AGENTS_ISOPRO', 'INTERVENTION_48_HRS_RVAD', 'SGOT_AST', 'CC_CHRONIC_COAGULOPATHY', 'LVEDD', 'ACTIVITY_MAIN_CONSIDERED', 'MED_PRE_IMP_LOOP_DIURETICS', 'SEC_DGN_DIL_MYO_POST_PART', 'HIV_ART_EMTRICITABINE', 'SEC_DGN_SV_HLF', 'HIV_ART_NEVIRAPINE', 'CC_SEVERE_DEPRESSION_M', 'MED_PRE_IMP_DIURETICTYPE_OTHER', 'PREV_CARD_OP_CON_SUR_SEP_DEFCT', 'CV_PRES_I', 'SIX_MIN_WALK', 'CC2_THORACIC_AORTIC_DIS_M', 'CC2_CURR_SMOKING_M', 'SYS_BP', 'CC2_LIVER_DYSFUNCTION_M', 'RACE_ASIAN', 'RVEF', 'CONCOM_SURG_RVAD_IMPLANT', 'MED_PRE_IMP_DIURETICTYPE_FUROS', 'HEMOGLOBIN_G_L', 'SEC_DGN_DIL_MYO_IDIOPATHIC', 'INFECTION_LOC', 'CC2_FREQUENT_ICD_SHOCKS', 'IV_INO_THERAPY_AGENTS_OTHER', 'LOS', 'EVENT_HOSP_CAR_ARREST', 'CONCOM_SURG_MVS_REP_BIO', 'HGT_IN', 'SEC_DGN_REST_MYO_RAD_CHEMO', 'CHOLESTEROL_MG_DL', 'DIA_BP', 'PC_PUMP_EXCHANGE', 'HIV_ART_STAVUDINE', 'ACTIVITIES', 'PREV_CARDIAC_OPER_OTHER', 'CONCOM_SURG_TVS_R_RING', 'CONCOM_SURG_TVS_R_DEVEGA']
isCategoricalPredictor = [true, true, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, false, true, false, true, true, false, true, true, false, false, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, false, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, false, true, true, true, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, false, true, false, true, true, false, true, false, false, true, true, true, true, true, true, false, true, true, true, true, false, true, true, true, true, true, false, true, true, true, true, true, true, true, true, false, true, true, false, true, true, true, true, true, false, true, true, true, true, false, true, true, true, false, true, false, true, true, true, true, true, true, true, true, true, false, true, false, true, true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, false, true, false, false, true, true, false, true, true, false, true, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, true, false, true, false, true, true, false, true, true, true, true, true, false, true, false, true, true, false, true, true, false, true, true, true, true, true, false, true, true, true, true, true, true, false, true, true, false, true, true, true, true, true, true, true, true, true, false, true, true, true, true, true, true, true, true, false, true, true, true, true, false, true, true, true, true, true, false, true, false, true, true, false, true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, true, true, false, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, true, true, true, false, false, false, true, true, true, false, true, false, true, true, true, true, true, true, true, true, false, true, true, true, true, true, false, true, true, false, true, true, true, true, false, true, true, true, true, false, true, true, false, false, true, true, true, false, true, false, true, true, true, true, true, true, true, false, true, false, true, true, true, true, true, true, true, true, true, true, false, true, true, false, true, true, true, true, true, false, true, true, true, true, false, true, true, false, true, false, false, true, true, true, true, true, true]

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Model interface 

initilize example:
    treeModel = model(fileToData("book1.csv",fs1),tree1)
model.train()   return the model(variable types depends on model)
model.getPerformance()  (return the ROC and accuracy in dictionary format)
model.predict(v): pass in a 1xn array and return prediction value


"""

#1 


# ========== PRE-PROCESSING ========== #
def fileToData(fileName,FeaMethod): #get matrix from csv after feature selection/feature extraction method
    return FeaMethod(genfromtxt(fileName,delimiter=',')) 

def encoding(nominal_col):
    """
    transform a column of categorical values to one of numeric values
    """
    le = preprocessing.LabelEncoder()
    le.fit(nominal_col)
    return le.transform(nominal_col)
def encoding_wrapper(X):
    for col_name in X.select_dtypes(include=['object']):
        X.loc[:,col_name] = model.encoding(X[col_name])
    return X


def filling_NAs(X):
    """
    fill in missing values with column means
    """
    fill = preprocessing.Imputer(strategy='mean')
    X_imputed = fill.fit_transform(X)

    return pd.DataFrame(X_imputed, columns = X.columns)

def floatFormatter(x):
    return '{0:.5f}'.format(float(x))
    
def clean_training_data(X, y):
 
  regr = linear_model.LinearRegression()

  # Train the model using the training sets
  regr.fit(X, y)
  y_pred = regr.predict(X)
  #print("before:")
  #print(np.mean((y - y_pred)**2)) 
  for i in range(499,-1,-1): #This removes observations that 
  #are strongly affecting the entire model fit
    resi = (y_pred[i]-y[i])**2
    if(resi>900):
      X=np.delete(X,i,0)
      y=np.delete(y,i,0)
  
  #print("after:")
  regr2 = linear_model.LinearRegression()

  # Train the model using the training sets
  regr2.fit(X, y)
  y_pred2 = regr.predict(X)
  
  #print(np.mean((y - y_pred2)**2)) 
  #print("SVR rbf MSE :")
  
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  svr_rbf.fit(X,y)
  pre=svr_rbf.predict(X)
 # print(np.mean((y - pre)**2))
  sX = csr_matrix(X) #sparse matrix format
  
  return sX , y
  
def transform(x):
  """
  Apply a transformation to a feature vector for a single instance
  :param x: a feature vector for a single instance
  :return: a modified feature vector
  """
  # Attemped use of kernel estimator 
  #rbf_feature = RBFSampler(gamma=0.1, random_state=1)
  #x = rbf_feature.fit_transform(x)
  #if(x[13]>3000):x[13]=x[13]//3000 #scale extreme data
  #without hurting much for its effect on model
  
  poly = PolynomialFeatures(2)
  x=poly.fit_transform(x) #2nd degree polynomial interpolation
  #Kernel methods extend this idea and can induce very high
  #(even infinite) dimensional feature spaces.
  x = Normalizer().fit_transform(x) #normalize the features
  imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
  imp.fit(x)
  x=imp.transform(x) #fill in expected values
  scaler = RobustScaler()
  scaler.fit(x)
  #x=scaler.transform(x)
  return x
  
def rov(thres,tV, pV):
    TP,TN,FP,FN= 0
    for i in range (len(tV)):
        TP= TP+1 if(tV[i]<=thres and pV[i] <=thres) else TP
        TN= TN+1 if(tV[i]> thres and pV[i] > thres) else TN
        FN= FN+1 if(tV[i]<=thres and pV[i] > thres) else FN
        FP= FP+1 if(tV[i]> thres and pV[i] <=thres) else FP
    return ((TP+TN)/(TP+TN+FP+FN))
 
        
    # ============== MODELS ============== #        
        
        
class model(object):
    def __init__(self,dataIn,ModelName):
        self.data=dataIn
        self.target= dataIn[:,[len(dataIn)-1]]
        self.thres
        self.name = ModelName
        self.model
        self.predict
        self.performance
    def __repr__ (self):
       return ((self.name,self.performance))
        
    def __hash__(self):   
       return hash((self.name))
       
    def __eq__(self,other):
       return(isinstance(other,model)and(self.name==other.name) \
              and (self.data==other.data))
    def train(self):
       pass
    def predict(self):
       pass
    def performance(self):
       cv = cross_val_score(clf, self.data, self.target, cv=5)
       rovAccu = rov(3,self.target,self.predict)
       rovauc = roc_auc_score(self.target,self.predict)
       
       
class linear(model)  : 
    def train(X, y):
      """
      Train a model
      :parma X: n x p design matrix
      :param y: response vector of length n
      :return weights: weight vector of length p
      """
      
      """
      n, p = X.shape
      
      weights = np.zeros(p)
      regr3 = linear_model.LinearRegression(fit_intercept=False)
    
      # Train the model using the training sets
      regr3.fit(X, y)
      
      weights = regr3.coef_ 
      print (weights)
      
      svr_rbf = SVR(kernel='linear', C=1e3)
      svr_rbf.fit(X,y)
      weights= svr_rbf.coef_.T
      print(svr_rbf.support_vectors_.shape)
      """
      #attempted use of SVR 
      svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
      svr_rbf.fit(X,y)
      pre=svr_rbf.predict(X)
      #weights =np.linalg.lstsq(X,pre)[0]
      #weights=pre
      #print(weights)
      
      #weights= np.linalg.solve(X,pre) #X inverse times b
      
      #clf = linear_model.SGDRegressor()
      #clf.fit(X, y)
      #weights=clf.coef_
      
      
      
      regr3 = linear_model.LinearRegression(fit_intercept=False)
      # Train the Linear model using the training sets 
      regr3.fit(X, y)
      print("cross validation score:")
      scores = cross_val_score(regr3, X, y, cv=10)
      print(scores)
      
      weights = regr3.coef_ 
      return weights
      
    def predict(X, weights):
      
      y = np.dot(X, weights)
      return y
    
    def MSE(y, predictions):
    
      mse = np.mean((y - predictions)**2)
      return mse

def train_and_predict(X_train, y_train, X_new):

  # clean the training data
  X_train_clean, y_train_clean = clean_training_data(X_train, y_train)
  # transform the training data
  X_train_transformed = np.vstack([transform(x) for x in X_train_clean])
  # transform the new data
  X_new_transformed = np.vstack([transform(x) for x in X_new])
  # learn a model
  weights = train(X_train_transformed, y_train_clean)
  # make predictions on the training data
  predictions_train = predict(X_train_transformed, weights)
  # make predictions on the new data
  predictions = predict(X_new_transformed, weights)
  # report the MSE on the training data
  train_MSE = MSE(y_train_clean, predictions_train)
  print("MSE on training data = %0.4f" % train_MSE)
  # return the predictions on the new data
  return predictions

def linReg(X_train, X_test, y_train, y_test, modelObj):
    """
    train a linear regression model
    """

    # get model parameters
    normalize = modelObj["parameters"][0]

    # train the classifier
    model = LinearRegression(normalize = normalize)
    model.fit(X=X_train, y=y_train)

    # predict
    y_pred = model.predict(X_test)

    # score the model
    coef = np.around(model.coef_, decimals=5).tolist()
    mse = metrics.mean_squared_error(y_test, y_pred) 
    var_score = metrics.r2_score(y_test, y_pred) 

    # format numbers '{0:.5f}'.format
    mse = floatFormatter(mse)
    var_score = floatFormatter(var_score)

    return (y_pred, coef, mse, var_score)

  #==============tree=================#  
    
class dt(model):
    def __init__(self,leaf_size=4):
        self.leaf_size= leaf_size
        
    def train(self,dataX,dataY):
        if dataX.shape[0] <= self.leaf_size: 
            return np.array([-1,np.mean(dataY),np.nan,np.nan])
        if np.all(dataY[:]==dataY[0]): 
            return np.array([-1,dataY[0],np.nan,np.nan])
        if np.all(dataX[:]==dataX[0]): 
            return np.array([-1,dataX[0],np.nan,np.nan])
        
        #Calc the index of the best feature.
        
        index = 0;
        cor_list = [];
        for i in range(len(dataX[0])):
            cor_list.append(abs(np.corrcoef(dataX[:,i], dataY)[0,1]))
        index = cor_list.index(max(cor_list))
        
       
        #split the tree by most significant feature.
        
        leaf= np.array([-1, Counter(dataY).most_common(1)[0][0], np.nan, np.nan])
        split_value=np.medium(dataX[:,index])
        left=dataX[:,index]<=split_value
        right=dataX[:,index]>split_value
        ldataX=dataX[left,:]
        ldataY=dataY[left]
        rdataX=dataX[right,:]
        rdataY=dataY[right]
        if(len(rdataY)==0 or len(ldataX)==0 ): return leaf
        #recursive train left and right node of the tree then merge at the end 
        ltree=self.train(ldataX,ldataY)
        rtree=self.train(rdataX,rdataY)
        if ltree.ndim==1:
            root=np.array([index,split_value,1,2])
        else:
            root=np.array([index,split_value,1,ltree.shape[0]+1])
        tree=np.vstack((root,ltree,rtree))
                                                
        return tree
    def predict(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        res=[]
        start=int(self.tree[0,0])
        tree_height=self.tree.shape[0]
        for point in points:
            index=start
            i=0
            while(i<tree_height):
                index=self.tree[i,0]
                if index==-1:
                    break
                else:
                    index=int(index)
                if point[index] <= self.tree[i,1]:
                    i = i + 1
                    
                else:
                    i = i + int(self.tree[i,3])
                
            if index==-1:
                res.append(self.tree[i,1])
            else:
                res.append(np.nan)
        return np.array(res)
        
        
class Bag(dt):

    def __init__(self, bags=30,  **kwargs):
        self.learners =np.zeros(bags)
        self.kwargs = kwargs
        self.bags = bags
        
    def bagtrain(self):
        #set bags and  do trees 
        num_samples = self.data.shape[0]
        for i in range(self.bags):
            idx = np.random.choice(num_samples, num_samples)
            bagX = dataX[idx]
            bagY = dataY[idx]
            self.learners a.append(self.train(bagX,bagY))
        
        
    def bagPredict(self, val):
        preds = np.array([learner.predict(val) for learner in self.learners])
        return np.mean(preds, axis=0)

        
#sk decision tree        
def treeClf(X, y, modelObj):
    """
    train a decision tree model
    """

    # get model parameters

    max_depth = float(modelObj['max_depth'])
    min_samples_leaf = modelObj['min_samples_leaf']

    # encode categorical features
    # ! must before dealing with NaNs

    X = encoding_wrapper(X)

    # deal with NaNs

    X = filling_NAs(X)
    
    # split train, test set

    (X_train, X_test, y_train, y_test) = train_test_split(X, y,
            test_size=0.3, random_state=17)

    # train the classifier

    model = tree.DecisionTreeClassifier(max_depth=max_depth,
            min_samples_leaf=min_samples_leaf)
    model.fit(X=X_train, y=y_train)
    
    return (model, X_test, y_test)
    
class ensembleTree(model):
    def train(self):
        eng = matlab.engine.start_matlab()
        template = eng.templateTree('MaxNumSplits', 20);
        classificationEnsemble = fitcensemble(self.target, self.data, 
                                              'Method', 'AdaBoostM1', 
                                              'NumLearningCycles', 30, 
                                              'Learners', template,
                                              'LearnRate', 0.1, 
                                              'ClassNames', categorical('No', 'Yes'))

    def performance(self):
        eng = matlab.engine.start_matlab()
        partitionedModel = eng.crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);



class clustering(model):
    def __init__(self, geo_locs_, k_):
        self.geo_locations = geo_locs_
        self.k = k_
        self.clusters = []  #clusters of nodes
        self.means = []     #means of clusters
        self.debug = False  #debug flag
    #this method returns the next random node
    def next_random(self, index, points, clusters):
        #pick next node that has the maximum distance from other nodes
        dist = {}
        for point_1 in points:
            for cluster in clusters.values():
                point_2 = cluster[0]
                if point_1 not in dist:
                    dist[point_1] = math.sqrt(math.pow(point_1.latit - point_2.latit,2.0) + math.pow(point_1.longit - point_2.longit,2.0))       
                else:
                    dist[point_1] += math.sqrt(math.pow(point_1.latit - point_2.latit,2.0) + math.pow(point_1.longit - point_2.longit,2.0))
       
        # return the point that has the maximum distance from previous nodes
        count_ = 0
        max_ = 0
        for key, value in dist.items():
            if count_ == 0:
                max_ = value
                max_point = key
                count_ += 1
            else:
                if value > max_:
                    max_ = value
                    max_point = key
        return max_point
    #this method computes the initial means
    def initial_means(self, points):
        #pick the first node at random
        point_ = rand.choice(points)
        clusters = dict()
        clusters.setdefault(0, []).append(point_)
        points.remove(point_)
        #now let's pick k-1 more random points
        for i in range(1, self.k):
            point_ = self.next_random(i, points, clusters)
            #clusters.append([point_])
            clusters.setdefault(i, []).append(point_)
            points.remove(point_)
        #compute mean of clusters
        #self.print_clusters(clusters)
        self.means = self.compute_mean(clusters)
  
    def compute_mean(self, clusters):
        means = []
        for cluster in clusters.values():
            mean_point = Point(0.0, 0.0)
            cnt = 0.0
            for point in cluster:
                #print "compute: point(%f,%f)" % (point.latit, point.longit)
                mean_point.latit += point.latit
                mean_point.longit += point.longit
                cnt += 1.0
            mean_point.latit = mean_point.latit/cnt
            mean_point.longit = mean_point.longit/cnt
            means.append(mean_point)
        return means
    #this method assign nodes to the cluster with the smallest mean
    def assign_points(self, points):
    
        clusters = dict()
        for point in points:
            dist = []
          
            #find the best cluster for this node
            for mean in self.means:
                dist.append(math.sqrt(math.pow(point.latit - mean.latit,2.0) + math.pow(point.longit - mean.longit,2.0)))
            cnt_ = 0
            index = 0
            min_ = dist[0]
            for d in dist:
                if d < min_:
                    min_ = d
                    index = cnt_
                cnt_ += 1
  
            clusters.setdefault(index, []).append(point)
        return clusters
    def update_means(self, means, threshold):
        #check the current mean with the previous one to see if we should stop
        for i in range(len(self.means)):
            mean_1 = self.means[i]
            mean_2 = means[i]      
            if math.sqrt(math.pow(mean_1.latit - mean_2.latit,2.0) + math.pow(mean_1.longit - mean_2.longit,2.0)) > threshold:
                return False
        return True

    def k_means(self, plot_flag):
        if len(self.geo_locations) < self.k:
            return -1   #error
        points_ = [point for point in self.geo_locations]
        #compute the initial means
        self.initial_means(points_)
        stop = False
        while not stop:
            #assignment step: assign each node to the cluster with the closest mean
            points_ = [point for point in self.geo_locations]
            clusters = self.assign_points(points_)
            if self.debug:
                self.print_clusters(clusters)
            means = self.compute_mean(clusters)
            
        self.clusters = clusters
       
        return 0

class deepLearn(model): #then use deep learning to train each clusters 
    def inference(self):
        #first convolutinary layer 
        with tf.variable_scope('conv1') as scope:
            weights = tf.get_variable('weights', 
                                      shape = [3, 3, 3, 663],
                                      dtype = tf.float32, 
                                      initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32)) 
            biases = tf.get_variable('biases', 
                                     shape=[663],
                                     dtype=tf.float32, 
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(data, weights, strides=[1,1,1,1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name= scope.name)
        
        
        #pooling layer 1 and normalization 1   
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                   padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                              beta=0.75, name='norm1')
        
        
        #conv2
        with tf.variable_scope('conv2') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3,3,663, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[64], 
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')
        
        
        #pool2 and norm2
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                              beta=0.75,name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                                   padding='SAME',name='pooling2')
            
           #conv 3
        with tf.variable_scope('conv3') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3,3,663, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[64], 
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name='conv3')
              
    
        #pool3 and norm3
        with tf.variable_scope('pooling3_lrn') as scope:
            norm3 = tf.nn.lrn(conv3, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                              beta=0.75,name='norm2')
            pool3 = tf.nn.max_pool(norm3, ksize=[1,3,3,1], strides=[1,1,1,1],
                                   padding='SAME',name='pooling3')
        
        #local3
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable('weights',
                                      shape=[dim,384],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[384],
                                     dtype=tf.float32, 
                                     initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        
         
        #local4
        with tf.variable_scope('local4') as scope:
            weights = tf.get_variable('weights',
                                      shape=[384,192],
                                      dtype=tf.float32, 
                                      initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[192],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
         
            
        #fully conected layer 
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('softmax_linear',
                                      shape=[192, 10],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
            biases = tf.get_variable('biases', 
                                     shape=[10],
                                     dtype=tf.float32, 
                                     initializer=tf.constant_initializer(0.1))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
        
        return softmax_linear
        
    def train(self):
        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        data, labels = self.data ,self.target
        logits = inference(data)
        loss = losses(logits, labels)
        optimizer = tf.train.AdaOptimizer(2)
        train_op = optimizer.minimize(loss, global_step= my_global_step)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
            _,loss_value = sess.run([train_op, loss])
            if step %1000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        
        
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
        
        