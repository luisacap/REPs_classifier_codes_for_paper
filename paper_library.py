#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:14:47 2022

@author: luisacap
"""
#Routines needed for paper_load_model_poes_predictions.py

import numpy as np
import scipy
from scipy import stats


def preproc(raw_df):
    index_neg0 = np.where((raw_df['e1_0'] <= 0))
    raw_df['e1_0'][index_neg0[0]]  = 0.01
    index_neg90 = np.where((raw_df['e1_90'] <= 0))
    raw_df['e1_90'][index_neg90[0]]  = 100.0
    index_neg0 = np.where((raw_df['e2_0'] <= 0))
    raw_df['e2_0'][index_neg0[0]]  = 0.01
    index_neg90 = np.where((raw_df['e2_90'] <= 0))
    raw_df['e2_90'][index_neg90[0]]  = 100.0
    index_neg0 = np.where((raw_df['e3_0'] <= 0))
    raw_df['e3_0'][index_neg0[0]]  = 0.01
    index_neg90 = np.where((raw_df['e3_90'] <= 0))
    raw_df['e3_90'][index_neg90[0]]  = 100.0
    index_neg0 = np.where((raw_df['e4_0'] <= 0))
    raw_df['e4_0'][index_neg0[0]]  = 0.01
    index_neg90 = np.where((raw_df['e4_90'] <= 0))
    raw_df['e4_90'][index_neg90[0]]  = 100.0
    
    cleaned_df = raw_df.copy()
    
    eps = 0.001
    cleaned_df['Log e1_0']  = np.log(cleaned_df.pop('e1_0')  + eps)
    cleaned_df['Log e1_90'] = np.log(cleaned_df.pop('e1_90') + eps)
    cleaned_df['Log e2_0']  = np.log(cleaned_df.pop('e2_0')  + eps)
    cleaned_df['Log e2_90'] = np.log(cleaned_df.pop('e2_90') + eps)
    cleaned_df['Log e3_0']  = np.log(cleaned_df.pop('e3_0')  + eps)
    cleaned_df['Log e3_90'] = np.log(cleaned_df.pop('e3_90') + eps)
    cleaned_df['Log e4_0']  = np.log(cleaned_df.pop('e4_0')  + eps)
    cleaned_df['Log e4_90'] = np.log(cleaned_df.pop('e4_90') + eps)
    
    return cleaned_df, raw_df


def create_shifted_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        labels = y[i: i + time_steps]
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys)


def clean_up(x, th1, th2):
    window = False
    acc = 0
    count = 0
    l_acc = np.zeros(len(x))
    pulse = []
    loc = []
    for i in range(len(x)):
        if window == False and x[i] >= th2:
            window = True
            acc = acc + x[i]
            count = count + 1
            pulse.append(i)
        elif window == True:
            if x[i] >= th2:
                acc = acc + x[i]                            
            elif x[i] <= th1:
                window = False
                acc = 0
                pulse.append(i-1)
                loc.append(pulse)
                pulse = []
        l_acc[i] = acc        
    return l_acc , count, loc 


def filter_Lshell(locs,L,L_in,L_fin):
    locs_filtered = []
    for locs_temp in locs:
        #Consider a wider range of i so that if L has a gap at the event range, it looks at nearby points
        for i in range(locs_temp[0]-10,locs_temp[1]+1+10):
            if L[i] >= L_in and L[i] <= L_fin:
                locs_filtered.append(locs_temp)
                break
            
    return locs_filtered