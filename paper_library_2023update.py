#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:20:27 2021

@author: luisacap
"""
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
            if i <= 0:
                i=0
            if i >= len(L):               
                i = len(L)-1
            if L[i] >= L_in and L[i] <= L_fin:
                locs_filtered.append(locs_temp)
                break
            
    return locs_filtered


def shift_locs(locs,num_points):
    locs_shifted_start = np.array([x[0]+num_points for x in locs])
    locs_shifted_stop  = np.array([x[1]+num_points for x in locs])

    return locs_shifted_start,locs_shifted_stop


def join_nearby_events(start,stop,distance):
    """
    This function joins nearby events (within the distance set by the user)
    Parameters
    ----------
    start : start index of event 
    stop : stop index of event 
    distance : max distance of points for allowing 2 separate events

    Returns
    -------
    start_new : np.array, new start index
    stop_new : np.array, new stop index
    rep_num : new number of events

    """
    delta = start[1:] - start[0:-1]
    delta_i = np.where(delta <= distance)
    if len(delta_i[0]) != 0:
        for d_i in np.arange(len(delta_i[0])):
            start[delta_i[0][d_i]] = min(start[delta_i[0][d_i]],start[delta_i[0][d_i]+1])
            stop[delta_i[0][d_i]]  = max(stop[delta_i[0][d_i]],stop[delta_i[0][d_i]+1])
    start_new = np.copy(start)
    start_new = np.delete(start_new,delta_i[0][:]+1)
    stop_new = np.copy(stop)
    stop_new = np.delete(stop_new,delta_i[0][:]+1)
    rep_num = len(start_new)

    return start_new,stop_new,rep_num



def remove_unphysical(dataset_day,dataset_day_copy,start,stop,rep_num):
    """
    This function removes the unphysical events
    Parameters
    ----------
    dataset_day : POES dataset
    start : start index of event 
    stop : stop index of event 
    rep_num : number of REPs

    Returns
    -------
    start_new : np.array, new start index
    stop_new : np.array, new stop index
    num_new : new number of events


    """
    start_new = []
    stop_new = []
    for ind in np.arange(rep_num):
        i = start[ind]
        f = stop[ind]
        if i == f: f = i+1
        E1_0 = np.array(dataset_day['e1_0'][i:f])
        E2_0 = np.array(dataset_day['e2_0'][i:f])
        E3_0 = np.array(dataset_day['e3_0'][i:f])
        E4_0 = np.array(dataset_day['e4_0'][i:f])
        E4_90 = np.array(dataset_day_copy['e4_90'][i:f])
        if max(E4_0) < 181.81818*2.: continue
        elif np.average(E4_90) <= 0: continue #discard event if E4_0 is < 2 count/s
        elif max(E4_0) <= max(E3_0): #and max(E3_0) <= max(E2_0) and max(E2_0) <= max(E1_0):
            start_new = np.append(start_new,start[ind])
            stop_new = np.append(stop_new,stop[ind])
        elif max(E4_0) <= max(E2_0) and max(E4_0) <= max(E1_0):
            start_new = np.append(start_new,start[ind])
            stop_new = np.append(stop_new,stop[ind])
    num_new = len(start_new)

    return start_new,stop_new,num_new