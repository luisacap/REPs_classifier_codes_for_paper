#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:20:27 2021

@author: luisacap
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

import scipy
from scipy import stats

from joblib import dump


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_metrics(history):
  metrics = ['precision', 'recall','loss','auc', 'prc']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(3,3,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
              color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.tight_layout()
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
      plt.locator_params(axis="y", nbins=4)
    elif metric == 'prc':
      plt.ylim([0.8,1])
      plt.locator_params(axis="y", nbins=4)
    else:
      plt.ylim([0,1])

    plt.legend()
    
def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  # print('NO-CSS-events Detected (True Negatives): ', cm[0][0])
  # print('NO-CSS-events Incorrectly Detected (False Positives): ', cm[0][1])
  # print('CSS-events Missed (False Negatives): ', cm[1][0])
  # print('CSS-events Detected (True Positives): ', cm[1][1])
  # print('Total CSS-events: ', np.sum(cm[1]))


def plot_cm_multiple(labels, predictions):
  cm = confusion_matrix(labels, predictions)
  df_cm = pd.DataFrame(cm,index=['no events','REPs','CSSs'],columns=['no events','REPs','CSSs'])
  plt.figure(figsize=(5,5))
  sns.heatmap(df_cm, annot=True, fmt="d",vmin=0,vmax=200)
  plt.title('Confusion matrix')
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  print(cm)
  return


def plot_multiple_roc(name, labels, predictions, **kwargs):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot ROC curves for the multilabel problem
    # ..........................................
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    
    # Then interpolate all ROC curves at this points
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= 3
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    from itertools import cycle
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()
    
    # Area under ROC for the multiclass problem
    # .........................................
    # The :func:`sklearn.metrics.roc_auc_score` function can be used for
    # multi-class classification. The multi-class One-vs-One scheme compares every
    # unique pairwise combination of classes. In this section, we calculate the AUC
    # using the OvR and OvO schemes. We report a macro average, and a
    # prevalence-weighted average.
    #y_prob = classifier.predict_proba(X_test)
    
    
    macro_roc_auc_ovo = roc_auc_score(labels, predictions, multi_class="ovo",
                                      average="macro")
    weighted_roc_auc_ovo = roc_auc_score(labels, predictions, multi_class="ovo",
                                          average="weighted")
    macro_roc_auc_ovr = roc_auc_score(labels, predictions, multi_class="ovr",
                                      average="macro")
    weighted_roc_auc_ovr = roc_auc_score(labels, predictions, multi_class="ovr",
                                          average="weighted")
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
    return
      

def preproc(raw_df):
    #Replace <=0 values
    #Calculate ratios
    #Apply logs
    
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
    
    cleaned_df['r1'] = cleaned_df['e1_0']/cleaned_df['e1_90']
    cleaned_df['r2'] = cleaned_df['e2_0']/cleaned_df['e2_90']
    cleaned_df['r3'] = cleaned_df['e3_0']/cleaned_df['e3_90']
    cleaned_df['r4'] = cleaned_df['e4_0']/cleaned_df['e4_90']
    
    #transform to log (base e) all e_ variables
    eps = 0.001 # makes 0s into 0.001
    cleaned_df['Log e1_0']  = np.log(cleaned_df.pop('e1_0')  + eps)
    cleaned_df['Log e1_90'] = np.log(cleaned_df.pop('e1_90') + eps)
    cleaned_df['Log e2_0']  = np.log(cleaned_df.pop('e2_0')  + eps)
    cleaned_df['Log e2_90'] = np.log(cleaned_df.pop('e2_90') + eps)
    cleaned_df['Log e3_0']  = np.log(cleaned_df.pop('e3_0')  + eps)
    cleaned_df['Log e3_90'] = np.log(cleaned_df.pop('e3_90') + eps)
    cleaned_df['Log e4_0']  = np.log(cleaned_df.pop('e4_0')  + eps)
    cleaned_df['Log e4_90'] = np.log(cleaned_df.pop('e4_90') + eps)
    cleaned_df['Log r1'] = np.log(cleaned_df.pop('r1') + eps)
    cleaned_df['Log r2'] = np.log(cleaned_df.pop('r2') + eps)
    cleaned_df['Log r3'] = np.log(cleaned_df.pop('r3') + eps)
    cleaned_df['Log r4'] = np.log(cleaned_df.pop('r4') + eps)
    
    return cleaned_df, raw_df


def split_onehot_norm(cleaned_df,timestep,train_percentage,val_percentage):
    
    
    #Split into train, val, test
    # b = train_percentage + val_percentage
    # train_df = cleaned_df[0:int(len(cleaned_df)*train_percentage)]
    # val_df   = cleaned_df[int(len(cleaned_df)*train_percentage):int(len(cleaned_df)*b)]
    # test_df  = cleaned_df[int(len(cleaned_df)*b):int(len(cleaned_df))]
    
    #  not using the python function because I need to keep blocks of time_steps
    n_events = int(len(cleaned_df)/timestep) #num of events
    a = int(np.rint(n_events*train_percentage)) #80% of num_events
    b = int(np.rint(n_events*val_percentage)) #10% of num_events
    train_df = cleaned_df[0:a*timestep] #take first 80% of events (of time_steps)
    val_df   = cleaned_df[a*timestep:(a+b)*timestep] #take next 10% of events (of time_steps)
    test_df  = cleaned_df[(a+b)*timestep:] #take next 10% of events (of time_steps) 

    train_labels = np.array(train_df.pop('Class'))
    val_labels   = np.array(val_df.pop('Class'))
    test_labels  = np.array(test_df.pop('Class'))

    #One hot encoding
    from sklearn.preprocessing import LabelEncoder
    train_labels = LabelEncoder().fit_transform(train_labels)
    val_labels   = LabelEncoder().fit_transform(val_labels)
    test_labels  = LabelEncoder().fit_transform(test_labels)
    train_labels = tf.keras.utils.to_categorical(train_labels, 3)
    test_labels  = tf.keras.utils.to_categorical(test_labels,  3)
    val_labels   = tf.keras.utils.to_categorical(val_labels,   3)
    train_features = np.array(train_df)
    val_features   = np.array(val_df)
    test_features  = np.array(test_df)
   
    #Normalize
    scaler = StandardScaler()
    #scaler = RobustScaler()
    train_features = scaler.fit_transform(train_features)
    val_features   = scaler.transform(val_features)
    test_features  = scaler.transform(test_features)
    
    return train_features,train_labels,val_features,val_labels,test_features,test_labels


def split(cleaned_df,timestep,train_percentage,val_percentage):
    
    #Split into train, val, test
    # b = train_percentage + val_percentage
    # train_df = cleaned_df[0:int(len(cleaned_df)*train_percentage)]
    # val_df   = cleaned_df[int(len(cleaned_df)*train_percentage):int(len(cleaned_df)*b)]
    # test_df  = cleaned_df[int(len(cleaned_df)*b):int(len(cleaned_df))]
    
    #  not using the python function because I need to keep blocks of time_steps
    n_events = int(len(cleaned_df)/timestep) #num of events
    a = int(np.rint(n_events*train_percentage)) #80% of num_events
    b = int(np.rint(n_events*val_percentage)) #10% of num_events
    train_df = cleaned_df[0:a*timestep] #take first 80% of events (of time_steps)
    val_df   = cleaned_df[a*timestep:(a+b)*timestep] #take next 10% of events (of time_steps)
    test_df  = cleaned_df[(a+b)*timestep:] #take next 10% of events (of time_steps) 

    train_labels = np.array(train_df.pop('Class'))
    val_labels   = np.array(val_df.pop('Class'))
    test_labels  = np.array(test_df.pop('Class'))
    train_features = np.array(train_df)
    val_features   = np.array(val_df)
    test_features  = np.array(test_df)
   
    return train_features,train_labels,val_features,val_labels,test_features,test_labels,n_events


def onehot(train_features,train_labels,val_features,val_labels,test_features,test_labels):
    #One hot encoding
    from sklearn.preprocessing import LabelEncoder
    train_labels = LabelEncoder().fit_transform(train_labels)
    val_labels   = LabelEncoder().fit_transform(val_labels)
    test_labels  = LabelEncoder().fit_transform(test_labels)
    train_labels = tf.keras.utils.to_categorical(train_labels, 3)
    test_labels  = tf.keras.utils.to_categorical(test_labels,  3)
    val_labels   = tf.keras.utils.to_categorical(val_labels,   3)
    
    return train_features,train_labels,val_features,val_labels,test_features,test_labels


def norm(method,filename,train_features,train_labels,val_features,val_labels,test_features,test_labels):
    #Normalize
    if method == 'StandardScaler':
        scaler = StandardScaler()
    
    if method == 'RobustScaler':
        scaler = RobustScaler()
    
    scaler.fit(train_features)
    dump(scaler, filename+'.joblib')
    train_features = scaler.fit_transform(train_features)
    val_features   = scaler.transform(val_features)
    test_features  = scaler.transform(test_features)
    
    return train_features,train_labels,val_features,val_labels,test_features,test_labels

#Adapted from website: https://towardsdatascience.com/time-series-classification-for-human-activity-recognition-with-lstms-using-tensorflow-2-and-keras-b816431afdff
#"We choose the label (category) by using the mode of all categories in the sequence. 
#That is, given a sequence of length time_steps, we're are classifying it as the category 
#that occurs most often."
#def create_dataset(X, y, time_steps=1, step=1):
def create_shifted_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        labels = y[i: i + time_steps]
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys)



def temporalize(X, y, lookback):
    '''
    it forms the data sets in many chunks of timesteps shifted by 1
    '''
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records up to the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1]) ### EDITED FROM ORIGINAL FUNCTION
    output_X = np.array(output_X)
    output_y = np.array(output_y)
    return output_X, output_y

# def windowing(X, y, lookback):
#     '''
#     it forms the data sets in many chunks of timesteps
#     '''
#     output_X = []
#     output_y = []
#     for i in range(0, len(X)-lookback-1, lookback):
#         t = []
#         for j in range(1,lookback+1):
#             # Gather past records upto the lookback period
#             t.append(X[[(i+j+1)], :])
#         output_X.append(t)
#         output_y.append(y[i+lookback-1]) ### EDITED FROM ORIGINAL FUNCTION
#     output_X = np.array(output_X)
#     output_y = np.array(output_y)
#     return output_X, output_y

def windowing(features,timestep):
    '''
    separate the features into chunks of timestep
       this is a simple reshaping
    features_windowed has shape (num_events,timestep,1,num_features)
    '''
    n = int(len(features)/timestep)
    features_windowed = features.reshape(n,timestep,1,features.shape[-1])
    
    return features_windowed


def one_pulse_filter(x, token):
    '''
    Parameters
    ----------
    x : 
        has ONLY the token different from ZERO.
    token : 
        the only value present in x apart zero

    Returns
    -------
    return a copy of the input but cancel all
        pulses with 1 sample-only

    '''
    for i in range(0, len(x)-1):
        if x[i-1] != token and x[i] == token and x[i+1] != token:
            x[i] = 0
    return x  

def clean_up(x, th1, th2):
    """
    Parameters
    ----------
    x : TYPE
        prediction.
    th1 : TYPE
        threshold down.
    th2 : TYPE
        threshold up.
    Returns
    -------
    l_acc : TYPE
        "Power of the CSSs".
    count : is the number total number
    of the events
    loc : is WHERE each event starts and stops    
    """
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


def filter_over_mean(rep_in, df, th_mean):
    """
    Parameters
    ----------
    rep_in : TYPE
        list of events locations
    df : TYPE
        dataframe of day considered for predictions 
    th_mean : TYPE
        mean used as a threshold to filter out events

    Returns
    -------
    rep_out : TYPE
        list of events location with at least one e4_0 point >= th_mean

    """
    rep_out = []
    for rep in rep_in:
        for i in range(rep[0],rep[1]+1):
            if df['e4_0'][i] >= th_mean:
                rep_out.append(rep)
                break
    return rep_out


def filter_Lshell(locs,L,L_in,L_fin):
    """
    Parameters
    ----------
    locs : list of events locations
    L : L shell from data file 
    L_in,L_fin : range of L shell to filter out

    Returns
    -------
    locs_filtered : TYPE
        list of events location within L shell range

    """
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
    """
    Parameters
    ----------
    locs : list of events locations
    num_points : num points to add

    Returns
    -------
    locs_shifted_start : np.array
        start index of event location shifted by num_points

    """
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