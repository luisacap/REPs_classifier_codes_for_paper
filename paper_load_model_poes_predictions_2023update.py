#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:49:16 2023

@author: luisacap
"""

# THIS SAMPLE CODE
# 1. Loads the saved model and normalization
# 2. Loads one date of POES/MetOp data
# 3. Pre-processes the POES/MetOp data
# 4. Provides the REPs and CSSs in the outer radiation belt (2.5<l<8.5) for the selected POES/MetOp date
# 5. Post-processing of the results


#%% 1. Load the saved model and normalization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from joblib import load

model_str = 'REPs_classifier_model'
model = models.load_model(model_str)
timestep = 7

print('Model ')
print(model_str)
print('LOADED!')
print()
#print('Model summary :')
#print()
#model.summary()

filename = 'LSTM_scaler' #normalization 
scaler = load(filename+'.joblib')

#%% 2. Load one date of POES/MetOp data
import xarray as xr
import paper_library_2023update as lib
import pandas as pd
import numpy as np

yy = ['2017']              #year
mm = ['11']                #month
dd = ['21']                #day
sc = ['m01']               #spacecraft abbreviation
sc_extended = ['metop01']  #spacecraft extended name

data_folder = '...PATH_OF_POES_or_METOP_data...'+yy[0]+'/'+sc_extended[0]+'/'
file_id = 'poes_'+sc[0]+'_'+yy[0]+mm[0]+dd[0]+'_proc.nc'
print('###############################################################')
print('Finding REPs/CSSs for file:'+data_folder+file_id)

day = xr.open_dataset(data_folder + file_id)
E1_0 = day.variables['mep_ele_tel0_flux_e1'][:]
E1_90 = day.variables['mep_ele_tel90_flux_e1'][:]
E1_0 = E1_0.values.reshape(len(E1_0), 1)
E1_90 = E1_90.values.reshape(len(E1_90), 1)
E2_0 = day.variables['mep_ele_tel0_flux_e2'][:]
E2_90 = day.variables['mep_ele_tel90_flux_e2'][:]
E2_0 = E2_0.values.reshape(len(E2_0), 1)
E2_90 = E2_90.values.reshape(len(E2_90), 1)
E3_0 = day.variables['mep_ele_tel0_flux_e3'][:]
E3_90 = day.variables['mep_ele_tel90_flux_e3'][:]
E3_0 = E3_0.values.reshape(len(E3_0), 1)
E3_90 = E3_90.values.reshape(len(E3_90), 1)
E4_0 = day.variables['mep_ele_tel0_flux_e4'][:]
E4_90 = day.variables['mep_ele_tel90_flux_e4'][:]
E4_0 = E4_0.values.reshape(len(E4_0), 1)
E4_90 = E4_90.values.reshape(len(E4_90), 1)

dataset = np.concatenate((E1_0,E1_90,E2_0,E2_90,E3_0,E3_90,E4_0,E4_90),axis=1)
dataset_day = pd.DataFrame(dataset, columns = ['e1_0','e1_90','e2_0','e2_90','e3_0','e3_90','e4_0','e4_90'])
dataset_day_copy = dataset_day.copy()

#%% 3. Pre-processes the POES/MetOp data
cleaned_dataset_day, dataset_day = lib.preproc(dataset_day)
test_day = np.array(cleaned_dataset_day)
test_day = scaler.transform(test_day)
y_dummy = test_day[:,0]
test_day, _ = lib.create_shifted_dataset(test_day, y_dummy,  timestep)


#%% 4. Provides the REPs and CSSs in the outer radiation belt (2.5<L<8.5) for the selected POES/MetOp date
predictions_day = model.predict(test_day)
flag_day = predictions_day.argmax(1)
dataset_day_plot = dataset_day[0:len(flag_day)]

css_label = 2
CSSs = np.where(flag_day == css_label, css_label, 0)
_ , css_count, css_locs = lib.clean_up(CSSs, 0, css_label) 

rep_label = 1
REPs = np.where(flag_day == rep_label, rep_label, 0)    
_ , rep_count, rep_locs = lib.clean_up(REPs, 0, rep_label) 

L  = day.variables['L_IGRF'][:]
MLT = day.variables['MLT'][:]
L  = L.values.reshape(len(L), 1)
#interpolate negative values of L
ind_L = np.where(L < 0)
if len(ind_L) > 0:
    L[ind_L[0]]= np.nan
    ok = ~np.isnan(L)
    xp = ok.ravel().nonzero()[0]
    fp = L[~np.isnan(L)]
    x = np.isnan(L).ravel().nonzero()[0]
    from scipy.interpolate import CubicSpline
    CS = CubicSpline(xp,fp)
    L = CS(np.arange(len(L)))
    L  = np.float32(L.reshape(len(L), 1))

MLT = MLT.values.reshape(len(MLT), 1)
MLT = MLT.astype('float32')
MLT = MLT/1e9/3600. # because MLT in the file is in ns
dataset_day['MLT'] = MLT
dataset_day['L'] = L

css_locs_Lshell = lib.filter_Lshell(css_locs,L,2.5,8.5)
rep_locs_Lshell = lib.filter_Lshell(rep_locs,L,2.5,8.5)

print('Number of identified REPs :', len(rep_locs_Lshell))
print('Most probable REPs indices (locations) :', rep_locs_Lshell)
print()
print('Number of identified CSSs :', len(css_locs_Lshell))
print('Most probable CSSs indices (locations):', css_locs_Lshell)
print()

                
#%% 5. Post-processing
# Add 3 points to the start-stop indices
# Join REPs if separated by 5 points
# Remove unphysical events
  # only keep events if max(E4_0)>=2 count/s
  # only keep events if max(E4_0) <= max(E3_0)
  #   if this is not true, then keep event if max(E4_0) <= max(E2_0) and max(E4_0) <= max(E1_0)
rep_shifted_start,rep_shifted_stop = lib.shift_locs(rep_locs_Lshell,3)
css_shifted_start,css_shifted_stop = lib.shift_locs(css_locs_Lshell,3)
rep_shifted_start,rep_shifted_stop,rep_num = lib.join_nearby_events(rep_shifted_start,rep_shifted_stop,5)
css_shifted_start,css_shifted_stop,css_num = lib.join_nearby_events(css_shifted_start,css_shifted_stop,5)
rep_start_new,rep_stop_new,rep_num = lib.remove_unphysical(dataset_day,dataset_day_copy,rep_shifted_start,rep_shifted_stop,rep_num)
css_start_new,css_stop_new,css_num = lib.remove_unphysical(dataset_day,dataset_day_copy,css_shifted_start,css_shifted_stop,css_num)
            
print('After post-processing....')
print('Number of identified REPs :', rep_num)
print('Most probable REPs start indices (locations) :', rep_start_new)
print('Most probable REPs stop indices (locations) :', rep_stop_new)
print()
print('Number of identified CSSs :', css_num)
print('Most probable CSSs start indices (locations):', css_start_new)
print('Most probable CSSs stop indices (locations):', css_stop_new)
print()

 
    

