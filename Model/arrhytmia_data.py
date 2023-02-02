"""
Author: Alexander Koch
@github: alexkoch14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import wfdb                            
import os

# Global Parameters
num_sec = 3 #recording duration / 2
sample_rt = 360 #sample frequency
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\Model") #Path to project location

# Importing Data
data = os.getcwd() + '/arrhythmia_database/'
patients = np.loadtxt("arrhythmia_database/RECORDS", dtype=str)

nonbeat = [
            'Q',                            #Unclassifiable beat
            '~',                          #Signal quality change
            '|',                     #Isolated QRS-like artifact
            's',                                      #ST change
            'T',                                  #T-wave change
            '*',                                        #Systole
            'D',                                       #Diastole
            '"',                             #Comment annotation
            '=',                         #Measurement annotation
            'p',                                    #P-wave peak
            '^',                      #Non-conducted pacer spike
            't',                                    #T-wave peak
            '+',                                  #Rhythm change
            '?',                                       #Learning
            '!',                       #Ventricular flutter wave
            '[',      #Start of ventricular flutter/fibrillation
            ']',        #End of ventricular flutter/fibrillation
            '@',  #Link to external data (aux_note contains URL)
            '(',                                 #Waveform onset
            ')'                                    #Waveform end
            ]

abnormal = [
            'L',                  #Left bundle branch block beat
            'R',                 #Right bundle branch block beat
            'a',                #Aberrated atrial premature beat
            'V',              #Premature ventricular contraction
            'F',          #Fusion of ventricular and normal beat
            'J',              #Nodal (junctional) premature beat
            'A',                   #Atrial premature contraction
            'S',     #Premature or ectopic supraventricular beat
            'E',                        #Ventricular escape beat
            'j',                 #Nodal (junctional) escape beat
            '/',                                     #Paced beat
            'r',       #R-on-T premature ventricular contraction
            'B',              #Left or right bundle branch block
            'e',                             #Atrial escape beat
            'n',                   #Supraventricular escape beat
            'x',             #Non-conducted P-wave (blocked APB)
            'f',                #Fusion of paced and normal beat
            '-'    #Normal beat with abnormal within obs. window
            ]

normal = [
            '·',                                    #Normal beat
            'N'                                     #Normal beat
         ]    

def check_symbols():
# Check signal annotations in dataset

    #r_test = wfdb.rdrecord(data + '100')
    #a_test = wfdb.rdann(data + '100', 'atr')
    #print(wfdb.show_ann_classes())   Supported file types by wfdb
    #print(wfdb.show_ann_labels())   Supported labels by wfdb *NOT ALL INCLUDED MIT-BIH*
    #print(a_test.__dict__)   Annotation file (Array of symbol types and locations)
    #print(r_test.__dict__)   Signal file (Info on enoding, digitalization, etc)

    symbols_df = pd.DataFrame()
    # Reading all annotation files (.atr)
    for pts in patients:
        # Generating filepath for all .atr file names
        file = data + pts
        # Saving annotation object
        annotation = wfdb.rdann(file, 'atr')
        # Extracting symbols from the object
        sym = annotation.symbol
        # Saving value counts
        values, counts = np.unique(sym, return_counts=True)
        # Writing data points into dataframe
        df_sub = pd.DataFrame({'symbol':values, 'Counts':counts, 'Patient Number':[pts]*len(counts)})
        # Concatenating all data points  
        symbols_df = pd.concat([symbols_df, df_sub],axis = 0)
    #remove non beat annotation
    symbols_df = symbols_df[~symbols_df['symbol'].isin(nonbeat)]
    # Value Counts of abnocrmal and normal symbols in data
    print(symbols_df.groupby('symbol').Counts.sum().sort_values(ascending = False))

def load_ecg(file):    
# load ecg and annotation file for each patient
# Extract the symbols and annotation indexes in entire time series signal 
# Returns a list of the entire beat signal
# and symbols with beat indices (stacked vertically)

    # load the ecg file
    record = wfdb.rdrecord(file)
    # load the annotation file
    annotation = wfdb.rdann(file, 'atr')
    # extracting the signal
    ecg_signal = record.p_signal
    # extracting symbols and annotation indexes
    beat_symbols = annotation.symbol
    beat_indices = annotation.sample
    
    return ecg_signal, beat_symbols, beat_indices

def get_sequences(ecg_signal, beats, num_cols):
# Build the X,Y matrices for each beat (num_beats X num_sec*2*sample_rt)
# Returns the X,Y matrices and original symbols for any patient

    # Declare the number of beats in a patient's file (one row per beat)
    num_rows = len(beats)
    # Build X,Y matrices
    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows,1))
    beat_symbols = []
    # Keep track of number of rows
    max_row = 0
    # For every index and symbol pair (ie: every annotated signal)
    for beat_index, beat_symbol in zip(beats.beat_indices.values, beats.beat_symbols.values):
        left = max([0,(beat_index - num_sec*sample_rt) ])
        right = min([len(ecg_signal),(beat_index + num_sec*sample_rt) ])
        x = ecg_signal[left: right] # Compute the sliding window for beat X of patient X
        if (len(x) == num_cols): # Ensure we aren't at the end or start of the ecg reading
            X[max_row,:] = x
            Y[max_row,:] = int(beat_symbol in abnormal)
            beat_symbols.append(beat_symbol)
            max_row += 1
    X = X[:max_row,:]
    Y = Y[:max_row,:]
    
    # Z-score each row
    X = stats.zscore(X, axis=1)
    
    return X, Y, beat_symbols

def make_dataset():
# Makes the dataset of all patients (ignoring non-beats annotations)
# output: 
#   X_all   = ecg signal                (nbeats rows, [num_sec * 2 * sample_rt] columns)
#   Y_all   = binary label good/bad     (nbeats rows, 1 column)
#   symbols = beat symbol annotation    (nbeats rows, 1 column)

    # Number of columns per reading
    num_cols = num_sec * sample_rt * 2
    # Stack all signals, annotations and symbols vertically
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    symbols = []

    for patient in patients:
        # Get the path to extract patient X's info
        file = data + patient
        # Load patient X's entire ecg signal, all beat symbols and beat indices
        ecg_signal, beat_symbols, beat_indices = load_ecg(file)
        # Grab the MLII signal
        ecg_signal = ecg_signal[:,0]
        
        # Make df to exclude the nonbeats
        beats = pd.DataFrame({'beat_symbols':beat_symbols, 'beat_indices':beat_indices})
        beats = beats.loc[beats.beat_symbols.isin(abnormal + normal)]

        abnormal_temp = abnormal.copy()
        abnormal_temp.remove('-')

        #### Sliding window code: if any beat in the sequence window is abnormal, then the whole sequence is flagged abnormal ###
        # Check to see if adjascent beats in the observation window are bad
        for i in range (0, beats.shape[0] - 4): # For every heart beat X
            if beats.iloc[i]['beat_symbols'] in normal: # If the current beat is normal
                for j in range (1, 4): # We know that the current beat is normal
                    if beats.iloc[i+j]['beat_symbols'] in normal: # And if one of the next beats in the observation window is also  normal
                        pass # Then do nothing
                    elif ((beats.iloc[i+j]['beat_indices'] - beats.iloc[i]['beat_indices']) <= num_sec*sample_rt): #If the next beat is bad and within the window
                        beats.iloc[i, beats.columns.get_loc('beat_symbols')] = '-' # Then set the current good beat to bad (ie: flag the sequence)

            elif beats.iloc[i]['beat_symbols'] in abnormal_temp: # If current beat is abnormal
                for j in range (1, 4): #we know that the current beat is abnormal
                    if beats.iloc[i+j]['beat_symbols'] in abnormal_temp: # And if one of the next beats is also abnormal
                        pass # Then do nothing
                    elif ((beats.iloc[i+j]['beat_indices'] - beats.iloc[i]['beat_indices']) <= num_sec*sample_rt): # If the next beat is good and within window
                        beats.iloc[i+j, beats.columns.get_loc('beat_symbols')] = '-' #Then set the adjacent good beat to ('bad beat adjacent')

        # Get a dataframe for all of patient X's beats
        X, Y, symbol = get_sequences(ecg_signal, beats, num_cols)

        symbols = symbols+symbol # Append the symbol of reading X to the list of all symbols
        X_all = np.append(X_all,X,axis = 0) # Append ecg signal of every beat 
        Y_all = np.append(Y_all,Y,axis = 0) # Append encoded value of that beat
        print('Patient ' + patient + ' added to the dataset')
        
    # Drop the first zero row to account for first beat not having num_sec buffer on the left side
    X_all = X_all[1:,:]
    Y_all = Y_all[1:,:]

    print('Dataset created successfully')

    return X_all, Y_all, symbols

def plot_heartbeat(plot_beat = 'N'):
    # Plot the ecg signal of desired heart beat type
    # Default argument is of a normal beat

    for patient in patients:
        # Get the path to extract patient X
        file = data + patient

        ecg_signal, beat_symbols, beat_indices = load_ecg(file)

        if plot_beat in beat_symbols:

            ab_index = [b for a,b in zip(beat_symbols,beat_indices) if a in plot_beat][10]

            # Generating evenly spaced values
            x = np.arange(len(ecg_signal))

            left = ab_index-(num_sec * sample_rt)
            right = ab_index+(num_sec * sample_rt)

            plt.figure(figsize=(12,6))
            plt.plot(x[left:right],ecg_signal[left:right,0],'-')
            plt.plot(x[beat_indices],ecg_signal[beat_indices,0], 'none')
            plt.plot(x[ab_index],ecg_signal[ab_index,0],'ro',label=plot_beat)

            plt.xlim(left,right)
            plt.ylim(ecg_signal[left:right].min()-0.05,ecg_signal[left:right, 0].max()+0.05)
            plt.xlabel('Time Index')
            plt.ylabel('ECG signal')
            #plt.legend(bbox_to_anchor = (1.04,1), loc = 'upper left')
            plt.show()
            break
        else: 
            pass

def plot_distribution():
    # Function to plot the target label distribution
    
    values, counts = np.unique(Y_all, return_counts=True)
    print(np.asarray((values, counts)))
    plt.pie(counts, labels= values)
    plt.title('Distrubition of Normal/Abnormal Ratio')
    plt.show()

    values, counts = np.unique(symbols, return_counts=True)
    print(np.asarray((values, counts)))
    plt.pie(counts, labels= values)
    plt.title('Distrubition of Abnormalities')
    plt.show()
    return

# Create the datasets
# X_all, Y_all, symbols = make_dataset()

# Save the datasets
# np.savez_compressed('arr_data',a=X_all, b=Y_all, c=symbols)

# Import datasets for visualization
training_data = np.load('arr_data.npz')
X_all = training_data['a']
Y_all = training_data['b']
symbols = training_data['c']

plot_distribution()

plot_heartbeat('N')