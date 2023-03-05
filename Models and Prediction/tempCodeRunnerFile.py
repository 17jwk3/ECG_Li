"""
Author: Alexander Koch
@github: alexkoch14
"""

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import NuSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
import os
import random
import warnings

# Global Parameters
num_sec = 3 #recording duration / 2
sample_rt = 360 #sample frequency
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\ECG_Li\\Models and Prediction") #Path to project location
random.seed(42) #Set random state
warnings.filterwarnings("ignore") #Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Ignore TensorFlow warnings


data = pd.read_csv('stress_database/swell-hrv.csv') 
data = data[['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'HR', 'pNN50', 'SD1', 'SD2', # Time domain features
            #'VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', #Frequency domain features
            'Condition Label', 'subject_id']]

print(data.dtypes)