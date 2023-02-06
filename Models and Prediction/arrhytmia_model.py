"""
Author: Alexander Koch
@github: alexkoch14
"""

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPool1D, ELU, BatchNormalization
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import random
import warnings

# Global Parameters
num_sec = 3 #recording duration / 2
sample_rt = 360 #sample frequency
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\ECG_Li\\Model") #Path to project location
random.seed(42) #Set random state
warnings.filterwarnings("ignore") #Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Ignore TensorFlow warnings

def model_report(y_actual, y_pred):
    # Print evaluation metrics and confusion matrix

    # Classification is done with X% likelihood, determined by defined thereshold value
    # https://stackoverflow.com/questions/48619132/binary-classification-predict-method-sklearn-vs-keras
    threshold = 0.5

    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > threshold))
    recall = recall_score(y_actual, (y_pred > threshold))
    precision = precision_score(y_actual, (y_pred > threshold))
    specificity = sum((y_pred < threshold) & (y_actual == 0)) /sum(y_actual ==0)
    prevalence = (sum(y_actual)/len(y_actual))
    print('AUC:%.3f'%auc)
    print('Accuracy:%.3f'%accuracy)
    print('Recall:%.3f'%recall)
    print('Precision:%.3f'%precision)
    print('Specificity:%.3f'%specificity)
    print('Prevalence:%.3f'%prevalence)
    print(' ')

    conf_matrix = confusion_matrix(y_actual, (y_pred > threshold))
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['Normal', 'Abnormal'])
    cm_display.plot()
    plt.show()

    return auc, accuracy, recall, precision, specificity

# Load training data
training_data = np.load('arr_data.npz')
X_all = training_data['a'] #tuple of len 109,291 each with 2,160 nested elements
Y_all = training_data['b'] #tuple of len 109,291 each with 1 nested element
symbols = training_data['c']

# X_train is a 2D array, each element is the array of one heartbeat
# ie: X_train has 73,224 element in dimension 0, and 2160 elements in dimension 1 (ie: 73,224 heart beats)
X_train, X_valid, y_train, y_valid = train_test_split(X_all, Y_all, test_size=0.33, random_state=42)

# Nest each dimension 1 element so that all 2,160 elements are nested in seperate lists, not appended to the same list
X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_valid_cnn = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

def arr_model_v1():
    # Single Layer CNN

    model = Sequential()
    model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu', input_shape = (sample_rt*num_sec*2,1))) # Regardless of how many beats I have, each beat has this shape 
    model.add(Dropout(rate = 0.25))
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))
    print('Model V1 Built')

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print('Model V1 Compiled')

    print(model.summary())

    # Fitting data in model - longest step
    model.fit(X_train_cnn, y_train, batch_size = 32, epochs= 2)
    print('Model V1 Fitted') 

    #model.save("arr_model_v1")
    #print('Model V1 Saved')

    return model

def arr_model_v2():
    # VGGNet proposed by Tae Joon Jun et al.
    # https://arxiv.org/pdf/1804.06812.pdf

    model = Sequential()
    model.add(Conv1D(filters= 64, kernel_size= 3, strides= 1, input_shape = (sample_rt*num_sec*2,1))) # Regardless of how many beats I have, each beat has this shape 
    model.add(ELU())
    model.add(BatchNormalization())

    model.add(Conv1D(filters= 64, kernel_size= 3, strides= 1))
    model.add(ELU())
    model.add(BatchNormalization())

    model.add(MaxPool1D(pool_size=2, strides= 2))

    model.add(Conv1D(filters= 128, kernel_size= 3, strides= 1))
    model.add(ELU())
    model.add(BatchNormalization())

    model.add(Conv1D(filters= 128, kernel_size= 3, strides= 1))
    model.add(ELU())
    model.add(BatchNormalization())

    model.add(MaxPool1D(pool_size=2, strides= 2))

    model.add(Conv1D(filters= 256, kernel_size= 3, strides= 1))
    model.add(ELU())
    model.add(BatchNormalization())

    model.add(Conv1D(filters= 256, kernel_size= 3, strides= 1))
    model.add(ELU())
    model.add(BatchNormalization())

    model.add(MaxPool1D(pool_size=2, strides= 2))
    model.add(Flatten())

    model.add(Dense(2048))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    print('Model V2 Built')

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print('Model V2 Compiled')

    print(model.summary())

    # Fitting data in model - longest step
    model.fit(X_train_cnn, y_train, batch_size = 32, epochs= 2)
    print('Model V2 Fitted') 

    model.save("arr_model_v2")
    print('Model V2 Saved')
    return model

# Build the model
#model = arr_model_v2()

# Import saved model
model = load_model('arr_model_v1')

# Metriction and metrics
print('Training Predict')
y_train_preds_cnn = model.predict(X_train_cnn,verbose = 1)
model_report(y_train, y_train_preds_cnn)


print('Validation Predict')
y_valid_preds_cnn = model.predict(X_valid_cnn,verbose = 1)
model_report(y_valid, y_valid_preds_cnn)