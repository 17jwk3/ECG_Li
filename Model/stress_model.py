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
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\Model") #Path to project location
random.seed(42) #Set random state
warnings.filterwarnings("ignore") #Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Ignore TensorFlow warnings


data = pd.read_csv('stress_database/swell-hrv.csv') 
data = data[['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD',
            'HR', 'pNN50', 'SD1', 'SD2','Condition Label']]

X = data.iloc[:, :len(data.columns)-1].values
y = data.iloc[:, len(data.columns)-1:len(data.columns)].values

def plot_distribution(Y):
    # Function to plot the stress label distribution

    values, counts = np.unique(Y, return_counts=True)
    print(np.asarray((values, counts)))
    plt.pie(counts, labels= values)
    plt.title('Distrubition of Stress Classes')
    plt.show()

plot_distribution(y)

# Resample majority classes
#rs = RandomUnderSampler()
rs = RandomOverSampler()
rs.fit(X, y)
X, y = rs.fit_resample(X, y)

#plot_distribution(y)

# Preprocessing needed for stress_model_v1
#ohe = OneHotEncoder()
#Y = ohe.fit_transform(Y).toarray()
#sc = StandardScaler()
#X = sc.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33, random_state=42)

# LazyPredict Python API to assess baseline performance of various models
#clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
#models,predictions = clf.fit(X_train, X_valid, y_train, y_valid)
#print(models)

def model_report(y_actual, y_pred):
    # Print evaluation metrics and confusion matrix for Neural Network (model_v1)

    # Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    
    # Converting one hot encoded test label to label
    actual = list()
    for i in range(len(y_actual)):
        actual.append(np.argmax(y_actual[i]))


    accuracy = accuracy_score(actual, pred)
    recall = recall_score(actual, pred, average='macro')
    precision = precision_score(actual, pred, average='macro')

    print('Accuracy:%.3f'%accuracy)
    print('Recall:%.3f'%recall)
    print('Precision:%.3f'%precision)
    print(' ')

    conf_matr(actual, pred)

    return accuracy, recall, precision

def conf_matr(y_actual, y_pred):
    # Plot confusion matrix

    conf_matrix = confusion_matrix(y_actual, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['Normal', 'Low Stress', 'High Stress'])
    cm_display.plot()
    plt.show()

def stress_model_v1():
    # Proof of concept NN
    
    model = Sequential()
    model.add(Dense(16, input_shape=(len(data.columns)-1,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    print('Model V1 Built')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Model V1 Compiled')

    print('Model Summary')
    print(model.summary())

    # Fitting data in model - longest step
    model.fit(X_train, y_train, batch_size=32, epochs=10)
    print('Model V1 Fitted')

    #model.save("stress_model_v1")
    #print('Model V1 Saved')

    print('Training Prediction')
    y_train_pred = model.predict(X_train, verbose=1)
    model_report(y_train, y_train_pred)

    print('Validation Prediction')
    y_valid_pred = model.predict(X_valid, verbose=1)
    model_report(y_valid, y_valid_pred)

    return model

def stress_model_v2():
    # XGB Model

    #Model V2.1: 79% Accuracy
    #model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight=1) 

    #Model V2.2: 99% Accuracy
    model = XGBClassifier(max_depth=12, learning_rate =0.05, n_estimators=250, min_child_weight=5)

    print('Model V2 Built')
    print('Model V2 Compiled')

    model.fit(X_train, y_train)
    print('Model V2 Fitted')
    
    model.save_model("stress_model_v2.json")
    print('Model V2 Saved')
    
    print('Training Prediction')
    y_predict = model.predict(X_train)
    print(classification_report(y_train, y_predict, digits=3))
    conf_matr(y_train, y_predict)

    print('Validation Prediction')
    y_predict = model.predict(X_valid)
    print(classification_report(y_valid, y_predict, digits=3))
    conf_matr(y_valid, y_predict)
    
    return model

#model = stress_model_v1()
model = stress_model_v2()
#model = load_model('stress_model_v1')