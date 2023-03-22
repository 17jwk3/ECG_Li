"""
Author: Alexander Koch
@github: alexkoch14
"""

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import NuSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
import joblib
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

#sns.heatmap(data.corr(), annot=True)
#plt.show()

data_train = data[data['subject_id'] < 23]
data_test = data[data['subject_id'] >= 23] #create hold out set

X = data_train.iloc[:, :len(data.columns)-2].values
y = data_train.iloc[:, len(data.columns)-2:len(data.columns)-1].values

X_test = data_test.iloc[:, :len(data.columns)-2].values
y_test = data_test.iloc[:, len(data.columns)-2:len(data.columns)-1].values


def plot_distribution(Y):
    # Function to plot the stress label distribution

    values, counts = np.unique(Y, return_counts=True)
    print('Distribution of Stress Classes')
    print(np.asarray((values, counts)))
    plt.pie(counts, labels= values)
    plt.title('Distrubition of Stress Classes')
    plt.show()

plot_distribution(y)

'''
### No resampling done as F1 was used for measuring performance###

# Resample majority classes
rs = RandomUnderSampler()
#rs = RandomOverSampler()

rs.fit(X, y)
X, y = rs.fit_resample(X, y)

rs.fit(X_test, y_test)
X_test, y_test = rs.fit_resample(X_test, y_test)
'''

#plot_distribution(y)

# Preprocessing needed for stress_model_v1
#ohe = OneHotEncoder()
#Y = ohe.fit_transform(Y).toarray()
#sc = StandardScaler()
#X = sc.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33, random_state=42)


# LazyPredict Python API to assess baseline performance of various models
#clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
#models,predictions = clf.fit(X_train, X_test, y_train, y_test)
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

    model = XGBClassifier(max_depth=12, learning_rate =0.05, n_estimators=250, min_child_weight=5)

    print('Model V2 Built')
    print('Model V2 Compiled')

    model.fit(X_train, y_train)
    print('Model V2 Fitted')
    
    #model.save_model("stress_model_v2.json")
    #print('Model V2 Saved')
    
    print('Training Prediction')
    y_predict = model.predict(X_train)
    print(classification_report(y_train, y_predict, digits=3))
    conf_matr(y_train, y_predict)

    print('Validation Prediction')
    y_predict = model.predict(X_valid)
    print(classification_report(y_valid, y_predict, digits=3))
    conf_matr(y_valid, y_predict)

    print('Testing Prediction')
    y_predict = model.predict(X_test)
    print(classification_report(y_test, y_predict, digits=3))
    conf_matr(y_test, y_predict)
    
    return model

def stress_model_v3():
    # NuSVC model

    model = NuSVC()

    print('Model V3 Built')
    print('Model V3 Compiled')

    model.fit(X_train, y_train)
    print('Model V3 Fitted')
    
    model.save_model("stress_model_v3.json")
    print('Model V3 Saved')
    
    print('Training Prediction')
    y_predict = model.predict(X_train)
    print(classification_report(y_train, y_predict, digits=3))
    conf_matr(y_train, y_predict)

    print('Validation Prediction')
    y_predict = model.predict(X_valid)
    print(classification_report(y_valid, y_predict, digits=3))
    conf_matr(y_valid, y_predict)

    print('Testing Prediction')
    y_predict = model.predict(X_test)
    print(classification_report(y_test, y_predict, digits=3))
    conf_matr(y_test, y_predict)

    return model

def stress_model_v4():
    # Ridge Classifier

    model = RidgeClassifier()

    print('Model V4 Built')
    print('Model V4 Compiled')

    model.fit(X_train, y_train)
    print('Model V4 Fitted')
    
    #model.save_model("stress_model_v4.json")
    #print('Model V4 Saved')
    
    print('Training Prediction')
    y_predict = model.predict(X_train)
    print(classification_report(y_train, y_predict, digits=3))
    conf_matr(y_train, y_predict)

    print('Validation Prediction')
    y_predict = model.predict(X_valid)
    print(classification_report(y_valid, y_predict, digits=3))
    conf_matr(y_valid, y_predict)

    print('Testing Prediction')
    y_predict = model.predict(X_test)
    print(classification_report(y_test, y_predict, digits=3))
    conf_matr(y_test, y_predict)

    return model

def stress_model_v5():
    # GaussianNB

    model = GaussianNB()

    print('Model V5 Built')
    print('Model V5 Compiled')

    model.fit(X_train, y_train)
    print('Model V5 Fitted')
    
    #joblib.dump(model, "stress_model_v5.joblib")
    #print('Model V5 Saved')
    
    print('Training Prediction')
    y_predict = model.predict(X_train)
    print('Precision, Recall, F1 Score, Support')
    print('Note: Metrics calculated globally by counting the total true positives, false negatives and false positives')
    print(precision_recall_fscore_support(y_train, y_predict, average='micro'))
    print('Note: Metrics calculated for each label, and their average weight by support (the number of true instances for each label)')
    print(precision_recall_fscore_support(y_train, y_predict, average='weighted'))
    print('Classification Report')
    print(classification_report(y_train, y_predict, digits=3))
    conf_matr(y_train, y_predict)

    print('Validation Prediction')
    y_predict = model.predict(X_valid)
    print('Precision, Recall, F1 Score, Support')
    print('Note: Metrics calculated globally by counting the total true positives, false negatives and false positives')
    print(precision_recall_fscore_support(y_valid, y_predict, average='micro'))
    print('Note: Metrics calculated for each label, and their average weight by support (the number of true instances for each label)')
    print(precision_recall_fscore_support(y_valid, y_predict, average='weighted'))
    print('Classification Report')
    print(classification_report(y_valid, y_predict, digits=3))
    conf_matr(y_valid, y_predict)

    print('Testing Prediction')
    y_predict = model.predict(X_test)
    print('Precision, Recall, F1 Score, Support')
    print('Note: Metrics calculated globally by counting the total true positives, false negatives and false positives')
    print(precision_recall_fscore_support(y_test, y_predict, average='micro'))
    print('Note: Metrics calculated for each label, and their average weight by support (the number of true instances for each label)')
    print(precision_recall_fscore_support(y_test, y_predict, average='weighted'))
    print('Classification Report')
    print(classification_report(y_test, y_predict, digits=3))
    conf_matr(y_test, y_predict)

    return model


model = stress_model_v5()
#model = load_model('stress_model_v1')