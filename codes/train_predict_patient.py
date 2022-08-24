import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import random
from helper import feature_extractor, csv_db
import pydicom as dicom
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump
from patient_list import get_test_patient_ids


def load_data():
    X = np.load('../data/X_quality_train.npy')
    y = np.load('../data/y_quality_train.npy')
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
    # print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}, x_train.max: {x_train.max()}, x_test.max: {x_test.max()}')
    return X, y


def train_iqa(image_features, y_train):
    #Reshape to a vector for Random Forest / SVM training
    n_features = image_features.shape[1]
    image_features = np.expand_dims(image_features, axis=0)
    X_for_RF = np.reshape(image_features, (len(y_train), -1))  #Reshape to #images, features
    # print(f'X_for_RF.shape: {X_for_RF.shape}')
    
    #Define the classifier
    RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

    #Can also use SVM but RF is faster and may be more accurate.
    #from sklearn import svm
    #SVM_model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification
    #SVM_model.fit(X_for_RF, y_train)
    print(f'before fit: X_for_RF.shape: {X_for_RF.shape}, y_train.shape: {y_train.shape}')
    # Fit the model on training data
    RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding
    joblib.dump(RF_model, "../models/rf_patient.joblib")
    
    return RF_model

    
    
def predict_image_quality(rf_model):
    
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    # load model and scaler
    run_note = 'more train data, train as test'
    scaler = joblib.load('../models/scaler_patient.joblib')  
    csv_path = "../data/all_labels.xlsx" # T2 path
    patient_test_result_path = f'../reports/patient_test_results.csv'
    df = pd.read_excel(csv_path)
    rootdir = '../data/iqa_images'
    patient_test_list = get_test_patient_ids()
    counter = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            counter += 1
            # # T2 and not ADC nor loc
            if 'DS_Store' not in file and 'DICOMDIR' not in file and "T2" in os.path.join(subdir, file) and 'ADC' not in os.path.join(subdir, file) and 'LOC' not in os.path.join(subdir, file):
            # ADC and not T2 nor loc
            # if 'DS_Store' not in file and 'DICOMDIR' not in file and "ADC" in os.path.join(subdir, file) and 'T2' not in os.path.join(subdir, file) and 'LOC' not in os.path.join(subdir, file):
                # print(f'------- {subdir}')
                # print(f'------- {subdir}')
                image_path = os.path.join(subdir, file)
                start_index = subdir.lower().find('al study id')+21
                end_index = -2 # -2 for T2 and -3 for ADC
                # print(f"subdir: {subdir}")
                # print(f'start_index: {start_index}')
                print(f"ID: {subdir[start_index:end_index]}")
                ID = subdir[start_index:end_index]
                if int(ID) in patient_test_list:
                    # print(f'ID: {ID}')
                    dc_ar = dicom.dcmread(image_path).pixel_array
                    print(f'dc_ar.shape: {dc_ar.shape}')
                    label = df[df['Study ID']==int(ID)]['label'].values[0]
                    print(f'label: {label}')
                    dc_ar = resize(dc_ar, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                    scaled_img = scaler.transform(dc_ar)
                    scaled_img = np.expand_dims(scaled_img, axis=0)
                    scaled_img = np.expand_dims(scaled_img, axis=3)
                    image_features = feature_extractor(scaled_img)

                    n_features = image_features.shape[1]
                    image_features = np.expand_dims(image_features, axis=0)
                    X_for_RF = np.reshape(image_features, (scaled_img.shape[0], -1))  #Reshape to #images, features
                    # print(f'X_for_RF.shape: {X_for_RF.shape}')

                    img_pred = rf_model.predict(X_for_RF)
                    # save to csv
                    case_list = {}
                    case_list['ID'] = ID
                    case_list['label'] = label
                    case_list['prediction'] = img_pred
                    df_single = pd.DataFrame(case_list)
                    df_single['run_note'] = run_note
                    csv_db(df_single, patient_test_result_path)

    df_test_all = pd.read_csv(patient_test_result_path)
    df_test = df_test_all.query('run_note==@run_note')
    result_df = pd.DataFrame(classification_report(df_test['label'], df_test['prediction'], output_dict=True))
    result_df['run_note'] = run_note
    csv_db(result_df, '../reports/classification_results.csv')


if __name__ == "__main__":
    
    X, y = load_data()
    #Extract features from training images
    image_features = feature_extractor(X)
    print(f'image_features.shape: {image_features.shape}')
    rf_model = train_iqa(image_features, y)
    # test_model(X,y, RF_model)
    predict_image_quality(rf_model)
    
    