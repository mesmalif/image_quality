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
from sklearn.metrics import confusion_matrix, accuracy_score
import random
from helper import feature_extractor, csv_db
import pydicom as dicom
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump
from patient_list import get_test_patient_ids


IMG_HEIGHT = 256
IMG_WIDTH = 256

# load model and scaler
scaler = joblib.load('./models/scaler_patient.joblib') 
rf_model = joblib.load('./models/rf_patient.joblib') 
csv_path = "../data/all_labels.xlsx" # T2 path
# csv_path = "/Users/neginpiran/OneDrive/Documents/ImageQuality/img_quality_adc.xlsx" # ADC path
df = pd.read_excel(csv_path)

rootdir = '../data/iqa_images'

img_list, label_list = [], []
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
                scaled_img = scaler.transform(dc_ar)
                img_pred = rf_model.predict(scaled_img)
                # save to csv
                case_list = {}
                case_list['ID'] = ID
                case_list['label'] = label
                case_list['prediction'] = img_pred
                df_single = pd.DataFrame(case_list)
                csv_db(df_single, f'patient_test_results.csv')
                
