import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pylab as plt
import os
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump
from sklearn.model_selection import train_test_split
from loguru import logger
import joblib
from patient_list import get_test_patient_ids

IMG_HEIGHT = 256
IMG_WIDTH = 256
    
def read_from_folders():

    patient_test_list = get_test_patient_ids()

    csv_path = "../data/all_labels.xlsx" # T2 path
    # csv_path = "/Users/neginpiran/OneDrive/Documents/ImageQuality/img_quality_adc.xlsx" # ADC path
    df = pd.read_excel(csv_path)

    rootdir = '../data/iqa_images'

    img_list, label_list, id_list = [], [], []
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
                #print(f"subdir: {subdir}")
                #print(f'end_index: {end_index}')
                #print(f'start_index: {start_index}')
                print(f"ID: {subdir[start_index:end_index]}")
                ID = subdir[start_index:end_index]
                if int(ID) not in patient_test_list:
                    # print(f'ID: {ID}')
                    dc_ar = dicom.dcmread(image_path).pixel_array
                    label = df[df['Study ID']==int(ID)]['label'].values[0]
                    img_list.append(dc_ar)
                    label_list.append(label)
                    id_list.append(int(ID))
                    
    return img_list, label_list, id_list

def resize_scale_save(img_list, label_list, id_list):
    # resize all images
    X = np.zeros((len(img_list), IMG_HEIGHT, IMG_WIDTH))
    y = np.zeros((len(img_list), 2))
    for i in range(len(img_list)):
        img = resize(img_list[i], (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[i,:,:] = img
        y[i,0] = label_list[i]
        y[i,1] = id_list[i]

    scaler = MinMaxScaler()

    # X_all = np.concatenate((X_peripheral, X_anterior), axis=0)
    # y_all = np.concatenate((y_peripheral, y_anterior), axis=0)
    X_scld = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    joblib.dump(scaler, "../models/scaler_patient.joblib")
    X_final = np.expand_dims(X_scld, axis=3)
    # y_final = np.expand_dims(y, axis=1)

    print(f'X_final.shape: {X_final.shape}')
    print(f'y_final.shape: {y_final.shape}')

    np.save('../data/X_quality_train', X_final)
    np.save('../data/y_quality_train', y_final)

if __name__ == "__main__":
    img_list, label_list, id_list = read_from_folders()
    resize_scale_save(img_list, label_list, id_list)
    
