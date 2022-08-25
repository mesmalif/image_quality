import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import random
from helper import feature_extractor, csv_db
from sklearn import svm

def load_data():
    x_train = np.load('../data/X_quality_train.npy')
    y_train = np.load('../data/y_quality_train.npy')
    x_test = np.load('../data/X_quality_test.npy')
    y_test = np.load('../data/y_quality_test.npy')
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
    # print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}, x_train.max: {x_train.max()}, x_test.max: {x_test.max()}')
    return x_train, x_test, y_train, y_test


def train_iqa(image_features, y, method='random_forest'):
    y_train = y[:,0].copy()
    #Reshape to a vector for Random Forest / SVM training
    # n_features = image_features.shape[1]
    image_features = np.expand_dims(image_features, axis=0)
    X_for_RF = np.reshape(image_features, (len(y_train), -1))  #Reshape to #images, features
    # print(f'X_for_RF.shape: {X_for_RF.shape}')
    
    #Define the classifier
    if method=='random_forest':
        model = RandomForestClassifier(n_estimators = 250, random_state = 42)
    elif method=='svm':
        model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification

    print(f'before fit: X_for_RF.shape: {X_for_RF.shape}, y_train.shape: {y_train.shape}')
    # Fit the model on training data
    model.fit(X_for_RF, y_train) #For sklearn no one hot encoding
    joblib.dump(model, "../models/patient_model.joblib")
    
    return model
    
def test_model(x_test, y, model):
    patient_test_result_path = f'../reports/patient_test_results.csv'
    y_test = y[:,0].copy()
    y_label = y[:,1].copy()
    test_features = feature_extractor(x_test)
    test_features = np.expand_dims(test_features, axis=0)
    test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

    #Predict on test
    test_prediction = model.predict(test_for_RF)
    #Inverse le transform to get original label back. 
    # test_prediction = le.inverse_transform(test_prediction)

    #Print confusion matrix
    cm = confusion_matrix(y_test, test_prediction)
    print(f'confusion matrix (pred on test dataset): {cm}')
    clf_report = pd.DataFrame(classification_report(y_test, test_prediction, output_dict=True))
    print(f'clf_report: {clf_report}')
    result_df = pd.DataFrame()
    result_df.loc[:, 'y_test'] = y_test.copy()
    result_df.loc[:, 'test_prediction'] = test_prediction.copy()
    result_df.loc[:, 'label'] = y_label.copy()
    
    csv_db(result_df, patient_test_result_path)

if __name__ == "__main__":
    
    x_train, x_test, y_train, y_test = load_data()
    #Extract features from training images
    image_features = feature_extractor(x_train)
    print(f'image_features.shape: {image_features.shape}')
    model = train_iqa(image_features, y_train, 'svm')
    test_model(x_test, y_test, model)
