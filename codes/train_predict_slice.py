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

def load_data():
    X = np.load('../data/temp_x.npy')
    y = np.load('../data/temp_y.npy')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
    print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}, x_train.max: {x_train.max()}, x_test.max: {x_test.max()}')
    return x_train, x_test, y_train, y_test


def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):  #iterate through each file
        if image in set(range(1,4000,1000)):
            print(f'feature extraction for image: {image}/{len(dataset)}')
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.   
        input_img = x_train[image, :,:,:]
        img = input_img
    ################################################################
    #START ADDING DATA TO THE DATAFRAME
    #Add feature extractors, e.g. edge detection, smoothing, etc. 
            
         # FEATURE 1 - Pixel values
         
        #Add pixel values to the data frame
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values   #Pixel value itself as a feature
        #df['Image_Name'] = image   #Capture image name as we read multiple images
        
        # FEATURE 2 - Bunch of Gabor filter responses
        
                #Generate Gabor features
        num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):   #Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  #Sigma with 1 and 3
            #sigma = 1
            
                lamda = np.pi/4
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                
         
        # # FEATURE 3 Sobel
        # edge_sobel = sobel(img)
        # edge_sobel1 = edge_sobel.reshape(-1)
        # df['Sobel'] = edge_sobel1
       
        #Add more filters as needed
        
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        
    return image_dataset


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
    
def test_model(x_test,y_test, RF_model):
    test_features = feature_extractor(x_test)
    test_features = np.expand_dims(test_features, axis=0)
    test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

    #Predict on test
    test_prediction = RF_model.predict(test_for_RF)
    #Inverse le transform to get original label back. 
    # test_prediction = le.inverse_transform(test_prediction)
    print(f'test_prediction.shape:{test_prediction.shape}')
    #Print overall accuracy
    print ("Accuracy = ", accuracy_score(y_test, test_prediction))

    #Print confusion matrix
    cm = confusion_matrix(y_test, test_prediction)
    print(f'confusion matrix: {cm}')

if __name__ == "__main__":
    
    x_train, x_test, y_train, y_test = load_data()
    #Extract features from training images
    image_features = feature_extractor(x_train)
    print(f'image_features.shape: {image_features.shape}')
    RF_model = train_iqa(image_features, y_train)
    test_model(x_test,y_test, RF_model)