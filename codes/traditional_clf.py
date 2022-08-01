import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from tqdm import tqdm
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def Binarypattern(im):                               # creating function to get local binary pattern
    img= np.zeros_like(im)
    n=3                                              # taking kernel of size 3*3
    for i in range(0,im.shape[0]-n):                 # for image height
        for j in range(0,im.shape[1]-n):               # for image width
            x  = im[i:i+n,j:j+n]                     # reading the entire image in 3*3 format
            center       = x[1,1]                    # taking the center value for 3*3 kernel
            img1        = (x >= center)*1.0          # checking if neighbouring values of center value is greater or less than center value
            img1_vector = img1.T.flatten()           # getting the image pixel values 
            img1_vector = np.delete(img1_vector,4)  
            digit = np.where(img1_vector)[0]         
            if len(digit) >= 1:                     # converting the neighbouring pixels according to center pixel value
                num = np.sum(2**digit)              # if n> center assign 1 and if n<center assign 0
            else:                                    # if 1 then multiply by 2^digit and if 0 then making value 0 and aggregating all the values of kernel to get new center value
                num = 0
            img[i+1,j+1] = num
    return(img)

def create_LBP_features(data):
    Feature_data = np.zeros(data.shape)

    for i in range(len(data)):
        img = data[i]
        imgLBP=Binarypattern(img)  
        Feature_data[i] = imgLBP
    
    return Feature_data

X = np.load('./data/X_quality.npy')
# X = X1[:100, :,:,:]
# y = np.array('./data/y_quality.npy')
y = pd.read_csv('./data/y_remap.csv').values
# y = y1[:100, :]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42, shuffle = True, stratify=y)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("-------------------------------")
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

# X_train_Gabor=create_Gabor_features(X_train)
# X_test_Gabor=create_Gabor_features(X_test)

# print("X_train_Gabor shape: ", X_train_Gabor.shape)
# print("X_test_Gabor shape: ", X_test_Gabor.shape)

Feature_X_train = create_LBP_features(X_train)
Feature_X_test = create_LBP_features(X_test)

print("Feature_X_train shape: ", Feature_X_train.shape)
print("Feature_X_test shape: ", Feature_X_test.shape)

X_train_HOG_Flat = np.zeros((len(X_train), 256*256))
for i in range(len(Feature_X_train)):
    img = Feature_X_train[0]
    img = img.flatten()
    X_train_HOG_Flat[i] = img

print(f'X_train_HOG_Flat.shape: {X_train_HOG_Flat.shape}')

X_test_HOG_Flat = np.zeros((len(X_test), 256*256))
for i in range(len(Feature_X_test)):
    img = Feature_X_test[0]
    img = img.flatten()
    X_test_HOG_Flat[i] = img

print(f'X_test_HOG_Flat.shape: {X_test_HOG_Flat.shape}')

scaler = StandardScaler()
X_train_HOG_Flat = scaler.fit_transform(X_train_HOG_Flat)


param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True, verbose = True )
model=GridSearchCV(svc,param_grid)
a = np.squeeze(y_train)

model.fit(X_train_HOG_Flat,a)
y_pred = model.predict(X_test_HOG_Flat)
y_ground = np.squeeze(y_test)
print(f'y_ground.shape: {y_ground.shape}, y_pred.shape: {y_pred.shape}')
accuracy_svm = accuracy_score(y_ground,y_pred)

print(f'accuracy_svm: {accuracy_svm}')