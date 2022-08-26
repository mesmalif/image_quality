from tensorflow.keras import backend as K
import os
import pandas as pd
import time
from datetime import datetime
import cv2
from skimage.transform import resize
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.stats import uniform, truncnorm, randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import logging
import sys
from skimage.restoration import estimate_sigma,denoise_bilateral
from scipy import ndimage as nd
from skimage.exposure import equalize_adapthist
from skimage.filters import sobel
from sklearn.metrics import f1_score, classification_report

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def f1_negative(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_negatives = K.sum(K.round(K.clip(abs(y_true-1) * abs(y_pred-1), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(abs(y_pred-1), 0, 1)))
        recall = true_negatives / (possible_negatives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_negatives = K.sum(K.round(K.clip(abs(y_true-1) * abs(y_pred-1), 0, 1)))
        predicted_negatives = K.sum(K.round(K.clip(abs(y_true-1), 0, 1)))
        precision = true_negatives / (predicted_negatives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def csv_db(df, fname):
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    df.loc[:, 'save_date'] = current_time
    df.loc[:, 'save_ts'] = time.time()
    file_exists = os.path.isfile(fname) 
    if file_exists:
        df.to_csv(fname, mode='a', index=True, header=False)
    else:
        df.to_csv(fname, mode='a', index=True, header=True)
        
def img_resize(imgs, img_rows, img_cols, equalize=False):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def get_folder_list(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
def get_file_list(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def read_resize_XY(root_path, img_height, img_width):
    img_list, label_list = [], []
    # root_path = "./data/MRI_2classes_val"
    folders = get_folder_list(root_path)
    for directory in folders:
        # print(f'---------- directory: {directory}')

        sub_dir = os.path.join(root_path, directory)
        files = get_file_list(sub_dir)
        for file in files:
            file_path = os.path.join(sub_dir, file)
            img = cv2.imread(file_path)
            img_list.append(img)
            label = [1 if directory=='recurrence' else 0]
            # print(label)
            label_list.append(label) 
            
    # resize all images
    X = np.zeros((len(img_list), img_height, img_width, 3))
    y = np.zeros(len(img_list))
    for i in range(len(img_list)):
        img = resize(img_list[i], (img_height, img_width, 3), mode='constant', preserve_range=True)
        X[i,:,:] = img
        # print(label_list[i][0])
        y[i] = label_list[i][0]
        
    return X, y

def delete_file(files_to_be_deleted):
    for file in files_to_be_deleted:
        if os.path.isfile(file):
            os.remove(file)
            
def rf_params():
    return  {
    # randomly sample numbers from 4 to 204 estimators
    'n_estimators': randint(4,200),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'min_samples_split': uniform(0.01, 0.199)
}

def svc_params():
    return {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

def lr_params():
    return {'penalty' : ['l2'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['newton-cg']}


def model_a(img_size):

    INPUT_SHAPE = (img_size, img_size, 3)   #change to (SIZE, SIZE, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1, f1_negative])
    return model
    

def model_b(img_size):

    INPUT_SHAPE = (img_size, img_size, 3)   #change to (SIZE, SIZE, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1, f1_negative])

    return model
    
def model_c(img_size):

    INPUT_SHAPE = (img_size, img_size, 3)   #change to (SIZE, SIZE, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1, f1_negative])

    return model
    
def log_run_info(filename, log_level=logging.INFO):
    '''
    ex:
    logging = log_run_info(filename=f'./logs/log_file.txt')
    logging.info(f'--------log any f string ---')
    '''
        
    logging.basicConfig(filename=filename,
                        filemode='a',
                        format='[%(asctime)s] {%(pathname)s:%(lineno)d} [%(levelname)s]: \n %(message)s',
                        datefmt='%m/%d/%Y, %H:%M:%S',
                        level=log_level)
    return logging

def upsample_recurrence(X, y):
    print(f'y.sum() start: {y.sum()}')
    image_size = X.shape[1]
    X = X.reshape(-1,image_size, image_size)
    # filter for recurrence samples
    X_rec = X[y!=0,:,:]
    y_rec = y[y!=0]
    print(f'X_rec.shape: {X_rec.shape}')
    aug_mult=int(len(y)/len(y_rec))-1

    datagen = ImageDataGenerator(width_shift_range=[-200,200], 
                                 height_shift_range=0.5, 
                                 horizontal_flip=True,
                                 rotation_range=90,
                                 zoom_range=[0.5,1.0])
    X_rec = np.expand_dims(X_rec, 0)
    it = datagen.flow(X_rec, batch_size=1)
    image_list, y_clf_list = list(), list()
    for i in range(aug_mult):
        batch = it.next()
        image = batch[0]#.astype('uint8')
        image_list.append(image)
        y_clf_list.append(y_rec)
        print(f' augmented image sum: {image.sum()} and shape: {image.shape}')
    X_rec_upsmpld = np.asarray(image_list).reshape(-1,image_size, image_size)
    y_rec_upsmpld = np.asarray(y_clf_list).reshape(-1)
    
    print(f'X.shape: {X.shape}, y.shape: {y.shape}')
    print(f'X_rec_upsmpld.shape: {X_rec_upsmpld.shape}')
    X_upsmpld = np.append(X, X_rec_upsmpld, axis=0)
    y_upsmpld = np.append(y, y_rec_upsmpld, axis=0)
    print(f'y_upsmpld.sum() end: {y_upsmpld.sum()}')
    print(f'X_upsmpld.shape: {X_upsmpld.shape}')
    X_upsmpld = np.expand_dims(X_upsmpld, 3)
    return X_upsmpld, y_upsmpld

def summarize_diagnostics(history, run_note, pic_path):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    plt.savefig(pic_path + f'{run_note}.png')
    plt.close()

def denoise_img(noisy_img):
    
    # gausian denoising
    denoised = nd.gaussian_filter(noisy_img, sigma=1)
    
    #bilateral denoising
    sigma_est = estimate_sigma(noisy_img, multichannel=True, average_sigmas=True)
    #denoised = denoise_bilateral(noisy_img, sigma_spatial=1, multichannel=False)
    
    return denoised



def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):  #iterate through each file
        if image in set(range(1,4000,100)):
            print(image)
        
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


def patient_results(file_path):
    df_vote = pd.DataFrame()
    df_results = pd.read_csv(file_path)
    last_ts = df_results['save_date'].iloc[-1]
    df_fltd = df_results.query('save_date==@last_ts')
    df_vote = pd.DataFrame()
    for patient_id in df_fltd['patient_id'].unique():
        df_id = df_fltd.query('patient_id==@patient_id')
        vote = df_id.test_prediction.value_counts().index[0]
        y_test = df_id.y_test.values[0]
        df_vote.loc[len(df_vote.index),['patient_id','vote','g_truth']] = [patient_id, vote, y_test]

        
    clf_report = pd.DataFrame(classification_report(df_vote['g_truth'], df_vote['vote'], output_dict=True))
    print(f'patient-based clf report: {clf_report}')
    return clf_report
