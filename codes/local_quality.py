# 1- adam opt

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.metrics import top_k_categorical_accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from helper import csv_db, img_resize
from models import load_model
from sklearn.utils import class_weight
from collections import Counter

def main(model_name, run_note):
    batch_size = 32
    img_rows = 224 
    img_cols = 224
    # X = np.load('../../classification/data/X_quality.npy') # T2 images
    X = np.load('../../classification/data/X_quality_adc.npy') # ADC images
    # X = denoise_img(X)
    X = img_resize(X, img_rows, img_cols)
    X = np.repeat(X[..., np.newaxis], 3, -1)
    # y = pd.read_csv("../../classification/data/y_remap.csv").values # T2 images
    y = np.load('../../classification/data/y_quality_adc.npy') # ADC images
    
    print(f'X.shape: {X.shape}, y.shape: {y.shape}')

    # Split dataset to train/validation/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=1)
    
#     class_weights = class_weight.compute_class_weight('balanced',
#                                                  classes = np.unique(y),
#                                                  y = y.reshape(-1))
    
#     print(f'class_weights: {class_weights}')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, shuffle=True, random_state=1)
    print(f'X.min(): {X.min()}, X.max(): {X.max()}')
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
    print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')
    print(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')

    ######### Data augmentation

    train_datagen = ImageDataGenerator(rotation_range=45,
        width_shift_range=0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
    train_datagen.fit(X_train)

    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size = batch_size)
    
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    model = load_model(model_name, input_shape)
    print(model.summary())

    history = model.fit_generator(
            train_generator,
            steps_per_epoch = len(X_train)//batch_size,
            # class_weight = class_weight,
            epochs = 1000,
           # class_weight = {0:10.37, 1:1.35, 2:.46},
            validation_data = (X_test, y_test)
    )

    hist_df = pd.DataFrame(history.history)
    print(f'hist_df: {hist_df}')
    # hist_df.to_csv(f'../reports/quality_history.csv')
    hist_df.loc[:,'run_note'] = run_note
    csv_db(hist_df, f'../reports/quality_history.csv')
    ypred = model.predict(X_val).round()

    print(f'X_val.shape: {X_val.shape}, ypred.shape: {ypred.shape}, y_val.shape: {y_val.shape}')
    y_val_value = np.argmax(y_val, axis=1)
    ypred_value = np.argmax(ypred, axis=1)
    print(f'y_val_value.shape: {y_val_value.shape}, ypred_value.shape: {ypred_value.shape}')
    clf_rep_df = pd.DataFrame(classification_report(y_val_value, ypred_value, output_dict=True))
    clf_rep_df.loc[:,'run_note'] = run_note
    csv_db(clf_rep_df, f'../reports/quality_predictions.csv')


    
if __name__=="__main__":
    # model_names = {'a', 'ResNet50', 'InceptionV3', 'VGG16'}
    model_names = {'a'}
    for model_name in model_names:
        run_note = model_name + "_b32_adc_NoWeights_1000"
        main(model_name, run_note)
