from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from helper import f1, f1_negative, csv_db, read_resize_XY, upsample_recurrence, summarize_diagnostics
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
    

def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3) 
    
def load_model(model_name, input_shape):
    model = Sequential()
        
    if model_name=='a':
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

        
    if model_name=='ResNet50':
        base_model = ResNet50V2(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
        base_model.trainable = False
        model = Sequential()

        model.add(base_model)
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))


    if model_name=='InceptionV3':
        
        base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
        base_model.trainable = False
        model = Sequential()

        model.add(base_model)
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))
        
        
    if model_name=='VGG16':
        
        base_model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False, pooling="avg")
        base_model.trainable = False
        model = Sequential()

        model.add(base_model)
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))

    model.add(Dense(3, activation='softmax'))    
    # opt = SGD(lr=0.001, momentum=0.9)
    opt = Adam(learning_rate=0.001)
    # opt = 'adam'   
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1, f1_negative])
    model.compile(optimizer = opt,loss = 'categorical_crossentropy', metrics = ['accuracy', top_3_categorical_accuracy])
        
    return model