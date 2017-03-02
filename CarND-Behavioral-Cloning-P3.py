
# coding: utf-8

# In[1]:

import pandas as pd 
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random
get_ipython().magic('matplotlib inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Load the data
DIR ='data/data/'

df = pd.read_csv('data/data/driving_log.csv')

#Shuffle the data
df = df.iloc[np.random.permutation(len(df))]

#Split the data into Train and Validation set
X_train, X_Validation = train_test_split(df, test_size=0.20)


# In[2]:

def read_img(path):
    """
    Read image from the filepath and convert it to RGB format.
    """
    img = cv2.imread(path)
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img


# In[3]:

def image_flip(image,steering):
    """
    Randomly flip the images and change the measurement by multiplying the steering angle by negative 1.
    """
    
    if random.randrange(2) == 1: 
        return cv2.flip(image,1), -1 * steering
    else: 
        return image,steering


# In[4]:

def resize(image):
    """ resize the image to 64 * 64 to improve the training perfomance. """
    img = cv2.resize(image,(64,64),interpolation = cv2.INTER_AREA)
    return img


# In[5]:

def generator(data, batch_size): 
    
    while 1:
        
        #Creating empty list to store images and steering angle
        bat_cnt = 0 
        X = []
        Y = []
        
        for line in data.iterrows():
            #randomly select either right or center or left image from the data frame.
            
            rnd = np.random.randint(0, 3)
            if rnd == 0 : 
                image = read_img(DIR + line[1].left.strip() )
                angle = line[1].steering +  0.25 
                
            elif rnd == 1:
                image = read_img(DIR + line[1].center.strip() )
                angle = line[1].steering  
                                
            else:
                image = read_img (DIR + line[1].right.strip() )
                angle = line[1].steering   - 0.25  
                
            #Crop the image height by (65,140)
            image = image[65:140,:,:]
            
            #Randomly flip the image to generalized the data.
            image, angle = image_flip(image, angle)
            
            #resize the image
            image = resize(image)
            
            #Append to list
            X.append(image)
            Y.append(angle)
            
            bat_cnt +=1

            if bat_cnt == batch_size: 
                #Since Keras needs inputs the to be array format , convert the list to numpy array.
                X_data = np.array(X)
                Y_data = np.array(Y) 
                yield (X_data, Y_data)
                
                #Reset the counter value 
                bat_cnt = 0
                X = []
                Y = []
            


# In[6]:

#Import Keras modules 

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, Dense,Lambda,Activation
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K


# In[7]:


def get_model():
    
    model = Sequential()

    # Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

    # starts with five convolutional and maxpooling layers
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))

    model.add(Dense(1))

    # model.summary()

    model.compile(optimizer=Adam(1e-4), loss="mse", )

    
    return model

model =  get_model()


# In[8]:

"""Define Callback to view on internal states and statistics of the model during training """

early_stopping = EarlyStopping (monitor= 'val_loss',patience = 3 , verbose = 1 , mode='min')
save_weights = ModelCheckpoint('model.h5',monitor ='val_loss',save_best_only = True)


# In[9]:

#Train and Evaluate the model
model.fit_generator(generator(X_train,128),
                    samples_per_epoch = 20096, nb_epoch =30,
                    validation_data = generator(X_Validation,64),nb_val_samples=6400,
                    callbacks = [early_stopping,save_weights]
                   )


# In[ ]:



