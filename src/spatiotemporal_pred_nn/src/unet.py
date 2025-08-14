import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Dropout, Dense, Reshape, concatenate
from tensorflow.keras.models import Model
from data_generator import DataGenerator
import os

def UNet(input_size =(100,100,1), num_params=3):
    
    """
    Build UNet model with additional scalar parameters input
    
    Args:
        input_size: Tuple of (height, width, channels)
        num_params: Number of scalar parameters (velocity, time, etc.)
        
    Returns:
        Keras model with the UNet architecture
    """
    # Grid input
    inputs = Input(input_size)
    
    # Scalar parameters input
    param_inputs = Input((num_params,))
    
    # Encoder path
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottom
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.3)(conv5)
    
    # Process scalar parameters and inject them
    # The feature map size at the bottom is (6, 6, 1024) for 100x100 input
    param_dense = Dense(128, activation='relu')(param_inputs)
    param_dense = Dense(256, activation='relu')(param_dense)
    param_features = Dense(6*6*64, activation='relu')(param_dense)
    param_features = Reshape((6, 6, 64))(param_features)
    
    # Concatenate parameters with the bottleneck features
    merged_features = concatenate([drop5, param_features], axis=3)
    
    # Decoder path
    # up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(merged_features))
    up6 = Conv2DTranspose(256, 2, strides=(2, 2), activation='relu', padding='same')(merged_features)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    # up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    up7 = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same')(conv6)
    up7_pad = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(up7)  # Adjust padding for concatenation
    merge7 = concatenate([conv3, up7_pad], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    # up8 = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same')(conv7)
    up8 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    # up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    up9 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='tanh')(conv9)  # tanh for [-1, 1] output range

    # Create and compile model
    model = Model(inputs=[inputs, param_inputs], outputs=outputs)
    
    return model


    
