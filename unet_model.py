
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from keras.metrics import MeanIoU

kernel_initializer =  'he_uniform'  # also try 'he_normal' but model not converging... 
hidden_activation = 'relu'
padding = 'same'

################################################################
def UNet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=padding)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding=padding)(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding=padding)(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding=padding)(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation=hidden_activation, kernel_initializer=kernel_initializer, padding=padding)(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model