from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv3D, BatchNormalization, Flatten, Dropout, Dense

def createModel():
    input_shape=(1, 22, 59, 114)
    model = Sequential()
    #C1
    model.add(Conv3D(16, (22, 5, 5), strides=(1, 2, 2), padding='valid',activation='relu',data_format= "channels_first", input_shape=input_shape))
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2),data_format= "channels_first",  padding='same'))
    model.add(BatchNormalization())
    
    #C2
    model.add(Conv3D(32, (1, 3, 3), strides=(1, 1,1), padding='valid',data_format= "channels_first",  activation='relu'))#incertezza se togliere padding
    model.add(keras.layers.MaxPooling3D(pool_size=(1,2, 2),data_format= "channels_first", ))
    model.add(BatchNormalization())
    
    #C3
    model.add(Conv3D(64, (1,3, 3), strides=(1, 1,1), padding='valid',data_format= "channels_first",  activation='relu'))#incertezza se togliere padding
    model.add(keras.layers.MaxPooling3D(pool_size=(1,2, 2),data_format= "channels_first", ))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    opt_adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])
    
    return model