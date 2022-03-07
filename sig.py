from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt


imag_shape = (100, 100, 3)     # 3-> r,g,b
nb_epoch = 50               # it reduces error, more value will make overfit
learning_rate = 1.0e-4



# modelling
model = Sequential()      # sequential neural network, function from line 2
# 1st Layer
model.add(Conv2D(filters=24, kernel_size=3, activation='relu', input_shape=imag_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=36, kernel_size=3, activation='relu', input_shape=imag_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=48, kernel_size=3, activation='relu', input_shape=imag_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))   #
model.add(Dense(32, activation='relu'))

#Readout Layer
model.add(Dense(2, activation='sigmoid')) # two class here created



# saving only best result in all epoch run

checkpoint = ModelCheckpoint('model-{epoch:03d}-{val_loss:03f}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only='true',
                             mode='auto')                  # It saves data   

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate))        // adam optimizer



# making simple change in picture to make more variation in data

datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")    # training input augmnetationj
datagen_v = ImageDataGenerator()       #  testing input from class-0, class-1



# it is training, earlier pre-processing

history = model.fit_generator(datagen.flow_from_directory(directory="./Train/",target_size=(100,100),color_mode='rgb',batch_size=15,
														  class_mode="categorical",shuffle=True,seed=42), steps_per_epoch=15,
							  epochs=nb_epoch, validation_data=datagen_v.flow_from_directory(directory="./Test/",
							target_size=(100,100),color_mode='rgb',batch_size=15,class_mode="categorical",shuffle=True,seed=42),
							  validation_steps= 3,callbacks=[checkpoint],verbose=1)	  
							  
							  
							  
							  
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.title('Model loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('model loss.png')
