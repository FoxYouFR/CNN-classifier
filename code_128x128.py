# Partie 1 - Construction du CNN

# Importation des modules
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

# Initialisation
classifier = Sequential()

# Convolution - couche 1
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), strides=1, input_shape=(256,256,3), activation='relu'))

# Pooling - couche 1
classifier.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Convolution - couche 2
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), strides=1, activation='relu'))

# Pooling - couche 2
classifier.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Convolution - couche 3
classifier.add(Convolution2D(filters=64, kernel_size=(3,3), strides=1, activation='relu'))

# Pooling - couche 3
classifier.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Convolution - couche 4
classifier.add(Convolution2D(filters=64, kernel_size=(3,3), strides=1, activation='relu'))

# Pooling - couche 4
classifier.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Flattening
classifier.add(Flatten())

# Couche complètement connecté
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilation
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Augmentation du dataset et entraînement sur les images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset\\training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset\\test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=100,
        validation_data=test_set,
        validation_steps=64)

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset\\single_prediction\\cat_or_dog_1.jpg', target_size=(256, 256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'chien'
else:
    prediction = 'chat'
print(prediction)