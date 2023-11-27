import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. Data Processing
# Image shape definition
im_size = (100, 100, 3)

# Directories
train_dir = "C:\\Users\\user\\OneDrive\\Documents\\AER850\\Data\\Train"
validation_dir = "C:\\Users\\user\\OneDrive\\Documents\\AER850\\Data\\Validation"
test_dir = "C:\\Users\\user\\OneDrive\\Documents\\AER850\\Data\\Test"
categories = ["Large", "Medium", "Small", "None"]  # Classes of crack images

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2
)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Train and validation generator
train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=im_size[:2],
    batch_size=32,
    class_mode='categorical'
)
validation_set = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=im_size[:2],
    batch_size=32,
    class_mode='categorical'
)

# 2. Neural Network Architecture Design
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=im_size))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# 3. Hyperparameter Analysis
# Addition of dense layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))  # 4 neurons for 4 classes

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Model Training
history = model.fit(
    train_set,
    epochs=4,
    validation_data=validation_set
)

# 5. Model Evaluation
test_loss, test_acc = model.evaluate(validation_set, verbose=2)

# Plotting Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# 6. Model Testing
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=im_size[:2],  # Corrected to im_size
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
