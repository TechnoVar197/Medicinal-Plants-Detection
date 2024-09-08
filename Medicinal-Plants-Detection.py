import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import shutil
import random

# Define paths
dataset_path = 'path_to_your_dataset'
train_path = 'path_to_save_train_dataset'
validation_path = 'path_to_save_validation_dataset'

# Create train and validation directories
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(validation_path):
    os.makedirs(validation_path)

# Split data into train and validation
for plant in os.listdir(dataset_path):
    plant_path = os.path.join(dataset_path, plant)
    if os.path.isdir(plant_path):
        images = os.listdir(plant_path)
        random.shuffle(images)
        train_size = int(0.8 * len(images))
        for i, img in enumerate(images):
            img_path = os.path.join(plant_path, img)
            if i < train_size:
                shutil.copy(img_path, os.path.join(train_path, plant))
            else:
                shutil.copy(img_path, os.path.join(validation_path, plant))

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Save the model
model.save('medicinal_plants_classifier.h5')
