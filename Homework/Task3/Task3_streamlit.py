import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from keras import optimizers
from keras import layers
import os
import time
import requests
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.request

headers = {}
#response = requests.get(f'https://api.github.com/repos/kieran31415/AI/main/Homework/Task3/data', headers=headers)
map = 'https://raw.githubusercontent.com/kieran31415/AI/main/Homework/Task3/data'
# Define a Streamlit app
def main():
    st.title("Vehicle Classifier EDA and Training Controls")

    # EDA Section
    st.header("Exploratory Data Analysis (EDA)")
    
    # Display some sample images
    st.subheader("Sample Images")
    vehicles = ['Boat', 'Bus', 'Car', 'Motorbike', 'Plane']
    image_data = [
    {"path": map + "testing_set/boat/boat.13.jpg", "title": "Boat"},
    {"path": map + "testing_set/bus/bus.2.jpg", "title": "Bus"},
    {"path": map + "testing_set/car/car.2.jpg", "title": "Car"},
    {"path": map + "testing_set/motorbike/motorbike.0.jpg", "title": "Motorbike"},
    {"path": map + "testing_set/plane/plane.26.jpg", "title": "Plane"}
    ]

# Create a Streamlit layout
    st.title("Image Gallery")

# Loop through the image data and display each image with a title
    for image_info in image_data:
        st.subheader(image_info["title"])
        st.image(image_info["path"])

    # Display the figure using Streamlit
    st.pyplot(fig)

    # Training Controls Section
    st.header("Training Controls")

    # Slider for the number of training epochs
    num_epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=20)

    # Checkbox for regularization options
    use_regularization = st.checkbox("Use Regularization")

    if use_regularization:
        # You can add more regularization options here
        st.write("Regularization options enabled")

    # Button to start training
    if st.button("Start Training"):
        st.text(f"Training model for {num_epochs} epochs...")

        # Include your training code here, updating the model with the selected options
        train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_val_datagen.flow_from_directory('data/training_set',
                                                 subset='training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical') 

        validation_set = train_val_datagen.flow_from_directory('data/training_set',
                                                 subset='validation',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

        test_set = test_datagen.flow_from_directory('data/testing_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
        # For example, update the `num_epochs` and `use_regularization` in your training code
        NUM_CLASSES = 5

# Create a sequential model with a list of layers
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="sigmoid"),
            layers.Dense(NUM_CLASSES, activation="softmax")
            ])

# Compile and train your model as usual
        model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

        print(model.summary())
        # Once training is done, you can display the loss and accuracy plots as you did before.
        history = model.fit(training_set,
                validation_data = validation_set,
                epochs = 20
                )
        # Display the loss and accuracy plots (similar to your code)
        st.subheader("Training Progress")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the loss curves on the first subplot
        ax1.plot(history.history['loss'], label='training loss')
        ax1.plot(history.history['val_loss'], label='validation loss')
        ax1.set_title('Loss curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot the accuracy curves on the second subplot
        ax2.plot(history.history['accuracy'], label='training accuracy')
        ax2.plot(history.history['val_accuracy'], label='validation accuracy')
        ax2.set_title('Accuracy curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Display the figure using Streamlit
        st.pyplot(fig)

if __name__ == "__main__":
    main()