import streamlit as st
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from keras import optimizers
from keras import layers
import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np

# Define a Streamlit app
def main():
    st.title("Vehicle Classifier EDA and Training Controls")

    # EDA Section
    st.header("Exploratory Data Analysis (EDA)")
    
    # Display some sample images
    st.subheader("Sample Images")
    vehicles = ['Boat', 'Bus', 'Car', 'Motorbike', 'Plane']
    
    # Define a map to your GitHub repository
    map = 'https://raw.githubusercontent.com/kieran31415/AI/main/Homework/Task3/data'

    image_data = [
        {"path": f"{map}/testing_set/boat/boat.13.jpg", "title": "Boat"},
        {"path": f"{map}/testing_set/bus/bus.2.jpg", "title": "Bus"},
        {"path": f"{map}/testing_set/car/car.2.jpg", "title": "Car"},
        {"path": f"{map}/testing_set/motorbike/motorbike.0.jpg", "title": "Motorbike"},
        {"path": f"{map}/testing_set/plane/plane.26.jpg", "title": "Plane"}
    ]

    # Display images using Streamlit
    st.title("Image Gallery")

    for image_info in image_data:
        st.subheader(image_info["title"])
        st.image(image_info["path"])

    # Training Controls Section
    st.header("Training Controls")

    # Slider for the number of training epochs
    num_epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=20)

    # Button to start training
    if st.button("Start Training"):
        st.text(f"Training model for {num_epochs} epochs...")

        # Include your training code here, updating the model with the selected options
        base_url = "https://github.com/kieran31415/AI/tree/main/Homework/Task3/data/"

    # Create an ImageDataGenerator
        train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                           rescale=1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

    # Create a list of categories
        categories = ['boat', 'bus', 'car', 'motorbike', 'plane']

    # Create the training and validation sets
        training_images = []
        validation_images = []

        for category in categories:
            category_url = base_url + f"training_set/{category}/"
            response = requests.get(category_url)

            if response.status_code == 200:
                category_images = [category_url + img for img in response.text.splitlines()]

            # Split the category images into training and validation sets
                split_index = int(0.8 * len(category_images))
                training_images.extend(category_images[:split_index])
                validation_images.extend(category_images[split_index:])
            else:
                st.write(f"Failed to fetch images from the '{category}' category.")

    # Create the training and validation sets using the fetched images
        train_datagen = ImageDataGenerator(rescale=1./255)
        validation_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_dataframe(
            dataframe=None,  # You can provide a dataframe if available, or use None
            x=np.array([]),  # You can provide image data as a NumPy array or an empty array
            y=None,  # You can provide labels if available, or use None
            directory=None,  # No need to specify a directory when using x as NumPy array
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical',
            subset='training',
        )

        validation_set = validation_datagen.flow_from_dataframe(
            dataframe=None,  # You can provide a dataframe if available, or use None
            x=np.array([]),  # You can provide image data as a NumPy array or an empty array
            y=None,  # You can provide labels if available, or use None
            directory=None,  # No need to specify a directory when using x as NumPy array
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical',
            subset='validation',
        )


        # For example, update the `num_epochs` and `use_regularization` in your training code
        NUM_CLASSES = 5

        # Create a sequential model with a list of layers
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"),
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
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

        print(model.summary())

        # Once training is done, you can display the loss and accuracy plots as you did before.
        history = model.fit(training_set,
                            validation_data=validation_set,
                            epochs=num_epochs
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