import cv2
import numpy as np
import os
import pathlib
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
   
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    
    file_name = data_dir

    imgData = os.path.join('/','Users', 'pushpitbhardwaj','Downloads','CS50_AI','traffic',file_name,'*')

    folders = glob.glob(imgData)
    imgList = []
    images = []
    labels = []
    for folder in folders:
        #print(folder)
        counter = 0
        for f in glob.glob(folder+'/*'):
            f = (cv2.imread(f))

            f = cv2.resize(f, dsize=(30, 30), interpolation=cv2.INTER_CUBIC)
            images.append(f)
            
            counter += 1

        labels.append(counter)
    
    return (images, labels)
    
    images = []
    labels = []

    filepath = os.path.dirname(os.path.abspath(data_dir))

    for i in range(NUM_CATEGORIES):
        os.chdir(os.path.join(filepath,data_dir,str(i)))
        for image in os.listdir(os.getcwd()):
            image = (cv2.imread(image))
            
            image_new = cv2.resize(image, dsize=(IMG_HEIGHT, IMG_WIDTH))
            images.append(image_new)
            labels.append(str(i))
        os.chdir(os.path.join(filepath, data_dir))

    return (images, labels)

    """
    os.chdir(data_dir)

    images = []
    labels = []
    parent_path = os.getcwd()
    print(os.getcwd())
    for dir_name in os.listdir('.'):
        if dir_name != ".DS_Store":
            path = os.path.join(parent_path, dir_name)
            os.chdir(path)

            print(f"loading data from {dir_name}")
            for image in os.listdir('.'):
                im = cv2.imread(image)
                im_resized = cv2.resize(im,(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA).astype('float32')
                images.append(im_resized)
                labels.append(int(dir_name))
    return (images, labels)
    


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.ZeroPadding2D(padding=1),
            tf.keras.layers.Conv2D(
                64, (3,3), activation="relu", input_shape=(IMG_WIDTH,IMG_HEIGHT,3)
            ),

            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(
                128, (3,3), activation="relu"
            ),

            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(
                256, (3,3), activation="relu"
            ),

            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    main()
