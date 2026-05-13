import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
#saving model
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# Pad space before resizing
def pad_to_space(image, pad_color=(255, 255, 255)):
    # Find shape
    h, w = image.shape[:2]

    # Initialize padding
    top = 0
    bottom = 0
    left = 0
    right = 0

    #Ideal length and height
    length =  4 * h
    height = w // 4 

    # Update
    if (length > w):
        left = (length - w) // 2 
        right = length - w - left
    
    if (height > h):
        top = (height - h) // 2
        bottom = height - h - top

    # Apply padding
    new_image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return new_image

# Print image from np array
def print_image(image):
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Read in CSV file
df = pd.read_csv("train_test_data_with_stop_words_new.csv")

# Define image directory
image_dir = "Dataset"

# Initialize empty arrays for images and labels
train_images = []
test_images = []
train_labels = []
test_labels = []

max_train = 0
max_test = 0
# Load and preprocess images
for index, row in df.iterrows():

    image_filename = row["FullFilePaths"]

    image_path = os.path.join("Dataset/", image_filename)
    print(image_path)
    # Read in image from file path in greyscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Add padding to sides to follow 1:4 height:width ratio
    image = pad_to_space(image)
  

    # resize to fit shape - w, h
    image = cv2.resize(image, (128, 32))
    # print_image(image)

    image = np.array(image) / 255.0  

    if row["Train"] == 0:
        train_images.append(image)
        train_labels.append(row['Label_Encoded'])
        #if len(row['Label']) > max_train :
        #    max_train = len(row['Label'])
    else:
        test_images.append(image)
        test_labels.append(row['Label_Encoded'])
        #if len(row['Label']) > max_test :
        #    max_test = len(row['Label'])
    

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#train_labels[1] = max_train
#test_labels[1] = max_test

# Validate np - h, w
train_images = train_images.reshape((len(train_images), 32, 128, 1)).astype('float32') / 255
test_images = test_images.reshape((len(test_images), 32, 128, 1)).astype('float32') / 255


# Print shapes to verify
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)


# Model modeled after the given architecture in the document
model = models.Sequential([ 
    #input
    layers.Input(shape=(128, 32, 1)),

    # CNN
    layers.Conv2D(64, (3, 3), activation='relu', padding = "same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding = "same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding = "same"),
    layers.Conv2D(256, (3, 3), activation='relu', padding = "same"),
    layers.MaxPooling2D((2, 1)),

    # Added Layers
    layers.Conv2D(512, (3, 3), activation='relu', padding = "same"),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), activation='relu', padding = "same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 1)),

    layers.Conv2D(512, (3,2), activation='relu', padding = "same"),

    layers.Lambda(lambda t: tf.squeeze(t, axis=1)),

    # BiLSTM
    layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(256, return_sequences=True)),

    layers.Dense(26, activation='softmax'),
])

model.compile(optimizer='adam', loss='CTC', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=50, batch_size=64)
#model.summary()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Model Accuracy: {test_acc*100:.2f}%")

# Save the trained Python Keras model
model_path = 'model_crnn.h5'
model.save(model_path)
print(f"Model saved to {model_path}")
