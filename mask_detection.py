import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Load and preprocess the dataset
# Please make sure to replace the paths with your dataset paths
with_mask_path = 'C:/Users/DELL/OneDrive/Desktop/data/with_mask'
without_mask_path = 'C:/Users/DELL/OneDrive/Desktop/data/without_mask'

data = []
labels = []

# Load images with mask
for img_file in os.listdir(with_mask_path):
    image_path = os.path.join(with_mask_path, img_file)
    image = Image.open(image_path).resize((128, 128)).convert('RGB')
    data.append(np.array(image))
    labels.append(1)  # With mask label

# Load images without mask
for img_file in os.listdir(without_mask_path):
    image_path = os.path.join(without_mask_path, img_file)
    image = Image.open(image_path).resize((128, 128)).convert('RGB')
    data.append(np.array(image))
    labels.append(0)  # Without mask label

X = np.array(data)
Y = np.array(labels)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train / 255.0, Y_train, validation_split=0.1, epochs=5)

# Evaluate the model
loss, accuracy = model.evaluate(X_test / 255.0, Y_test)
print('Test Accuracy:', accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
model.save('mask_detection_model.h5')
