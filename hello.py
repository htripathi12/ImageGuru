# example of loading the cifar10 dataset
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


# LOAD DATASET AND TEST

(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()
# # plt.imshow(training_images[2])
# # plt.show()


# print(training_images.shape)
# print(training_labels.shape, "\n")

# print(testing_images.shape)
# print(testing_labels.shape)

# fig = plt.figure(figsize=(12, 8))
# columns = 5
# rows = 3

# for i in range(1, columns*rows + 1):
#       img = training_images[i]
#       fig.add_subplot(rows, columns, i) # create subplot (row index, col index, which number of plot)
#       plt.title("Label:" + str(training_labels[i])) # plot the image, along with its label
#       plt.imshow(img, cmap=plt.cm.binary)
# plt.show()

# X_train, X_valid, y_train, y_valid = train_test_split(training_images, training_labels, random_state = 0, test_size = 0.2)

# MODEL BELOW

# LOAD DATASET AND TEST
(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(training_images, training_labels, random_state=0, test_size=0.2)

# MODEL BELOW
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),  # hidden layer
    tf.keras.layers.Dense(80, activation='relu'),  # hidden layer
    tf.keras.layers.Dense(70, activation='relu'),
    tf.keras.layers.Dense(70, activation='relu'),  # hidden layer
    tf.keras.layers.Dense(70, activation='relu'),
    tf.keras.layers.Dense(70, activation='relu'),
    tf.keras.layers.Dense(70, activation='relu'),  # hidden layer
    tf.keras.layers.Dense(10, activation='softmax')  # output layer
])

optimizer = tf.keras.optimizers.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-05,
    name='Nadam',
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fit the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), verbose=1)

# Plotting accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.xlim([0, 30])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

# Confusion Matrix
predictions = model.predict(testing_images)
predictions_for_cm = predictions.argmax(1)
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
cm = confusion_matrix(testing_labels, predictions_for_cm)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)

# Classification Report
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predictions = model.predict(testing_images)
predictions = np.argmax(predictions, axis=1)
print(classification_report(testing_labels, predictions, target_names=class_names))
