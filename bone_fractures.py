import tensorflow as tf
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# Define dataset paths
train_dir = "dataset/train"
val_dir = "dataset/validation"

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

# Load validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = train_dataset.class_names
print("Class names:", class_names)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to datasets
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Check a batch of images
for images, labels in train_dataset.take(1):
    print("Batch shape:", images.shape)
    print("Labels:", labels.numpy())
    

# Define Data Augmentation Layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # Flip images randomly
    layers.RandomRotation(0.2),       # Rotate up to 20%
    layers.RandomZoom(0.2),           # Randomly zoom in
])

# Apply data augmentation to training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Set training parameters
epochs = 10  
batch_size = 32  

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)
