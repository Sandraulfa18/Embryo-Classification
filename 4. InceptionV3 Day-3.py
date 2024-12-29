import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Paths to dataset folders
train_dir = "/content/drive/MyDrive/Dataset Hari ke-3/Train"
val_dir = "/content/drive/MyDrive/Dataset Hari ke-3/Validation"
test_dir = "/content/drive/MyDrive/Dataset Hari ke-3/Test"

# Image size and input shape for InceptionV3
image_size = (299, 299)
input_shape = (299, 299, 3)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=16,
    class_mode='sparse'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=16,
    class_mode='sparse'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=16,
    class_mode='sparse',
    shuffle=False
)

# Define the model
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

# Fine-tuning: Freeze all but the last few layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callback to save the best model
checkpoint_callback = ModelCheckpoint('/content/drive/MyDrive/Dataset Hari ke-3/3inceptionv3_model.h5', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[checkpoint_callback]
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Smooth curve function
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# Smoothed metrics
smoothed_training_accuracy = smooth_curve(history.history['accuracy'])
smoothed_val_accuracy = smooth_curve(history.history['val_accuracy'])
smoothed_training_loss = smooth_curve(history.history['loss'])
smoothed_val_loss = smooth_curve(history.history['val_loss'])

# Plotting function for individual metrics
def plot_single_metric(metric, title, ylabel, color):
    plt.figure(figsize=(6, 4))
    plt.plot(metric, label=title, color=color)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot metrics
plot_single_metric(smoothed_training_accuracy, 'Training Accuracy', 'Accuracy', 'blue')
plot_single_metric(smoothed_val_accuracy, 'Validation Accuracy', 'Accuracy', 'cyan')
plot_single_metric(smoothed_training_loss, 'Training Loss', 'Loss', 'orange')
plot_single_metric(smoothed_val_loss, 'Validation Loss', 'Loss', 'red')

# Save metrics to Excel
data = {
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy'],
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
}
df = pd.DataFrame(data)
df.to_excel('/content/drive/MyDrive/Dataset Hari ke-3/3inceptionv3_metrics.xlsx', index=False)

# Print average training metrics
avg_training_accuracy = np.mean(history.history['accuracy'])
avg_training_loss = np.mean(history.history['loss'])
print(f"Average Training Accuracy: {avg_training_accuracy:.4f}")
print(f"Average Training Loss: {avg_training_loss:.4f}")
