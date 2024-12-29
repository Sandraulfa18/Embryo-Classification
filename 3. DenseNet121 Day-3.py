import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
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

# Image size and input shape
image_size = (224, 224)
input_shape = (224, 224, 3)

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
    shuffle=False  # Shuffle off for evaluation
)

# Define the model
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)

# Fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = False

# Add custom layers on top
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

# Callbacks
checkpoint_callback = ModelCheckpoint('/content/drive/MyDrive/Dataset Hari ke-3/densenet121_model.keras', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[checkpoint_callback]
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Smooth function
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# Smooth metrics
smoothed_training_accuracy = smooth_curve(history.history['accuracy'])
smoothed_val_accuracy = smooth_curve(history.history['val_accuracy'])
smoothed_training_loss = smooth_curve(history.history['loss'])
smoothed_val_loss = smooth_curve(history.history['val_loss'])


# Plot Training Accuracy
plt.figure(figsize=(6, 4))
plt.plot(smoothed_training_accuracy, label='Training Accuracy', color='blue')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(6, 4))
plt.plot(smoothed_val_accuracy, label='Validation Accuracy', color='cyan')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

# Plot Training Loss
plt.figure(figsize=(6, 4))
plt.plot(smoothed_training_loss, label='Training Loss', color='orange')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, max(smoothed_training_loss) + 0.1)
plt.legend()
plt.grid(True)
plt.show()

# Plot Validation Loss
plt.figure(figsize=(6, 4))
plt.plot(smoothed_val_loss, label='Validation Loss', color='red')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, max(smoothed_val_loss) + 0.1)
plt.legend()
plt.grid(True)
plt.show()

# Save metrics to Excel
data = {
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy'],
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
}

df = pd.DataFrame(data)
df.to_excel('/path/to/save/densenet121_metrics.xlsx', index=False)

# Print average training accuracy and loss
print(f"Average Training Accuracy: {np.mean(history.history['accuracy']):.4f}")
print(f"Average Training Loss: {np.mean(history.history['loss']):.4f}")
