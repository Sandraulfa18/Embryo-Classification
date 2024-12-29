import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pandas as pd

# Path to dataset folders
train_dir = "/content/drive/MyDrive/Dataset Hari ke - 5/Train"
val_dir = "/content/drive/MyDrive/Dataset Hari ke - 5/Validation"
test_dir = "/content/drive/MyDrive/Dataset Hari ke - 5/Test"

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
    shuffle=False
)

# Define CNN Architecture with Added 256 Filter Layer
def custom_cnn_with_256(input_shape):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Block 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    # Global average pooling and dropout
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    # Model definition
    model = Model(inputs, outputs)
    return model

# Build and compile the model
model = custom_cnn_with_256(input_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
)

# Display model summary
model.summary()

# Callbacks
checkpoint_callback = ModelCheckpoint('/content/drive/MyDrive/Dataset Hari ke - 5/106custom_cnn_with_256_model.keras', save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[checkpoint_callback]
)

# Evaluate on test set
test_loss, test_accuracy, test_top_k_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Top-K Accuracy: {test_top_k_accuracy:.4f}")

# Save metrics to Excel
data = {
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy'],
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss'],
    # 'Training Top-K Accuracy': history.history['sparse_top_k_categorical_accuracy'],
    # 'Validation Top-K Accuracy': history.history['val_sparse_top_k_categorical_accuracy']
}

history_df = pd.DataFrame(data)
history_df.to_excel("/content/drive/MyDrive/Dataset Hari ke - 6/106custom_cnn_with_256_training_history.xlsx", index=False)
print("Training history saved to 'custom_cnn_with_256_training_history.xlsx'")

# Calculate average accuracy and loss
average_training_accuracy = sum(history.history['accuracy']) / len(history.history['accuracy'])
average_training_loss = sum(history.history['loss']) / len(history.history['loss'])

print(f"Average Training Accuracy: {average_training_accuracy:.4f}")
print(f"Average Training Loss: {average_training_loss:.4f}")
