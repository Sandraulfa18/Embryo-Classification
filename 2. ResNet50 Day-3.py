import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path ke dataset
train_dir = "/content/drive/MyDrive/11. New Data set/Dataset Hari ke-3/Train"
val_dir = "/content/drive/MyDrive/11. New Data set/Dataset Hari ke-3/Validation"
test_dir = "/content/drive/MyDrive/11. New Data set/Dataset Hari ke-3/Test"

# Ukuran gambar dan input
image_size = (224, 224)
input_shape = (224, 224, 3)

# Augmentasi data untuk training
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

# Preprocessing data validasi dan pengujian
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data generator
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

# Model arsitektur menggunakan ResNet50
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

# Membekukan sebagian layer untuk fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Menambahkan layer kustom
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint_callback = ModelCheckpoint(
    '/content/drive/MyDrive/11. New Data set/Dataset Hari ke-3/rev12resnet50_model.keras',
    save_best_only=True
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.0001,
    restore_best_weights=True,
    verbose=1
)

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Evaluasi pada dataset pengujian
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Fungsi smoothing untuk hasil plot
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# Smoothing metrics
smoothed_training_accuracy = smooth_curve(history.history['accuracy'])
smoothed_val_accuracy = smooth_curve(history.history['val_accuracy'])
smoothed_training_loss = smooth_curve(history.history['loss'])
smoothed_val_loss = smooth_curve(history.history['val_loss'])

# Plot hasil pelatihan
plt.figure(figsize=(6, 4))
plt.plot(smoothed_training_accuracy, label='Training Accuracy', color='blue')
plt.plot(smoothed_val_accuracy, label='Validation Accuracy', color='cyan')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(smoothed_training_loss, label='Training Loss', color='orange')
plt.plot(smoothed_val_loss, label='Validation Loss', color='red')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Simpan metrik ke file Excel
data = {
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy'],
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
}

df = pd.DataFrame(data)
df.to_excel('/content/drive/MyDrive/11. New Data set/Dataset Hari ke-3/Rev12resnet50_metrics.xlsx', index=False)

# Print rata-rata akurasi dan loss
print(f"Rata-rata Training Accuracy: {np.mean(history.history['accuracy']):.4f}")
print(f"Rata-rata Training Loss: {np.mean(history.history['loss']):.4f}")
