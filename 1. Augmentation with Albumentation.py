import os
import cv2
import numpy as np
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate,
    RandomBrightnessContrast, Blur, CoarseDropout, HueSaturationValue, Compose
)

# Folder input/output
input_output_dir = 'E:/S2 SANDRA ULFA/Thesis/program baru/trains-20241015T163246Z-001/Data Augmentasi/Good'  # Ganti dengan folder yang diinginkan

# Target total gambar
target_total_images = 500

# Definisikan transformasi augmentasi
transform = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    Blur(blur_limit=3, p=0.2),
    CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3)
])

# Dapatkan jumlah gambar asli di folder
original_images = [f for f in os.listdir(input_output_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
num_original_images = len(original_images)

# Tentukan jumlah augmentasi per gambar untuk mencapai target
augment_per_image = (target_total_images - num_original_images) // num_original_images

# Fungsi untuk melakukan augmentasi dan menyimpan gambar
def augment_and_save_in_place(image_path, augment_count):
    # Baca gambar
    image = cv2.imread(image_path)
    filename = os.path.basename(image_path).split('.')[0]
    file_extension = os.path.splitext(image_path)[1]

    # Loop untuk menghasilkan beberapa gambar augmented
    for i in range(augment_count):
        # Terapkan transformasi augmentasi
        augmented = transform(image=image)['image']
        
        # Simpan hasil augmentasi dalam folder yang sama
        augmented_filename = os.path.join(input_output_dir, f"{filename}_aug_{i}{file_extension}")
        cv2.imwrite(augmented_filename, augmented)

# Loop melalui semua gambar asli di folder input/output
for image_file in original_images:
    image_path = os.path.join(input_output_dir, image_file)
    augment_and_save_in_place(image_path, augment_count=augment_per_image)

print(f"Augmentasi selesai! Total gambar sekarang: {num_original_images + num_original_images * augment_per_image}")
