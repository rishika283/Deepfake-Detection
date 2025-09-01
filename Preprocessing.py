import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN
from tqdm import tqdm

IMG_SIZE = (224, 224)
real_path = "training_real"
fake_path = "training_fake"

detector = MTCNN()

def load_and_crop_faces(folder, label):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(img_rgb)

            if len(detections) == 0:
                continue

            # Get first face bounding box
            x, y, w, h = detections[0]['box']
            x, y = max(0, x), max(0, y)
            face = img_rgb[y:y+h, x:x+w]

            # Resize & normalize
            face_resized = cv2.resize(face, IMG_SIZE)
            face_normalized = face_resized.astype('float32') / 255.0

            images.append(face_normalized)
            labels.append(label)

    return images, labels

real_imgs, real_labels = load_and_crop_faces(real_path, 1)
fake_imgs, fake_labels = load_and_crop_faces(fake_path, 0)

images = np.array(real_imgs + fake_imgs)
labels = np.array(real_labels + fake_labels)

print(f"Total faces loaded: {len(images)}")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator()  # No augmentation for validation

train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)

print("Preprocessing and augmentation complete.")
