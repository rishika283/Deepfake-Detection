import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_auc_score
import tensorflowjs as tfjs
import numpy as np
import os

base_dir = "rvf10k/rvf10k"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)
classes = np.unique(train_gen.classes)
class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights_values))
print(f"Class weights: {class_weights}")
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
optimizer = optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
#training custom head
EPOCHS_HEAD = 10
history_head = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler]
)
#fine tuning
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
optimizer_fine = optimizers.Adam(learning_rate=1e-6)
model.compile(optimizer=optimizer_fine, loss='binary_crossentropy', metrics=['accuracy'])
EPOCHS_FINE = 5
history_fine = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler]
)
val_gen.reset()
val_pred_probs = model.predict(val_gen)
val_pred = (val_pred_probs > 0.5).astype("int32")
val_true = val_gen.classes
final_acc = accuracy_score(val_true, val_pred)
final_prec = precision_score(val_true, val_pred)
final_rec = recall_score(val_true, val_pred)
final_f1 = f1_score(val_true, val_pred)
conf_matrix = confusion_matrix(val_true, val_pred)
TN, FP, FN, TP = conf_matrix.ravel()
specificity = TN / (TN + FP)
roc_auc = roc_auc_score(val_true, val_pred_probs)

print(f"Accuracy   : {final_acc:.4f}")
print(f"Precision  : {final_prec:.4f}")
print(f"Recall     : {final_rec:.4f}")
print(f"F1 Score   : {final_f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"ROC-AUC    : {roc_auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("========================================\n")
os.makedirs("tfjs_model", exist_ok=True)
if not model.built:
    model.build(input_shape=(None, 224, 224, 3))
model.predict(np.zeros((1, 224, 224, 3)))
tfjs.converters.save_keras_model(model, "tfjs_model")
print("Model saved")
convert.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import os
MODEL_PATH = r"C:\Users\Bhavana\Downloads\deepfake_mobilenetv2_safe_finetune.keras"
OUTPUT_PATH = r"C:\Users\Bhavana\Downloads\deepfake_mobilenetv2_repaired.h5"
print("Attempting to load model in safe mode OFF...")
try:
    model = load_model(MODEL_PATH, safe_mode=False)
    print("Model loaded successfully (no errors).")
except Exception as e:
    print(f"Load failed due to: {e}")
    print("Rebuilding the architecture manually using MobileNetV2 backbone...")
    base = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    try:
        model.load_weights(MODEL_PATH, skip_mismatch=True)
        print("Model weights loaded successfully.")
    except Exception as e2:
        print(f"Warning: Could not load some weights ({e2}).")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
model.save(OUTPUT_PATH)
print(f"Model successfully rebuilt and saved as: {OUTPUT_PATH}")
app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
@st.cache_resource
def load_model():
    MODEL_PATH = r"C:\Users\Bhavana\Downloads\deepfake_mobilenetv2_repaired.h5"
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model
model = load_model()
def preprocess_image(image: Image.Image):
    IMG_SIZE = 224
    image = image.convert("RGB")
    image_np = np.array(image)
    resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)
    return image_np, input_tensor
def compute_clarity(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance_of_laplacian < 100:
        clarity_score = 0.4
    elif variance_of_laplacian < 300:
        clarity_score = 0.7
    else:
        clarity_score = 1.0
    return variance_of_laplacian, clarity_score
def predict_image(model, input_tensor, clarity_score, variance_of_laplacian):
    pred = model.predict(input_tensor, verbose=0)[0][0]
    threshold = 0.5
    if variance_of_laplacian < 100:
        label = "Real"
        confidence = (1 - pred) * clarity_score + 0.2
    else:
        if pred > threshold:
            label = "Deepfake"
            confidence = pred * clarity_score
        else:
            label = "Real"
            confidence = (1 - pred) * clarity_score
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return label, confidence, pred
st.set_page_config(page_title="Deepfake Detection App", layout="centered")
st.title("Deepfake Detection using MobileNetV2")
st.markdown("Upload an image to check if it’s **Real** or **Deepfake**.")

uploaded_file = st.file_uploader("📂 Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image_np, input_tensor = preprocess_image(image)
    variance_of_laplacian, clarity_score= compute_clarity(image_np)
    label, confidence, raw_pred = predict_image(model, input_tensor, clarity_score, variance_of_laplacian)
    st.markdown("---")
    st.subheader("Prediction Result")
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence:** {confidence:.2f}")
    st.progress(confidence)
        st.markdown("---")
else:
    st.info("Please upload an image to begin analysis.")

