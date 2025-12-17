import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model # Untuk diagram arsitektur
from sklearn.utils.class_weight import compute_class_weight

# --- KONFIGURASI ---
DATASET_PATH = "data/cnn_rumah_train"    
OUTPUT_MODEL_DIR = "models"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, "model_rumah_4grafik.h5")
OUTPUT_IMAGES_DIR = "gambar_laporan" 

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

IMAGE_SIZE = (128, 128)    
BATCH_SIZE = 16
EPOCHS = 25 
VALIDATION_SPLIT = 0.2
SEED = 42

# --- 1. PERSIAPAN DATA ---
print("Menyiapkan Data...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# Hitung Class Weight
classes = list(train_gen.classes)
class_weights_vals = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights = {i: w for i, w in enumerate(class_weights_vals)}

# --- 2. BANGUN MODEL ---
print("Membangun Model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'   
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 3. SIMPAN DIAGRAM ARSITEKTUR  ---
try:
    print("Menyimpan Diagram Arsitektur...")
    plot_path = os.path.join(OUTPUT_IMAGES_DIR, "arsitektur_cnn.png")
    plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=False)
    print(f"[OK] Diagram arsitektur tersimpan di {plot_path}")
except Exception as e:
    print(f"[SKIP] Gagal membuat diagram arsitektur (perlu graphviz): {e}")

# --- 4. TRAINING ---
print("Mulai Training Ulang...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ModelCheckpoint(OUTPUT_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks
)

# Fine Tuning (Opsional, agar grafik lebih panjang/bagus)
print("Mulai Fine Tuning...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks
)

# --- 5. PLOT & SIMPAN GRAFIK TRAINING ---
print("Menyimpan Grafik Training...")

# Gabungkan history awal dan fine tuning
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 5))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

grafik_path = os.path.join(OUTPUT_IMAGES_DIR, "grafik_training_cnn.png")
plt.savefig(grafik_path, dpi=300)
print(f"[OK] Grafik training tersimpan di {grafik_path}")

print("\nSELESAI. Silakan cek folder 'gambar_laporan'")