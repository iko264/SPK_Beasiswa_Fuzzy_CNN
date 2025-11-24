import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Lokasi dataset
DATASET_PATH = "data/cnn_rumah_train/"

# Cek folder
print("Folder dataset:", os.listdir(DATASET_PATH))

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training generator
train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# Validation generator
val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Informasi kelas
print("Class indices:", train_gen.class_indices)
# Contoh output:
# {'kelas_mewah': 0, 'kelas_sederhana': 1, 'kelas_sedang': 2}

# Model CNN
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")  # ada 3 kelas
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Training
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=[es]
)

# Simpan model
model.save("model_rumah.h5")

print("Training selesai. Model disimpan sebagai model_rumah.h5")
