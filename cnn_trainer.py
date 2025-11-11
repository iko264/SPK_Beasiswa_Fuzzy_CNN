import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_TYPE = 'sertifikat' 

if MODEL_TYPE == 'sertifikat':
    TRAIN_DIR = 'data/cnn_sertifikat_train'
    MODEL_SAVE_PATH = 'models/model_cnn_sertifikat.h5'
    NUM_CLASSES = 3 
else:
    TRAIN_DIR = 'data/cnn_rumah_train'
    MODEL_SAVE_PATH = 'models/model_cnn_rumah.h5'
    NUM_CLASSES = 3 

IMAGE_WIDTH, IMAGE_HEIGHT = 150, 150
BATCH_SIZE = 16
EPOCHS = 20

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    subset='training' 
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"Mapping Kelas untuk '{MODEL_TYPE}': {train_generator.class_indices}")


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print(f"\nMemulai Pelatihan Model: {MODEL_TYPE}...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

model.save(MODEL_SAVE_PATH)
print(f"\nModel berhasil disimpan di: {MODEL_SAVE_PATH}")