import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import plot_model

input_shape = (128, 128, 3)
inputs = Input(shape=input_shape, name="Input_Citra")

x = layers.Lambda(lambda x: x, name="MobileNetV2_Feature_Extractor")(inputs)

x = layers.GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)
x = layers.Dropout(0.4, name="Dropout_1")(x)
x = layers.Dense(128, activation='relu', name="Dense_ReLU_128")(x)
x = layers.Dropout(0.3, name="Dropout_2")(x)
outputs = layers.Dense(3, activation='softmax', name="Output_Layer_3_Class")(x)

model_simple = models.Model(inputs=inputs, outputs=outputs, name="Arsitektur_CNN_Sederhana")

plot_model(
    model_simple,
    to_file="arsitektur_cnn_ringkas.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  
    dpi=300
)

print("Gambar arsitektur ringkas berhasil dibuat: arsitektur_cnn_ringkas.png")