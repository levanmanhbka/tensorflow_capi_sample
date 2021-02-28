# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# %%
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_label) = fashion_mnist.load_data()


# %%
train_images.shape


# %%
train_labels.shape


# %%
test_images.shape


# %%
test_label.shape


# %%
train_images = train_images / 255.0
test_images = test_images / 255.0


# %%
def create_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(filters= 4, kernel_size=(3, 3), strides=(1, 1), padding= "same")(inputs)
    x = tf.keras.layers.BatchNormalization(axis= -1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters= 4, kernel_size=(3, 3), strides=(1, 1), padding= "same")(inputs)
    x = tf.keras.layers.BatchNormalization(axis= -1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters= 4, kernel_size=(3, 3), strides=(1, 1), padding= "same")(inputs)
    x = tf.keras.layers.BatchNormalization(axis= -1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units= 128, activation= "relu")(x)
    x = tf.keras.layers.Dropout(rate= 0.5)(x)
    x = tf.keras.layers.Dense(units= 64, activation= "relu")(x)
    x = tf.keras.layers.Dropout(rate= 0.5)(x)
    
    outputs = tf.keras.layers.Dense(units= 10, activation= "softmax")(x)
    model = tf.keras.models.Model(inputs= inputs, outputs= outputs)
    return model


# %%
model = create_model()
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001)
model.compile(optimizer= optimizer, loss= "categorical_crossentropy", metrics=["accuracy"])


# %%
train_labels = tf.keras.utils.to_categorical(y= train_labels, num_classes= 10)


# %%
train_labels.shape


# %%
model.fit(x= train_images, y= train_labels, batch_size= 8, epochs=10)


# %%
model.save("image_classification", save_format="tf")


