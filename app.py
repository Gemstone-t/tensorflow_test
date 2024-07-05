import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras

# Function to load and preprocess the image
def load_and_prep_image(filename, img_shape=224, scale=False):
    img = tf.io.read_file(filename)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img / 255.0
    else:
        return img

# Load the saved model with error handling

    # custom_objects = {"CustomLayer": CustomLayer, "custom_fn": custom_fn}

    # with keras.saving.custom_object_scope(custom_objects):
model = keras.models.load_model("fine_tuned.keras")
st.write("Model loaded successfully.")