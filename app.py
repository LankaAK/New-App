import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model (MobileNet)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Preprocess the input image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Decode predictions
def decode_predictions(preds):
    return tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]

# Title of the application
st.title("Image Classification with Streamlit")

# Sidebar for user inputs
st.sidebar.title("Upload and Classify Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Display progress bar
    with st.spinner('Loading model...'):
        predictions = model.predict(processed_image)
    
    # Decode the predictions
    decoded_preds = decode_predictions(predictions)
    
    st.write(f"Prediction: {decoded_preds[0][1]}, Confidence: {decoded_preds[0][2]*100:.2f}%")

    # Progress and status update
    st.progress(100)
    st.success('Classification Complete')

# Sidebar inputs
st.sidebar.title("Other Options")
if st.sidebar.button('Show Example Image'):
    st.sidebar.image("example.jpg", caption='Example Image', use_column_width=True)

# Container for graphs
with st.container():
    st.header("Model Performance Visualization")
    x = np.random.rand(100)
    y = np.random.rand(100)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    st.pyplot(fig)

st.sidebar.text("Created by [Your Name]")

