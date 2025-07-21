import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the model
model = load_model("fashion_mnist_model.h5")

st.title("Fashion MNIST Classifier")
st.write("Upload a 28x28 grayscale image")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted class: {predicted_class}")
