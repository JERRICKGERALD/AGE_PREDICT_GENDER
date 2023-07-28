import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load the Keras model
model = load_model('model.h5')

# Dictionary to map gender predictions to labels
gender_dict = {0: 'Male', 1: 'Female'}

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to the input size of the model (e.g., 128x128)
    image = image.resize((128, 128))

    # Convert the image to grayscale and normalize the pixel values
    image_array = np.array(image.convert('L')) / 255.0

    # Reshape the image to match the model input shape (1, 128, 128, 1)
    image_array = image_array.reshape((1, 128, 128, 1))

    return image_array

# Function to make predictions using the model
def predict(image_array):
    # Use the loaded model to make predictions on the input image
    # Replace this with your prediction code based on your model architecture
    prediction = model.predict(image_array)

    return prediction

def main():
    st.title("AGE AND GENDER PREDICTOR")

    # File uploader widget for user to upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Preprocess the uploaded image
        image = Image.open(uploaded_image)
        image_array = preprocess_image(image)

        # Make predictions using the model
        predictions = predict(image_array)

        # Convert the predictions to human-readable format
        pred_gender = gender_dict[round(predictions[0][0][0])]
        pred_age = round(predictions[1][0][0])

        # Display the original image using matplotlib with Instagram-like layout
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Display the predictions
        st.write("Predicted Gender:", pred_gender)
        st.write("Predicted Age:", pred_age)

if __name__ == '__main__':
    main()
