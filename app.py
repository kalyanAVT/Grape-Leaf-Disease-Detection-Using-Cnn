import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = load_model('notebooks/models/CNN_leaf_disease_model.keras')

# Define the class labels
class_labels = ['Black_rot', 'Esca_(Black_Measles)', 'Healthy', 'leaf_blight']

# Function to predict the disease from the uploaded image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict class probabilities
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)
    
    # Return the predicted class label and probability
    return class_labels[class_idx[0]], prediction[0][class_idx[0]]

# Streamlit App UI
def main():
    st.title("🍇Grape Leaf Disease Detection")
    st.markdown("Upload a grape leaf image to predict the disease type.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Save the uploaded image temporarily
        img_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Make prediction
        predicted_class, predicted_prob = predict_image(img_path)

        # Display prediction result
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Prediction Confidence: {predicted_prob:.2f}")

        # Provide some information about the prediction
        st.markdown(f"### {predicted_class} Overview:")
        st.write("Here you can provide details about the predicted disease type.")

        # Optional: Plot prediction confidence
        plt.bar(class_labels, [predicted_prob if label == predicted_class else 1 - predicted_prob for label in class_labels])
        plt.title("Prediction Confidence")
        plt.ylabel("Confidence")
        plt.xlabel("Disease Type")
        st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    main()
