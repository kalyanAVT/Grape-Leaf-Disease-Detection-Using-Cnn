import os
import numpy as np
from PIL import Image
from tensorflow import image

# 📌 Class label mapping (used consistently in training & prediction)
class_indices = {'Black_rot': 0, 'Esca_(Black_Measles)': 1, 'Healthy': 2, 'Leaf_blight': 3}
index_to_class = {v: k for k, v in class_indices.items()}

# 📌 Load and preprocess a single image
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# 📌 Predict disease from image using loaded model
def predict_image(model, img_array):
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = index_to_class[predicted_index]
    confidence = prediction[predicted_index]
    return predicted_class, confidence

# 📌 Load all images and labels from folder (for custom scripts)
def load_images_from_folder(folder_path, target_size=(224, 224)):
    image_data = []
    labels = []

    for class_name in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize(target_size)
                img_array = np.array(img) / 255.0
                image_data.append(img_array)
                labels.append(class_indices[class_name])
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    return np.array(image_data), np.array(labels)
