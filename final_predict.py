import sys
from helpers import preprocess_image, predict_image
from tensorflow.keras.models import load_model

MODEL_PATH = "models/VGG16_model.keras"

def main(image_path):
    model = load_model(MODEL_PATH)
    img_array = preprocess_image(image_path)
    if img_array is None:
        print("Image loading failed.")
        return
    prediction, confidence = predict_image(model, img_array)
    print(f"Prediction: {prediction} ({confidence * 100:.2f}% confidence)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python final_predict.py <image_path>")
    else:
        main(sys.argv[1])
