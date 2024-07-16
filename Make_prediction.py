import argparse
#import cv2 
import numpy as np
import joblib

from tensorflow.keras.applications.vgg19 import VGG19

def load_and_preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (180, 180))
        images.append(img)
    data = np.array(images)
    return data

def make_prediction(image_paths, model_path):
    # Load and preprocess images
    data = load_and_preprocess_images(image_paths)

    # Load VGG19 model
    vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

    # Extract features using VGG19
    features = vgg_model.predict(data)
    x = features.reshape(data.shape[0], -1)

    feature_difference = x[1] - x[0]

    # Load the trained model
    RF = joblib.load(model_path)

    # Make prediction
    prediction = RF.predict(feature_difference.reshape(1, -1))

    return prediction

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("image1", help="Path to the first image.")
    parser.add_argument("image2", help="Path to the second image.")
    parser.add_argument("model_path", help="Path to the trained model file.")
    args = parser.parse_args()

    # Example usage:
    image_paths = [args.image1, args.image2]
    model_path = args.model_path

    prediction = make_prediction(image_paths, model_path)
    print("Prediction:", prediction)

