# Go through all the  folders in the images folder and classify the images

import os
import cv2
import mediapipe
import numpy as np
from tensorflow.keras.models import load_model


# Go through all the  folders in the images folder and classify the images
sign_language_detector_model = load_model("application\\trained_models\model_1\keras_model.h5", compile=False)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']

mediapipe_hands = mediapipe.solutions.hands
hands = mediapipe_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

mediapipe_drawing = mediapipe.solutions.drawing_utils

if not os.path.exists("classified_images"):
    os.makedirs("classified_images")

# Function to preprocess the input image
def preprocess_image(image_path):
    # Load the input image
    input_image = cv2.imread(image_path)

    if input_image is None:
        print("Error: Unable to load the input image")
        return None

    # Convert the image to RGB
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Process the image using the MediaPipe Hands model
    results = hands.process(input_image_rgb)

    # If hands are detected, draw the hand landmarks on the image
    if results.multi_hand_landmarks:
        print("Hand landmarks detected in the input image")
        for hand_landmarks in results.multi_hand_landmarks:
            mediapipe_drawing.draw_landmarks(input_image, hand_landmarks, mediapipe_hands.HAND_CONNECTIONS)
            landmarks_x = [landmark.x for landmark in hand_landmarks.landmark]
            landmarks_y = [landmark.y for landmark in hand_landmarks.landmark]
            x_min = min(landmarks_x)
            x_max = max(landmarks_x)
            y_min = min(landmarks_y)
            y_max = max(landmarks_y)

        padding = .1 # 10% padding around the hand

        x_min = max(0, int((x_min - padding) * input_image_rgb.shape[1]))
        x_max = min(input_image_rgb.shape[1], int((x_max + padding) * input_image_rgb.shape[1]))
        y_min = max(0, int((y_min - padding) * input_image_rgb.shape[0]))
        y_max = min(input_image_rgb.shape[0], int((y_max + padding) * input_image_rgb.shape[0]))

        input_image = input_image[y_min:y_max, x_min:x_max] # Crop the image to the bounding box


        # Check if the cropped image size is valid
        if input_image.size == 0:
            print("Error: Cropped image size is invalid")
            return None

        # Resize the image to the required input size for the model
        resized_image = cv2.resize(input_image, (224, 224))

        # Remove the alpha channel from the image
        resized_image = resized_image[:,:,:3]

        # Convert the image to a NumPy array


        # Normalize the image array
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Save the preprocessed image to the classified_images folder
        cv2.imwrite("classified_images/" + image_path.split("\\")[-1], input_image) # Save the preprocessed image to the classified_images folder

# In images folder, go through all the folders and classify the images

for folder in os.listdir("images"):
        for image in os.listdir("images\\" + folder):
            input_data = preprocess_image("images\\" + folder + "\\" + image)