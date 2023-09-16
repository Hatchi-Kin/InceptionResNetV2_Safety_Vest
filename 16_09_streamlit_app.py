import cv2
import streamlit as st
import tensorflow as tf
import numpy as np
import time

###################################################################################################

# Function to load the model
def load_model():
    model = tf.keras.models.load_model('my_inception_model.tf')
    return model

model = load_model()

###################################################################################################

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image while maintaining its aspect ratio
    h, w, _ = image.shape
    if h > w:
        ratio = 224 / h
    else:
        ratio = 224 / w
    resized_img = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

    # Add padding to make the image 224x224
    h, w, _ = resized_img.shape
    pad_h = (224 - h) // 2
    pad_w = (224 - w) // 2
    padded_img = cv2.copyMakeBorder(resized_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

    # Preprocess the image
    image = np.expand_dims(padded_img, axis=0)
    image = image / 255.0

    # Resize the image to (224, 224, 3)
    image = cv2.resize(image[0], (224, 224), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)

    return image

###################################################################################################

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    prediction_label = "Not Wearing Vest" if prediction < 0.5 else "Wearing Safety Vest"
    return prediction_label

###################################################################################################

# (main) Function to display the webcam feed with prediction text
def display_webcam():

    st.header("Real-time prediction of safety vest detection")

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    # Create a Streamlit button to start the webcam feed
    start_button = col1.button("Start Webcam Feed")

    # Create a Streamlit button to stop the webcam feed
    stop_button = col2.button("Stop Webcam Feed")

    if start_button and not stop_button:
        cap = cv2.VideoCapture(0)
        # Variables to store the previous prediction and its timestamp
        prediction_label = None
        prev_prediction_time = 0

        # Create a streamlit canvas to display the webcam feed
        canvas = st.image([], use_column_width=True)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Convert the image from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make prediction every 2 seconds
            current_time = time.time()
            if current_time - prev_prediction_time >= 2:
                prediction_label = predict(frame)
                prev_prediction_time = current_time

            # Add the prediction text on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_color = (0, 0, 255) if prediction_label == "Not Wearing Vest" else (0, 255, 0)
            font_thickness = 2
            text = prediction_label
            text_x = 10
            text_y = frame.shape[0] - 10
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

            # Display the frame with prediction text on the streamlit canvas
            canvas.image(frame, channels="RGB")

            # Add a delay of 0.4 seconds between each frame capture
            time.sleep(0.4)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

###################################################################################################

if __name__ == "__main__":
    display_webcam()


# streamlit run 16_09_app.py