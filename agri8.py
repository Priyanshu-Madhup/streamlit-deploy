import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras_preprocessing import image

# Set random seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Classification function
def classification(image_path):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(38, activation='softmax')  # Corrected to 38 classes
    ])

    LABELS = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
              'Cherry___healthy', 'Cherry___Powdery_mildew', 'Grape___Black_rot', 'Grape___Esca_Black_Measles', 'Grape___healthy',
              'Grape___Leaf_blight_Isariopsis_Leaf_Spot', 'Orange___Haunglongbing', 'Peach___Bacterial_spot', 'Peach___healthy',
              'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
              'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
              'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus']

    model.load_weights("plant_disease_prediction_model.h5")  # Load pretrained model

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize the image

    predictions = model.predict(x)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = LABELS[predicted_class_index]

    return predicted_class_name

# Streamlit app
def main():
    st.title("Plant Disease Classification")
    st.text("Upload an image of a plant leaf to classify the disease")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Save the uploaded image to a temporary file
        temp_file_path = "temp_image.jpg"
        img.save(temp_file_path)

        if st.button('Classify'):
            with st.spinner('Classifying...'):
                predicted_class_name = classification(temp_file_path)
                st.success(f'Predicted Class Name: {predicted_class_name}')

if __name__ == "__main__":
    main()