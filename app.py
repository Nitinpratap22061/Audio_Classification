import streamlit as st


import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model.h5')

# Your class labels (modify as per your dataset)
class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
                'siren', 'street_music']

# Streamlit app setup
st.title("🔊 Urban Sound Classification App")
st.write("Upload a WAV file and get prediction using your trained ANN model.")

# File upload
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])               


if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Load audio file with librosa
    y, sr = librosa.load(uploaded_file, sr=None)

    # Extract features (match your training feature extractor)
    # Example: 40 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # shape (40,)

    # Reshape for prediction
    mfcc_mean = mfcc_mean.reshape(1, -1)  # shape (1, 40)

    # Predict
    prediction = model.predict(mfcc_mean)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display result
    st.success(f"Predicted class: **{predicted_class}**")
