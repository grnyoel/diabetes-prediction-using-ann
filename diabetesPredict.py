import streamlit as st

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install TensorFlow
try:
    import tensorflow as tf
except ImportError:
    install('tensorflow==2.15.0')
    import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Nama file model
model_filename = 'diabetes_model1.h5'

# Fungsi untuk memuat model dengan penanganan eksepsi
def load_model(model_filename):
    try:
        model = tf.keras.models.load_model(model_filename)
        st.write("Model berhasil dimuat.")
        return model
    except OSError as e:
        st.error(f"Gagal memuat model: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
    return None

# Muat model
model = load_model(model_filename)

# Function untuk prediksi diabetes dengan penanganan eksepsi
def predict_diabetes(input_data):
    if model:
        # Normalize the input data
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)
        
        # Make prediction
        try:
            prediction = model.predict(input_data)
            return prediction
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")
            return None
    else:
        st.warning("Model tidak berhasil dimuat. Tidak bisa melakukan prediksi.")


# # Pastikan untuk menggunakan path absolut ke file .h5
# model_path = 'diabetes_model1.h5'

# # Log the model path to verify
# st.write(f"Model path: {model_path}")

# # Try loading the model and catch any exceptions
# try:
#     model = tf.keras.models.load_model(model_path)
#     st.write("Model loaded successfully.")
# except Exception as e:
#     st.error(f"Failed to load model: {e}")

# # Load the saved model
# model = tf.keras.models.load_model('E:\UAS_AI\diabetes_model1.h5')

# Define a function for prediction
# def predict_diabetes(input_data):
#     # Normalize the input data
#     scaler = StandardScaler()
#     input_data = scaler.fit_transform(input_data)
#     try:
#         prediction = model.predict(input_data)
#         return prediction
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

# Streamlit interface
st.title("Diabetes Prediction")

# Collect user input
def user_input_features():
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    
    data = {'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display input data
st.subheader('User Input parameters')
st.write(input_df)

# Predict
if st.button('Predict'):
    prediction = predict_diabetes(input_df)
    st.subheader('Prediction')
    st.write(prediction)
