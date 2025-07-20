import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import h5py

# Define custom objects if needed for model loading
custom_objects = {
    'binary_accuracy': tf.keras.metrics.binary_accuracy,
    'binary_crossentropy': tf.keras.losses.binary_crossentropy
}

# Set page config
st.set_page_config(
    page_title="Chest X-Ray Classification",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_classification_model():
    """Load the trained model."""
    model = load_model('best_weights.hdf5', custom_objects=custom_objects)
    return model

def preprocess_image(img):
    """Preprocess the image for model prediction."""
    # Resize image to 150x150 (the expected input size for your model)
    img = img.resize((150, 150))
    
    # Convert to RGB if the image is in grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

def main():
    st.title("Chest X-Ray Classification System")
    st.write("Upload a chest X-ray image to classify whether it shows pneumonia or is normal.")

    # Load model
    try:
        model = load_classification_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image for classification"
    )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image_bytes = uploaded_file.read()
                img = Image.open(io.BytesIO(image_bytes))
                st.image(img, use_column_width=True)

            # Preprocess and predict
            with col2:
                st.subheader("Classification Results")
                with st.spinner('Analyzing image...'):
                    processed_img = preprocess_image(img)
                    prediction = model.predict(processed_img)
                    
                    # Assuming binary classification (Normal vs Pneumonia)
                    probability = prediction[0][0]
                    
                    # Create a progress bar for visualization
                    if probability >= 0.5:
                        result = "Pneumonia"
                        prob_display = probability
                    else:
                        result = "Normal"
                        prob_display = 1 - probability

                    st.markdown(f"### Diagnosis: **{result}**")
                    st.progress(float(prob_display))
                    st.markdown(f"Confidence: **{prob_display:.2%}**")

                    # Additional information
                    st.info("""
                    Note: This is an AI-assisted diagnosis tool and should not be used as the sole basis for medical decisions. 
                    Please consult with a healthcare professional for proper medical diagnosis.
                    """)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()

with h5py.File('best_weights.hdf5', 'r') as f:
    print(list(f.keys())) 