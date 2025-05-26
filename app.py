import streamlit as st
import pandas as pd
import numpy as np
import joblib



# Load model and data
@st.cache_data
def load_model():
    model = joblib.load('models/model.pkl')
    return model

model = load_model()

# UI Components
st.title('🏠 Airbnb Price Recommender')
st.write("Get optimal pricing suggestions for your property")

# Input form
with st.form("prediction_form"):
    st.subheader("Property Details")
    
    # Essential inputs
    neighbourhood = st.selectbox(
        'Neighbourhood*',
        ['Brooklyn', 'Manhattan', 'Staten Island', 'Queens', 'Bronx'],
        index=0
    )
    
    room_type = st.selectbox(
        'Room Type*',
        ['Entire home/apt', 'Private room', 'Shared room'],
        index=0
    )
    
    minimum_nights = st.slider(
        'Minimum Nights Stay*',
        1, 30, 1
    )

    submit_button = st.form_submit_button("Predict Price")

    if submit_button:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'neighbourhood_group': [neighbourhood],
            'room_type': [room_type],
            'minimum_nights': [minimum_nights]
        })

        # Make prediction
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"The recommended price for your property is ${predicted_price:.2f} per night.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
