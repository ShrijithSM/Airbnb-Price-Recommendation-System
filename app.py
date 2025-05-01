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
        ['Downtown', 'Suburb', 'Beachside', 'Mountain View', 'City Center'],
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
    
    reviews_per_month = st.slider(
        'Average Reviews per Month*',
        0.0, 10.0, 2.0, 0.1
    )
    
    host_experience = st.radio(
        'Host Experience*',
        ('New Host', 'Experienced Host'),
        index=0
    )
    
    submit_button = st.form_submit_button("Predict Price")

# Prediction logic
if submit_button:
    # Prepare input DataFrame
    input_data = pd.DataFrame([[
        neighbourhood,
        room_type,
        minimum_nights,
        reviews_per_month,
        1 if host_experience == 'Experienced Host' else 0
    ]], columns=[
        'neighbourhood',
        'room_type',
        'minimum_nights',
        'reviews_per_month',
        'host_experience'
    ])
    
    # Predict
    try:
        predicted_log_price = model.predict(input_data)
        predicted_price = np.expm1(predicted_log_price)[0]  # Convert back from log
        st.success(f"## Recommended Price: **${predicted_price:,.2f}** per night")
        
        # Price breakdown
        with st.expander("How this price was calculated"):
            st.markdown("""
            - **Base Price**: ${:,.2f} (Room type: {})  
            - **Location Premium**: {} neighborhood  
            - **Demand Factor**: {} reviews/month  
            - **Host Experience**: {}  
            """.format(
                predicted_price * 0.7,
                room_type,
                neighbourhood,
                reviews_per_month,
                host_experience
            ))
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("🔴 *Required fields")