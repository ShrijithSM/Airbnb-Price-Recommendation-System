import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
filepath = './data/listings.csv'
def load_data(filepath):
    """Load and clean raw Airbnb data"""
    df = pd.read_csv(filepath)
    
    # Clean price
    df = df[df['price'].between(50, 1000)]  # Remove outliers
    
    # Create simplified features
    df['host_experience'] = np.where(
        df['calculated_host_listings_count'] > 3, 
        1,  # Experienced
        0   # New host
    )
    
    # Fill missing values
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    
    return df

def engineer_features(df):
    """Create features for simplified model"""
    # Bin neighborhoods into broader categories
    df['neighbourhood'] = np.where(
        df['neighbourhood'].str.contains('Downtown|Central', case=False),
        'Downtown',
        np.where(
            df['neighbourhood'].str.contains('West|East|North|South', case=False),
            'Suburb',
            'Other'
        )
    )
    
    return df

def build_pipeline():
    """Create ML pipeline with preprocessing"""
    numeric_features = ['minimum_nights', 'reviews_per_month']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['neighbourhood', 'room_type']
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ])

def main():
    # Load and prepare data
    print("Loading data...")
    df = load_data('data/listings.csv')
    df = engineer_features(df)
    
    # Define features and target
    features = [
        'neighbourhood',
        'room_type',
        'minimum_nights',
        'reviews_per_month',
        'host_experience'
    ]
    target = 'price'
    
    X = df[features]
    y = np.log1p(df[target])  # Log transform for better performance
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = build_pipeline()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R²: {train_score:.3f}")
    print(f"Test R²: {test_score:.3f}")
    
    # Save artifacts
    joblib.dump(model, 'models/model.pkl')
    print("Model saved to models/model.pkl")
    
    # Save sample data for Streamlit options
    sample_data = {
        'neighbourhoods': sorted(df['neighbourhood'].unique()),
        'room_types': sorted(df['room_type'].unique())
    }
    joblib.dump(sample_data, 'models/metadata.pkl')

if __name__ == "__main__":
    main()