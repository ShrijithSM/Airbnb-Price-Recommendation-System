import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
flepath = './data/listings.csv'
def load_and_clean_data(filepath):
    """Load, clean, and preprocess raw Airbnb data."""
    df = pd.read_csv(filepath)
    
    # Remove unwanted columns
    columns_to_keep = ['neighbourhood_group', 'room_type', 'minimum_nights', 'price']
    df = df[columns_to_keep]
    
    # Clean price (remove outliers)
    df = df[df['price'].between(50, 1000)]
    
    # Save cleaned data
    df.to_csv('./data/preprocessed_data.csv', index=False)
    return df

def build_pipeline():
    """Create ML pipeline with preprocessing."""
    numeric_features = ['minimum_nights']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['neighbourhood_group', 'room_type']
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
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data('./data/listings.csv')
    
    # Define features and target
    features = ['neighbourhood_group', 'room_type', 'minimum_nights']
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
    
    # Save model
    joblib.dump(model, './models/model.pkl')
    print("Model saved to ./models/model.pkl")

if __name__ == "__main__":
    main()
