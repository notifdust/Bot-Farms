import pandas as pd
import numpy as np

def feature_engineering(df):
    """
    Creates new features from the raw user data.
    """
    print("Starting feature engineering...")
    
    # --- Basic Features (from our synthetic data) ---
    features_df = df.copy()
    
    # --- Engineered Features ---
    
    # 1. Posting Frequency (as you suggested)
    # Adding +1 to age to avoid division by zero for very new accounts
    features_df['posting_frequency'] = features_df['statuses_count'] / (features_df['account_age_days'] + 1)
    
    # 2. Network Ratios (as you suggested)
    # The classic bot signal: follow many, followed by few.
    features_df['followers_to_friends_ratio'] = features_df['followers_count'] / (features_df['friends_count'] + 1)
    
    # 3. Simple Network Degree Proxy (as you suggested)
    features_df['network_degree_proxy'] = features_df['followers_count'] + features_df['friends_count']
    
    # 4. Linguistic Markers (Proxy)
    # A simple proxy. Advanced version would use NLP on tweet/profile text.
    features_df['has_description'] = (features_df['description_length'] > 0).astype(int)

    # --- Feature Selection ---
    # Define the target variable
    y = features_df['label']
    
    # Define the columns to be used as features for the model
    feature_columns = [
        'followers_count',
        'friends_count',
        'statuses_count',
        'account_age_days',
        'default_profile_image',
        'description_length',
        'posting_frequency',
        'followers_to_friends_ratio',
        'network_degree_proxy',
        'has_description'
    ]
    
    X = features_df[feature_columns]
    
    print(f"Feature engineering complete. {X.shape[1]} features created.")
    return X, y