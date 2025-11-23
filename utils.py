import pandas as pd
import numpy as np
import joblib
import os

def create_synthetic_data(n_samples=2000):
    """
    Generates a synthetic DataFrame of user features.
    This simulates the user-level data we'd get from a dataset.
    """
    np.random.seed(42)
    
    # --- Generate Humans (label=0) ---
    human_data = {
        'followers_count': np.random.randint(50, 2000, size=n_samples // 2),
        'friends_count': np.random.randint(50, 1000, size=n_samples // 2),
        'statuses_count': np.random.randint(100, 20000, size=n_samples // 2),
        'account_age_days': np.random.randint(365, 4000, size=n_samples // 2),
        'default_profile_image': np.random.choice([0, 1], size=n_samples // 2, p=[0.9, 0.1]),
        'description_length': np.random.randint(20, 160, size=n_samples // 2),
        'label': 0
    }
    df_humans = pd.DataFrame(human_data)
    
    # --- Generate Bots (label=1) ---
    bot_data = {
        'followers_count': np.random.randint(10, 500, size=n_samples // 2),
        'friends_count': np.random.randint(500, 5000, size=n_samples // 2), # Bots follow many
        'statuses_count': np.random.randint(5000, 100000, size=n_samples // 2), # High frequency
        'account_age_days': np.random.randint(30, 730, size=n_samples // 2), # Newer accounts
        'default_profile_image': np.random.choice([0, 1], size=n_samples // 2, p=[0.2, 0.8]), # Use default image
        'description_length': np.random.choice([0, 10, 20], size=n_samples // 2), # Template/no description
        'label': 1
    }
    df_bots = pd.DataFrame(bot_data)
    
    # Combine and shuffle
    df = pd.concat([df_humans, df_bots]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated synthetic data: {len(df)} samples")
    return df

def save_model(model, filepath):
    """Saves a trained model to a file."""
    print(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)

def load_model(filepath):
    """Loads a model from a file."""
    if not os.path.exists(filepath):
        print(f"Error: Model file not found at {filepath}")
        return None
    print(f"Loading model from {filepath}...")
    return joblib.load(filepath)