import shap
import matplotlib.pyplot as plt

# Note: SHAP needs to be "told" about the pipeline structure
def get_model_from_pipeline(pipeline):
    """Extracts the classifier from the scikit-learn pipeline."""
    return pipeline.named_steps['model']

def get_preprocessor_from_pipeline(pipeline):
    """Extracts the preprocessor (scaler) from the pipeline."""
    return pipeline.named_steps['scaler']

def explain_predictions(pipeline, X_test):
    """
    Uses SHAP to explain model predictions.
    """
    print("\n--- Model Explainability Analysis (SHAP) ---")
    
    # Extract model and scaler
    model = get_model_from_pipeline(pipeline)
    scaler = get_preprocessor_from_pipeline(pipeline)
    
    # We must run SHAP on the *scaled* data, as that's what the model was trained on
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Initialize SHAP Explainer
    # TreeExplainer is fast and optimized for tree-based models like RandomForest
    explainer = shap.TreeExplainer(model)
    
    # 2. Calculate SHAP values
    # This explains every prediction in the test set
    shap_values = explainer.shap_values(X_test_scaled)
    
    print("Generating SHAP summary plot...")
    
    # 3. Create a summary plot
    # We use shap_values[1] because we care about the "Bot" class (label=1)
    # The plot shows:
    # - Which features are most important (top to bottom)
    # - How a feature's value (low=blue, high=red) impacts the prediction
    
    plt.figure()
    shap.summary_plot(shap_values[1], X_test, feature_names=X_test.columns, show=False)
    plt.title("SHAP Summary Plot (Impact on 'Bot' Prediction)")
    plt.savefig('shap_summary.png', bbox_inches='tight')
    print("Saved shap_summary.png")
    plt.show()

    print(
    """
    --- How to Read the SHAP Plot ---
    * Y-axis: Features, ranked by importance.
    * X-axis: SHAP Value (Impact on prediction. Positive = more likely 'Bot', Negative = more likely 'Human').
    * Color: Feature Value (Red = high value, Blue = low value).
    
    Example: If 'followers_to_friends_ratio' is at the top, and blue dots (low ratio)
    are on the right (positive SHAP), it means a LOW ratio strongly pushes the
    model to predict 'Bot'.
    """
    )