import utils
import features
import train
import explain

def main():
    print("===== Bot-vs-Human Classifier Pipeline =====")
    
    # 1. Load Data
    # Using synthetic data for this example.
    # TODO: Replace this with your function to load TwiBot-22 or Botometer data.
    raw_df = utils.create_synthetic_data(n_samples=5000)
    
    # 2. Feature Engineering
    X, y = features.feature_engineering(raw_df)
    
    # 3. Model Training
    model_pipeline, X_test, y_test = train.train_classifier(X, y)
    
    # 4. Model Evaluation
    train.evaluate_model(model_pipeline, X_test, y_test)
    
    # 5. Save the trained model
    utils.save_model(model_pipeline, 'bot_classifier_pipeline.joblib')
    
    # 6. Explain the model's predictions
    # We pass the original X_test (unscaled) for easier interpretation of SHAP plots
    explain.explain_predictions(model_pipeline, X_test)
    
    print("\n===== Pipeline Finished Successfully =====")
    print("Next steps: Try loading your real dataset in utils.py and updating features.py!")

if __name__ == "__main__":
    main()