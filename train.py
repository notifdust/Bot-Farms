import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def train_classifier(X, y):
    """
    Trains a classifier, evaluates it, and returns the trained model.
    """
    print("Splitting data and training model...")
    
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Create a model pipeline
    # A pipeline simplifies steps:
    # 1. StandardScaler: Scales features (good practice for many models)
    # 2. RandomForestClassifier: The model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # 3. Train the model
    pipeline.fit(X_train, y_train)
    
    print("Model training complete.")
    return pipeline, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints a report and confusion matrix.
    """
    print("\n--- Model Evaluation ---")
    
    # 1. Get predictions
    y_pred = model.predict(X_test)
    
    # 2. Print Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # 3. Print Classification Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    # 'target_names' maps the labels (0, 1) to human-readable names
    print(classification_report(y_test, y_pred, target_names=['Human (0)', 'Bot (1)']))
    
    # 4. Plot Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Human', 'Predicted Bot'],
                yticklabels=['Actual Human', 'Actual Bot'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    plt.show()