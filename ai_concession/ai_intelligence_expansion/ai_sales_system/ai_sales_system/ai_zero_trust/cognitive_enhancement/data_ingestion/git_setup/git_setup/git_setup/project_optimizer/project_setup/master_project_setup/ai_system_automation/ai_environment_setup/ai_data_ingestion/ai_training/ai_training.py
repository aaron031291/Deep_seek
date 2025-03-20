#!/usr/bin/env python3
ximport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('../ai_data_ingestion/ingested_data.csv')

    # Define features and target (replace with actual column names)
    X = df.drop('target_column', axis=1)  
    y = df['target_column']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    print("Evaluating the model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Model accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Save trained model
    joblib.dump(model, 'trained_model.joblib')
    print("Trained model saved as 'trained_model.joblib'.")

if __name__ == "__main__":
    train_model()
