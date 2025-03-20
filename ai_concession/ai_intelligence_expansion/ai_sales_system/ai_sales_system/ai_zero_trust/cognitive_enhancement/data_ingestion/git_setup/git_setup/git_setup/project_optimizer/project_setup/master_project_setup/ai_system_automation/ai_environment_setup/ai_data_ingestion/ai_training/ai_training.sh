#!/usr/bin/env python3

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ai_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ai_training")

def load_config():
    """Load configuration from config file"""
    try:
        with open("config/app_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def load_latest_data():
    """Load the latest processed data"""
    try:
        with open("data/processed/latest_data_reference.json", "r") as f:
            data_ref = json.load(f)
            
        features_file = f"data/processed/{data_ref['features_file']}"
        labels_file = f"data/processed/{data_ref['labels_file']}"
        
        features = pd.read_csv(features_file)
        labels = pd.read_csv(labels_file)
        
        logger.info(f"Loaded data from {features_file} and {labels_file}")
        return features, labels
    except Exception as e:
        logger.error(f"Error loading latest data: {e}")
        sys.exit(1)

def train_model(X_train, y_train, config):
    """Train the model with hyperparameter tuning"""
    try:
        logger.info("Starting model training with hyperparameter tuning")
        
        # Define model and hyperparameters
        model = RandomForestClassifier(random_state=42)
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='f1_weighted'
        )
        
        grid_search.fit(X_train, y_train.values.ravel())
        
        # Get best model
        best_model = grid_search.best_estimator_
        logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        
        return best_model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    try:
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None

def save_model(model, metrics):
    """Save the trained model and its metrics"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/trained/model_{timestamp}.joblib"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save model metadata
        metadata = {
            'timestamp': timestamp,
            'metrics': metrics,
            'model_path': model_path,
            'model_type': type(model).__name__,
            'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
        }
        
        with open(f"models/trained/model_{timestamp}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Update latest model reference
        with open("models/trained/latest_model_reference.json", "w") as f:
            json.dump({
                'model_path': model_path,
                'metadata_path': f"models/trained/model_{timestamp}_metadata.json",
                'timestamp': timestamp
            }, f, indent=4)
        
        logger.info(f"Model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def main():
    """Main function to orchestrate model training process"""
    logger.info("Starting AI model training process")
    
    # Load configuration
    config = load_config()
    
    # Load data
    features, labels = load_latest_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train, config)
    if model is None:
        logger.error("Model training failed. Exiting.")
        sys.exit(1)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    if metrics is None:
        logger.error("Model evaluation failed. Exiting.")
        sys.exit(1)
    
    # Save model
    if not save_model(model, metrics):
        logger.error("Failed to save model. Exiting.")
        sys.exit(1)
    
    logger.info("AI model training completed successfully")

if __name__ == "__main__":
    main()
