#!/usr/bin/env python3

import os
import sys
import json
import logging
import requests
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_ingestion.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("data_ingestion")

def load_config():
    """Load configuration from config file"""
    try:
        with open("config/app_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def fetch_data(config):
    """Fetch data from external sources"""
    try:
        logger.info("Fetching data from API...")
        api_endpoint = config["data_sources"]["api_endpoint"]
        api_key = config["data_sources"]["api_key"]
        
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(api_endpoint, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}")
            return None
            
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def validate_data(data):
    """Validate the fetched data"""
    if data is None:
        logger.error("No data to validate")
        return False
        
    # Check for required fields
    required_fields = ["features", "labels"]
    for field in required_fields:
        if field not in data:
            logger.error(f"Required field '{field}' missing from data")
            return False
            
    # Check data size
    if len(data["features"]) == 0:
        logger.error("Empty features dataset")
        return False
        
    if len(data["features"]) != len(data["labels"]):
        logger.error("Features and labels have different lengths")
        return False
        
    logger.info("Data validation passed")
    return True

def preprocess_data(data):
    """Preprocess the data for model training"""
    try:
        logger.info("Preprocessing data...")
        
        # Convert to pandas DataFrame
        features_df = pd.DataFrame(data["features"])
        labels_df = pd.DataFrame(data["labels"])
        
        # Handle missing values
        features_df.fillna(0, inplace=True)
        
        # Normalize numerical features
        numerical_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            features_df[col] = (features_df[col] - features_df[col].mean()) / features_df[col].std()
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_df.to_csv(f"data/processed/features_{timestamp}.csv", index=False)
        labels_df.to_csv(f"data/processed/labels_{timestamp}.csv", index=False)
        
        # Save reference to latest data
        with open("data/processed/latest_data_reference.json", "w") as f:
            json.dump({
                "features_file": f"features_{timestamp}.csv",
                "labels_file": f"labels_{timestamp}.csv",
                "timestamp": timestamp
            }, f)
            
        logger.info(f"Preprocessed data saved with timestamp {timestamp}")
        return True
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return False

def main():
    """Main function to orchestrate data ingestion process"""
    logger.info("Starting data ingestion process")
    
    # Load configuration
    config = load_config()
    
    # Fetch data
    data = fetch_data(config)
    
    # Validate data
    if not validate_data(data):
        logger.error("Data validation failed. Exiting.")
        sys.exit(1)
    
    # Preprocess data
    if not preprocess_data(data):
        logger.error("Data preprocessing failed. Exiting.")
        sys.exit(1)
    
    logger.info("Data ingestion completed successfully")

if __name__ == "__main__":
    main()
