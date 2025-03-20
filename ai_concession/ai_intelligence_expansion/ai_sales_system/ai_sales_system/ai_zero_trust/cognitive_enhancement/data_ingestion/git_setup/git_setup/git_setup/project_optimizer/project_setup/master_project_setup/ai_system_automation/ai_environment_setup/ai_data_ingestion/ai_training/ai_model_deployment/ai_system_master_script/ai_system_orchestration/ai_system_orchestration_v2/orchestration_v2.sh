#!/usr/bin/env python3

import os
import sys
import json
import logging
import requests
import pandas as pd
import numpy as np
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
        
        # For demonstration, create synthetic data if API endpoint is default
        if api_endpoint == "https://example.com/api":
            logger.info("Using synthetic data for demonstration")
            # Create synthetic data
            n_samples = 1000
            n_features = 10
            
            # Generate features
            features = np.random.randn(n_samples, n_features)
            
            # Generate labels (binary classification for simplicity)
            labels = np.random.randint(0, 2, size=(n_samples, 1))
            
            data = {
                "features": features.tolist(),
                "labels": labels.tolist()
            }
            return data
        
        # Real API call
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

