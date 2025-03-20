#!/bin/bash

# Exit on any error
set -e

echo "Starting environment setup..."

# Create necessary directories
mkdir -p ./data/{raw,processed,backup}
mkdir -p ./models/{trained,deployed,archived}
mkdir -p ./logs
mkdir -p ./config
mkdir -p ./scripts
mkdir -p ./temp

# Set permissions
chmod 755 ./scripts
chmod 750 ./data
chmod 750 ./models
chmod 755 ./logs

# Install dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo '{
        "numpy": ">=1.20.0",
        "pandas": ">=1.3.0",
        "scikit-learn": ">=1.0.0",
        "requests": ">=2.25.0",
        "joblib": ">=1.0.0",
        "matplotlib": ">=3.4.0",
        "flask": ">=2.0.0",
        "psutil": ">=5.8.0"
    }' > requirements.txt
    pip install -r requirements.txt
fi

# Install system packages
echo "Installing system packages..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-dev build-essential libssl-dev libffi-dev
elif command -v yum &> /dev/null; then
    sudo yum update -y
    sudo yum install -y python3-devel gcc openssl-devel libffi-devel
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Create config files if they don't exist
if [ ! -f "./config/app_config.json" ]; then
    echo "Creating default configuration..."
    echo '{
        "data_sources": {
            "api_endpoint": "https://example.com/api",
            "api_key": "YOUR_API_KEY"
        },
        "model_params": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": 10
        },
        "deployment": {
            "port": 8000,
            "host": "0.0.0.0",
            "staging_url": "http://staging-server:8000",
            "production_url": "http://production-server:8000"
        },
        "monitoring": {
            "log_level": "INFO",
            "alert_email": "admin@example.com",
            "metrics_interval": 60
        },
        "security": {
            "scan_interval": 86400,
            "vulnerability_threshold": "MEDIUM"
        },
        "backup": {
            "interval": 86400,
            "retention_days": 30,
            "storage_path": "./data/backup"
        }
    }' > ./config/app_config.json
fi

echo "Environment setup completed successfully!"
