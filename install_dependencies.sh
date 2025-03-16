#!/bin/bash

# Exit on error
set -e

echo "�� Ensuring all dependencies are installed..."

# Ensure pip is up to date
echo "📦 Updating pip..."
pip install --upgrade pip

# Define required dependencies
REQUIRED_PIP_PACKAGES=(
  redis
  hvac
  pyjwt
  cryptography
  scikit-learn
  joblib
  psutil
  fastapi
  uvicorn
  pytest
  fabric-sdk-py  # Corrected Hyperledger Fabric SDK
)

# Install pip dependencies
echo "📥 Installing required Python dependencies..."
pip install "${REQUIRED_PIP_PACKAGES[@]}"

# Install system dependencies if missing
SYS_DEPENDENCIES=("redis-server" "vault")

echo "🛠️ Installing system dependencies..."
for package in "${SYS_DEPENDENCIES[@]}"; do
  if ! command -v "$package" &> /dev/null; then
    echo "Installing $package..."
    sudo apt-get install -y "$package"
  fi
done

echo "✅ All dependencies are installed!"
