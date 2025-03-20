#!/bin/bash

# 1️⃣ Install Dependencies
echo "🔹 Installing dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# 2️⃣ Run Code Quality Checks
echo "🔹 Running code quality checks..."
flake8 . || {
    echo "❌ Code quality issues found. Please fix them."
    exit 1
}

# 3️⃣ Run Tests
echo "🔹 Running tests..."
pytest || {
    echo "❌ Tests failed. Please fix them."
    exit 1
}

# 4️⃣ Set Up GitHub Actions
echo "🔹 Setting up GitHub Actions..."
mkdir -p .github/workflows
cat <<EOL > .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
