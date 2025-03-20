#!/bin/bash

# Navigate to the ai_concession directory
cd /workspaces/Deep_seek/deepseek/ai_concession

# Ensure python dependencies are installed
echo "Installing required packages..."
pip install tensorflow queue

# Fix Python syntax issues using autopep8
echo "Fixing Python syntax issues..."
pip install autopep8
autopep8 --in-place --aggressive --aggressive ai_concession.py

# Run pylint to highlight any additional issues
echo "Checking with pylint..."
pip install pylint
pylint ai_concession.py

# Set correct permissions for the Python file
echo "Setting permissions..."
chmod +x ai_concession.py

# Git Commit & Push (optional - only if using GitHub)
echo "Committing to Git..."
git add .
git commit -m "Automated setup and fixes applied to ai_concession.py"
git push

echo "Setup completed successfully!"
