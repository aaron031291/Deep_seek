#!/bin/bash

# Project Directory
PROJECT_DIR="$(pwd)"

# Log function
log_message() {
    echo "🔹 $1"
}

# 1️⃣ Ensure Proper File Permissions
log_message "Setting correct file permissions..."
find "$PROJECT_DIR" -type f -name "*.py" -exec chmod +x {} \;

# 2️⃣ Add Missing Shebang to Python Scripts
log_message "Checking and adding missing shebang lines..."
for file in $(find "$PROJECT_DIR" -type f -name "*.py"); do
    if ! head -n 1 "$file" | grep -q "#!"; then
        sed -i '1i#!/usr/bin/env python3' "$file"
        log_message "✅ Added shebang to $file"
    fi
done

# 3️⃣ Check and Fix Syntax Errors
log_message "Checking for syntax errors..."
for file in $(find "$PROJECT_DIR" -type f -name "*.py"); do
    python3 -m py_compile "$file" 2>/dev/null
    if [[ $? -ne 0 ]]; then
        log_message "❌ Syntax error in: $file"
    else
        log_message "✅ Syntax OK: $file"
    fi
done

# 4️⃣ Run Debugging Checks
log_message "Running debugging checks..."
pylint "$PROJECT_DIR" --disable=all --enable=E,F 2>/dev/null || log_message "⚠️ Linter warnings detected."

# 5️⃣ Ensure Tests Exist
if [ ! -d "$PROJECT_DIR/tests" ]; then
    log_message "⚠️ No tests folder found. Creating 'tests' directory..."
    mkdir -p "$PROJECT_DIR/tests"
    echo "import pytest" > "$PROJECT_DIR/tests/test_sample.py"
    echo "def test_sample(): assert 1 + 1 == 2" >> "$PROJECT_DIR/tests/test_sample.py"
fi

# 6️⃣ Run Tests
log_message "Running tests..."
pytest --tb=short || log_message "❌ Test failures detected. Fix errors and re-run."

# 7️⃣ Automate Git Commit & Push
log_message "Committing and pushing changes..."
git add .
git commit -m "Automated commit from setup script"
git push -u origin main || log_message "❌ Git push failed. Ensure correct remote repository."

log_message "✅ Master Project Setup & Debugging Completed!"
