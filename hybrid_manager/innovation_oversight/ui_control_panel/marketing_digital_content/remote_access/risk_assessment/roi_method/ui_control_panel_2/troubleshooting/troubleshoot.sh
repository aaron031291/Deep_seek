#!/bin/bash

# Directory to scan
DIR_TO_SCAN="."

# Log file
LOG_FILE="execution_log.txt"

# Add shebang line if missing
add_shebang() {
    for file in $(find $DIR_TO_SCAN -type f -name "*.py"); do
        if ! head -n 1 "$file" | grep -q "#!"; then
            sed -i '1i#!/usr/bin/env python3' "$file"
        fi
    done
}

# Check and fix syntax errors
check_syntax_errors() {
    for file in $(find $DIR_TO_SCAN -type f -name "*.py"); do
        python3 -m py_compile "$file" 2>> $LOG_FILE
    done
}

# Set permissions
set_permissions() {
    find $DIR_TO_SCAN -type f -name "*.py" -exec chmod +x {} \;
}

# Log execution
log_execution() {
    echo "$(date): $1" >> $LOG_FILE
}

# Main function
main() {
    log_execution "Starting troubleshooting script"
    add_shebang
    check_syntax_errors
    set_permissions
    log_execution "Troubleshooting script completed"
}

# Run the main function
main
