#!/bin/bash

LOG_FILE="troubleshoot_master.log"
ERROR_LOG="error_master.log"

exec > >(tee -a ${LOG_FILE}) 2> >(tee -a ${ERROR_LOG} >&2)

echo "========== Master Troubleshooting Script =========="
echo "Run started at: $(date)"
echo "==================================================="

# Function to add shebang line if missing
add_shebang_if_missing() {
    FILE="$1"
    SHEBANG="$2"
    
    if [[ $(head -n 1 "$FILE") != "#!"* ]]; then
        echo "[INFO] Adding shebang ($SHEBANG) to $FILE"
        sed -i "1i\\$SHEBANG" "$FILE"
    else
        echo "[OK] Shebang exists in $FILE"
    fi
}

# Function to check syntax for Python scripts
check_python_syntax() {
    FILE="$1"
    python3 -m py_compile "$FILE"
    if [ $? -eq 0 ]; then
        echo "[OK] Python syntax correct: $FILE"
    else
        echo "[ERROR] Python syntax issue in $FILE"
    fi
}

# Process all .sh and .py files recursively
find .. -type f \( -name "*.sh" -o -name "*.py" \) | while read -r script; do
    if [[ $script == *.sh ]]; then
        add_shebang_if_missing "$script" "#!/bin/bash"
        echo "[CHECK] Testing Bash syntax for: $script"
        bash -n "$script" && echo "[OK] Bash syntax correct: $script" || echo "[ERROR] Bash syntax issue in $script"
    elif [[ $script == *.py ]]; then
        add_shebang_if_missing "$script" "#!/usr/bin/env python3"
        check_python_syntax "$script"
    fi
done

echo "==================================================="
echo "Run ended at: $(date)"
echo "Logs: ${LOG_FILE} | Errors: ${ERROR_LOG}"
