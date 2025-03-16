#!/bin/bash

# Define project structure
PROJECT_NAME="DeepSeek"
BASE_DIR="$HOME/workspaces/$PROJECT_NAME"

# List of directories to create
DIRECTORIES=(
    "deepseek/core"
    "deepseek/core/config"
    "deepseek/core/security"
    "deepseek/core/storage"
    "deepseek/core/telemetry"
    "deepseek/core/errors"
    "deepseek/ai"
    "deepseek/ai/engine"
    "deepseek/ai/models"
    "deepseek/ai/training"
    "deepseek/ai/inference"
    "deepseek/api"
    "deepseek/api/rest"
    "deepseek/api/graphql"
    "deepseek/api/websocket"
    "deepseek/api/middleware"
    "deepseek/cli"
    "deepseek/cli/utils"
    "deepseek/orchestration"
    "deepseek/orchestration/scheduler"
    "deepseek/orchestration/workers"
    "deepseek/orchestration/scaling"
    "deepseek/plugins"
    "deepseek/plugins/loader"
    "deepseek/plugins/registry"
    "deepseek/tests"
    "deepseek/tests/unit"
    "deepseek/tests/integration"
)

# Create directories
echo "📁 Creating directories..."
for DIR in "${DIRECTORIES[@]}"; do
    mkdir -p "$BASE_DIR/$DIR"
    echo "✅ Created: $BASE_DIR/$DIR"
done

# Create nano script templates inside each module
echo "📝 Generating script templates..."
for DIR in "${DIRECTORIES[@]}"; do
    SCRIPT_PATH="$BASE_DIR/$DIR/setup_${DIR##*/}.sh"
    echo "#!/bin/bash" > "$SCRIPT_PATH"
    echo "# Setup script for ${DIR##*/}" >> "$SCRIPT_PATH"
    chmod +x "$SCRIPT_PATH"
    echo "✅ Script created: $SCRIPT_PATH"
done

echo "�� Setup complete! Navigate to $BASE_DIR and start coding!"
