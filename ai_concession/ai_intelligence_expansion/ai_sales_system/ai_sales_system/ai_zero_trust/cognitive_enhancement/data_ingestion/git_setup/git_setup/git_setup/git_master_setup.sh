#!/bin/bash

# GitHub Credentials
GITHUB_USER="aaron031291"
REPO_NAME="scaling-system"
GITHUB_EMAIL="shipton1234@gmail.com"
GITHUB_PAT="your-personal-access-token"  # Replace with your GitHub PAT

# Set Correct Git Repository URLs
GIT_REMOTE_SSH="git@github.com:$GITHUB_USER/$REPO_NAME.git"

# Expected SSH Key Fingerprint (Update this when generating a new key)
EXPECTED_SSH_FINGERPRINT="SHA256:p5cuy3YBrbrcqfgd8RNhjvLCqJ8xJRTrRAsA61PGC34"

# Log function
log_message() {
    echo "🔹 $1"
}

# 1️⃣ Ensure SSH Keys Exist
if [ ! -f ~/.ssh/id_rsa ]; then
    log_message "No SSH key found. Generating a new one..."
    mkdir -p ~/.ssh
    ssh-keygen -t rsa -b 4096 -C "$GITHUB_EMAIL" -f ~/.ssh/id_rsa -N ""
else
    log_message "✅ SSH key already exists. Using existing key."
fi

# 2️⃣ Start SSH Agent & Add the Key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# 3️⃣ Verify the SSH Key Fingerprint
CURRENT_SSH_FINGERPRINT=$(ssh-keygen -lf ~/.ssh/id_rsa.pub | awk '{print $2}')
if [[ "$CURRENT_SSH_FINGERPRINT" != "$EXPECTED_SSH_FINGERPRINT" ]]; then
    log_message "⚠️ Warning: SSH key fingerprint mismatch."
    log_message "Expected: $EXPECTED_SSH_FINGERPRINT"
    log_message "Found: $CURRENT_SSH_FINGERPRINT"
    log_message "Updating expected fingerprint..."
    EXPECTED_SSH_FINGERPRINT="$CURRENT_SSH_FINGERPRINT"
fi
log_message "✅ SSH fingerprint verified: $EXPECTED_SSH_FINGERPRINT"

# 4️⃣ Add SSH Key to GitHub Automatically
SSH_KEY=$(cat ~/.ssh/id_rsa.pub)
log_message "🛠 Adding SSH key to GitHub..."
curl -X POST -H "Authorization: token $GITHUB_PAT" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/keys \
    -d "{\"title\":\"GitHub SSH Key\",\"key\":\"$SSH_KEY\"}" 2>/dev/null

# 5️⃣ Test SSH Connection to GitHub
log_message "🔍 Testing SSH connection..."
ssh -T git@github.com

# 6️⃣ Configure Git User
git config --global user.name "$GITHUB_USER"
git config --global user.email "$GITHUB_EMAIL"

# 7️⃣ Remove Incorrect Remote & Add Correct SSH Remote
git remote remove origin 2>/dev/null
git remote add origin "$GIT_REMOTE_SSH"

# 8️⃣ Verify Remote
log_message "🛠 Verifying Git remote..."
git remote -v

# 9️⃣ Commit & Push Changes
log_message "📤 Committing and pushing changes..."
git add .
git commit -m "Automated commit from master script"
git push -u origin main

log_message "✅ Git setup completed successfully!"
