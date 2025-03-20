#!/bin/bash

# GitHub Credentials
GITHUB_USER="aaron031291"
REPO_NAME="scaling-system"
GITHUB_EMAIL="shipton1234@gmail.com"
GITHUB_PAT="your-personal-access-token"  # Replace this with your GitHub PAT

# SSH Key Fingerprint
SSH_FINGERPRINT="SHA256:S+7Fec9UHVZ7sOLD/3RfDDK1Y7TPGw4BAYeVa1wMTZ4"

# Set the correct repository URL
GIT_REMOTE_SSH="git@github.com:$GITHUB_USER/$REPO_NAME.git"

# 1️⃣ Ensure the SSH Key Exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "🔑 No SSH key found. Generating one..."
    mkdir -p ~/.ssh
    ssh-keygen -t rsa -b 4096 -C "$GITHUB_EMAIL" -f ~/.ssh/id_rsa -N ""
else
    echo "✅ SSH key already exists. Using existing key."
fi

# 2️⃣ Start SSH Agent & Add the Key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# 3️⃣ Verify the SSH Key Fingerprint
EXISTING_FINGERPRINT=$(ssh-keygen -lf ~/.ssh/id_rsa.pub | awk '{print $2}')
if [[ "$EXISTING_FINGERPRINT" != "$SSH_FINGERPRINT" ]]; then
    echo "⚠️ Warning: SSH key fingerprint does not match the expected value."
    echo "Expected: $SSH_FINGERPRINT"
    echo "Found: $EXISTING_FINGERPRINT"
    echo "❌ Exiting script."
    exit 1
fi
echo "✅ SSH fingerprint verified: $SSH_FINGERPRINT"

# 4️⃣ Add SSH Key to GitHub Automatically
SSH_KEY=$(cat ~/.ssh/id_rsa.pub)
echo "🛠 Adding SSH key to GitHub..."
curl -X POST -H "Authorization: token $GITHUB_PAT" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/keys \
    -d "{\"title\":\"GitHub SSH Key\",\"key\":\"$SSH_KEY\"}"

# 5️⃣ Test SSH Connection to GitHub
echo "🔍 Testing SSH connection..."
ssh -T git@github.com

# 6️⃣ Configure Git User
git config --global user.name "$GITHUB_USER"
git config --global user.email "$GITHUB_EMAIL"

# 7️⃣ Remove Incorrect Remote & Add Correct SSH Remote
git remote remove origin 2>/dev/null
git remote add origin "$GIT_REMOTE_SSH"

# 8️⃣ Verify Remote
echo "🛠 Verifying Git remote..."
git remote -v

# 9️⃣ Commit & Push Changes
echo "📤 Committing and pushing changes..."
git add .
git commit -m "Automated commit from master script"
git push -u origin main

echo "✅ Git setup completed successfully!"
