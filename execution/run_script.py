#!/usr/bin/env python3
import os
import subprocess
import pathlib

class DeepSeekManager:
    def __init__(self, root_dir="deepseek"):
        self.root_dir = pathlib.Path(root_dir)
        self.execution_path = self.root_dir / "execution"

    def setup_directories(self):
        """Ensure necessary directories exist before running."""
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.execution_path, exist_ok=True)

    def setup_git(self):
        """Initialize Git and create a .gitignore file."""
        self.setup_directories()  # Ensure directories exist
        gitignore_path = self.root_dir / ".gitignore"
        
        if not (self.root_dir / ".git").exists():
            subprocess.run(["git", "init", str(self.root_dir)], check=True)
        
        with open(gitignore_path, 'w') as f:
            f.write("__pycache__/\n")
            f.write(".DS_Store\n")

        print("Git initialized and .gitignore created.")

    def execute(self):
        """Main execution function"""
        self.setup_git()
        print("DeepSeek execution started.")

if __name__ == "__main__":
    manager = DeepSeekManager()
    manager.execute()
