#!/usr/bin/env python
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict

class DeepSeekEnvironmentManager:
    def __init__(self, root_dir: str = "deepseek"):
        self.root_dir = Path(root_dir)
        self.directory_structure = {
            'py': ['training.py', 'inference.py'],
            'api': ['__init__.py', 'rest.py', 'graphql.py', 'websocket.py', 'middleware.py'],
            'orchestration': ['__init__.py', 'scheduler.py', 'workers.py', 'scaling.py'],
            'plugins': ['__init__.py', 'loader.py', 'registry.py'],
            'cli': ['__init__.py', 'commands.py', 'utils.py'],
            'tests': {
                '.': ['__init__.py'],
                'unit': ['__init__.py', 'test_core.py', 'test_ai.py', 'test_api.py', 'test_plugins.py'],
                'integration': ['__init__.py', 'test_e2e.py'],
                'security': ['__init__.py', 'test_security.py']
            },
            'docs': ['architecture.md', 'security.md', 'api.md', 'examples.md'],
            'examples': ['basic_usage.py', 'ai_pipeline.py', 'plugin_development.py'],
            'deployment': {
                'docker': ['Dockerfile', 'docker-compose.yml'],
                'kubernetes': ['deployment.yaml', 'service.yaml', 'configmap.yaml']
            },
            'benchmarks': ['__init__.py', 'core_benchmarks.py', 'ai_benchmarks.py'],
            'core': ['__init__.py', 'config.py', 'security.py', 'storage.py', 'telemetry.py', 'errors.py'],
            'ai': ['__init__.py', 'engine.py', 'models.py']
        }
        
    def create_directory_structure(self):
        """Creates the entire directory structure"""
        def create_nested_structure(base_path: Path, structure: Dict):
            for key, value in structure.items():
                current_path = base_path / key
                current_path.mkdir(parents=True, exist_ok=True)
                
                if isinstance(value, list):
                    for file in value:
                        file_path = current_path / file
                        if not file_path.exists():
                            file_path.touch()
                elif isinstance(value, dict):
                    create_nested_structure(current_path, value)
        
        create_nested_structure(self.root_dir, self.directory_structure)
        
        # Create root level files
        root_files = ['setup.py', 'requirements.txt', 'README.md', '__init__.py']
        for file in root_files:
            file_path = self.root_dir / file
            if not file_path.exists():
                file_path.touch()

    def setup_git(self):
        """Initializes git repository and creates .gitignore"""
        os.chdir(self.root_dir)
        
        # Initialize git if not already initialized
        if not (self.root_dir / '.git').exists():
            subprocess.run(['git', 'init'])
        
        # Create .gitignore
        gitignore_content = """
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.env
.venv
venv/
ENV/
.idea/
.vscode/
"""
        with open(self.root_dir / '.gitignore', 'w') as f:
            f.write(gitignore_content.strip())

    def setup_virtual_environment(self):
        """Creates and activates virtual environment"""
        if not (self.root_dir / 'venv').exists():
            subprocess.run(['python', '-m', 'venv', 'venv'])

    def install_dependencies(self):
        """Installs project dependencies"""
        pip_cmd = str(self.root_dir / 'venv' / 'bin' / 'pip')
        subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'])

    def run_tests(self):
        """Executes test suite"""
        pytest_cmd = str(self.root_dir / 'venv' / 'bin' / 'pytest')
        subprocess.run([pytest_cmd, 'tests'])

    def execute(self):
        """Executes all setup steps in the correct order"""
        print("Setting up DeepSeek CodeSpaces Environment...")
        self.create_directory_structure()
        self.setup_git()
        self.setup_virtual_environment()
        self.install_dependencies()
        self.run_tests()
        print("DeepSeek CodeSpaces Environment setup completed successfully!")

if __name__ == "__main__":
    manager = DeepSeekEnvironmentManager()
    manager.execute()
