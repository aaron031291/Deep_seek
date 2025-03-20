#!/usr/bin/env python3
import os
import sys
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import subprocess
import requests
from pathlib import Path
import tempfile
import threading
import queue

logger = logging.getLogger(__name__)

class AIBuilder:
    """System for AI-driven development of DeepSeek components using DeepSeek's own AI."""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "deepseek-coder",  # Using DeepSeek's coding model
                 output_dir: Path = Path("./generated"),
                 test_mode: bool = True,
                 api_base: str = "https://api.deepseek.com/v1"):  # DeepSeek API endpoint
        """Initialize the AI builder.
        
        Args:
            api_key: API key for the DeepSeek AI service
            model: DeepSeek model to use for code generation
            output_dir: Directory for generated code
            test_mode: Whether to run tests on generated code
            api_base: Base URL for the DeepSeek API
        """
        self.api_key = api_key
        self.model = model
        self.output_dir = output_dir
        self.test_mode = test_mode
        self.api_base = api_base
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Component definitions
        self.components = self._load_component_definitions()
        
        # Dependencies between components
        self.dependency_graph = self._build_dependency_graph()
        
        # Results tracking
        self.results = {
            "generated": [],
            "failed": [],
            "pending": list(self.components.keys())
        }
        
        logger.info(f"Initialized AIBuilder with DeepSeek AI model: {model}")
    
    def _load_component_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load component definitions from specification files."""
        components_dir = Path("./specifications/components")
        components = {}
        
        if not components_dir.exists():
            logger.warning(f"Components directory not found: {components_dir}")
            return components
        
        for spec_file in components_dir.glob("*.json"):
            with open(spec_file, "r") as f:
                component_spec = json.load(f)
                components[component_spec["name"]] = component_spec
        
        logger.info(f"Loaded {len(components)} component definitions")
        return components
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build a graph of dependencies between components."""
        graph = {}
        
        for name, component in self.components.items():
            graph[name] = component.get("dependencies", [])
        
        return graph
    
    def _get_component_order(self) -> List[str]:
        """Determine the order in which to generate components based on dependencies."""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            
            for dependency in self.dependency_graph.get(node, []):
                visit(dependency)
            
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        
        for component in self.components:
            if component not in visited:
                visit(component)
        
        return order
    
    def generate_component(self, component_name: str) -> bool:
        """Generate code for a specific component using DeepSeek AI.
        
        Args:
            component_name: Name of the component to generate
            
        Returns:
            True if generation was successful
        """
        component = self.components.get(component_name)
        if not component:
            logger.error(f"Component {component_name} not found")
            return False
        
        logger.info(f"Generating component: {component_name}")
        
        # Prepare prompt for AI
        prompt = self._prepare_generation_prompt(component)
        
        # Call DeepSeek AI service
        try:
            response = self._call_deepseek_ai_service(prompt)
            code_blocks = self._extract_code_blocks(response)
            
            if not code_blocks:
                logger.error(f"No code blocks found in AI response for {component_name}")
                self.results["failed"].append(component_name)
                return False
            
            # Save generated code
            success = self._save_generated_code(component_name, code_blocks)
            
            if not success:
                logger.error(f"Failed to save generated code for {component_name}")
                self.results["failed"].append(component_name)
                return False
            
            # Test generated code if enabled
            if self.test_mode:
                test_success = self._test_generated_code(component_name)
                if not test_success:
                    logger.error(f"Tests failed for {component_name}")
                    self.results["failed"].append(component_name)
                    return False
            
            # Update results
            self.results["generated"].append(component_name)
            self.results["pending"].remove(component_name)
            
            logger.info(f"Successfully generated component: {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating component {component_name}: {str(e)}")
            self.results["failed"].append(component_name)
            return False
    
    def _prepare_generation_prompt(self, component: Dict[str, Any]) -> str:
        """Prepare a prompt for DeepSeek AI to generate a component."""
        # Basic component information
        prompt = f"""
        Generate code for the {component['name']} component of the DeepSeek system.
        
        Component description:
        {component['description']}
        
        Requirements:
        """
        
        # Add requirements
        for req in component.get('requirements', []):
            prompt += f"\n- {req}"
        
        # Add interfaces
        if 'interfaces' in component:
            prompt += "\n\nInterfaces:"
            for interface in component['interfaces']:
                prompt += f"\n- {interface['name']}: {interface['description']}"
        
        # Add dependencies
        if 'dependencies' in component:
            prompt += "\n\nDependencies:"
            for dep in component['dependencies']:
                dep_component = self.components.get(dep, {})
                prompt += f"\n- {dep}: {dep_component.get('description', 'No description')}"
        
        # Add examples if available
        if 'examples' in component:
            prompt += "\n\nExamples:"
            for example in component['examples']:
                prompt += f"\n- {example}"
        
        # Add specific instructions
        prompt += f"""
        
        Please generate complete, production-ready Python code for this component.
        Include proper error handling, logging, and documentation.
        The code should follow PEP 8 style guidelines and include type hints.
        
        For each file, include the full file path in the format:
        ```python:path/to/file.py
        # Code here
        ```
        
        Include unit tests in a separate file with the naming convention test_{filename}.py
        """
        
        return prompt
    
    def _call_deepseek_ai_service(self, prompt: str) -> str:
        """Call the DeepSeek AI service to generate code."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert Python developer specializing in AI systems, distributed computing, and enterprise software architecture."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 4000
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API call failed with status {response.status_code}: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _extract_code_blocks(self, response: str) -> Dict[str, str]:
        """Extract code blocks from the AI response."""
        import re
        
        # Pattern to match code blocks with file paths
        pattern = r"```python:([^\n]+)\n(.*?)```"
        matches = re.finditer(pattern, response, re.DOTALL)
        
        code_blocks = {}
        for match in matches:
            file_path = match.group(1).strip()
            code = match.group(2)
            code_blocks[file_path] = code
        
        return code_blocks
    
    def _save_generated_code(self, component_name: str, code_blocks: Dict[str, str]) -> bool:
        """Save generated code to files."""
        component_dir = self.output_dir / component_name
        os.makedirs(component_dir, exist_ok=True)
        
        try:
            for file_path, code in code_blocks.items():
                # Create directory structure
                full_path = self.output_dir / file_path
                os.makedirs(full_path.parent, exist_ok=True)
                
                # Write code to file
                with open(full_path, "w") as f:
                    f.write(code)
                
                logger.info(f"Saved file: {full_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving code: {str(e)}")
            return False
    
    def _test_generated_code(self, component_name: str) -> bool:
        """Run tests on generated code."""
        # Find test files
        test_files = []
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py") and component_name in root:
                    test_files.append(os.path.join(root, file))
        
        if not test_files:
            logger.warning(f"No test files found for component {component_name}")
            return True
        
        # Run tests
        try:
            for test_file in test_files:
                logger.info(f"Running tests in {test_file}")
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", test_file, "-v"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Tests failed in {test_file}:\n{result.stdout}\n{result.stderr}")
                    return False
                
                logger.info(f"Tests passed in {test_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return False
    
    def generate_all_components(self) -> Dict[str, Any]:
        """Generate all components in dependency order."""
        component_order = self._get_component_order()
        
        for component_name in component_order:
            # Check if dependencies are satisfied
            dependencies = self.dependency_graph.get(component_name, [])
            dependencies_satisfied = all(dep in self.results["generated"] for dep in dependencies)
            
            if not dependencies_satisfied:
                logger.warning(f"Skipping {component_name} due to unsatisfied dependencies")
                continue
            
            self.generate_component(component_name)
        
        return self.results
    
    def generate_components_parallel(self, max_workers: int = 4) -> Dict[str, Any]:
        """Generate components in parallel where possible."""
        component_order = self._get_component_order()
        
        # Queue for components ready to be generated
        ready_queue = queue.Queue()
        
        # Initial components with no dependencies
        for component_name in component_order:
            dependencies = self.dependency_graph.get(component_name, [])
            if not dependencies:
                ready_queue.put(component_name)
        
        # Lock for thread safety
        results_lock = threading.Lock()
        
        def worker():
            while True:
                try:
                    component_name = ready_queue.get(block=False)
                except queue.Empty:
                    break
                
                success = self.generate_component(component_name)
                
                if success:
                    # Check if any components are now ready
                    with results_lock:
                        for next_component in component_order:
                            if next_component in self.results["pending"]:
                                dependencies = self.dependency_graph.get(next_component, [])
                                if all(dep in self.results["generated"] for dep in dependencies):
                                    ready_queue.put(next_component)
                
                ready_queue.task_done()
        
        # Start worker threads
        threads = []
        for _ in range(min(max_workers, len(component_order))):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        
        return self.results
