import os
import sys
import platform
import tempfile
import shutil
import subprocess
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import threading
import multiprocessing

logger = logging.getLogger(__name__)

class PlatformManager:
    """Manages platform-specific functionality and provides a unified interface."""
    
    # Platform identifiers
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    
    def __init__(self):
        """Initialize platform manager and detect current platform."""
        self.platform = self._detect_platform()
        self.platform_handlers = {
            self.WINDOWS: WindowsHandler(),
            self.MACOS: MacOSHandler(),
            self.LINUX: LinuxHandler()
        }
        self.current_handler = self.platform_handlers[self.platform]
        
        logger.info(f"Initialized platform manager for {self.platform}")
    
    def _detect_platform(self) -> str:
        """Detect the current operating system."""
        system = platform.system().lower()
        
        if system == "darwin":
            return self.MACOS
        elif system == "windows":
            return self.WINDOWS
        elif system == "linux":
            return self.LINUX
        else:
            logger.warning(f"Unknown platform: {system}, defaulting to Linux compatibility")
            return self.LINUX
    
    def get_data_directory(self) -> Path:
        """Get the platform-specific data directory for DeepSeek."""
        return self.current_handler.get_data_directory()
    
    def get_config_directory(self) -> Path:
        """Get the platform-specific configuration directory for DeepSeek."""
        return self.current_handler.get_config_directory()
    
    def get_temp_directory(self) -> Path:
        """Get the platform-specific temporary directory for DeepSeek."""
        return self.current_handler.get_temp_directory()
    
    def get_log_directory(self) -> Path:
        """Get the platform-specific log directory for DeepSeek."""
        return self.current_handler.get_log_directory()
    
    def execute_command(self, command: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute a system command with platform-specific adjustments."""
        return self.current_handler.execute_command(command, **kwargs)
    
    def create_process(self, target: Callable, args: tuple = (), **kwargs) -> Any:
        """Create a platform-optimized process."""
        return self.current_handler.create_process(target, args, **kwargs)
    
    def get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        return self.current_handler.get_available_memory()
    
    def get_available_cores(self) -> int:
        """Get number of available CPU cores."""
        return self.current_handler.get_available_cores()
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get information about available GPUs."""
        return self.current_handler.get_gpu_info()
    
    def is_admin(self) -> bool:
        """Check if the current process has administrative privileges."""
        return self.current_handler.is_admin()
    
    def elevate_privileges(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a command with elevated privileges (if possible)."""
        return self.current_handler.elevate_privileges(command)
    
    def setup_ipc(self, name: str) -> Any:
        """Set up inter-process communication mechanism."""
        return self.current_handler.setup_ipc(name)
    
    def get_platform_name(self) -> str:
        """Get the current platform name."""
        return self.platform


class PlatformHandler:
    """Base class for platform-specific handlers."""
    
    def get_data_directory(self) -> Path:
        """Get the platform-specific data directory for DeepSeek."""
        raise NotImplementedError
    
    def get_config_directory(self) -> Path:
        """Get the platform-specific configuration directory for DeepSeek."""
        raise NotImplementedError
    
    def get_temp_directory(self) -> Path:
        """Get the platform-specific temporary directory for DeepSeek."""
        return Path(tempfile.gettempdir()) / "deepseek"
    
    def get_log_directory(self) -> Path:
        """Get the platform-specific log directory for DeepSeek."""
        raise NotImplementedError
    
    def execute_command(self, command: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute a system command with platform-specific adjustments."""
        return subprocess.run(command, **kwargs)
    
    def create_process(self, target: Callable, args: tuple = (), **kwargs) -> Any:
        """Create a platform-optimized process."""
        return multiprocessing.Process(target=target, args=args, **kwargs)
    
    def get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        import psutil
        return psutil.virtual_memory().available
    
    def get_available_cores(self) -> int:
        """Get number of available CPU cores."""
        return multiprocessing.cpu_count()
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get information about available GPUs."""
        try:
            import torch
            gpus = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory,
                        "compute_capability": torch.cuda.get_device_capability(i)
                    })
            return gpus
        except ImportError:
            logger.warning("PyTorch not available, cannot get GPU info")
            return []
    
    def is_admin(self) -> bool:
        """Check if the current process has administrative privileges."""
        raise NotImplementedError
    
    def elevate_privileges(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a command with elevated privileges (if possible)."""
        raise NotImplementedError
    
    def setup_ipc(self, name: str) -> Any:
        """Set up inter-process communication mechanism."""
        raise NotImplementedError


class WindowsHandler(PlatformHandler):
    """Windows-specific platform handler."""
    
    def get_data_directory(self) -> Path:
        """Get the Windows-specific data directory for DeepSeek."""
        return Path(os.environ.get("APPDATA", "")) / "DeepSeek"
    
    def get_config_directory(self) -> Path:
        """Get the Windows-specific configuration directory for DeepSeek."""
        return Path(os.environ.get("LOCALAPPDATA", "")) / "DeepSeek" / "Config"
    
    def get_log_directory(self) -> Path:
        """Get the Windows-specific log directory for DeepSeek."""
        return Path(os.environ.get("LOCALAPPDATA", "")) / "DeepSeek" / "Logs"
    
    def execute_command(self, command: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute a system command with Windows-specific adjustments."""
        # Windows needs shell=True for some commands
        if "shell" not in kwargs and any(">" in arg for arg in command):
            kwargs["shell"] = True
        
        return subprocess.run(command, **kwargs)
    
    def is_admin(self) -> bool:
        """Check if the current process has administrative privileges on Windows."""
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    
    def elevate_privileges(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a command with elevated privileges on Windows."""
        if self.is_admin():
            return self.execute_command(command)
        
        # Use PowerShell to elevate privileges
        ps_command = [
            "powershell.exe", 
            "-Command", 
            f"Start-Process -FilePath '{command[0]}' -ArgumentList '{' '.join(command[1:])}' -Verb RunAs -Wait"
        ]
        return self.execute_command(ps_command)
    
    def setup_ipc(self, name: str) -> Any:
        """Set up inter-process communication mechanism for Windows."""
        from multiprocessing.connection import Listener, Client
        address = fr'\\.\pipe\{name}'
        
        class WindowsIPC:
            def create_server(self):
                return Listener(address)
                
            def connect_client(self):
                return Client(address)
        
        return WindowsIPC()


class MacOSHandler(PlatformHandler):
    """macOS-specific platform handler."""
    
    def get_data_directory(self) -> Path:
        """Get the macOS-specific data directory for DeepSeek."""
        return Path.home() / "Library" / "Application Support" / "DeepSeek"
    
    def get_config_directory(self) -> Path:
        """Get the macOS-specific configuration directory for DeepSeek."""
        return Path.home() / "Library" / "Preferences" / "DeepSeek"
    
    def get_log_directory(self) -> Path:
        """Get the macOS-specific log directory for DeepSeek."""
        return Path.home() / "Library" / "Logs" / "DeepSeek"
    
    def is_admin(self) -> bool:
        """Check if the current process has administrative privileges on macOS."""
        try:
            return os.geteuid() == 0
        except AttributeError:
            return False
    
    def elevate_privileges(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a command with elevated privileges on macOS."""
        if self.is_admin():
            return self.execute_command(command)
        
        # Use sudo to elevate privileges
        sudo_command = ["sudo"] + command
        return self.execute_command(sudo_command)
    
    def setup_ipc(self, name: str) -> Any:
        """Set up inter-process communication mechanism for macOS."""
        from multiprocessing.connection import Listener, Client
        address = f'/tmp/{name}'
        
        class MacOSIPC:
            def create_server(self):
                return Listener(address)
                
            def connect_client(self):
                return Client(address)
        
        return MacOSIPC()


class LinuxHandler(PlatformHandler):
    """Linux-specific platform handler."""
    
    def get_data_directory(self) -> Path:
        """Get the Linux-specific data directory for DeepSeek."""
        xdg_data_home = os.environ.get("XDG_DATA_HOME", "")
        if xdg_data_home:
            return Path(xdg_data_home) / "deepseek"
        return Path.home() / ".local" / "share" / "deepseek"
    
    def get_config_directory(self) -> Path:
        """Get the Linux-specific configuration directory for DeepSeek."""
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME", "")
        if xdg_config_home:
            return Path(xdg_config_home) / "deepseek"
        return Path.home() / ".config" / "deepseek"
    
    def get_log_directory(self) -> Path:
        """Get the Linux-specific log directory for DeepSeek."""
        xdg_state_home = os.environ.get("XDG_STATE_HOME", "")
        if xdg_state_home:
            return Path(xdg_state_home) / "deepseek" / "logs"
        return Path.home() / ".local" / "state" / "deepseek" / "logs"
    
    def is_admin(self) -> bool:
        """Check if the current process has administrative privileges on Linux."""
        try:
            return os.geteuid() == 0
        except AttributeError:
            return False
    
    def elevate_privileges(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a command with elevated privileges on Linux."""
        if self.is_admin():
            return self.execute_command(command)
        
        # Use sudo to elevate privileges
        sudo_command = ["sudo"] + command
        return self.execute_command(sudo_command)
    
    def setup_ipc(self, name: str) -> Any:
        """Set up inter-process communication mechanism for Linux."""
        from multiprocessing.connection import Listener, Client
        address = f'/tmp/{name}'
        
        class LinuxIPC:
            def create_server(self):
                return Listener(address)
                
            def connect_client(self):
                return Client(address)
        
        return LinuxIPC()
