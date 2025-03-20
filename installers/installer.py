#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import logging
import shutil
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import tempfile
import zipfile
import tarfile
import requests
import hashlib

from deepseek.platform.compatibility import PlatformManager
from deepseek.core import Config, SecurityProvider
logger
 = logging.getLogger(__name__)

class Installer:
    """Cross-platform installer for DeepSeek."""
    
    def __init__(self):
        """Initialize installer with platform-specific functionality."""
        self.platform_manager = PlatformManager()
        self.platform = self.platform_manager.get_platform_name()
        self.security = SecurityProvider()
        
        # Create platform-specific installer
        if self.platform == PlatformManager.WINDOWS:
            self.platform_installer = WindowsInstaller(self.platform_manager)
        elif self.platform == PlatformManager.MACOS:
            self.platform_installer = MacOSInstaller(self.platform_manager)
        elif self.platform == PlatformManager.LINUX:
            self.platform_installer = LinuxInstaller(self.platform_manager)
        else:
            raise ValueError(f"Unsupported platform: {self.platform}")
    
    def install(self, version: str = "latest", install_dir: Optional[Path] = None) -> bool:
        """Install DeepSeek on the current system.
        
        Args:
            version: Version to install, or "latest" for the latest version
            install_dir: Custom installation directory (platform-specific default if None)
            
        Returns:
            True if installation was successful
        """
        logger.info(f"Installing DeepSeek version {version}")
        
        # Get package URL and checksum
        package_info = self._get_package_info(version)
        
        # Download package
        package_path = self._download_package(package_info["url"], package_info["checksum"])
        
        # Install package
        result = self.platform_installer.install(package_path, install_dir)
