#!/bin/bash
# Make the script executable
chmod +x "$SCRIPT"
# Run the script
./"$SCRIPT"

#!/usr/bin/env python3
"""
DeepSeek Core Module - Foundation components for the DeepSeek Integrated System
Provides security, configuration, and memory management functionality
"""

import os
import json
import hashlib
import jwt
import redis
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from typing import Dict, Any, Optional

class SecurityError(Exception):
    """Raised for security violations and data integrity issues"""
    pass

class Config:
    """Centralized configuration management with secure defaults"""
    # Security settings
    JWT_SECRET = os.getenv("DEEPSEEK_JWT_SECRET", hashlib.sha3_256(os.urandom(32)).hexdigest())
    ENCRYPTION_KEY = os.getenv("DEEPSEEK_ENC_KEY", "")
    
    # Storage paths
    CONFIG_DIR = "/etc/deepseek"
    CONFIG_FILE = f"{CONFIG_DIR}/config.key"
    AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "/var/lib/deepseek/ai_model.joblib")
    
    # Service connections
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    VAULT_URL = os.getenv("VAULT_URL", "http://localhost:8200")
    VAULT_TOKEN = os.getenv("VAULT_TOKEN", "")
    
    # System settings
    LOG_LEVEL = os.getenv("DEEPSEEK_LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        return bool(cls.ENCRYPTION_KEY)

class AuthProvider:
    """Authentication and authorization management"""
    
    def __init__(self):
        self.secret = Config.JWT_SECRET
        
    def create_token(self, user_id: str, role: str = "guest") -> str:
        """Create a time-limited JWT token with role-based permissions"""
        return jwt.encode({
            "sub": user_id,
            "role": role,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=15)
        }, self.secret, algorithm="HS256")
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate token and return payload if valid"""
        try:
            return jwt.decode(token, self.secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            raise SecurityError("Invalid or expired token")
    
    def check_permission(self, token: str, required_permission: str) -> bool:
        """Check if token has required permission based on role"""
        try:
            payload = self.validate_token(token)
            role = payload.get("role", "guest")
            
            # Role-based permission mapping
            permissions = {
                "admin": ["create", "read", "update", "delete"],
                "operator": ["read", "update"],
                "guest": ["read"]
            }
            
            return required_permission in permissions.get(role, [])
        except:
            return False

class MemoryController:
    """Secure memory management with encryption and integrity verification"""
    
    def __init__(self):
        if not Config.ENCRYPTION_KEY:
            raise SecurityError("Encryption key not configured")
            
        self.cipher = Fernet(Config.ENCRYPTION_KEY.encode())
        self.redis = redis.from_url(Config.REDIS_URL)
    
    def write(self, key: str, value: str) -> bool:
        """Write encrypted data with integrity protection"""
        try:
            data_package = {
                "value": value,
                "sig": hashlib.sha3_256(value.encode()).hexdigest(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            encrypted = self.cipher.encrypt(json.dumps(data_package).encode())
            self.redis.set(f"mem:{key}", encrypted)
            return True
        except Exception as e:
            raise SecurityError(f"Failed to write data: {str(e)}")
    
    def read(self, key: str) -> str:
        """Read and verify encrypted data"""
        try:
            encrypted_data = self.redis.get(f"mem:{key}")
            if not encrypted_data:
                raise KeyError(f"Key not found: {key}")
                
            data = json.loads(self.cipher.decrypt(encrypted_data))
            
            # Verify data integrity
            if hashlib.sha3_256(data["value"].encode()).hexdigest() != data["sig"]:
                raise SecurityError("Data integrity check failed - possible tampering")
                
            return data["value"]
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            raise SecurityError(f"Failed to read data: {str(e)}")
