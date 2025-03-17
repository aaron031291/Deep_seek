
from .config import Config
from .security import SecurityProvider, require_permission
from .storage import StorageProvider
from .telemetry import TelemetryProvider, Metrics, Logger, Tracer
from .errors import (
    DeepSeekError, ValidationError, AuthenticationError, AuthorizationError,
    NotFoundError, ConflictError, RateLimitError, ConfigurationError,
    StorageError, SecurityError, AIError
)

__all__ = [
    'Config', 
    'SecurityProvider', 'require_permission',
    'StorageProvider',
    'TelemetryProvider', 'Metrics', 'Logger', 'Tracer',
    'DeepSeekError', 'ValidationError', 'AuthenticationError', 'AuthorizationError',
    'NotFoundError', 'ConflictError', 'RateLimitError', 'ConfigurationError',
    'StorageError', 'SecurityError', 'AIError'
]

"""
Error hierarchy for the DeepSeek system.

This module defines a consistent error hierarchy that is used throughout
the DeepSeek system to provide clear error messages and proper error handling.

Example:
    try:
        result = perform_operation()
    except ValidationError as e:
        # Handle validation error
        print(f"Validation error: {e.message}, field: {e.field}")
    except AuthenticationError as e:
        # Handle authentication error
        print(f"Authentication error: {e.message}")
    except DeepSeekError as e:
        # Handle any DeepSeek error
        print(f"Error: {e.message}, code: {e.code}, status: {e.status_code}")
"""

class DeepSeekError(Exception):
    """Base exception for all DeepSeek errors."""
    
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500):
        """Initialize a DeepSeekError.
        
        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            status_code: HTTP status code for API responses
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)
    
    def to_dict(self):
        """Convert error to a dictionary for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "status": self.status_code
        }


class ValidationError(DeepSeekError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None):
        """Initialize a ValidationError.
        
        Args:
            message: Human-readable error message
            field: The field that failed validation
        """
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400
        )
        self.field = field
        
    def to_dict(self):
        """Convert error to a dictionary for API responses."""
        result = super().to_dict()
        if self.field:
            result["field"] = self.field
        return result


class AuthenticationError(DeepSeekError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        """Initialize an AuthenticationError.
        
        Args:
            message: Human-readable error message
        """
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationError(DeepSeekError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Not authorized to perform this action"):
        """Initialize an AuthorizationError.
        
        Args:
            message: Human-readable error message
        """
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=403
        )


class NotFoundError(DeepSeekError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, message: str = "Resource not found", resource_type: str = None, resource_id: str = None):
        """Initialize a NotFoundError.
        
        Args:
            message: Human-readable error message
            resource_type: Type of resource that was not found
            resource_id: ID of resource that was not found
        """
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=404
        )
        self.resource_type = resource_type
        self.resource_id = resource_id
        
    def to_dict(self):
        """Convert error to a dictionary for API responses."""
        result = super().to_dict()
        if self.resource_type:
            result["resource_type"] = self.resource_type
        if self.resource_id:
            result["resource_id"] = self.resource_id
        return result


class ConflictError(DeepSeekError):
    """Raised when there is a conflict with the current state of a resource."""
    
    def __init__(self, message: str = "Resource conflict"):
        """Initialize a ConflictError.
        
        Args:
            message: Human-readable error message
        """
        super().__init__(
            message=message,
            code="CONFLICT",
            status_code=409
        )


class RateLimitError(DeepSeekError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        """Initialize a RateLimitError.
        
        Args:
            message: Human-readable error message
            retry_after: Seconds until the client can retry
        """
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=429
        )
        self.retry_after = retry_after
        
    def to_dict(self):
        """Convert error to a dictionary for API responses."""
        result = super().to_dict()
        if self.retry_after:
            result["retry_after"] = self.retry_after
        return result


class ConfigurationError(DeepSeekError):
    """Raised when there is an issue with the system configuration."""
    
    def __init__(self, message: str = "Invalid configuration"):
        """Initialize a ConfigurationError.
        
        Args:
            message: Human-readable error message
        """
        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            status_code=500
        )


class StorageError(DeepSeekError):
    """Raised when there is an issue with storage operations."""
    
    def __init__(self, message: str = "Storage operation failed", operation: str = None):
        """Initialize a StorageError.
        
        Args:
            message: Human-readable error message
            operation: The storage operation that failed
        """
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            status_code=500
        )
        self.operation = operation
        
    def to_dict(self):
        """Convert error to a dictionary for API responses."""
        result = super().to_dict()
        if self.operation:
            result["operation"] = self.operation
        return result


class SecurityError(DeepSeekError):
    """Raised for security violations and data integrity issues."""
    
    def __init__(self, message: str = "Security violation detected"):
        """Initialize a SecurityError.
        
        Args:
            message: Human-readable error message
        """
        super().__init__(
            message=message,
            code="SECURITY_ERROR",
            status_code=500
        )


class AIError(DeepSeekError):
    """Raised when there is an issue with AI operations."""
    
    def __init__(self, message: str = "AI operation failed", model: str = None):
        """Initialize an AIError.
        
        Args:
            message: Human-readable error message
            model: The AI model that failed
        """
        super().__init__(
            message=message,
            code="AI_ERROR",
            status_code=500
        )
        self.model = model
        
    def to_dict(self):
        """Convert error to a dictionary for API responses."""
        result = super().to_dict()
        if self.model:
            result["model"] = self.model
        return result

"""
Configuration management with secure defaults, validation, and hot-reloading.

This module provides a centralized configuration system that supports:
- Loading from environment variables
- Loading from JSON/YAML configuration files
- Hot-reloading of configuration changes
- Type conversion and validation
- Secure handling of sensitive values
- Configuration change notifications

Example:
    # Initialize configuration
    from deepseek.core import Config
    Config.initialize("/etc/deepseek/config.json")
    
    # Get configuration values
    redis_url = Config.get("REDIS_URL")
    worker_count = Config.get("WORKER_THREADS", 4)  # With default
    
    # Set configuration values
    Config.set("LOG_LEVEL", "DEBUG")
    
    # Watch for configuration changes
    def on_log_level_change(key, value):
        print(f"Log level changed to {value}")
    
    Config.watch("LOG_LEVEL", on_log_level_change)
