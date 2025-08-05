"""
Configuration management for nanook-curator.

This module provides centralized configuration management with environment variable
handling, validation, and default values for the nanook-curator system.
"""

import os
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from dotenv import load_dotenv


class Configuration(BaseModel):
    """
    Configuration class for nanook-curator system.
    
    Handles API keys, search parameters, quality thresholds, and other settings
    with environment variable support and validation.
    """
    
    # API Configuration
    youtube_api_key: str = Field(..., description="YouTube Data API v3 key")
    openai_api_key: str = Field(..., description="OpenAI API key for script generation")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model for script generation")
    
    # Content Discovery Settings
    max_videos: int = Field(default=10, ge=1, le=50, description="Maximum videos to analyze per search")
    days_back: int = Field(default=7, ge=1, le=30, description="Days back to search for videos")
    quality_threshold: float = Field(default=80.0, ge=0.0, le=100.0, description="Quality score threshold")
    min_quality_videos: int = Field(default=1, ge=1, le=10, description="Minimum quality videos required")
    max_search_attempts: int = Field(default=3, ge=1, le=5, description="Maximum search refinement attempts")
    
    # Search Configuration
    default_search_keywords: List[str] = Field(
        default=["AI news", "AI tools", "AI agents", "artificial intelligence", "machine learning"],
        description="Default search keywords for video discovery"
    )
    
    # Script Generation Settings
    target_word_count_min: int = Field(default=750, ge=100, le=2000, description="Minimum script word count")
    target_word_count_max: int = Field(default=1500, ge=500, le=3000, description="Maximum script word count")
    script_language: str = Field(default="en", description="Language code for script generation")
    
    # Storage Configuration
    results_storage_path: Path = Field(default=Path("./output"), description="Directory for storing results")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Path = Field(default=Path("./output/nanook-curator.log"), description="Log file path")
    
    # Development Settings
    debug: bool = Field(default=False, description="Enable debug mode")
    mock_apis: bool = Field(default=False, description="Use mock API responses for testing")
    
    # Proxy Configuration
    proxy_username: Optional[str] = Field(default=None, description="Proxy username for transcript fetching")
    proxy_password: Optional[str] = Field(default=None, description="Proxy password for transcript fetching")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
        
    @field_validator('youtube_api_key')
    @classmethod
    def validate_youtube_api_key(cls, v):
        """Validate YouTube API key format."""
        if not v or v == "your_youtube_api_key_here":
            raise ValueError("YouTube API key must be provided and cannot be the placeholder value")
        if len(v) < 20:
            raise ValueError("YouTube API key appears to be invalid (too short)")
        return v
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_api_key(cls, v):
        """Validate OpenAI API key format."""
        if not v or v == "your_openai_api_key_here":
            raise ValueError("OpenAI API key must be provided and cannot be the placeholder value")
        if not v.startswith(('sk-', 'sk-proj-')):
            raise ValueError("OpenAI API key must start with 'sk-' or 'sk-proj-'")
        return v
    
    @field_validator('openai_model')
    @classmethod
    def validate_openai_model(cls, v):
        """Validate OpenAI model name format."""
        if not v or not v.strip():
            raise ValueError("OpenAI model must be provided")
        
        # List of known valid OpenAI models (as of current knowledge)
        valid_models = [
            'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-4-32k',
            'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-instruct'
        ]
        
        # Allow any model that starts with known prefixes for future compatibility
        valid_prefixes = ['gpt-4', 'gpt-3.5', 'gpt-4o']
        
        model_lower = v.lower().strip()
        if model_lower not in [m.lower() for m in valid_models] and not any(model_lower.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(f"OpenAI model '{v}' may not be valid. Common models: {', '.join(valid_models[:5])}")
        
        return v.strip()
    
    @field_validator('target_word_count_max')
    @classmethod
    def validate_word_count_range(cls, v, info):
        """Ensure max word count is greater than min word count."""
        if info.data and 'target_word_count_min' in info.data and v <= info.data['target_word_count_min']:
            raise ValueError("Maximum word count must be greater than minimum word count")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is one of the standard levels."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    @field_validator('script_language')
    @classmethod
    def validate_language_code(cls, v):
        """Validate language code format (basic ISO 639-1 check)."""
        if len(v) != 2 or not v.isalpha():
            raise ValueError("Language code must be a 2-letter ISO 639-1 code (e.g., 'en', 'es')")
        return v.lower()
    
    @field_validator('default_search_keywords')
    @classmethod
    def validate_search_keywords(cls, v):
        """Validate search keywords are not empty."""
        if not v or len(v) == 0:
            raise ValueError("At least one search keyword must be provided")
        # Remove empty strings and strip whitespace
        cleaned_keywords = [keyword.strip() for keyword in v if keyword.strip()]
        if not cleaned_keywords:
            raise ValueError("Search keywords cannot be empty after cleaning")
        return cleaned_keywords
    
    def __init__(self, **data):
        """Initialize configuration with environment variable loading."""
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Override with environment variables
        env_data = self._load_from_environment()
        data.update(env_data)
        
        super().__init__(**data)
        
        # Ensure storage directories exist
        self._ensure_directories()
    
    def _load_from_environment(self) -> dict:
        """Load configuration values from environment variables."""
        env_mapping = {
            'youtube_api_key': 'YOUTUBE_API_KEY',
            'openai_api_key': 'OPENAI_API_KEY',
            'openai_model': 'OPENAI_MODEL',
            'max_videos': 'MAX_VIDEOS',
            'days_back': 'DAYS_BACK',
            'quality_threshold': 'QUALITY_THRESHOLD',
            'min_quality_videos': 'MIN_QUALITY_VIDEOS',
            'max_search_attempts': 'MAX_SEARCH_ATTEMPTS',
            'default_search_keywords': 'DEFAULT_SEARCH_KEYWORDS',
            'target_word_count_min': 'TARGET_WORD_COUNT_MIN',
            'target_word_count_max': 'TARGET_WORD_COUNT_MAX',
            'script_language': 'SCRIPT_LANGUAGE',
            'results_storage_path': 'RESULTS_STORAGE_PATH',
            'log_level': 'LOG_LEVEL',
            'log_file': 'LOG_FILE',
            'debug': 'DEBUG',
            'mock_apis': 'MOCK_APIS',
            'proxy_username': 'PROXY_USERNAME',
            'proxy_password': 'PROXY_PASSWORD',
        }
        
        env_data = {}
        for field_name, env_var in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Handle special cases for type conversion
                if field_name in ['max_videos', 'days_back', 'min_quality_videos', 'max_search_attempts', 
                                'target_word_count_min', 'target_word_count_max']:
                    try:
                        env_data[field_name] = int(env_value)
                    except ValueError:
                        raise ValueError(f"Environment variable {env_var} must be an integer")
                elif field_name == 'quality_threshold':
                    try:
                        env_data[field_name] = float(env_value)
                    except ValueError:
                        raise ValueError(f"Environment variable {env_var} must be a number")
                elif field_name in ['debug', 'mock_apis']:
                    env_data[field_name] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif field_name == 'default_search_keywords':
                    # Split comma-separated keywords
                    env_data[field_name] = [kw.strip() for kw in env_value.split(',') if kw.strip()]
                elif field_name in ['results_storage_path', 'log_file']:
                    env_data[field_name] = Path(env_value)
                else:
                    env_data[field_name] = env_value
        
        return env_data
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        # Create results storage directory
        self.results_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_config(cls, config_file: Optional[Path] = None) -> 'Configuration':
        """
        Load configuration from environment variables and optional config file.
        
        Args:
            config_file: Optional path to .env file to load
            
        Returns:
            Configuration instance
            
        Raises:
            ValidationError: If configuration validation fails
            FileNotFoundError: If specified config file doesn't exist
        """
        if config_file and not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Load from specified file if provided
        if config_file:
            load_dotenv(config_file)
        
        try:
            return cls()
        except ValidationError as e:
            raise e
    
    def validate_api_keys(self) -> bool:
        """
        Validate that API keys are properly configured.
        
        Returns:
            True if all API keys are valid, False otherwise
        """
        try:
            # Basic validation - more thorough validation would require API calls
            if not self.youtube_api_key or self.youtube_api_key == "your_youtube_api_key_here":
                return False
            if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
                return False
            return True
        except Exception:
            return False
    
    def get_search_keywords(self, custom_keywords: Optional[List[str]] = None) -> List[str]:
        """
        Get search keywords, with optional custom keywords override.
        
        Args:
            custom_keywords: Optional custom keywords to use instead of defaults
            
        Returns:
            List of search keywords to use
        """
        if custom_keywords:
            return [kw.strip() for kw in custom_keywords if kw.strip()]
        return self.default_search_keywords
    
    def get_quality_settings(self) -> dict:
        """
        Get quality evaluation settings as a dictionary.
        
        Returns:
            Dictionary with quality-related configuration
        """
        return {
            'quality_threshold': self.quality_threshold,
            'min_quality_videos': self.min_quality_videos,
            'max_search_attempts': self.max_search_attempts,
        }
    
    def get_script_settings(self) -> dict:
        """
        Get script generation settings as a dictionary.
        
        Returns:
            Dictionary with script-related configuration
        """
        return {
            'target_word_count_min': self.target_word_count_min,
            'target_word_count_max': self.target_word_count_max,
            'script_language': self.script_language,
        }
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary, excluding sensitive data.
        
        Returns:
            Dictionary representation of configuration (API keys masked)
        """
        config_dict = self.dict()
        # Mask sensitive information
        config_dict['youtube_api_key'] = '***masked***'
        config_dict['openai_api_key'] = '***masked***'
        # Convert Path objects to strings for serialization
        config_dict['results_storage_path'] = str(self.results_storage_path)
        config_dict['log_file'] = str(self.log_file)
        return config_dict


# Global configuration instance
_config_instance: Optional[Configuration] = None


def get_config() -> Configuration:
    """
    Get the global configuration instance.
    
    Returns:
        Configuration instance
        
    Raises:
        RuntimeError: If configuration hasn't been initialized
    """
    global _config_instance
    if _config_instance is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")
    return _config_instance


def init_config(config_file: Optional[Path] = None) -> Configuration:
    """
    Initialize the global configuration instance.
    
    Args:
        config_file: Optional path to .env file to load
        
    Returns:
        Configuration instance
        
    Raises:
        ValidationError: If configuration validation fails
    """
    global _config_instance
    _config_instance = Configuration.load_config(config_file)
    return _config_instance


def reset_config():
    """Reset the global configuration instance (mainly for testing)."""
    global _config_instance
    _config_instance = None