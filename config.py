#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
Configuration management for Anti Buzzer Detection System
Loads settings from .env file
"""

import os
from pathlib import Path
from typing import Optional, Union
import logging

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[!] python-dotenv not installed. Install with: pip install python-dotenv")

class Config:
    """Configuration class that loads settings from .env file"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory and parent directories
        """
        if env_file:
            self.env_path = Path(env_file)
        else:
            # Look for .env in current directory and parent directories
            current_dir = Path.cwd()
            for path in [current_dir] + list(current_dir.parents):
                env_path = path / ".env"
                if env_path.exists():
                    self.env_path = env_path
                    break
            else:
                self.env_path = current_dir / ".env"
        
        # Load environment variables
        if DOTENV_AVAILABLE and self.env_path.exists():
            load_dotenv(self.env_path)
            print(f"[CONFIG] Loaded configuration from: {self.env_path}")
        elif self.env_path.exists():
            print(f"[CONFIG] Found .env file but python-dotenv not available: {self.env_path}")
        else:
            print(f"[CONFIG] No .env file found. Using default values.")
    
    def get(self, key: str, default: Union[str, int, float, bool] = None, cast_type: type = str):
        """Get configuration value with type casting
        
        Args:
            key: Environment variable name
            default: Default value if not found
            cast_type: Type to cast the value to (str, int, float, bool)
            
        Returns:
            Configuration value cast to specified type
        """
        value = os.getenv(key)
        
        if value is None:
            return default
            
        # Type casting
        try:
            if cast_type == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif cast_type == int:
                return int(value)
            elif cast_type == float:
                return float(value)
            else:
                return str(value)
        except (ValueError, TypeError):
            print(f"[CONFIG] Warning: Cannot cast {key}='{value}' to {cast_type.__name__}, using default: {default}")
            return default
    
    # Database Configuration
    @property
    def database_type(self) -> str:
        return self.get('DATABASE_TYPE', 'sqlite3')
    
    @property
    def database_path(self) -> str:
        return self.get('DATABASE_PATH', 'buzzer_detection.db')
    
    @property
    def database_host(self) -> str:
        return self.get('DATABASE_HOST', 'localhost')
    
    @property
    def database_port(self) -> int:
        return self.get('DATABASE_PORT', 5432, int)
    
    @property
    def database_name(self) -> str:
        return self.get('DATABASE_NAME', 'buzzer_detection')
    
    @property
    def database_user(self) -> str:
        return self.get('DATABASE_USER', 'postgres')
    
    @property
    def database_password(self) -> str:
        return self.get('DATABASE_PASSWORD', 'password')
    
    # MySQL Configuration
    @property
    def mysql_host(self) -> str:
        return self.get('MYSQL_HOST', 'localhost')
    
    @property
    def mysql_port(self) -> int:
        return self.get('MYSQL_PORT', 3306, int)
    
    @property
    def mysql_user(self) -> str:
        return self.get('MYSQL_USER', 'root')
    
    @property
    def mysql_password(self) -> str:
        return self.get('MYSQL_PASSWORD', 'password')
    
    @property
    def mysql_database(self) -> str:
        return self.get('MYSQL_DATABASE', 'buzzer_detection')
    
    # Server Configuration
    @property
    def server_host(self) -> str:
        return self.get('SERVER_HOST', '0.0.0.0')
    
    @property
    def server_port(self) -> int:
        return self.get('SERVER_PORT', 8080, int)
    
    @property
    def gin_mode(self) -> str:
        return self.get('GIN_MODE', 'debug')
    
    # API Configuration
    @property
    def api_version(self) -> str:
        return self.get('API_VERSION', 'v1')
    
    @property
    def cors_allow_origins(self) -> str:
        return self.get('CORS_ALLOW_ORIGINS', '*')
    
    @property
    def cors_allow_methods(self) -> str:
        return self.get('CORS_ALLOW_METHODS', 'GET,POST,PUT,DELETE,OPTIONS')
    
    @property
    def cors_allow_headers(self) -> str:
        return self.get('CORS_ALLOW_HEADERS', 'Content-Type,Authorization')
    
    # Pagination Configuration
    @property
    def default_page_size(self) -> int:
        return self.get('DEFAULT_PAGE_SIZE', 20, int)
    
    @property
    def max_page_size(self) -> int:
        return self.get('MAX_PAGE_SIZE', 100, int)
    
    # Security Configuration
    @property
    def jwt_secret(self) -> str:
        return self.get('JWT_SECRET', 'your-secret-key-here-change-in-production')
    
    @property
    def api_rate_limit(self) -> int:
        return self.get('API_RATE_LIMIT', 100, int)
    
    @property
    def rate_limit_window(self) -> int:
        return self.get('RATE_LIMIT_WINDOW', 60, int)
    
    # AI Analysis Configuration
    @property
    def embed_model_name(self) -> str:
        return self.get('EMBED_MODEL_NAME', 'all-MiniLM-L6-v2')
    
    @property
    def use_embeddings(self) -> bool:
        return self.get('USE_EMBEDDINGS', True, bool)
    
    @property
    def ai_analysis_enabled(self) -> bool:
        return self.get('AI_ANALYSIS_ENABLED', True, bool)
    
    # Scraping Configuration
    @property
    def default_scraping_mode(self) -> str:
        return self.get('DEFAULT_SCRAPING_MODE', 'playwright')
    
    @property
    def max_posts_default(self) -> int:
        return self.get('MAX_POSTS_DEFAULT', 2000, int)
    
    @property
    def default_limit(self) -> int:
        return self.get('DEFAULT_LIMIT', 1000, int)
    
    @property
    def headful_mode(self) -> bool:
        return self.get('HEADFUL_MODE', False, bool)
    
    @property
    def use_firefox(self) -> bool:
        return self.get('USE_FIREFOX', True, bool)
    
    # Model Configuration
    @property
    def model_dir(self) -> str:
        return self.get('MODEL_DIR', 'models')
    
    @property
    def training_history_dir(self) -> str:
        return self.get('TRAINING_HISTORY_DIR', 'results')
    
    @property
    def save_model(self) -> bool:
        return self.get('SAVE_MODEL', True, bool)
    
    # Performance Configuration
    @property
    def n_jobs(self) -> int:
        return self.get('N_JOBS', -1, int)
    
    @property
    def random_state(self) -> int:
        return self.get('RANDOM_STATE', 42, int)
    
    @property
    def cross_val_folds(self) -> int:
        return self.get('CROSS_VAL_FOLDS', 5, int)
    
    # Logging Configuration
    @property
    def log_level(self) -> str:
        return self.get('LOG_LEVEL', 'info').upper()
    
    @property
    def log_format(self) -> str:
        return self.get('LOG_FORMAT', 'json')
    
    def get_database_config(self) -> dict:
        """Get database configuration as dictionary"""
        if self.database_type.lower() == 'mysql':
            return {
                'type': 'mysql',
                'host': self.mysql_host,
                'port': self.mysql_port,
                'user': self.mysql_user,
                'password': self.mysql_password,
                'database': self.mysql_database
            }
        else:
            return {
                'type': 'sqlite3',
                'path': self.database_path
            }
    
    def setup_logging(self):
        """Setup logging configuration"""
        level = getattr(logging, self.log_level, logging.INFO)
        
        if self.log_format.lower() == 'json':
            # JSON format for structured logging
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
            )
        else:
            # Simple format
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
            )
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='[%(asctime)s] %(levelname)s: %(message)s' if self.log_format != 'json' else None,
            handlers=[logging.StreamHandler()]
        )
        
        return logging.getLogger(__name__)

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance"""
    return config

# For backward compatibility
def info(msg): 
    logger = logging.getLogger(__name__)
    logger.info(msg)

def warn(msg): 
    logger = logging.getLogger(__name__)
    logger.warning(msg)

def error(msg): 
    logger = logging.getLogger(__name__)
    logger.error(msg)