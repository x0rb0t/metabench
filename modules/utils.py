"""
Utility functions for environment variables and logging setup.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Tuple


def resolve_env_var(value: str) -> str:
    """
    Resolve environment variable if value starts with 'env:'.
    
    Args:
        value: String that may be in format 'env:VARIABLE_NAME' or regular value
        
    Returns:
        Resolved environment variable value or original value
        
    Raises:
        ValueError: If environment variable is not found
    """
    if value.startswith('env:'):
        env_var_name = value[4:]  # Remove 'env:' prefix
        env_value = os.getenv(env_var_name)
        if env_value is None:
            raise ValueError(f"Environment variable '{env_var_name}' is not set")
        return env_value
    return value


def setup_logging(log_file: str = None) -> Tuple[logging.Logger, logging.Logger]:
    """Setup dual logging configuration: file-only logger and console-only logger"""
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create unique log filename if none provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"benchmark_log_{timestamp}.log")
    
    # 1. Create FILE-ONLY logger for detailed logging
    file_logger = logging.getLogger('transformation_benchmark_file')
    file_logger.setLevel(logging.DEBUG)
    file_logger.handlers.clear()
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    file_logger.propagate = False
    
    # 2. Create CONSOLE-ONLY logger for emoji output
    console_logger = logging.getLogger('transformation_benchmark_console')
    console_logger.setLevel(logging.INFO)
    console_logger.handlers.clear()
    
    # Console handler with minimal formatting (just the message)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    console_logger.addHandler(console_handler)
    console_logger.propagate = False
    
    return file_logger, console_logger
