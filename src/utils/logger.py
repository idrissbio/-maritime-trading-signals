import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys

def setup_logging(log_level="INFO", log_file=None, max_bytes=10*1024*1024, backup_count=5):
    """
    Setup comprehensive logging for the maritime trading system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (defaults to logs/maritime_trading.log)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Default log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"maritime_trading_{timestamp}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error file handler (errors only)
    error_log_file = os.path.join(log_dir, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Log startup message
    logger.info("="*60)
    logger.info("Maritime Trading System - Logging Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log File: {log_file}")
    logger.info("="*60)
    
    return logger

def get_component_logger(component_name):
    """Get a logger for a specific component"""
    return logging.getLogger(f"maritime.{component_name}")

class LogFilter:
    """Custom log filter for specific components"""
    
    def __init__(self, component_name):
        self.component_name = component_name
    
    def filter(self, record):
        return record.name.startswith(f"maritime.{self.component_name}")

def setup_component_logging(component_name, log_level="INFO"):
    """Setup logging for a specific component"""
    
    logger = get_component_logger(component_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create component-specific log file
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"{component_name}_{datetime.now().strftime('%Y%m%d')}.log")
    
    # File handler for component
    handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler.setFormatter(formatter)
    handler.addFilter(LogFilter(component_name))
    
    logger.addHandler(handler)
    
    return logger