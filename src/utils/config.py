import json
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "database.host")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> bool:
    """
    Set nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        value: Value to set
        
    Returns:
        True if successful, False otherwise
    """
    
    keys = key_path.split('.')
    current = config
    
    try:
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        return True
        
    except Exception as e:
        logger.error(f"Error setting config value {key_path}: {e}")
        return False

def validate_config(config: Dict[str, Any]) -> tuple[bool, list]:
    """
    Validate configuration structure and required fields
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    
    errors = []
    required_sections = [
        "data_sources",
        "risk_management", 
        "alerts",
        "trading"
    ]
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate data_sources section
    if "data_sources" in config:
        data_config = config["data_sources"]
        
        if "update_interval_minutes" in data_config:
            interval = data_config["update_interval_minutes"]
            if not isinstance(interval, int) or interval < 1:
                errors.append("update_interval_minutes must be positive integer")
    
    # Validate risk_management section
    if "risk_management" in config:
        risk_config = config["risk_management"]
        
        if "account_balance" in risk_config:
            balance = risk_config["account_balance"]
            if not isinstance(balance, (int, float)) or balance <= 0:
                errors.append("account_balance must be positive number")
        
        if "risk_per_trade" in risk_config:
            risk = risk_config["risk_per_trade"]
            if not isinstance(risk, (int, float)) or risk <= 0 or risk > 1:
                errors.append("risk_per_trade must be between 0 and 1")
        
        if "max_daily_trades" in risk_config:
            trades = risk_config["max_daily_trades"]
            if not isinstance(trades, int) or trades < 1:
                errors.append("max_daily_trades must be positive integer")
    
    # Validate trading section
    if "trading" in config:
        trading_config = config["trading"]
        
        if "min_signal_confidence" in trading_config:
            confidence = trading_config["min_signal_confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                errors.append("min_signal_confidence must be between 0 and 1")
        
        if "symbols" in trading_config:
            symbols = trading_config["symbols"]
            if not isinstance(symbols, list) or len(symbols) == 0:
                errors.append("symbols must be non-empty list")
    
    return len(errors) == 0, errors

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def load_environment_overrides() -> Dict[str, Any]:
    """
    Load configuration overrides from environment variables
    
    Returns:
        Configuration overrides dictionary
    """
    
    overrides = {}
    
    # Map environment variables to config paths
    env_mappings = {
        "ACCOUNT_BALANCE": "risk_management.account_balance",
        "RISK_PER_TRADE": "risk_management.risk_per_trade", 
        "MAX_DAILY_TRADES": "risk_management.max_daily_trades",
        "MIN_SIGNAL_CONFIDENCE": "trading.min_signal_confidence",
        "UPDATE_INTERVAL_MINUTES": "data_sources.update_interval_minutes",
        "MOCK_MODE": "data_sources.mock_mode"
    }
    
    for env_var, config_path in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Convert to appropriate type
            try:
                if env_var in ["ACCOUNT_BALANCE", "RISK_PER_TRADE", "MIN_SIGNAL_CONFIDENCE"]:
                    value = float(env_value)
                elif env_var in ["MAX_DAILY_TRADES", "UPDATE_INTERVAL_MINUTES"]:
                    value = int(env_value)
                elif env_var == "MOCK_MODE":
                    value = env_value.lower() in ("true", "1", "yes")
                else:
                    value = env_value
                
                set_config_value(overrides, config_path, value)
                logger.info(f"Environment override: {config_path} = {value}")
                
            except ValueError:
                logger.warning(f"Invalid environment value for {env_var}: {env_value}")
    
    return overrides

def create_default_config_files():
    """Create default configuration files if they don't exist"""
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Main settings file
    settings_file = config_dir / "settings.json"
    if not settings_file.exists():
        default_settings = {
            "data_sources": {
                "mock_mode": True,
                "datalastic_api_key": None,
                "twelve_data_api_key": None,
                "update_interval_minutes": 5,
                "cache_duration_seconds": 300
            },
            "risk_management": {
                "account_balance": 100000,
                "risk_per_trade": 0.01,
                "max_daily_trades": 15,
                "max_positions": 10,
                "max_correlated_positions": 3,
                "max_tier3_allocation": 0.3
            },
            "alerts": {
                "email_enabled": True,
                "sms_enabled": True,
                "discord_enabled": True,
                "min_tier_for_sms": 2,
                "rate_limit_minutes": 5
            },
            "trading": {
                "min_signal_confidence": 0.65,
                "symbols": ["CL", "NG", "GC", "SI", "HG"],
                "ports": ["singapore", "houston", "rotterdam", "fujairah"],
                "max_signal_age_hours": 24
            },
            "logging": {
                "level": "INFO",
                "file_rotation_mb": 10,
                "backup_count": 5
            }
        }
        
        save_config(default_settings, str(settings_file))
        logger.info(f"Created default settings file: {settings_file}")
    
    # Markets configuration file
    markets_file = config_dir / "markets.json"
    if not markets_file.exists():
        markets_config = {
            "futures": {
                "CL": {
                    "name": "Crude Oil",
                    "exchange": "NYMEX",
                    "point_value": 1000,
                    "tick_size": 0.01,
                    "margin": 5000,
                    "trading_hours": "Sunday - Friday: 6:00 PM - 5:00 PM ET",
                    "related_ports": ["singapore", "houston", "fujairah"]
                },
                "NG": {
                    "name": "Natural Gas",
                    "exchange": "NYMEX",
                    "point_value": 10000,
                    "tick_size": 0.001,
                    "margin": 3000,
                    "trading_hours": "Sunday - Friday: 6:00 PM - 5:00 PM ET",
                    "related_vessels": ["lng_carrier"]
                },
                "GC": {
                    "name": "Gold",
                    "exchange": "COMEX",
                    "point_value": 100,
                    "tick_size": 0.10,
                    "margin": 8000,
                    "trading_hours": "Sunday - Friday: 6:00 PM - 5:00 PM ET",
                    "related_ports": []
                },
                "SI": {
                    "name": "Silver",
                    "exchange": "COMEX", 
                    "point_value": 5000,
                    "tick_size": 0.005,
                    "margin": 10000,
                    "trading_hours": "Sunday - Friday: 6:00 PM - 5:00 PM ET",
                    "related_ports": []
                },
                "HG": {
                    "name": "Copper",
                    "exchange": "COMEX",
                    "point_value": 25000,
                    "tick_size": 0.0005,
                    "margin": 4000,
                    "trading_hours": "Sunday - Friday: 6:00 PM - 5:00 PM ET",
                    "related_ports": ["singapore", "rotterdam"]
                }
            },
            "correlations": {
                "CL": {"HO": 0.85, "RB": 0.80, "NG": 0.25},
                "NG": {"HO": 0.45, "CL": 0.25},
                "GC": {"SI": 0.75},
                "SI": {"GC": 0.75},
                "HG": {"CL": 0.30}
            }
        }
        
        save_config(markets_config, str(markets_file))
        logger.info(f"Created default markets file: {markets_file}")

def get_full_config(config_path: str = None) -> Dict[str, Any]:
    """
    Get complete configuration with environment overrides
    
    Args:
        config_path: Path to main config file
        
    Returns:
        Complete configuration dictionary
    """
    
    # Create default files if needed
    create_default_config_files()
    
    # Load main config
    if config_path is None:
        config_path = "config/settings.json"
    
    try:
        base_config = load_config(config_path)
    except Exception:
        logger.warning("Could not load config file, using defaults")
        base_config = {}
    
    # Load environment overrides
    env_overrides = load_environment_overrides()
    
    # Merge configurations
    final_config = merge_configs(base_config, env_overrides)
    
    # Validate
    is_valid, errors = validate_config(final_config)
    if not is_valid:
        logger.warning(f"Configuration validation errors: {errors}")
    
    return final_config