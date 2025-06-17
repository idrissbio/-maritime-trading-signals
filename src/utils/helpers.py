import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional, Union
import hashlib
import json
import time
import functools
import logging

logger = logging.getLogger(__name__)

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two coordinates using Haversine formula
    
    Args:
        lat1, lon1: First coordinate pair
        lat2, lon2: Second coordinate pair
        
    Returns:
        Distance in nautical miles
    """
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in nautical miles
    r = 3440.065
    
    return c * r

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to 0-1 range
    
    Args:
        value: Value to normalize
        min_val: Minimum possible value
        max_val: Maximum possible value
        
    Returns:
        Normalized value between 0 and 1
    """
    
    if max_val == min_val:
        return 0.0
    
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))

def moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculate moving average
    
    Args:
        data: List of values
        window: Window size
        
    Returns:
        List of moving averages
    """
    
    if len(data) < window:
        return data.copy()
    
    result = []
    for i in range(len(data)):
        if i < window - 1:
            result.append(data[i])
        else:
            avg = sum(data[i-window+1:i+1]) / window
            result.append(avg)
    
    return result

def exponential_moving_average(data: List[float], alpha: float) -> List[float]:
    """
    Calculate exponential moving average
    
    Args:
        data: List of values
        alpha: Smoothing factor (0-1)
        
    Returns:
        List of exponential moving averages
    """
    
    if not data:
        return []
    
    result = [data[0]]
    
    for i in range(1, len(data)):
        ema = alpha * data[i] + (1 - alpha) * result[-1]
        result.append(ema)
    
    return result

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: List of prices
        period: RSI period
        
    Returns:
        List of RSI values
    """
    
    if len(prices) < period + 1:
        return [50.0] * len(prices)  # Return neutral RSI
    
    # Calculate price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # Separate gains and losses
    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]
    
    # Calculate initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values = [50.0] * (period + 1)  # Fill initial values
    
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    # Calculate subsequent RSI values
    for i in range(period + 1, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
    
    return rsi_values

def calculate_bollinger_bands(prices: List[float], period: int = 20, 
                            std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: List of prices
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    
    if len(prices) < period:
        middle = prices.copy()
        upper = prices.copy()
        lower = prices.copy()
        return upper, middle, lower
    
    middle = moving_average(prices, period)
    
    upper = []
    lower = []
    
    for i in range(len(prices)):
        if i < period - 1:
            upper.append(prices[i])
            lower.append(prices[i])
        else:
            window_data = prices[i-period+1:i+1]
            std = np.std(window_data)
            upper.append(middle[i] + (std_dev * std))
            lower.append(middle[i] - (std_dev * std))
    
    return upper, middle, lower

def detect_support_resistance(prices: List[float], window: int = 10, 
                            threshold: float = 0.02) -> Tuple[List[float], List[float]]:
    """
    Detect support and resistance levels
    
    Args:
        prices: List of prices
        window: Window for local extrema detection
        threshold: Minimum price difference (as percentage)
        
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    
    if len(prices) < window * 2:
        return [], []
    
    support_levels = []
    resistance_levels = []
    
    # Find local minima (support) and maxima (resistance)
    for i in range(window, len(prices) - window):
        
        # Check for local minimum
        is_minimum = all(prices[i] <= prices[j] for j in range(i - window, i + window + 1))
        if is_minimum:
            support_levels.append(prices[i])
        
        # Check for local maximum
        is_maximum = all(prices[i] >= prices[j] for j in range(i - window, i + window + 1))
        if is_maximum:
            resistance_levels.append(prices[i])
    
    # Filter levels by significance
    current_price = prices[-1]
    
    significant_support = [
        level for level in support_levels 
        if abs(level - current_price) / current_price >= threshold
    ]
    
    significant_resistance = [
        level for level in resistance_levels
        if abs(level - current_price) / current_price >= threshold
    ]
    
    return significant_support, significant_resistance

def calculate_volatility(prices: List[float], period: int = 20) -> float:
    """
    Calculate price volatility (standard deviation of returns)
    
    Args:
        prices: List of prices
        period: Period for calculation
        
    Returns:
        Volatility as decimal (e.g., 0.02 = 2%)
    """
    
    if len(prices) < 2:
        return 0.0
    
    # Calculate returns
    returns = [(prices[i] / prices[i-1] - 1) for i in range(1, min(len(prices), period + 1))]
    
    if not returns:
        return 0.0
    
    return np.std(returns)

def format_currency(amount: float, symbol: str = "$") -> str:
    """
    Format currency amount with appropriate symbol and commas
    
    Args:
        amount: Amount to format
        symbol: Currency symbol
        
    Returns:
        Formatted currency string
    """
    
    if abs(amount) >= 1e6:
        return f"{symbol}{amount/1e6:.1f}M"
    elif abs(amount) >= 1e3:
        return f"{symbol}{amount/1e3:.1f}K"
    else:
        return f"{symbol}{amount:,.2f}"

def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format percentage with appropriate sign
    
    Args:
        value: Decimal percentage (0.05 = 5%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    
    percentage = value * 100
    sign = "+" if percentage > 0 else ""
    return f"{sign}{percentage:.{decimal_places}f}%"

def time_ago(timestamp: datetime) -> str:
    """
    Convert timestamp to human-readable time ago format
    
    Args:
        timestamp: Timestamp to convert
        
    Returns:
        Human-readable time difference
    """
    
    now = datetime.now()
    if timestamp.tzinfo:
        now = now.replace(tzinfo=timezone.utc)
    
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"

def generate_hash(data: Any) -> str:
    """
    Generate MD5 hash of data
    
    Args:
        data: Data to hash (will be converted to string)
        
    Returns:
        MD5 hash string
    """
    
    if isinstance(data, dict) or isinstance(data, list):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier
        
    Returns:
        Decorator function
    """
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    delay = backoff_factor * (2 ** attempt)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
            
        return wrapper
    return decorator

def rate_limiter(calls_per_second: float):
    """
    Decorator for rate limiting function calls
    
    Args:
        calls_per_second: Maximum calls per second
        
    Returns:
        Decorator function
    """
    
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        
        return wrapper
    return decorator

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles zero denominator
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    
    if denominator == 0:
        return default
    
    return numerator / denominator

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    
    return max(min_value, min(max_value, value))

def weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted average
    
    Args:
        values: List of values
        weights: List of weights
        
    Returns:
        Weighted average
    """
    
    if len(values) != len(weights) or not values:
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight

def interpolate(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Linear interpolation between two points
    
    Args:
        x: X value to interpolate
        x1, y1: First point
        x2, y2: Second point
        
    Returns:
        Interpolated Y value
    """
    
    if x2 == x1:
        return y1
    
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def is_trading_hours(timestamp: datetime = None) -> bool:
    """
    Check if current time is within trading hours (simplified)
    
    Args:
        timestamp: Timestamp to check (defaults to now)
        
    Returns:
        True if within trading hours
    """
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Convert to UTC if timezone-aware
    if timestamp.tzinfo:
        timestamp = timestamp.astimezone(timezone.utc)
    
    # Basic check: Monday-Friday, 6 PM Sunday to 5 PM Friday ET
    # This is simplified - real implementation would be more complex
    weekday = timestamp.weekday()  # 0 = Monday
    hour = timestamp.hour
    
    # Weekend (Saturday = 5, Sunday = 6)
    if weekday >= 5:
        return False
    
    # Simplified trading hours check
    return True  # For demo purposes, always return True

def get_market_session(timestamp: datetime = None) -> str:
    """
    Determine current market session
    
    Args:
        timestamp: Timestamp to check (defaults to now)
        
    Returns:
        Market session name
    """
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Convert to UTC
    if timestamp.tzinfo:
        timestamp = timestamp.astimezone(timezone.utc)
    
    hour = timestamp.hour
    
    # Simplified session mapping (UTC)
    if 22 <= hour or hour < 6:
        return "asian"
    elif 6 <= hour < 14:
        return "european"
    else:
        return "american"