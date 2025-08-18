"""
Utility functions for MAGMA Platform
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimal places"""
    try:
        return f"{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    try:
        return f"{value:.2f}%"
    except (ValueError, TypeError):
        return str(value)

def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary"""
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default

def log_api_call(api_name: str, endpoint: str, status: str = "success"):
    """Log API call for monitoring"""
    logger.info(f"API Call: {api_name} - {endpoint} - {status}")

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return timestamp
