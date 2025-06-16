# qa_toolkit/config_loader.py
import yaml
import json
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def _deep_merge(source: Dict, destination: Dict) -> Dict:
    """Recursively merges source dict into destination dict."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def load_config(default_config_path: str, user_config_path: Optional[str] = None) -> Dict:
    """
    Loads configuration from a default YAML file and optionally merges
    with a user-provided YAML file.
    """
    if not os.path.exists(default_config_path):
        logger.error(f"Default configuration file not found at: {default_config_path}")
        raise FileNotFoundError(f"Default configuration file not found: {default_config_path}")

    with open(default_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded default configuration from {default_config_path}")

    if user_config_path and os.path.exists(user_config_path):
        try:
            with open(user_config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            if user_config: # Ensure user_config is not None
                config = _deep_merge(user_config, config)
                logger.info(f"Successfully loaded and merged user configuration from {user_config_path}")
            else:
                logger.warning(f"User configuration file {user_config_path} is empty. Using defaults.")
        except Exception as e:
            logger.error(f"Error loading user YAML config {user_config_path}: {e}. Using default or partially merged.", exc_info=True)
    elif user_config_path: # Path provided but not found
        logger.warning(f"User config file {user_config_path} not found. Using default configuration.")
    else: # No user path provided
        logger.info("No user config file specified. Using default configuration.")
    
    return config