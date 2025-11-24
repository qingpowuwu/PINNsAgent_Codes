# utils/config_loader.py

import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    """Configuration file loader"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../configs/default_config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_fixed_params(self) -> Dict[str, Any]:
        """Get fixed parameters"""
        return self.config.get("fixed_params", {})
    
    def get_pde_list(self, dimension: str) -> list:
        """Get PDE list for specified dimension"""
        return self.config.get("pde_lists", {}).get(dimension, [])
    
    def get_search_space(self) -> Dict[str, list]:
        """Get search space"""
        return self.config.get("search_space", {})
    
    def get_llm_config(self) -> Dict[str, str]:
        """Get LLM configuration"""
        return self.config.get("llm_config", {})
    
    def update_fixed_params(self, **kwargs):
        """Update fixed parameters"""
        self.config["fixed_params"].update(kwargs)