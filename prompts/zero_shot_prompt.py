# prompts/zero_shot_prompt.py

from typing import Dict, List, Any
from .base_prompt import BasePrompt


class ZeroShotPrompt(BasePrompt):
    """
    Zero-shot prompt strategy
    
    This strategy does not use historical experiment data and relies solely
    on the LLM's general knowledge about PINNs hyperparameters.
    """
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for zero-shot strategy"""
        return """You are a PINNs (Physics-Informed Neural Networks) hyperparameter optimization expert.
Your task is to recommend hyperparameter configurations for PINNs training based on best practices.

""" + self.get_hyperparameter_description() + """

Recommend configurations that are likely to achieve good performance based on general PINNs best practices.
Return only the configuration in JSON format, without other explanations."""
    
    def build_user_prompt(self, history: List[Dict[str, Any]] = None) -> str:
        """
        Build user prompt for zero-shot strategy
        
        Args:
            history: Historical experiment records (ignored in zero-shot)
            
        Returns:
            User prompt string
        """
        # Zero-shot ignores history and always asks for a fresh recommendation
        return """Please recommend a PINNs hyperparameter configuration based on best practices.

Return format:
""" + self.get_json_format_example()