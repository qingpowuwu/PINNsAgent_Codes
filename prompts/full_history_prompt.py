# prompts/full_history_prompt.py

import json
from typing import Dict, List, Any
from .base_prompt import BasePrompt


class FullHistoryPrompt(BasePrompt):
    """
    Full history prompt strategy
    
    This strategy uses all historical experiment results to guide the next configuration recommendation. (It sorts experiments results by MSE and does not provide any additional analysis as a guidance)
    """
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for full history strategy"""
        return """You are a PINNs (Physics-Informed Neural Networks) hyperparameter optimization expert.
Your task is to recommend the next experiment's hyperparameter configuration based on historical experimental results.

""" + self.get_hyperparameter_description() + """

Based on trends in historical experimental results, recommend a configuration that is likely to achieve lower MSE.
Return only the configuration in JSON format, without other explanations."""
    
    def build_user_prompt(self, history: List[Dict[str, Any]] = None) -> str:
        """
        Build user prompt with full history
        
        Args:
            history: Historical experiment records
            
        Returns:
            User prompt string
        """
        if not history:
            # First experiment: use best practices
            return """This is the first experiment, please recommend an initial configuration based on PINNs best practices.

Return format:
""" + self.get_json_format_example()
        
        # Sort historical records by MSE
        sorted_history = sorted(history, key=lambda x: x.get("mse", float('inf')))
        
        prompt = "Historical experimental results (sorted by MSE from low to high):\n\n"
        
        for i, exp in enumerate(sorted_history):
            mse = exp.get("mse", "N/A")
            config = exp.get("config", {})
            prompt += f"Experiment {i+1}: MSE = {mse:.2e}\n"
            prompt += f"Config: {json.dumps(config, indent=2)}\n\n"
        
        best_mse = sorted_history[0].get("mse", float('inf'))
        prompt += f"Current best MSE: {best_mse:.2e}\n\n"
        
        prompt += """Analyze the trends in the above experimental results and recommend the configuration for the next experiment, aiming to achieve lower MSE.

Return only the configuration in JSON format:"""
        
        return prompt