# prompts/pgkr_prompt.py

import json
from typing import Dict, List, Any
from prompts.base_prompt import BasePrompt


class PGKRPrompt(BasePrompt):
    """
    PGKR (Physics-Guided Knowledge Retrieval) prompt strategy
    
    This strategy uses:
    1. Best configurations from similar PDEs (retrieved via PGKR)
    2. Historical experiment results for the current PDE (sorted by MSE)
    
    The PGKR retrieval finds PDEs with similar physics characteristics and provides their best-performing configurations as references.

    Args:
        search_space: Hyperparameter search space dictionary
    """
    
    def __init__(self, search_space: Dict[str, List]):
        """
        Initialize PGKR prompt
        
        Args:
            search_space: Hyperparameter search space dictionary
        """
        super().__init__(search_space)
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for PGKR strategy"""
        return """You are a PINNs (Physics-Informed Neural Networks) hyperparameter optimization expert.
Your task is to recommend the next experiment's hyperparameter configuration based on:
1. Best configurations from similar PDEs (retrieved via PGKR - Physics-Guided Knowledge Retrieval)
2. Historical experimental results for the current PDE

""" + self.get_hyperparameter_description() + """

The PGKR system retrieves PDEs with similar physics characteristics (e.g., equation type, dimensionality, 
boundary conditions) and provides their best-performing configurations. These serve as strong starting 
points since similar PDEs often benefit from similar hyperparameter choices.

Analyze both the successful configurations from similar PDEs and the current PDE's experimental trends.
Recommend a configuration that is likely to achieve lower MSE for the current PDE.
Return only the configuration in JSON format, without other explanations."""
    
    def build_user_prompt(self, 
        history: List[Dict[str, Any]] = None,
        similar_pdes_configs: Dict[str, Dict[str, Any]] = None
    ) -> str:
        """
        Build user prompt with similar PDEs' configurations and current PDE full history
        
        Args:
            history: Historical experiment records for current PDE
            similar_pdes_configs: Dictionary of similar PDEs' best configurations
                                Format: {
                                    pde_name: {
                                        'similarity': float,
                                        'best_config': dict,
                                        'best_mse': float
                                    }
                                }
            
        Returns:
            User prompt string
        """
        prompt = ""
        
        # Part 1: Similar PDEs' best configurations (from PGKR)
        if similar_pdes_configs:
            prompt += "=" * 80 + "\n"
            prompt += "PART 1: Best Configurations from Similar PDEs (Retrieved via PGKR)\n"
            prompt += "=" * 80 + "\n\n"
            prompt += "The following PDEs have similar physics characteristics to the current target PDE.\n"
            prompt += "Their best-performing configurations provide valuable insights:\n\n"
            
            for pde_name, pde_info in similar_pdes_configs.items():
                similarity = pde_info['similarity']
                best_config = pde_info['best_config']
                best_mse = pde_info['best_mse']
                
                prompt += f"Similar PDE: {pde_name}\n"
                prompt += f"  • Physics Similarity: {similarity:.4f}\n"
                prompt += f"  • Best MSE: {best_mse:.2e}\n"
                prompt += f"  • Best Config:\n"
                for key, value in best_config.items():
                    prompt += f"      - {key}: {value}\n"
                prompt += "\n"
        
        # Part 2: Current PDE's historical experiments
        prompt += "=" * 80 + "\n"
        if similar_pdes_configs:
            prompt += "PART 2: Historical Experimental Results for Current PDE\n"
        else:
            prompt += "Historical Experimental Results for Current PDE\n"
        prompt += "=" * 80 + "\n\n"
        
        if not history:
            # First experiment: use PGKR guidance
            if similar_pdes_configs:
                prompt += "This is the first experiment for this PDE.\n\n"
                prompt += "Please recommend an initial configuration by:\n"
                prompt += "1. Considering the successful configurations from similar PDEs above\n"
                prompt += "2. Applying PINNs best practices\n\n"
                prompt += "Return format:\n"
                prompt += self.get_json_format_example()
            else:
                prompt += "This is the first experiment, please recommend an initial configuration based on PINNs best practices.\n\n"
                prompt += "Return format:\n"
                prompt += self.get_json_format_example()
        else:
            # Sort historical records by MSE
            sorted_history = sorted(history, key=lambda x: x.get("mse", float('inf')))
            
            prompt += "Historical experimental results (sorted by MSE from low to high):\n\n"
            
            for i, exp in enumerate(sorted_history):
                mse = exp.get("mse", "N/A")
                config = exp.get("config", {})
                prompt += f"Experiment {i+1}: MSE = {mse:.2e}\n"
                prompt += f"Config: {json.dumps(config, indent=2)}\n\n"
            
            best_mse = sorted_history[0].get("mse", float('inf'))
            prompt += f"Current best MSE: {best_mse:.2e}\n\n"
            
            # Task description
            if similar_pdes_configs:
                prompt += "Based on:\n"
                prompt += "1. The successful configurations from similar PDEs (Part 1)\n"
                prompt += "2. The experimental trends of the current PDE (Part 2)\n\n"
                prompt += "Recommend the configuration for the next experiment to achieve lower MSE.\n\n"
            else:
                prompt += "Analyze the trends in the above experimental results and recommend the configuration for the next experiment, aiming to achieve lower MSE.\n\n"
            
            prompt += "Return only the configuration in JSON format:"
        
        return prompt