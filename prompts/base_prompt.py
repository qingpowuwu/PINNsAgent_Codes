# prompts/base_prompt.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BasePrompt(ABC):
    """
    Base class for prompt generation strategies
    
    All prompt strategies should inherit from this class and implement
    the abstract methods for system prompt and user prompt generation.
    """
    
    def __init__(self, search_space: Dict[str, List], max_iterations: int = None):
        """
        Initialize base prompt
        
        Args:
            search_space: Hyperparameter search space dictionary
            max_iterations: Maximum number of iterations (optional, used by some strategies)
        """
        self.search_space = search_space
        self.max_iterations = max_iterations
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Generate system prompt
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def build_user_prompt(self, history: List[Dict[str, Any]] = None) -> str:
        """
        Build user prompt based on historical data
        
        Args:
            history: Historical experiment records
            
        Returns:
            User prompt string
        """
        pass
    
    def get_hyperparameter_description(self) -> str:
        """
        Get hyperparameter descriptions dynamically from search_space
        
        Returns:
            Hyperparameter description string
        """
        descriptions = []
        
        # Define descriptions for each parameter type
        param_descriptions = {
            "activation": "Activation function",
            "net": "Network architecture",
            "optimizer": "Optimizer",
            "lr": "Learning rate",
            "width": "Network width",
            "depth": "Network depth",
            "num_domain_points": "Domain sampling points",
            "num_boundary_points": "Boundary sampling points",
            "num_initial_points": "Initial condition sampling points",
            "initializer": "Weight initialization"
        }
        
        descriptions.append("Hyperparameter descriptions:")
        
        for param_name, param_values in self.search_space.items():
            if param_name in param_descriptions:
                desc = param_descriptions[param_name]
                
                # Format the values based on parameter type
                if param_name in ["lr"]:
                    # Learning rate: show as is (already strings like '1e-3')
                    values_str = ", ".join(str(v) for v in param_values)
                elif param_name in ["width", "depth", "num_domain_points", 
                                   "num_boundary_points", "num_initial_points"]:
                    # Numeric ranges: show min to max
                    min_val = min(param_values)
                    max_val = max(param_values)
                    
                    # Detect increment for cleaner description
                    if len(param_values) > 1:
                        increments = set()
                        sorted_values = sorted(param_values)
                        for i in range(len(sorted_values) - 1):
                            increments.add(sorted_values[i+1] - sorted_values[i])
                        
                        if len(increments) == 1:
                            # Uniform increment
                            increment = list(increments)[0]
                            values_str = f"{min_val} to {max_val}, increment: {increment}"
                        else:
                            # Non-uniform, just show range
                            values_str = f"{min_val} to {max_val}"
                    else:
                        values_str = str(param_values[0])
                else:
                    # Categorical parameters: list all options
                    values_str = ", ".join(str(v) for v in param_values)
                
                descriptions.append(f"- {param_name}: {desc} ({values_str})")
        
        return "\n".join(descriptions)
    
    def get_json_format_example(self) -> str:
        """
        Get JSON format example (shared across all strategies)
        
        Returns:
            JSON format example string
        """
        return """{
  "activation": "tanh",
  "net": "fnn",
  "optimizer": "adam",
  "lr": 1e-3,
  "width": 128,
  "depth": 3,
  "num_domain_points": 4100,
  "num_boundary_points": 1100,
  "num_initial_points": 1100,
  "initializer": "Glorot normal"
}"""