# prompts/memory_tree_prompt.py
from pathlib import Path
import os
import sys
current_dir = Path(__file__).parent.absolute()
pinnsagent_root = current_dir.parent
os.chdir(pinnsagent_root)
sys.path.insert(0, str(pinnsagent_root))

import json
from typing import Dict, List, Any
from prompts.base_prompt import BasePrompt


class MemoryTreePrompt(BasePrompt):
    """
    Memory Tree prompt strategy
    
    This strategy uses:
    1. UCT scores from MemoryTree to guide parameter selection
    2. Historical experiment results for the current PDE (sorted by MSE)
    
    The UCT scores indicate which parameters are most promising to explore based on historical data across all PDEs.

    Args:
        search_space: Hyperparameter search space dictionary
    """
    
    def __init__(self, search_space: Dict[str, List]):
        """
        Initialize Memory Tree prompt
        
        Args:
            search_space: Hyperparameter search space dictionary
        """
        super().__init__(search_space)
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for Memory Tree strategy"""
        return """You are a PINNs (Physics-Informed Neural Networks) hyperparameter optimization expert.
Your task is to recommend the next experiment's hyperparameter configuration based on:
1. UCT scores from historical data (indicating which parameters are most promising to explore)
2. Historical experimental results for the current PDE

""" + self.get_hyperparameter_description() + """

The UCT scores guide you on which parameters to prioritize when making changes.
Higher scores indicate parameters that are more important to adjust for improving performance.

Analyze both the UCT scores and the current PDE's experimental trends.
Recommend a configuration that is likely to achieve lower MSE.
Return only the configuration in JSON format, without other explanations."""
    
    def build_user_prompt(self, 
        history: List[Dict[str, Any]] = None,
        exploration_scores: Dict[str, float] = None
    ) -> str:
        """
        Build user prompt with UCT scores and current PDE history
        
        Args:
            history: Historical experiment records for current PDE
            exploration_scores: Dictionary of {param_name: uct_score} from MemoryTree
                              Scores are sorted by importance (higher = more important)
            
        Returns:
            User prompt string
        """
        prompt = ""
        
        # Part 1: UCT Scores
        if exploration_scores:
            prompt += "=" * 80 + "\n"
            prompt += "PART 1: UCT Scores (Parameter Importance Ranking)\n"
            prompt += "=" * 80 + "\n\n"
            prompt += "The following scores indicate which parameters are most promising to explore.\n"
            prompt += "Higher scores mean the parameter should receive more attention when tuning.\n"
            prompt += "Focus on adjusting high-score parameters first.\n\n"
            
            # Find max score for scaling the visualization bar
            max_score = max(exploration_scores.values()) if exploration_scores else 1.0
            
            for param_name, score in exploration_scores.items():
                # Scale bar based on relative score
                bar_length = int((score / max_score) * 20) if max_score > 0 else 0
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                prompt += f"{param_name:25s}: {score:.4f} {bar}\n"
            
            prompt += "\n"
            
            # Add interpretation guide
            prompt += "Interpretation:\n"
            prompt += "- Higher scores: Parameters that should receive more attention during tuning\n"
            prompt += "- Lower scores: Parameters that are relatively stable\n\n"
        
        # Part 2: Current PDE's historical experiments
        prompt += "=" * 80 + "\n"
        if exploration_scores:
            prompt += "PART 2: Historical Experimental Results for Current PDE\n"
        else:
            prompt += "Historical Experimental Results for Current PDE\n"
        prompt += "=" * 80 + "\n\n"
        
        if not history:
            # First experiment: use UCT scores + best practices
            if exploration_scores:
                prompt += "This is the first experiment for this PDE.\n\n"
                prompt += "Recommendation strategy:\n"
                prompt += "1. Review the UCT scores above - prioritize exploring high-score parameters\n"
                prompt += "2. For high-score parameters, choose values that historically performed well\n"
                prompt += "3. For low-score parameters, use standard/default values\n"
                prompt += "4. Apply PINNs best practices\n\n"
                
                # Suggest focusing on top parameters
                top_params = list(exploration_scores.keys())[:3]
                prompt += f"🎯 Suggestion: Pay special attention to these top-3 parameters: {', '.join(top_params)}\n\n"
                
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
            best_config = sorted_history[0].get("config", {})
            prompt += f"Current best MSE: {best_mse:.2e}\n"
            prompt += f"Current best config: {json.dumps(best_config, indent=2)}\n\n"
            
            # Task description with UCT guidance
            if exploration_scores:
                prompt += "Recommendation strategy:\n"
                prompt += "1. Review the UCT scores (Part 1) - they tell you which parameters matter most\n"
                prompt += "2. Analyze the experimental trends (Part 2) - identify what worked and what didn't\n"
                prompt += "3. Prioritize adjusting high-score parameters from the current best configuration\n"
                prompt += "4. Keep low-score parameters stable (they have less impact)\n\n"
                
                # Suggest focusing on top parameters
                top_params = list(exploration_scores.keys())[:3]
                prompt += f"🎯 Suggestion: Focus on exploring these top-3 parameters: {', '.join(top_params)}\n"
                prompt += f"   Consider trying different values for them while keeping other parameters closer to the best config.\n\n"
                
                prompt += "Based on the above analysis, recommend the configuration for the next experiment to achieve lower MSE.\n\n"
            else:
                prompt += "Analyze the trends in the above experimental results and recommend the configuration for the next experiment, aiming to achieve lower MSE.\n\n"
            
            prompt += "Return only the configuration in JSON format:"
        
        return prompt